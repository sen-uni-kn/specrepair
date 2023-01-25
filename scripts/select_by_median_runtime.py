import argparse
import os
import shutil
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from collect_experiment_results import ExperimentCase, ExperimentResult


float_value_rx = re.compile(r'(?P<value>-?\d+(\.\d+)?)')
bracket_name_rx = re.compile(r'\((?P<name>[^)]*)\)')


def loss_string_to_dict(loss_string) -> Dict[str, float]:
    # format: 123.4567 (name); 42.7712 (name2);
    return dict(
        (
            bracket_name_rx.search(val_str).group('name'),
            # is at start => match works fine
            float(float_value_rx.match(val_str).group('value'))
        )
        for val_str in map(str.strip, loss_string.split(';')) if val_str != ''
    )


def get_grouped_dir_name(arguments_, args_):
    arg_strs = []
    for argument_, value_ in zip(args.group_by, arguments_):
        if argument_ == "falsifiers":
            if value_ == "None":
                continue
            else:
                arg_str = value_.replace("[", "_").replace(",", "_").replace("]", "")
                arg_strs.append(arg_str)
        else:
            arg_strs.append(value_)
    prefix = args_.output_prefix + "_" if len(args_.output_prefix) > 0 else ""
    return prefix + "_".join(arg_strs)


def maxdev(series):
    return series.max() - series.min()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Group experiment runs by arguments and select the cases with"
        "the runtime next larger to the median runtime from each group of runs."
    )
    parser.add_argument("--experiment_dirs", nargs="+",
                        help="The directories containing the experiment runs. "
                             "The experiments need not be collected. "
                             "Try globs! For example output/acasxu_repair_1/2022-*")
    parser.add_argument("--group_by", nargs="+",
                        help="Arguments to group runs by. "
                             "For example --group_by verifier will group all "
                             "runs with the same verifier together and select "
                             "the cases with the median runtime among those "
                             "runs with the same verifier. "
                             "Using multiple arguments, such as "
                             "--group_by verifier falsifiers will group by all "
                             "combinations of the arguments.")
    parser.add_argument("--output_path",
                        help="Where to place the directories containing the "
                             "selected cases for each group.")
    parser.add_argument("--output_prefix", default="",
                        help="A prefix to add to the created directories in "
                             "the --output_path.")
    parser.add_argument("--loss_tolerance", default=0.01, type=float,
                        help="The threshold at which to emit a warning " 
                              "when loss values deviate by more than.")
    args = parser.parse_args()

    group_by_argument = args.group_by
    argument_rxs = [
        re.compile(rf' *({argument}): (?P<name>.*)')
        for argument in group_by_argument
    ]
    total_runtime_rx = re.compile(
        r"Executing overall repair took: (?P<time>[0-9]+\.[0-9]+) seconds\."
    )
    final_loss_rx = re.compile(
        r'Final loss: +(?P<values>(-?[0-9]+\.[0-9]+ \(.*\);?)+)'
    )
    repair_network_iteration_rx = re.compile(
        r'repair_network iteration (?P<repair_step>[0-9]+)'
    )
    repair_successful_rx = re.compile(
        r'(All properties verified! Repair successful\.)'
        r'|(Repair finished: success)'
    )
    backend_failure_rx = re.compile(
        r'(Backend could not repair counterexamples. Aborting repair\.)'
        r'|(Repair finished: failure)'
    )
    max_iterations_failure_rx = re.compile(
        r'Repair failed: Maximum number of iterations exhausted\.')
    verification_problem_rx = re.compile(
        r'The following properties could not be verified due to errors: '
        r'\[(?P<properties>.*)]'
    )
    timeout_rx = re.compile(r'Experiment timed out')

    num_warnings = 0
    print("Collecting experiment cases...")
    raw_cases = []
    for output_dir in tqdm(args.experiment_dirs, desc="output dirs"):
        if 'log.txt' in os.listdir(output_dir):
            # top level dir contains log file => is experiment case output dir itself
            raw_cases.append(ExperimentCase(name='main', dir=Path(output_dir)))
        else:
            for dir_element in os.listdir(output_dir):
                dir_element_path = Path(output_dir, dir_element)
                if dir_element_path.is_dir():
                    raw_cases.append(ExperimentCase(
                        name=dir_element, dir=dir_element_path)
                    )
                else:
                    print(f'Ignoring unknown directory element: {dir_element_path}')

    # arguments to (case names to list of cases)
    cases: Dict[Tuple[str, ...], Dict[str, List[ExperimentCase]]] = \
        defaultdict(lambda: defaultdict(list))

    print("Reading logfiles...")
    for experiment_case in tqdm(raw_cases, desc="total cases"):
        log_file_path = Path(experiment_case.dir, "log.txt")
        arguments = []
        repair_steps = None
        with open(log_file_path) as log_file:
            line = log_file.readline()
            while len(line) > 0:
                for argument_rx in argument_rxs:
                    match = argument_rx.match(line)
                    if match:
                        arguments.append(match.group("name"))

                match = total_runtime_rx.match(line)
                if match:
                    total_runtime = match.group("time")
                    experiment_case.total_runtime = float(total_runtime)

                match = final_loss_rx.match(line)
                if match:
                    loss_dict = loss_string_to_dict(match.group("values"))
                    experiment_case.final_loss = loss_dict

                match = repair_network_iteration_rx.match(line)
                if match:
                    repair_steps = int(match.group("repair_step"))

                if repair_successful_rx.match(line):
                    experiment_case.experiment_result = ExperimentResult.SUCCESS
                elif backend_failure_rx.match(line):
                    experiment_case.experiment_result = ExperimentResult.BACKEND_FAILURE
                elif max_iterations_failure_rx.match(line):
                    experiment_case.experiment_result = \
                        ExperimentResult.MAX_ITERATIONS_FAILURE
                elif verification_problem_rx.match(line):
                    experiment_case.experiment_result = \
                        ExperimentResult.VERIFICATION_PROBLEM
                elif timeout_rx.match(line):
                    experiment_case.experiment_result = ExperimentResult.TIMEOUT
                line = log_file.readline()

        # when the experiment_result wasn't set after reading the entire log file,
        # then the experiment was externally aborted.
        if experiment_case.experiment_result is None:
            experiment_case.experiment_result = ExperimentResult.ABORTED
        else:
            # only enter repair steps for orderly ends
            experiment_case.repair_steps = repair_steps + 1

        cases[tuple(arguments)][experiment_case.name].append(experiment_case)

    num_replicas_counts = defaultdict(int)
    for _, case_group in cases.items():
        for _, case_replicas in case_group.items():
            num_replicas_counts[len(case_replicas)] += 1

    if len(num_replicas_counts) != 1:
        most_frequent_count = max(
            num_replicas_counts, key=lambda num: num_replicas_counts[num]
        )
        print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: not all cases were repeated equally often!")
        print(f"Most frequent count: {most_frequent_count}. "
              f"Listing all cases with other counts.")
        for arguments, case_group in cases.items():
            for case_name, case_replicas in case_group.items():
                if len(case_replicas) != most_frequent_count:
                    print(f" - {arguments}, {case_name} ({len(case_replicas)})")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        num_warnings += 1

    print("Selecting cases...")
    runtime_spreads = []
    for arguments, cases_group in cases.items():
        arguments_dir_name = get_grouped_dir_name(arguments, args)
        arguments_output_path = Path(args.output_path, arguments_dir_name)
        arguments_output_path.mkdir(exist_ok=False, parents=True)
        median_runtimes: Dict[str, float] = {}

        for case_name, case_replicas in cases_group.items():
            runtimes = pd.Series([
                case.total_runtime for case in case_replicas
                if case.total_runtime is not None
            ])
            losses = pd.DataFrame([
                case.final_loss for case in case_replicas
                if case.final_loss is not None
            ])
            repair_steps = [
                case.repair_steps for case in case_replicas
                if case.repair_steps is not None
            ]
            results = [case.experiment_result for case in case_replicas]

            median_runtime = runtimes.median()
            median_runtimes[case_name] = median_runtime

            runtime_spread = maxdev(runtimes)
            runtime_spreads.append(runtime_spread)

            num_timeout = sum(
                result in (ExperimentResult.TIMEOUT, ExperimentResult.ABORTED)
                for result in results
            )
            non_timeout_results = [
                result for result in results
                if result not in (ExperimentResult.TIMEOUT, ExperimentResult.ABORTED)
            ]
            unique_non_timeout = set(non_timeout_results)

            unique_repair_steps = set(repair_steps)

            print("-------------------------------------------------------------------")
            print(f"{arguments_dir_name}: {case_name}")
            if len(unique_non_timeout) > 1:
                print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("WARNING: experiment replicas had different results!")
                print("Results: " + ", ".join(str(r) for r in unique_non_timeout))
                for case in case_replicas:
                    print(f"{case.dir}: {case.experiment_result}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                num_warnings += 1
            elif len(unique_non_timeout) == 0:
                print(f"Result: TIMEOUT")
            else:
                print(f"Result: {next(iter(unique_non_timeout))}, {num_timeout} timeouts")

            if len(unique_repair_steps) > 1:
                print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("WARNING: experiment replicas differed in number of repair steps!")
                print("Repair step counts: " + ", ".join(str(rs) for rs in repair_steps))
                for case in case_replicas:
                    print(f"{case.dir}: {case.repair_steps}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                num_warnings += 1

            loss_ranges = [(loss_name, maxdev(losses[loss_name])) for loss_name in losses]
            print("Loss Range: " + "  ".join(
                f"{loss_name}: {loss_range:.4f}"
                for loss_name, loss_range in loss_ranges)
            )
            for loss_name, loss_range in loss_ranges:
                if loss_range >= args.loss_tolerance:
                    print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
                          "WARNING")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                          "!!!!!!")
                    print(f"WARNING: loss {loss_name} had range >= {args.loss_tolerance}!")
                    print(f"{loss_name}: {loss_range:.4f}")
                    print(losses[loss_name])
                    for case in case_replicas:
                        print(f"{case.dir}: {case.final_loss}")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                          "!!!!!!")
                    num_warnings += 1

            selected_case: Optional[ExperimentCase] = None
            if num_timeout >= len(case_replicas) // 2.0:
                # select a run with timeout as representative
                for case in case_replicas:
                    if case.experiment_result in (
                        ExperimentResult.TIMEOUT, ExperimentResult.ABORTED
                    ):
                        selected_case = case
                        break
                print(f"Selecting {selected_case.dir} (TIMEOUT)")
            else:
                # select the case with the smallest runtime that is at least as large
                # as the median runtime
                closest_runtime = float("inf")
                for case in case_replicas:
                    if closest_runtime > case.total_runtime >= median_runtime:
                        closest_runtime = case.total_runtime
                        selected_case = case
                print(f"Selecting {selected_case.dir} "
                      f"({selected_case.total_runtime} >= {median_runtime})")

            shutil.copytree(
                selected_case.dir,
                arguments_output_path / case_name,
                dirs_exist_ok=False
            )

        median_runtimes_df = pd.DataFrame(
            median_runtimes.items(), columns=("case", "median_runtime")
        )
        median_runtimes_df.to_csv(
            arguments_output_path / "median_runtimes.csv", index=False,
        )

    print("===================================================================")
    print("Selecting runs finished.")
    if len(num_replicas_counts) == 0:
        print(f"Replica count: {next(iter(num_replicas_counts.keys()))}")
    else:
        print(f"Replica counts: {list(num_replicas_counts.keys())}")

    runtime_spreads = pd.Series(runtime_spreads)
    print(
        "Runtime range summary: "
        f"min: {runtime_spreads.min():.4f}, "
        f"10% quantile: {runtime_spreads.quantile(0.1):.4f}, "
        f"25% quantile: {runtime_spreads.quantile(0.25):.4f}, "
        f"median: {runtime_spreads.median():.4f}, "
        f"75% quantile: {runtime_spreads.quantile(0.75):.4f}, "
        f"90% quantile: {runtime_spreads.quantile(0.9):.4f}, "
        f"max: {runtime_spreads.max():.4f}\n"
        f"mean: {runtime_spreads.mean():.4f}, "
        f"std: {runtime_spreads.std():.4f}"
    )

    if num_warnings > 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Emitted {num_warnings} warnings! Check the output.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("No warnings emitted.")
