import argparse
import itertools
import os
from collections import defaultdict
from math import sqrt, ceil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ruamel.yaml as yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create Cactus-Plots for runtime and number of repair steps"
    )
    parser.add_argument("--optimal", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'optimal' verifier exit strategy.")
    parser.add_argument("--early_exit", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'early-exit' verifier exit strategy.")
    parser.add_argument("--switch1", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'switch1' verifier exit strategy.")
    parser.add_argument("--switch3", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'switch3' verifier exit strategy.")
    parser.add_argument("--switch5", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'switch5' verifier exit strategy.")
    parser.add_argument("--runtime_threshold", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'runtime_threshold' verifier exit strategy.")
    parser.add_argument("--runtime_threshold_decrease", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the 'runtime_threshold_decrease' verifier exit strategy.")
    parser.add_argument("--deep_opt_early_exit", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the deep_opt falsifier and the early-exit verifier.")
    parser.add_argument("--deep_opt_optimal", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the deep_opt falsifier and the optimal verifier.")
    parser.add_argument("--pgd_early_exit", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the PGD falsifier and the early-exit verifier.")
    parser.add_argument("--pgd_optimal", type=str, default=None,
                        help="The experiment case directory for the run with "
                             "the PGD falsifier and the optimal verifier.")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="The timeout of the experiment in hours. "
                             "Used for plotting. ")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where to place the produced csv files.")
    parser.add_argument("--export", action="store_true",
                        help="Export the data")
    parser.add_argument("--no_plot", action="store_true",
                        help="Don't plot the results.")
    parser.add_argument("--runtime_tolerance", type=float, default=30,
                        help="The threshold up to which to runtimes are considered "
                             "equal (in seconds).")
    args = parser.parse_args()

    successful_cases = {}
    all_cases = set()
    runtime = {}
    runtime_for_backend = {}
    runtime_for_cx_search = {}
    runtime_for_cx_search_first_repair_step = {}
    runtime_cx_search_by_tool = {}
    runtime_for_cx_search_timeseries = {}
    cx_search_names = ["ERAN", "DeepOpt", "FGSM", "PGDA+SDG", "PGDA+RMSprop", "PGDA+Adam"]
    repair_steps = {}
    cx_violations_at_generation = {}
    cx_violation_in_first_repair_step = {}
    cx_violation_in_last_repair_step = {}
    repaired_network_performance = {}
    newly_introduced_cx = {}

    for method in (
        "optimal", "early_exit", "switch1", "switch3", "switch5",
        "runtime_threshold", "runtime_threshold_decrease",
        "deep_opt_early_exit", "deep_opt_optimal", "pgd_early_exit", "pgd_optimal"
    ):
        directory = getattr(args, method)
        if directory is None:
            continue
        print(f"Reading data for {method}.")

        result_h5 = Path(directory, "results.h5")
        result_yaml = Path(directory, "results.yaml")
        with open(result_yaml) as file:
            results_overview = yaml.safe_load(file)

        cases = tuple(results_overview["cases"].keys())
        all_cases.update(cases)

        median_runtimes_file = Path(directory, "median_runtimes.csv")
        if median_runtimes_file.exists():
            print(f"Using stored median runtimes for {method}.")
            median_runtimes = pd.read_csv(median_runtimes_file)
            median_runtimes = dict(median_runtimes.iloc)
        else:
            median_runtimes = None

        grid_fidelities = None
        for file_name in os.listdir(directory):
            if file_name.startswith("grid_fidelity_calculation"):
                grid_fidelity_file = Path(directory, file_name)
                print(f"Using grid fidelity file: {grid_fidelity_file}")
                grid_fidelities = pd.read_csv(grid_fidelity_file)
                # convert to the ACAS XU experiment case names
                grid_fidelities = {
                    (f"property_"
                     f"{row['specification'].replace('(','').replace(')','').replace(',','_')[:-1]}_"
                     f"net_{row['network_i1']}_{row['network_i2']}"): {
                        "accuracy_fidelity": row["accuracy_fidelity"],
                        "mae_fidelity": row["mae_fidelity"],
                    }
                    for row in grid_fidelities.iloc
                }

        run_successful_cases = []
        run_repair_steps = {}
        run_runtimes = {}
        run_runtimes_for_backend = {}
        run_runtimes_for_cx_search = {}
        run_runtimes_for_cx_search_first_repair_step = {}
        run_runtime_cx_search_by_tool = defaultdict(list)
        run_runtime_for_cx_search_timeseries = {}
        run_cx_violations = {}
        run_cx_violations_first = {}
        run_cx_violations_last = {}
        run_repaired_network_performance = defaultdict(dict)
        run_newly_introduced_cx = {}
        for case in cases:
            if results_overview["cases"][case]["result"] != "success":
                continue

            run_successful_cases.append(case)
            run_repair_steps[case] = results_overview["cases"][case]["repair_steps"]
            cx_info = results_overview["cases"][case]["counterexamples"]
            run_newly_introduced_cx[case] = cx_info["newly_introduced"] / cx_info["total"]

            run_final_losses = results_overview["cases"][case]["final_loss"]
            if run_final_losses is not None:
                for key in (
                    # MNIST, CIFAR10
                    "training set loss", "training set accuracy",
                    "test set loss", "test set accuracy",
                    # ACAS Xu
                    "training loss", "validation loss",
                    # CollisionDetection
                    "training set", "test set", "test accuracy",
                    # C-MAPSS
                    "training MSE", "training RMSE", "test MSE", "test RMSE",
                    # Ouroboros RMI
                    "MSE", "accuracy", "MAE", "max error",
                ):
                    if key in run_final_losses:
                        run_repaired_network_performance[key][case] = run_final_losses[key]

            # Separately calculated training set accuracy.
            # Useful if you want to use a performance measure that wasn't calculated
            # in the experiment.
            if "final_training_set_accuracy" in results_overview["cases"][case]:
                run_repaired_network_performance["recalculated training set accuracy"][case] = \
                    results_overview["cases"][case]["final_training_set_accuracy"]

            if grid_fidelities is not None:
                for key, value in grid_fidelities[case].items():
                    run_repaired_network_performance[key][case] = value

            if len(run_repaired_network_performance) == 0:
                run_repaired_network_performance = None

            case_runtimes: Dict[str, Dict[int, str]] = \
                results_overview["cases"][case]["runtimes"]
            case_runtimes = {
                task: {
                    rs: float(rt)
                    for rs, rt in case_runtimes[task].items()
                } for task in case_runtimes
            }
            run_runtime_for_cx_search_timeseries[case] = [
                sum(rts)  # total runtime for cx search in each repair step
                for rs, rts in sorted(
                    # group the runtimes by their repair step
                    [(rs, [rt for _, rt in rts]) for rs, rts in itertools.groupby(
                        itertools.chain(
                            *[
                                ds.items()  # make (rs, rt) tuples
                                # get all repair step to runtime mappings
                                for task, ds in case_runtimes.items()
                                if any(task.startswith(cx_searcher) for cx_searcher in cx_search_names)
                            ]
                        ),
                        key=lambda t: t[0]
                    )],
                    key=lambda t: t[0]  # sort by repair step
                )
            ]
            if median_runtimes is not None:
                run_runtimes[case] = median_runtimes[case]
            else:
                total_runtime = results_overview["cases"][case]["total_runtime"]
                run_runtimes[case] = float(total_runtime)
            repair_backend_runtime = sum(rt for rt in case_runtimes["repair backend"].values())
            run_runtimes_for_backend[case] = repair_backend_runtime
            run_runtimes_for_cx_search[case] = 0.0
            # there may be multiple cx searches in the first repair step when there are
            # multiple properties.
            run_runtimes_for_cx_search_first_repair_step[case] = 0.0
            count_cx_searchers = 0
            tasks = case_runtimes.keys()
            for cx_searcher in cx_search_names:
                # In the YAML file, the cx search tasks aren't summarized.
                # For example "ERAN" will have multiple entries, one for each property.
                matching_tasks = [task for task in tasks if task.startswith(cx_searcher)]

                for task in matching_tasks:
                    all_task_runtimes = case_runtimes[task]
                    if 0 in all_task_runtimes:
                        run_runtimes_for_cx_search_first_repair_step[case] += float(all_task_runtimes[0])
                    total_searcher_runtime = sum(rt for rt in all_task_runtimes.values())
                    run_runtimes_for_cx_search[case] += total_searcher_runtime
                    run_runtime_cx_search_by_tool[cx_searcher].append(total_searcher_runtime)
                    count_cx_searchers += 1

            if count_cx_searchers == 0:
                print(f"WARNING: experiment with {method} lacks a "
                      f"cx search runtime for case {case}!")

            case_cx_violations: pd.Series = pd.read_hdf(
                result_h5, f"cases/{case}/counterexample_violations_at_generation"
            )
            run_cx_violations[case] = case_cx_violations.to_list()
            run_cx_violations_first[case] = case_cx_violations[0]
            run_cx_violations_last[case] = case_cx_violations[-1]

        successful_cases[method] = frozenset(run_successful_cases)
        repair_steps[method] = run_repair_steps
        runtime[method] = median_runtimes if run_runtimes is None else run_runtimes
        runtime_for_backend[method] = run_runtimes_for_backend
        runtime_for_cx_search[method] = run_runtimes_for_cx_search
        runtime_for_cx_search_first_repair_step[method] = \
            run_runtimes_for_cx_search_first_repair_step
        runtime_cx_search_by_tool[method] = run_runtime_cx_search_by_tool
        runtime_for_cx_search_timeseries[method] = run_runtime_for_cx_search_timeseries
        cx_violations_at_generation[method] = run_cx_violations
        cx_violation_in_first_repair_step[method] = run_cx_violations_first
        cx_violation_in_last_repair_step[method] = run_cx_violations_last
        repaired_network_performance[method] = run_repaired_network_performance
        newly_introduced_cx[method] = run_newly_introduced_cx

    successful = list(successful_cases.values())
    max_successful = max(len(successes) for successes in successful)
    all_successful = successful[0].intersection(*successful[1:])

    print("Aggregating Data...")

    success_rate = {
        method: len(cases) / len(all_cases)
        for method, cases in successful_cases.items()
    }

    all_successful_median_repaired_network_performance = {
        method: {
            name: pd.Series([
                value for case, value in case_values.items()
                if case in all_successful
            ]).median()
            for name, case_values in loss_values.items()
        }
        for method, loss_values in repaired_network_performance.items()
    }

    repair_steps_sorted = {
        method: sorted(data.values())
        for method, data in repair_steps.items()
    }

    timeout_value = args.timeout * 60 * 60
    runtime_sorted = {
        method: sorted(data.values())
        + [timeout_value + 60] * (len(all_cases) - len(data))
        for method, data in runtime.items()
    }

    cases_order = sorted(all_cases, key=lambda case_: 0 if case_ in all_successful else 1)
    runtime_matching = {
        method: [data[case] if case in data else timeout_value + 60
                 for case in cases_order]
        for method, data in runtime.items()
    }
    max_repair_step = max(max(data.values()) for data in repair_steps.values())
    repair_steps_matching = {
        method: [data[case] if case in data else max_repair_step + 10
                 for case in cases_order]
        for method, data in repair_steps.items()
    }

    cases_order_2 = sorted(all_cases)
    runtime_minimal = [
        min([data[case] for data in runtime.values() if case in data] + [timeout_value + 60])
        for case in cases_order_2
    ]

    def get_unique_best(candidates):
        if len(candidates) == 0:
            return "None"
        elif len(candidates) == 1:
            return candidates[0]
        else:
            return "Multiple"

    runtime_tolerance = args.runtime_tolerance
    runtime_best = {
        case: get_unique_best([
            method for method, data in runtime.items()
            if case in data and (
                -runtime_tolerance <= data[case] - runtime_minimal[i] <= runtime_tolerance
            )
        ])
        for i, case in enumerate(cases_order_2)
    }
    runtime_second_to_minimal = [
        min([
            data[case] for method, data in runtime.items()
            if case in data and method != runtime_best[case]
        ] + [timeout_value])
        for i, case in enumerate(cases_order_2)
    ]
    runtime_distance_to_second_best = {
        method: [
            0.0 if method != runtime_best[case]
            else runtime_second_to_minimal[i] - runtime_minimal[i]
            for i, case in enumerate(cases_order_2)
        ]
        for method, data in runtime.items()
    }
    runtime_fastest = {
        method: sum(best_method == method for best_method in runtime_best.values())
        for method in itertools.chain(runtime.keys(), ("None", "Multiple"))
    }
    runtime_fastest_percent = {
        method: 100 * fastest_count / len(all_cases)
        for method, fastest_count in runtime_fastest.items()
    }

    # print(cases_order_2)
    # print(runtime_distance_to_second_best)

    repair_steps_minimal = [
        # max_repair_step + 10 is the indicator for no successful repair
        min([data[case] for data in repair_steps.values() if case in data] + [max_repair_step + 10])
        for case in cases_order_2
    ]
    repair_steps_best = [
        get_unique_best([
            method for method, data in repair_steps.items()
            if case in data and data[case] == repair_steps_minimal[i]
        ])
        for i, case in enumerate(cases_order_2)
    ]
    repair_steps_second_to_minimal = [
        min([
            data[case] for method, data in repair_steps.items()
            if case in data and method != repair_steps_best[i]
        ] + [max_repair_step + 10])
        for i, case in enumerate(cases_order_2)
    ]
    repair_steps_distance_to_second_best = {
        method: [
            0.0 if method != repair_steps_best[i]
            else repair_steps_second_to_minimal[i] - repair_steps_minimal[i]
            for i, case in enumerate(cases_order_2)
        ]
        for method, data in repair_steps.items()
    }
    repair_steps_fastest = {
        method: sum(best_method == method for best_method in repair_steps_best)
        for method in itertools.chain(repair_steps.keys(), ("None", "Multiple"))
    }
    repair_steps_fastest_percent = {
        method: 100 * fastest_count / len(all_cases)
        for method, fastest_count in repair_steps_fastest.items()
    }

    repaired_network_performance_comparable = {
        method: {
            loss_name: [loss for case, loss in data.items() if case in all_successful]
            for loss_name, data in losses.items()
        }
        for method, losses in repaired_network_performance.items()
        if losses is not None  # problem with one MNIST run: losses were commented out
    }

    cx_violations_at_generation_comparable = {
        method: list(itertools.chain(
            *[viols for case, viols in data.items() if case in all_successful]
        ))
        for method, data in cx_violations_at_generation.items()
    }

    cx_violations_first_comparable = {
        method: [viol for case, viol in data.items() if case in all_successful]
        for method, data in cx_violation_in_first_repair_step.items()
    }

    cx_violations_last_comparable = {
        method: [viol for case, viol in data.items() if case in all_successful]
        for method, data in cx_violation_in_last_repair_step.items()
    }

    newly_introduced_cx_comparable = {
        method: [frac * 100 for case, frac in data.items() if case in all_successful]
        for method, data in newly_introduced_cx.items()
    }

    runtime_cx_search_first_comparable = {
        method: [time for case, time in data.items() if case in all_successful]
        for method, data in runtime_for_cx_search_first_repair_step.items()
    }

    runtime_cx_search_comparable = {
        method: [time for case, time in data.items() if case in all_successful]
        for method, data in runtime_for_cx_search.items()
    }

    runtime_backend_comparable = {
        method: [time for case, time in data.items() if case in all_successful]
        for method, data in runtime_for_backend.items()
    }

    if args.export:
        def to_csv(df, path):
            df.to_csv(path, index=True, index_label="index")

        output_dir = Path(args.output_dir)

        to_csv(
            pd.DataFrame([
                {"method": method, "success_rate": rate}
                for method, rate in success_rate.items()
            ]),
            output_dir / "success-rate.csv"
        )

        to_csv(
            pd.DataFrame([
                {"Method": method, **performances}
                for method, performances
                in all_successful_median_repaired_network_performance.items()
            ]),
            output_dir / "all_successful_median_performance.csv"
        )

        repair_steps_sorted_df = pd.DataFrame({
            method: data + [-1] * (len(all_cases) - len(data))
            for method, data in repair_steps_sorted.items()
        })
        to_csv(
            repair_steps_sorted_df,
            output_dir / "repair-steps-sorted.csv"
        )
        to_csv(
            pd.DataFrame(runtime_sorted),
            output_dir / "runtime-sorted.csv"
        )

        to_csv(
            pd.DataFrame([
                {"method": method, "was-fastest": num_fastest}
                for method, num_fastest in repair_steps_fastest_percent.items()
            ]),
            output_dir / "repair-steps-fastest.csv"
        )
        to_csv(
            pd.DataFrame([
                {"method": method, "was-fastest": num_fastest}
                for method, num_fastest in runtime_fastest_percent.items()
            ]),
            output_dir / "runtime-fastest.csv"
        )

        # runtime_distance_to_second_best_df = pd.DataFrame(
        #     {"case": cases_order_2, **runtime_distance_to_second_best}
        # )
        # to_csv(
        #     runtime_distance_to_second_best_df,
        #     output_dir / "runtime-distance-to-second-best.csv"
        # )
        # repair_steps_distance_to_second_best_df = pd.DataFrame(
        #     {"case": cases_order_2, **repair_steps_distance_to_second_best}
        # )
        # to_csv(
        #     repair_steps_distance_to_second_best_df,
        #     output_dir / "repair-steps-distance-to-second-best.csv"
        # )

    if not args.no_plot:
        # Create the cactus plots
        print("Plotting Success Rates...")
        fig = plt.figure(figsize=(15, 7))
        for i, (method, rate) in enumerate(success_rate.items()):
            plt.bar(i, height=rate, label=method)
        plt.title("Success Rate")
        plt.ylabel("Success Rate (%)")
        plt.ylim(bottom=0.0, top=1.0)
        plt.xlabel("Method")
        plt.legend(loc='upper left')

        plt.tight_layout()

        # Create the cactus plots
        print("Plotting Cactus Plots...")
        fig = plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        for method, data in repair_steps_sorted.items():
            plt.stairs(data, baseline=None, label=method)
        plt.title("Repair Steps")
        plt.ylabel("Repair Step")
        plt.xlabel("#Completed Repairs")
        plt.xlim(0, len(all_cases))
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.semilogy()
        for method, data in runtime_sorted.items():
            plt.stairs(data, baseline=None, label=method)
        plt.title("Runtime")
        plt.ylabel("Time (s)")
        plt.xlabel("#Completed Repairs")
        plt.ylim(top=timeout_value)
        plt.xlim(0, len(all_cases))
        plt.legend(loc='upper left')

        plt.tight_layout()

        print("Plotting Runtimes/Repair Steps Matching...")
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle("Runtime and Repair Steps Matched by Cases")

        plt.subplot(1, 3, 1)
        for method, data in repair_steps_matching.items():
            plt.plot(data, label=method)
        plt.title("Repair Steps")
        plt.ylabel("Repair Step")
        plt.xlabel("Experiment Case")
        plt.xlim(0, len(all_cases) - 1)
        plt.ylim(0, max_repair_step + 1)
        plt.legend(loc='upper left')

        plt.subplot(1, 3, 2)
        plt.semilogy()
        for method, data in runtime_matching.items():
            plt.plot(data, label=method)
        plt.title("Runtime")
        plt.ylabel("Time (s)")
        plt.xlabel("Experiment Case")
        plt.ylim(top=timeout_value)
        plt.xlim(0, len(all_cases) - 1)
        plt.legend(loc='upper left')

        plt.subplot(1, 3, 3)
        for method, data in runtime_matching.items():
            plt.plot(data, label=method)
        plt.title("Runtime (Non-Log)")
        plt.ylabel("Time (s)")
        plt.xlabel("Experiment Case")
        plt.ylim(top=timeout_value)
        plt.xlim(0, len(all_cases) - 1)
        plt.legend(loc='upper left')

        plt.tight_layout()

        print("Plotting Margin to Second Best")
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle("Runtime and Repair Steps Distance to Second Best")

        plt.subplot(1, 2, 1)
        for method, data in repair_steps_distance_to_second_best.items():
            plt.fill_between(x=range(len(data)), y1=data, label=method, step="mid")
        plt.title("Repair Steps")
        plt.ylabel("Repair Step")
        plt.xlabel("Experiment Case")
        plt.xlim(0, len(all_cases) - 1)
        plt.ylim(bottom=0)
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.semilogy()
        for method, data in runtime_distance_to_second_best.items():
            plt.fill_between(x=range(len(data)), y1=data, label=method, step="mid")
        plt.title("Runtime")
        plt.ylabel("Time (s)")
        plt.xlabel("Experiment Case")
        plt.ylim(0, top=timeout_value)
        plt.xlim(0, len(all_cases) - 1)
        plt.legend(loc='upper left')

        plt.tight_layout()

        print("Plotting Fastest Method Bar Charts")
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle("Who Often is a Method the Fastest Method?")

        plt.subplot(1, 2, 1)
        for i, (method, num_fastest) in enumerate(repair_steps_fastest.items()):
            plt.bar(i, height=num_fastest, label=method)
        plt.title("Repair Steps")
        plt.ylabel("#Fastest")
        plt.xlabel("Method")
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        for i, (method, num_fastest) in enumerate(runtime_fastest.items()):
            plt.bar(i, height=num_fastest, label=method)
        plt.title("Runtime")
        plt.ylabel("#Fastest")
        plt.xlabel("Method")
        plt.legend(loc='upper left')

        # Repaired Network Performance
        print("Plotting Repaired Network Performance...")
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("Repaired Network Performance")

        any_method = next(iter(repaired_network_performance.keys()))
        num_losses = len(repaired_network_performance[any_method].keys())
        for i, loss_name in enumerate(repaired_network_performance[any_method]):
            plt.subplot(1, num_losses, i+1)

            loss_values = {}
            for method, losses in repaired_network_performance_comparable.items():
                loss_values[method] = losses[loss_name]

            labels, data = zip(*loss_values.items())
            plt.violinplot(data)
            plt.boxplot(data, labels=labels)
            plt.xticks(rotation=90)
            plt.ylabel("Loss Value")
            plt.title(loss_name)

        # Counterexample Violation
        print("Plotting Counterexample Violation at Generation...")
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle("Counterexample Violation at Generation")

        plt.subplot(1, 3, 1)
        labels, data = zip(*cx_violations_at_generation_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("violation")
        plt.title("All Repair Steps")

        plt.subplot(1, 3, 2)
        labels, data = zip(*cx_violations_first_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("violation")
        plt.title("First Repair Step")

        plt.subplot(1, 3, 3)
        labels, data = zip(*cx_violations_last_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("violation")
        plt.title("Last Repair Step")

        plt.tight_layout()

        # Counterexample Violation
        print("Plotting Further Counterexample Statistics")
        fig = plt.figure(figsize=(5, 7))

        labels, data = zip(*newly_introduced_cx_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("newly introduced cx (%)")
        plt.title("Fraction of Newly Introduced Counterexamples")

        plt.tight_layout()

        # Runtimes
        print("Plotting Runtimes...")
        fig = plt.figure(figsize=(15, 7.5))
        fig.suptitle("Runtimes")

        plt.subplot(1, 3, 1)
        labels, data = zip(*runtime_cx_search_first_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("Time (s)")
        plt.title("Counterexample Search Runtime (First Repair Step)")

        plt.subplot(1, 3, 2)
        labels, data = zip(*runtime_cx_search_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("Time (s)")
        plt.title("Counterexample Search Runtime (Overall)")

        plt.subplot(1, 3, 3)
        labels, data = zip(*runtime_backend_comparable.items())
        plt.violinplot(data)
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=90)
        plt.ylabel("Time (s)")
        plt.title("Backend Runtime")

        print("Plotting Runtimes Split by Tool...")
        fig = plt.figure(figsize=(15, 7.5))
        fig.suptitle("Runtimes by Tool")

        max_runtime = max(
            max([max(data) for data in tool_data.values()])
            for tool_data in runtime_cx_search_by_tool.values()
        )

        for i, (method, tool_data) in enumerate(runtime_cx_search_by_tool.items()):
            plt.subplot(1, len(runtime_cx_search_by_tool), i+1)
            labels, data = zip(*tool_data.items())
            plt.violinplot(data)
            plt.boxplot(data, labels=labels)
            plt.xticks(rotation=90)
            plt.ylabel("Time (s)")
            plt.ylim(0.0, max_runtime + 100)
            plt.title(method)

        plt.tight_layout()

        print("Plotting Runtimes For Each Repair Case...")

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Cumulative Verification Time per Repair Case")

        columns = ceil(sqrt(len(all_cases)))
        rows = ceil(len(all_cases) / columns)
        for i, case in enumerate(all_cases):
            plt.subplot(rows, columns, i + 1)

            for method, case_data in runtime_for_cx_search_timeseries.items():
                if case in case_data:
                    cumulative_verification_time = np.cumsum(case_data[case])
                    plt.plot(cumulative_verification_time, label=method)

            plt.legend()
            plt.ylabel("Time (s)")
            best_method = runtime_best[case]
            plt.title(f"{case} / best: {best_method}")
        plt.tight_layout()

        plt.show()
