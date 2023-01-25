# get the data for the thesis tables and visualisations from the ACASXu output dirs
import os
from pathlib import Path
import re
from collections import OrderedDict, defaultdict
import argparse

import ruamel.yaml
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create ACAS Xu results table")
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
    args = parser.parse_args()

    # 0. get the data
    # dict of data frames

    # get all the yaml results (and others)
    yaml = ruamel.yaml.YAML(typ='safe')
    all_yaml_results = OrderedDict()
    grid_fidelity_results_per_backend = OrderedDict()
    all_cases = set()
    for method in (
        "optimal", "early_exit", "switch1", "switch3", "switch5",
        "runtime_threshold", "runtime_threshold_decrease",
        "deep_opt_early_exit", "deep_opt_optimal", "pgd_early_exit", "pgd_optimal"
    ):
        directory = getattr(args, method)
        if directory is None:
            continue
        with open(Path(directory, "results.yaml"), 'rt') as file:
            all_yaml_results[method] = yaml.load(file)
        grid_fidelity_file = \
            [element for element in os.listdir(directory) if element.startswith('grid_fidelity_calculation')][0]
        grid_fidelity_file = Path(directory, grid_fidelity_file)
        print(f'Found grid fidelity file: {grid_fidelity_file}')
        grid_fidelity_results_per_backend[method] = pd.read_csv(grid_fidelity_file)

        cases = tuple(all_yaml_results[method]["cases"].keys())
        all_cases.update(cases)

    all_cases = sorted(all_cases)

    all_successful = set(all_cases)
    for method, yaml_results in all_yaml_results.items():
        successful_cases = set(all_yaml_results[method]["summary"]["successful_experiments"]["cases"])
        all_successful &= successful_cases

    # 1. table
    # we need: repair status (with symbols), grid accuracy fidelity, grid mae accuracy
    # table has cases as rows
    num_methods = len(all_yaml_results)
    print('\n\n\n')
    print(r'\begin{tabular}{ll' + ('l' * 3 * num_methods) + r'}')
    print(r'\toprule')
    print(r'         &       & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{\textbf{Status}} & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{\textbf{Accuracy}} & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{\textbf{MAE}} \\')
    line = 'Property & Model'
    for name in all_yaml_results:
        line += f' & {name.replace("_", " ")}'
    for name in all_yaml_results:
        line += f' & {name.replace("_", " ")}'
    for name in all_yaml_results:
        line += f' & {name.replace("_", " ")}'
    line += r' \\'
    print(line)
    print(r'\midrule')

    status_to_symbol = {
        'success': r'\exsuccess',
        'backend_failure': r'\exfailure',
        'verification_problem': r'\exunknown',
        'max_iterations_failure': r'!!!',
        'timeout': r'\extimeout',
        'aborted': r'\extimeout'
    }
    case_name_rx = re.compile(r'property_(?P<prop>\d(_\d)*)_net_(?P<i1>\d)_(?P<i2>\d)')
    statuses_per_backend = defaultdict(list)
    accuracy_fidelity_per_backend = defaultdict(list)
    mae_fidelity_per_backend = defaultdict(list)
    for case in all_cases:
        match = case_name_rx.match(case)
        prop_is = match.group('prop').split('_')
        prop = rf'\(\phi_{prop_is[0]}\)'
        net_i1 = int(match.group('i1'))
        net_i2 = int(match.group('i2'))
        net = rf'\(N_{{{net_i1},{net_i2}}}\)'
        line = rf'{prop} & {net}'
        statuses = {}
        for name, yaml_results in all_yaml_results.items():
            status = yaml_results['cases'][case]['result']
            symbol = status_to_symbol[status]
            line += rf' & {symbol}'
            statuses[name] = status
            statuses_per_backend[name].append(status)
        for name, grid_fidelity in grid_fidelity_results_per_backend.items():
            prop_key = f'({",".join(map(str, prop_is))},)'
            row = grid_fidelity.loc[
                (grid_fidelity['specification'] == prop_key) &
                (grid_fidelity['network_i1'] == net_i1) &
                (grid_fidelity['network_i2'] == net_i2)
            ]
            if len(row) > 0 and statuses[name] in ('success', ):
                accuracy_fidelity = row["accuracy_fidelity"].item()
                line += rf' & \SI{{{accuracy_fidelity*100:4.1f}}}{{\percent}}'
                accuracy_fidelity_per_backend[name].append({'case': case, 'accuracy_fidelity': accuracy_fidelity})
            else:
                line += r' &  -- '
        for name, grid_fidelity in grid_fidelity_results_per_backend.items():
            prop_key = f'({",".join(map(str, prop_is))},)'
            row = grid_fidelity.loc[
                (grid_fidelity['specification'] == prop_key) &
                (grid_fidelity['network_i1'] == net_i1) &
                (grid_fidelity['network_i2'] == net_i2)
                ]
            if len(row) > 0 and statuses[name] in ('success', ):
                mae_fidelity = row["mae_fidelity"].item()
                line += rf' & \num{{{mae_fidelity:6.2f}}}'
                mae_fidelity_per_backend[name].append({'case': case, 'mae_fidelity': mae_fidelity})
            else:
                line += r' &    -- '
        line += r' \\'
        print(line)

    accuracy_fidelity_per_backend_all_successful = {
        name: pd.Series(value["accuracy_fidelity"] for value in values if value["case"] in all_successful)
        for name, values in accuracy_fidelity_per_backend.items()
    }
    mae_fidelity_per_backend_all_successful = {
        name: pd.Series(value["mae_fidelity"] for value in values if value["case"] in all_successful)
        for name, values in mae_fidelity_per_backend.items()
    }

    accuracy_fidelity_per_backend = {
        name: pd.DataFrame(values) for name, values in accuracy_fidelity_per_backend.items()
    }
    mae_fidelity_per_backend = {
        name: pd.DataFrame(values) for name, values in mae_fidelity_per_backend.items()
    }

    print(r'\midrule')
    line = '           &            '
    for name, statuses in statuses_per_backend.items():
        num_success = sum(status == 'success' for status in statuses)
        line += rf' & {num_success:2.0f}'
    for name, grid_fidelity in grid_fidelity_results_per_backend.items():
        median_acc_fidelity = accuracy_fidelity_per_backend[name]['accuracy_fidelity'].median()
        line += rf' & \SI{{{median_acc_fidelity * 100:4.1f}}}{{\percent}}'
    for name, grid_fidelity in grid_fidelity_results_per_backend.items():
        median_mae_fidelity = mae_fidelity_per_backend[name]['mae_fidelity'].median()
        line += rf' & \num{{{median_mae_fidelity:6.2f}}}'
    line += r' \\'
    print(line)
    print(r'% \midrule')
    print(r'           &             & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{success frequency} & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{median} & '
              r'\multicolumn{' + str(num_methods) + r'}{c}{median} \\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print('\n\n\n')

    print("Fidelity overview (only where all were successful):")
    print("===================================================")
    for method, grid_fidelity in grid_fidelity_results_per_backend.items():
        print(f"{method}:")
        print(
            f"Accuracy: "
            f"min: {accuracy_fidelity_per_backend_all_successful[method].min():.4f}, "
            f"25% quantile: {accuracy_fidelity_per_backend_all_successful[method].quantile(0.25):.4f}, "
            f"median: {accuracy_fidelity_per_backend_all_successful[method].median():.4f}, "
            f"75% quantile: {accuracy_fidelity_per_backend_all_successful[method].quantile(0.75):.4f}, "
            f"max: {accuracy_fidelity_per_backend_all_successful[method].max():.4f}"
        )
        print(
            f"MAE: "
            f"min: {mae_fidelity_per_backend_all_successful[method].min():.4f}, "
            f"25% quantile: {mae_fidelity_per_backend_all_successful[method].quantile(0.25):.4f}, "
            f"median: {mae_fidelity_per_backend_all_successful[method].median():.4f}, "
            f"75% quantile: {mae_fidelity_per_backend_all_successful[method].quantile(0.75):.4f}, "
            f"max: {mae_fidelity_per_backend_all_successful[method].max():.4f}"
        )
