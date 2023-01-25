import argparse
import pandas as pd
import ruamel.yaml as yaml

from collect_experiment_results import continuous_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Median Total Runtime")
    parser.add_argument("results_file", type=str,
                        help="Path to the results.yaml file containing the total runtimes.")
    args = parser.parse_args()

    with open(args.results_file, "rt") as file:
        results = yaml.safe_load(file)

    total_runtimes = [case["total_runtime"] for case in results["cases"].values()]
    total_runtimes = pd.Series(total_runtimes)

    summary = continuous_summary(total_runtimes)
    print(summary)
