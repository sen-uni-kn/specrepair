import sys
from pathlib import Path

import yaml
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Use like: python plot_acasxu_fidelity.py --label1 path/to/grid_fidelity/file \
    #           --label2 path/to/second/grid_fidelity/file ...
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise ValueError(f"Need an even number of arguments. Got {len(args)}.")

    successful_cases = {}
    grid_fidelity_results = {}
    for i in range(0, len(args), 2):
        name = args[i].strip("-")
        grid_fidelity_file = Path(args[i + 1])
        grid_fidelity_df = pd.read_csv(grid_fidelity_file)
        specifications = grid_fidelity_df["specification"].apply(
            lambda s: s.strip("(").strip(")").replace(",", "_").strip("_")
        )
        net_i1s = grid_fidelity_df["network_i1"].astype(str)
        net_i2s = grid_fidelity_df["network_i2"].astype(str)
        grid_fidelity_df["case"] = "property_" + specifications + "_net_" + net_i1s + "_" + net_i2s

        grid_fidelity_results[name] = grid_fidelity_df

        results_file = grid_fidelity_file.parent / "results.yaml"
        with open(results_file) as file:
            run_results = yaml.safe_load(file)
        run_successful_cases = run_results["summary"]["successful_experiments"]["cases"]
        successful_cases[name] = set(run_successful_cases)

    successful = list(successful_cases.values())
    all_successful = successful[0].intersection(*successful[1:])

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Performance after Repair")

    plt.subplot(1, 2, 1)
    accuracies_comparable = {
        method: [
            row["accuracy_fidelity"]
            for row in df.iloc if row["case"] in all_successful
        ]
        for method, df in grid_fidelity_results.items()
    }
    labels, data = zip(*accuracies_comparable.items())
    plt.violinplot(data)
    plt.boxplot(data, labels=labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Grid Fidelity Accuracy")

    plt.subplot(1, 2, 2)
    mae_comparable = {
        method: [
            row["mae_fidelity"]
            for row in df.iloc if row["case"] in all_successful
        ]
        for method, df in grid_fidelity_results.items()
    }
    labels, data = zip(*mae_comparable.items())
    plt.violinplot(data)
    plt.boxplot(data, labels=labels)
    plt.ylabel("Mean Absolute Error")
    plt.title("Grid Fidelity MAE")

    plt.show()
