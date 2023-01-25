import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from experiments.datasets import integer_dataset
from experiments.experiment_base import seed_rngs

if __name__ == "__main__":
    # Command line arguments: the RMI names (rmi_10_1, rmi_10_2, ...)
    rmi_names = sys.argv[1:]
    seed_rngs(2148522714424)

    spec_sizes = []
    for rmi in tqdm(rmi_names):
        rmi_dir = Path("..", "resources", "ouroboros_rmi", rmi)
        with open(rmi_dir / "params.yaml") as file:
            rmi_params = yaml.safe_load(file)

        first_stage_model = torch.load(rmi_dir / f"first_stage.pyt")
        dataset = integer_dataset(
            size=rmi_params["dataset"]["size"],
            distribution=rmi_params["dataset"]["distribution"],
            maximum=rmi_params["dataset"]["maximum"],
            seed=rmi_params["dataset"]["seed"],
        )
        second_stage_size = rmi_params["second_stage_size"]
        partition_size = len(dataset) // second_stage_size
        partitions = np.array_split(np.arange(len(dataset)), second_stage_size)
        first_stage_tolerance = rmi_params["tolerance"]["first_stage"]
        second_stage_tolerance = rmi_params["tolerance"]["second_stage"]

        for second_stage_i in range(second_stage_size):
            # take all the data from the partitions within the error tolerance of the
            # first stage model
            data_indices = set()
            for i in range(-first_stage_tolerance, first_stage_tolerance + 1):
                i = min(second_stage_size - 1, max(0, i + second_stage_i))
                data_indices.update(partitions[i])
            data_indices = list(data_indices)

            partition_dataset = Subset(dataset, data_indices)
            loader = DataLoader(
                partition_dataset, batch_size=len(partition_dataset), num_workers=0
            )
            full_keys, full_pos = next(iter(loader))

            # round instead of floor or trunc leads to specifications that are
            # similarly large as reported by Tan et. al.
            assigned = first_stage_model(full_keys.unsqueeze(-1)).round() == second_stage_i

            target_indices = partitions[second_stage_i]
            target_dataset = Subset(dataset, target_indices)
            target_loader = DataLoader(target_dataset, batch_size=len(target_dataset), num_workers=0)
            target_keys, target_pos = next(iter(target_loader))
            target_assigned = first_stage_model(target_keys.unsqueeze(-1)).round() == second_stage_i

            # All assigned + target partition size,
            # then subtract assigned in target to account for double counting.
            spec_size = assigned.sum() + len(target_dataset) - target_assigned.sum()
            spec_sizes.append(spec_size)

    spec_sizes = pd.Series(spec_sizes)
    print(f"Mean Spec Size: {spec_sizes.mean()}")
    print(f"Median Spec Size: {spec_sizes.median()}")
