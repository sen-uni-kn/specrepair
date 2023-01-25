import argparse
import logging
from logging import info, debug, warning, error
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import gurobipy
import ruamel.yaml as yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from deep_opt import BoxConstraint, Property
from experiments.datasets import integer_dataset
from experiments.experiment_base import seed_rngs
from nn_repair.utils.timing import LogExecutionTime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Ouroboros Second Stage Repair using QP'
    )
    parser.add_argument('--rmi', default="rmi_10_1",
                                  help="The RMI to repair. "
                                       "Needs to be a subdirectory of "
                                       "resources/ouroboros_rmi.")
    parser.add_argument('--part', default=0, type=int,
                        help="The part of the second stage model "
                             "to repair.")
    parser.add_argument('--second_stage_tolerance', default=None, type=int,
                        help="The second stage tolerance to use. "
                             "When not given, the tolerance recorded in the "
                             "params file is used.")
    parser.add_argument(
        '--timestamp', dest='timestamp', default=None, type=str,
        help="Set the timestamp of the experiment to use for "
             "determining the output directory. "
             "When omitted the current time is used. "
             "Use this argument when using bash's timeout for "
             "better enforcing timeouts for individual experiment "
             "cases. "
    )
    args = parser.parse_args()
    seed_rngs(2148522714424)

    rmi_dir = Path("..", "resources", "ouroboros_rmi", args.rmi)
    with open(rmi_dir / "params.yaml") as file:
        rmi_params = yaml.safe_load(file)
    second_stage_i = args.part
    model = torch.load(rmi_dir / f"second_stage_{second_stage_i}.pyt")
    # need this to build the specification
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
    second_stage_tolerance = args.second_stage_tolerance
    if second_stage_tolerance is None:
        second_stage_tolerance = rmi_params["tolerance"]["second_stage"]

    # take all the data from the partitions within the error tolerance of the
    # first stage model
    data_indices = []  # indices of the full dataset
    spec_indices = set()  # indices of the subset
    for i in range(-first_stage_tolerance, first_stage_tolerance + 1):
        i = second_stage_i + i
        if i < 0 or i >= second_stage_size:
            continue
        part = partitions[i].tolist()
        if i == second_stage_i:
            spec_indices.update(range(len(data_indices), len(data_indices) + len(part)))
        data_indices.extend(part)
    data_indices = sorted(data_indices)

    partition_dataset = Subset(dataset, data_indices)
    loader = DataLoader(
        partition_dataset, batch_size=len(partition_dataset), num_workers=0
    )
    full_keys, full_pos = next(iter(loader))

    # All wrongly assigned keys and the keys from the target partition.
    # Using round leads to the spec size reported by Tan et. al.
    # compared to trunc, which yields 1.4 times larger specifications.
    assigned = first_stage_model(full_keys.unsqueeze(-1)).round() == second_stage_i
    assigned.squeeze_()
    full_indices = torch.arange(len(full_keys))
    spec_indices.update(full_indices[assigned].tolist())
    spec_indices = sorted(spec_indices)

    before_first_key = (
        dataset[data_indices[0] - 1][0]
        if data_indices[0] > 0
        else dataset.data.amin().item() - 1
    )
    after_last_key = (
        dataset[data_indices[-1] + 1][0]
        if data_indices[-1] < len(dataset) - 1
        else dataset.data.amax().item() + 1
    )
    prev_key = [before_first_key] + full_keys[:-1].tolist()
    next_key = full_keys[1:].tolist() + [after_last_key]
    # The linear regression repair backends only support BoxConstraints,
    # not AndConstraints.
    # To mitigate this, just use two properties for each.
    specification = [
        Property(
            lower_bounds={0: min(prev_key[key_i] + 1, full_keys[key_i])},
            upper_bounds={0: max(next_key[key_i] - 1, full_keys[key_i])},
            output_constraint=BoxConstraint(0, ">=", full_pos[key_i] - second_stage_tolerance),
            property_name=f"Second Stage Error Bound {i} for Model {second_stage_i} (>=)"
        )
        for i, key_i in enumerate(spec_indices)
    ] + [
        Property(
            lower_bounds={0: min(prev_key[key_i] + 1, full_keys[key_i])},
            upper_bounds={0: max(next_key[key_i] - 1, full_keys[key_i])},
            output_constraint=BoxConstraint(0, "<=", full_pos[key_i] + second_stage_tolerance),
            property_name=f"Second Stage Error Bound {i} for Model {second_stage_i} (<=)"
        )
        for i, key_i in enumerate(spec_indices)
    ]

    for prop in specification:
        assert prop.lower_bounds[0] <= prop.upper_bounds[0]

    mse = torch.nn.MSELoss()

    def second_stage_preds_pos(data_loader=loader):
        keys, pos = next(iter(data_loader))

        keys = keys.unsqueeze(-1)
        pos = pos.unsqueeze(-1).float()

        preds = model(keys.float())
        return preds, pos

    def second_stage_loss(data_loader=loader):
        preds, pos = second_stage_preds_pos(data_loader)
        return mse(preds, pos)

    def second_stage_accuracy():
        preds, pos = second_stage_preds_pos()
        return (pos == preds.round()).float().mean()

    def second_stage_mae():
        preds, pos = second_stage_preds_pos()
        return torch.abs(preds.round() - pos).mean()

    def second_stage_max_error():
        preds, pos = second_stage_preds_pos()
        return torch.abs(preds.round() - pos).amax()

    timestamp = args.timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{args.rmi}_second_stage_{second_stage_i}"
    output_dir = Path("..", "output", "ouroboros_rmi_repair_second_stage_qp", timestamp, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s", stream=sys.stdout)
    logfile_handler = logging.FileHandler(Path(output_dir, f'log.txt'), encoding='utf-8')
    logging.root.addHandler(logfile_handler)

    repaired_network_file_path = Path(output_dir, f"repaired_network.pyt")
    assert not repaired_network_file_path.exists(), 'Repaired network output file already exists'

    info(
        f"RMI: {args.rmi}, part: {args.part}, "
        f"second stage tolerance: {second_stage_tolerance}"
    )

    def verify() -> bool:
        is_satisfied = True
        for prop_ in tqdm(specification):
            lower_ = prop_.lower_bounds[0]
            upper_ = prop_.upper_bounds[0]
            inputs = torch.tensor([[lower_], [upper_]])
            prop_satisfied = prop_.property_satisfied(inputs, model)
            is_satisfied = is_satisfied and prop_satisfied.all()
        return is_satisfied

    initial_loss_values = {
        "MSE": second_stage_loss(),
        "accuracy": second_stage_accuracy(),
        "MAE": second_stage_mae(),
        "max error": second_stage_max_error(),
    }

    with LogExecutionTime("overall repair"):
        spec_satisfied = verify()
        if spec_satisfied:
            info("Specification isn't violated")
            sys.exit()

        debug("Constructing Gurobi Model")
        opt_model = gurobipy.Model()

        linear_layer = model[0]
        weight_var = opt_model.addMVar(shape=(2,), lb=None, ub=None)

        debug("Adding constraints")
        sat_eps = 1e-2  # need this to avoid floating point problems
        for prop in tqdm(specification):
            lower = prop.lower_bounds[0]
            upper = prop.upper_bounds[0]
            lower = np.array([1, lower])
            upper = np.array([1, upper])
            output_constraint = prop.output_constraint
            bound = output_constraint.bound
            bound = np.array([bound])
            if output_constraint.less_than:
                # lower and upper are the two candidates for the
                # most-violating counterexamples
                opt_model.addConstr(lower @ weight_var <= bound - sat_eps)
                opt_model.addConstr(upper @ weight_var <= bound - sat_eps)
            else:
                opt_model.addConstr(lower @ weight_var >= bound + sat_eps)
                opt_model.addConstr(upper @ weight_var >= bound + sat_eps)

        debug("Adding objective")
        X = torch.hstack([torch.ones(len(full_keys), 1), full_keys.unsqueeze(-1)])
        X = X.float().numpy()
        y = full_pos.float().numpy()
        opt_model.setObjective(
            weight_var @ (X.T @ X) @ weight_var + 2 * y.T @ X @ weight_var + y.T@y,
            gurobipy.GRB.MINIMIZE
        )

        debug("Solving...")
        opt_model.update()
        opt_model.optimize()

        if opt_model.status in (gurobipy.GRB.INFEASIBLE, gurobipy.GRB.INF_OR_UNBD):
            info("Model can't satisfy specification.")
            info("Repair finished: failure")
            sys.exit()
        elif opt_model.status != gurobipy.GRB.OPTIMAL:
            warning("Error while solving the quadratic program.")
            sys.exit()

        bias, weight = weight_var.x
        with torch.no_grad():
            linear_layer.bias.set_(torch.tensor([bias], dtype=torch.float))
            linear_layer.weight.set_(torch.tensor([[weight]], dtype=torch.float))

        spec_satisfied = verify()

    final_loss_values = {
        "MSE": second_stage_loss(),
        "accuracy": second_stage_accuracy(),
        "MAE": second_stage_mae(),
        "max error": second_stage_max_error(),
    }
    if not spec_satisfied:
        error("Optimization succeeded, but result doesn't satisfy the specification.")
        info("Repair finished: error")
    else:
        info("All properties verified! Repair successful.")
        info("Repair finished: success")

    info(
        "Initial loss: " +
        "; ".join(f"{initial_loss_values[name]:.4f} ({name})" for name in initial_loss_values) + "\n" +
        "Final loss:   " +
        "; ".join(f"{final_loss_values[name]:.4f} ({name})" for name in final_loss_values) + "\n" +
        "Difference:   " +
        "; ".join(f"{abs(initial_loss_values[name] - final_loss_values[name]):.4f} ({name})"
                  for name in initial_loss_values)
    )

    repaired_network_file_path.touch(exist_ok=False)
    info(f"Saving repaired network in file: {repaired_network_file_path}")
    torch.save(model, repaired_network_file_path)
