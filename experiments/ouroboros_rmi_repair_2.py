from functools import partial
from pathlib import Path

import numpy as np
import torch
import argparse
import ruamel.yaml as yaml
from torch.utils.data import DataLoader, Subset

from deep_opt import Property, BoxConstraint

from experiment_base import (
    ExperimentBase, TrackingLossFunction, seed_rngs, L1_PENALTY_KEY,
    LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY,
)
from datasets import integer_dataset
from nn_repair.backends import (
    LinearModelDatasetAugmentationRepairDelegate,
    PenaltyFunctionRepairDelegate
)
from nn_repair.training import (
    Divergence, IterationMaximum, LogLoss, ResetOptimizer,
    TrainingLoop
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ouroboros Repair 2: Second Stage'
    )
    ExperimentBase.argparse_add_arguments(
        parser,
        [L1_PENALTY_KEY, LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY],
        default_falsifier_cascade="None",
        default_verifier="LinearRegressionVerifier"
    )
    experiment_group = parser.add_argument_group('experiment customization')
    experiment_group.add_argument('--rmi', default="rmi_10_1",
                                  help="The RMI to repair. "
                                       "Needs to be a subdirectory of "
                                       "resources/ouroboros_rmi.")
    experiment_group.add_argument('--part', default=0, type=int,
                                  help="The part of the second stage model "
                                       "to repair.")
    experiment_group.add_argument('--second_stage_tolerance', default=None, type=int,
                                  help="The second stage tolerance to use. "
                                       "When not given, the tolerance recorded in the "
                                       "params file is used.")
    args = parser.parse_args()
    seed_rngs(2148522714424)
    experiment_base = ExperimentBase(
        'ouroboros_rmi_repair_2', args,
    )

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

    # all wrongly assigned keys and the keys from the target partition
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

    train_loader = DataLoader(
        partition_dataset, batch_size=512, num_workers=0
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-13)
    task_loss = partial(second_stage_loss, train_loader)
    wrapped_task_loss = TrackingLossFunction(task_loss, "task loss")
    l1_penalty_training_loop = TrainingLoop(
        model, optimizer, wrapped_task_loss
    )
    l1_penalty_training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=25, average_training_loss=True,
        additional_losses=wrapped_task_loss.get_additional_losses()
    ))
    wrapped_task_loss.register_loss_resetting_hook(l1_penalty_training_loop)
    l1_penalty_training_loop.add_termination_criterion(
        IterationMaximum(150)
    )
    l1_penalty_training_loop.add_termination_criterion(Divergence(model.parameters()))
    l1_penalty_training_loop.add_pre_training_hook(ResetOptimizer(optimizer))
    experiment_base.register_hooks(l1_penalty_training_loop, training_checkpointing_frequency=100)
    l1_penalty_backend = PenaltyFunctionRepairDelegate(
        l1_penalty_training_loop,
        maximum_updates=25,
    )

    def pos_for_new_key(new_key):
        is_smaller = full_keys < new_key
        first_larger_index = torch.sum(is_smaller)
        return full_pos[first_larger_index]
    dataset_augmentation_backend = LinearModelDatasetAugmentationRepairDelegate(
        full_keys, full_pos,
        target_oracle=pos_for_new_key,
    )
    experiment_base.execute(
        model, specification,
        experiment_name=f"{args.rmi}_second_stage_{second_stage_i}",
        repair_backends={
            L1_PENALTY_KEY: lambda: l1_penalty_backend,
            LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY: lambda: dataset_augmentation_backend,
        },
        losses=(
            ("MSE", second_stage_loss),
            ("accuracy", second_stage_accuracy),
            ("MAE", second_stage_mae),
            ("max error", second_stage_max_error),
        ),
        repair_network_further_kwargs={
            'do_not_skip_properties': True,
            'abort_on_backend_error': not args.linear_regression_dataset_augmentation
        }
    )
