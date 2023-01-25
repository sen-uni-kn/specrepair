from functools import partial
from math import ceil, log10
from pathlib import Path

import numpy as np
import torch
import argparse
import ruamel.yaml as yaml
from torch.utils.data import DataLoader

from deep_opt import Property, BoxConstraint
from deep_opt.models import ConstraintAnd
from nn_repair.backends import PenaltyFunctionRepairDelegate, PenaltyFunction

from nn_repair.training import (
    TrainingLoop, IterationMaximum,
    LogLoss, ResetOptimizer, Divergence,
)

from experiment_base import (
    ExperimentBase, TrackingLossFunction, L1_PENALTY_KEY, seed_rngs,
)
from datasets import integer_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ouroboros Repair 1: First Stage'
    )
    ExperimentBase.argparse_add_arguments(
        parser, [L1_PENALTY_KEY], default_verifier="ERAN_plain"
    )
    experiment_group = parser.add_argument_group('experiment customization')
    experiment_group.add_argument('--rmi', default="rmi_10_1",
                                  help="The RMI to repair. "
                                       "Needs to be a subdirectory of "
                                       "resources/ouroboros_rmi.")
    args = parser.parse_args()
    seed_rngs(7103934170018)
    experiment_base = ExperimentBase(
        'ouroboros_rmi_repair_1', args,
    )

    rmi_dir = Path("..", "resources", "ouroboros_rmi", args.rmi)
    with open(rmi_dir / "params.yaml") as file:
        rmi_params = yaml.safe_load(file)
    first_stage_model = torch.load(rmi_dir / "first_stage.pyt")
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

    specification = [
        Property(
            lower_bounds={0: dataset.data[partitions[i]].amin()},
            upper_bounds={0: dataset.data[partitions[i]].amax()},
            output_constraint=ConstraintAnd(
                BoxConstraint(0, ">=", max(0, i - first_stage_tolerance)),
                BoxConstraint(0, "<=", min(second_stage_size - 1, i + first_stage_tolerance))
            ),
            property_name=f"First Stage Error Bound for Partition {i}"
        )
        for i in range(second_stage_size)
    ]

    mse = torch.nn.MSELoss()

    batch_size = 512
    first_stage_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_first_stage_loader = DataLoader(dataset, batch_size=batch_size * 4, shuffle=True, num_workers=0)
    full_first_stage_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)

    def first_stage_preds_parts(data_loader):
        keys, pos = next(iter(data_loader))
        parts = torch.div(pos, partition_size, rounding_mode='floor')

        keys = keys.unsqueeze(-1)
        parts = parts.unsqueeze(-1).float()

        preds = first_stage_model(keys.float())
        return preds, parts

    def first_stage_loss(data_loader=first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return mse(preds, parts)

    def first_stage_accuracy(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return (parts == preds.round()).float().mean() * 100

    def first_stage_mae(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return torch.abs(preds.round() - parts).mean()

    def first_stage_max_error(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return torch.abs(preds.round() - parts).amax()

    wrapped_loss = TrackingLossFunction(first_stage_loss, 'task loss')

    optimizer = torch.optim.Adam(first_stage_model.parameters(), lr=0.002)
    epoch_len = len(first_stage_loader)

    training_loop = TrainingLoop(
        first_stage_model, optimizer, wrapped_loss
    )
    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len // 5, epoch_length=epoch_len,
        average_training_loss=True,
        additional_losses=wrapped_loss.get_additional_losses(average=True) + (
            ("accuracy", first_stage_accuracy, False),
            ("mae", first_stage_mae, False),
            ("max error", first_stage_max_error, False),
        )
    ))
    wrapped_loss.register_loss_resetting_hook(training_loop)

    reset_optimizer = ResetOptimizer(optimizer)
    training_loop.add_pre_training_hook(reset_optimizer)

    training_loop.add_termination_criterion(Divergence(first_stage_model.parameters()))
    training_loop.add_termination_criterion(IterationMaximum(
        epoch_len * int(log10(second_stage_size) ** 1.5)
    ))

    experiment_base.register_hooks(training_loop)

    experiment_base.execute(
        first_stage_model, specification, experiment_name=args.rmi,
        repair_backends={
            L1_PENALTY_KEY: lambda: PenaltyFunctionRepairDelegate(
                training_loop, penalty_function=PenaltyFunction.L1, maximum_updates=25
            ),
        },
        losses=(
            ("MSE", partial(first_stage_loss, full_first_stage_loader)),
            ("accuracy", partial(first_stage_accuracy, full_first_stage_loader)),
            ("MAE", partial(first_stage_mae, full_first_stage_loader)),
            ("max error", partial(first_stage_max_error, full_first_stage_loader)),
        ),
        repair_network_further_kwargs={
            'do_not_skip_properties': True,
            'abort_on_backend_error': True
        }
    )
