import logging
import argparse

import torch

from nn_repair.backends import PenaltyFunctionRepairDelegate, PenaltyFunction, BarrierFunctionRepairDelegate, \
    BarrierFunction, NeuronFixationRepairDelegate
from nn_repair.training import TrainingLoop, IterationMaximum, TrainingLossChange, ValidationSet, LogLoss, \
    ResetOptimizer, Divergence
from nn_repair.training.optim import BacktrackingLineSearchSGD
from experiment_base import (
    ExperimentBase, TrackingLossFunction, L1_PENALTY_KEY,
    RECIPROCAL_BARRIER_KEY,
    NEURON_FIXATION_KEY, seed_rngs
)
from nn_repair.training.loss_functions import get_fidelity_loss_hcas_loss, ParameterChange

from acasxu_basics import acasxu_properties, load_acasxu_network, \
    acasxu_repair_case_dir_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACAS Xu repair 1')
    ExperimentBase.argparse_add_arguments(parser, [L1_PENALTY_KEY, RECIPROCAL_BARRIER_KEY,
                                                   NEURON_FIXATION_KEY])
    experiment_group = parser.add_argument_group('experiment customization')
    experiment_group.add_argument('--specification', type=str, required=True,
                                  help='Which ACAS Xu properties to repair.\n'
                                       'Examples: --specification 2 --specification 8'
                                       '--specification 1,2,3,4,8.')
    experiment_group.add_argument('--network', type=str, required=True,
                                  help='Which ACAS Xu network to repair.\n'
                                       'Examples: --network 2,1 --network 5,3')
    experiment_group.add_argument('--loss_function', type=str, required=True,
                                  help='The surrogate loss function to use for backends that '
                                       'require a task loss function. '
                                       'Options: "random_sample_hcas_loss", "parameter_change".'
                                       'This option will also cause different termination criteria to be set up.')
    args = parser.parse_args()
    seed_rngs(936238248673366)
    # In the first repair step, ERAN usually finds a counterexample after a little more
    # than 3 seconds.
    # Start at that threshold to have early-exit behaviour initially.
    # Then increase the runtime threshold linearly until reaching 5 minutes
    # (median ERAN runtime in optimal mode) in repair step 10 (median number of repair
    # steps with optimal mode ERAN).
    experiment_base = ExperimentBase(
        'acasxu_repair_1', args,
        initial_verifier_runtime_threshold=3,
        verifier_runtime_threshold_update=lambda i, _: 3 + (i+1) * ((5 * 60 - 3) / 10),
        initial_verifier_runtime_threshold_decrease=5 * 60,
        verifier_runtime_threshold_update_decrease=lambda i, _: 5 * 60 - (i+1) * ((5 * 60 - 3) / 10),
    )

    property_indices = [int(prop_i) for prop_i in args.specification.split(',')]
    net_i0, net_i1 = [int(net_i) for net_i in args.network.split(',')]
    properties = acasxu_properties()

    network = load_acasxu_network(net_i0, net_i1)
    specification = [properties[i] for i in property_indices]

    if args.loss_function is None:
        # only used to measure forgetting in this case
        task_loss = get_fidelity_loss_hcas_loss(network, specification)
        validation_loss = None
        optimizer = None
        setup_training_loop = False
        value_change_threshold = 0.0
    elif args.loss_function == 'parameter_change':
        task_loss = ParameterChange(module=network)
        optimizer = BacktrackingLineSearchSGD(network.parameters())
        validation_loss = None
        setup_training_loop = True
        value_change_threshold = 0.001
    elif args.loss_function == 'random_sample_hcas_loss':
        task_loss = get_fidelity_loss_hcas_loss(network, specification, seed=2703, num_samples=7000)
        validation_loss = get_fidelity_loss_hcas_loss(network, specification, seed=1455, num_samples=3000)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        setup_training_loop = True
        value_change_threshold = 0.01
    else:
        raise RuntimeError(f"Unknown loss function: {args.loss_function}")

    if setup_training_loop:
        wrapped_task_loss = TrackingLossFunction(task_loss, 'task loss')
        training_loop = TrainingLoop(network, optimizer, wrapped_task_loss)

        training_loop.add_termination_criterion(TrainingLossChange(
            change_threshold=value_change_threshold, iteration_block_size=1, num_blocks=25
        ))
        training_loop.add_termination_criterion(Divergence(network.parameters()))
        training_loop.add_termination_criterion(IterationMaximum(500))

        additional_losses = wrapped_task_loss.get_additional_losses(average=True)
        if validation_loss is not None:
            wrapped_validation_loss = TrackingLossFunction(validation_loss, 'validation loss')
            training_loop.add_termination_criterion(ValidationSet(
                wrapped_validation_loss, iterations_between_validations=5, acceptable_increase_length=5
            ))
            additional_losses += wrapped_validation_loss.get_additional_losses(average=True)
            wrapped_validation_loss.register_loss_resetting_hook(training_loop)

        loss_logger = LogLoss(log_frequency=5, additional_losses=additional_losses, log_level=logging.DEBUG)
        training_loop.add_post_iteration_hook(loss_logger)
        wrapped_task_loss.register_loss_resetting_hook(training_loop)

        training_loop.add_pre_training_hook(ResetOptimizer(optimizer))
        experiment_base.register_hooks(training_loop, training_checkpointing_frequency=5)

    def l1_penalty_backend():
        assert args.loss_function is not None, "l1 penalty backend requires a task loss function"
        return PenaltyFunctionRepairDelegate(
            training_loop, penalty_function=PenaltyFunction.L1,
            maximum_updates=25
        )

    experiment_base.execute(
        network, specification,
        repair_backends={
            L1_PENALTY_KEY: l1_penalty_backend,
        },
        experiment_name=acasxu_repair_case_dir_name(property_indices, net_i0, net_i1),
        losses=(('training loss', task_loss), ) +
               ((('validation loss', validation_loss), )
                if validation_loss is not None else ()),
        repair_network_further_kwargs={
            'abort_on_backend_error': True
        },
        raise_exceptions=False
    )
