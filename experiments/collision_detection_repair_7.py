import logging
from functools import partial

import torch
import argparse
from copy import deepcopy

from deep_opt import RobustnessPropertyFactory

from nn_repair.backends import (
    PenaltyFunctionRepairDelegate, PenaltyFunction,
    AugmentedLagrangianRepairDelegate, BarrierFunctionRepairDelegate, BarrierFunction,
    FineTuningRepairDelegate,
    LinearProgrammingMinimalModificationRepairDelegate, NeuronFixationRepairDelegate,
)

from nn_repair.training import (
    TrainingLoop, IterationMaximum, TrainingLossChange,
    LogLoss, ResetOptimizer, Divergence,
)
from nn_repair.training.loss_functions import accuracy2

from experiment_base import (
    ExperimentBase, TrackingLossFunction, QUADRATIC_PENALTY_KEY,
    L1_PENALTY_KEY,
    AUGMENTED_LAGRANGIAN_KEY, RECIPROCAL_BARRIER_KEY, FINE_TUNING_KEY,
    LP_MINIMAL_MODIFICATION_KEY,
    NEURON_FIXATION_KEY, seed_rngs,
)
from collision_detection_basics import (
    get_collision_detection_dataset,
    get_collision_detection_network,
    get_relu_ffnn_1_non_robust_samples_collision_detection,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CollisionDetection Repair 7: Robust on 10 Samples'
    )
    ExperimentBase.argparse_add_arguments(
        parser, [L1_PENALTY_KEY], default_verifier="ERAN_plain"
    )
    experiment_group = parser.add_argument_group('experiment customization')
    experiment_group.add_argument('--first_data_point', default=0, dest='first_data_point', type=int,
                              help='The first data point of the robustness specification. '
                                   'The specification if formed further contains the following '
                                   '19 data points from the training set.')
    experiment_group.add_argument(
        '--radius', type=float, required=True, default=0.05,
        help='Which radius to use in the robustness specification.\n'
             'Examples: --radius 0.04 --radius 0.03 --radius 0.05'
             'Radius 0.039 and above can lead to inconsistent specifications. '
             'However, radius 0.04 and radius 0.05 still lead to consistent '
             'specifications when using 20 consecutive data points from the '
             'training set. '
             'For radius 0.04, 66.35% of the training set are not robust for '
             'the network ReLU_FFNN_1. For radius 0.05, 70.20% are not robust '
             'and for radius 0.03, 56.70% are not robust. For radius 0.01, '
             'only 13.05% are not robust.'
    )
    args = parser.parse_args()
    seed_rngs(367672735164923)
    experiment_base = ExperimentBase(
        'collision_detection_repair_7', args,
        initial_verifier_runtime_threshold=0.1,
        # get to 0.3 after 10 repair steps
        verifier_runtime_threshold_update=lambda i, _: 0.1 + (i+1) * 0.02,
        initial_verifier_runtime_threshold_decrease=0.3,
        verifier_runtime_threshold_update_decrease=lambda i, _: 0.3 - (i+1) * 0.02
    )

    train_inputs, train_targets, test_inputs, test_targets = get_collision_detection_dataset()
    original_network = get_collision_detection_network('ReLU_FFNN_1')

    first_index = args.first_data_point
    sample_indices = [first_index + i for i in range(10)]
    robustness_samples_inputs = train_inputs[sample_indices, :]
    robustness_samples_labels = train_targets[sample_indices]

    robustness_factory = RobustnessPropertyFactory() \
        .name_prefix('CollisionDetection robustness') \
        .eps(args.radius).desired_extremum('strict_max')
    specification = tuple(
        robustness_factory.get_properties(
            robustness_samples_inputs, robustness_samples_labels
        )
    )

    loss_criterion = torch.nn.CrossEntropyLoss()
    network = deepcopy(original_network)

    def loss(inputs=train_inputs, targets=train_targets) -> torch.Tensor:
        outputs = network(inputs)
        return loss_criterion(outputs, targets)

    wrapped_loss = TrackingLossFunction(loss, 'task loss')
    test_loss = partial(loss, test_inputs, test_targets)

    def test_accuracy(inputs=test_inputs, targets=test_targets):
        output = network(inputs)
        return accuracy2(targets, output)

    optimizer = torch.optim.Adam(network.parameters())
    training_loop = TrainingLoop(network, optimizer, wrapped_loss)
    loss_logger = LogLoss(
        log_frequency=10,
        additional_losses=(
            wrapped_loss.get_additional_losses(average=True)
            + (('test loss', test_loss, True), ('test accuracy', test_accuracy, True))
        ),
        log_level=logging.DEBUG
    )
    training_loop.add_post_iteration_hook(loss_logger)
    wrapped_loss.register_loss_resetting_hook(training_loop)

    reset_optimizer = ResetOptimizer(optimizer)
    training_loop.add_pre_training_hook(reset_optimizer)

    training_loop.add_termination_criterion(
        TrainingLossChange(
            change_threshold=0.005, iteration_block_size=10, num_blocks=5
        )
    )
    training_loop.add_termination_criterion(Divergence(network.parameters()))
    training_loop.add_termination_criterion(
        IterationMaximum(5000)
    )  # make sure training ever stops

    experiment_base.register_hooks(training_loop)

    run_name = f"{first_index}-{first_index+10}"
    experiment_base.execute(
        network, specification, experiment_name=run_name,
        repair_backends={
            L1_PENALTY_KEY: lambda: PenaltyFunctionRepairDelegate(
                training_loop, penalty_function=PenaltyFunction.L1, maximum_updates=25
            ),
        },
        losses=(('training set', loss), ('test set', test_loss),
                ('test accuracy', test_accuracy)),
        # do not skip properties here because they tend to get violated again
        repair_network_further_kwargs={
            'do_not_skip_properties': True,
            'abort_on_backend_error': True
        }
    )
