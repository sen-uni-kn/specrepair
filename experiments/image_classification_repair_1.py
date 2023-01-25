import logging
import os.path
from functools import partial
from logging import info
from tqdm import tqdm

import dill
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nn_repair.backends import PenaltyFunctionRepairDelegate, PenaltyFunction

from nn_repair.training import TrainingLoop, IterationMaximum, TrainingLossChange, LogLoss, \
    ResetOptimizer, Divergence
from nn_repair.training.loss_functions import accuracy2
from experiment_base import ExperimentBase, TrackingLossFunction, L1_PENALTY_KEY, seed_rngs

from datasets import Datasets, get_dataset, get_test_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification repair 1: Robustness on <n> samples')
    ExperimentBase.argparse_add_arguments(
        parser, [L1_PENALTY_KEY],
        default_falsifier_cascade="PGD[SGD, 10]", default_verifier="ERAN_plain"
    )
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument('dataset', type=str,
                                  help='The name of the dataset for which a network should be repaired. '
                                       'Possible options: mnist, cifar10.')
    experiment_group.add_argument('network', type=str,
                                  help='The file name of the network to repair inside the resources/mnist or '
                                       'resources/cifar10 directory. Needs to be a stored pytorch NeuralNetwork. '
                                       'Omit the file extensions.')
    experiment_group.add_argument('--cpu', action="store_true",
                                  help="Train on CPU instead of CPU. When no GPU is available,"
                                       "training is always conducted on CPU.")
    experiment_group.add_argument('--cuda_device', type=int, default=0,
                                  help="The CUDA device to use for training when training on GPU.")
    inputs_group = parser.add_argument_group('Robust data points')
    inputs_group.add_argument('--num_data_points', default=1, dest='n', type=int,
                              help='The number of robustness properties in the specification.')
    inputs_group.add_argument('--first_property', default=0, dest='first_property', type=int,
                              help='The first property from the specification that should be repaired.')
    inputs_group.add_argument('--radius', default='001', dest='radius', type=str,
                              help='Determines the specification/non_robust file '
                                   '(generated with find_non_robust_inputs.py) '
                                   'to use as specification. The value is the suffix of the respective file.')

    args = parser.parse_args()
    seed_rngs(12426022)

    if args.dataset.upper() == 'MNIST':
        dataset = Datasets.MNIST
        dataset_name = 'mnist'
        batch_size = 32  # the batch size used for training
        batch_size_big = 5000  # the batch size used for logging losses after repair
        training_epochs = 0.1

        def get_optimizer(params):
            return torch.optim.SGD(params, lr=0.001, momentum=0.9)

        training_data, test_data = get_dataset(Datasets.MNIST)

        verifier_runtime_threshold_initial = 0.07  # minimal early exit runtime
        verifier_runtime_threshold_target = 15.07  # mean optimal verifier runtime
        verifier_runtime_threshold_target_at = 3  # median #repair steps for optimal
    elif args.dataset.upper() == 'CIFAR10':
        dataset = Datasets.CIFAR10
        dataset_name = 'cifar10'
        batch_size = 32
        # batch_size_big = 2048  # good for phobos90, too large for PC
        batch_size_big = 1024
        training_epochs = 0.1

        def get_optimizer(params):
            return torch.optim.Adam(params, lr=0.0001)

        # training_data = datasets.CIFAR10(
        #     root="../datasets", train=True, download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32, 4),
        #     ])
        # )
        # test_data = datasets.CIFAR10(
        #     root="../datasets", train=False, download=True,
        #     transform=transforms.ToTensor()
        # )
        training_data, test_data = get_dataset(Datasets.CIFAR10)

        verifier_runtime_threshold_initial = 0.07  # TODO
        verifier_runtime_threshold_target = 15.07  # TODO
        verifier_runtime_threshold_target_at = 3  # TODO
    else:
        raise RuntimeError(f"Dataset not supported by image_classification_repair_1: {args.dataset}")
    experiment_base = ExperimentBase(
        f'{dataset_name}_repair_1',
        args,
        initial_verifier_runtime_threshold=verifier_runtime_threshold_initial,
        verifier_runtime_threshold_update=lambda i, _:
        verifier_runtime_threshold_initial
        + (i+1) * (verifier_runtime_threshold_target - verifier_runtime_threshold_initial)
        / verifier_runtime_threshold_target_at,
        initial_verifier_runtime_threshold_decrease=verifier_runtime_threshold_target,
        verifier_runtime_threshold_update_decrease=lambda i, _:
        verifier_runtime_threshold_target - (i+1) *
        (verifier_runtime_threshold_target - verifier_runtime_threshold_initial)
        / verifier_runtime_threshold_target_at
    )

    if args.cpu or not torch.cuda.is_available():
        info("Training on CPU")
        device = torch.device('cpu')
        num_workers = 4
        pin_memory = False
    else:
        info("Training on GPU")
        device = torch.device(f'cuda:{args.cuda_device}')
        num_workers = 0
        pin_memory = True

    network_file = os.path.join('..', 'resources', dataset_name, args.network)
    network = torch.load(network_file + '.pyt').to(device)

    training_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    loss_criterion = torch.nn.CrossEntropyLoss().to(device)

    spec_file = network_file + '_not_robust_' + args.radius + '.dill'
    info(f'Loading properties from: {spec_file}')
    with open(spec_file, 'rb') as spec_file:
        properties = dill.load(spec_file)

    # select n properties from the loaded file, starting at property_index
    spec_start_index = args.first_property
    spec_end_index = spec_start_index + args.n
    specification = properties[spec_start_index:spec_end_index]

    def loss() -> torch.Tensor:
        inputs, targets = next(iter(training_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = network(inputs)
        return loss_criterion(outputs, targets)
    wrapped_loss = TrackingLossFunction(loss, 'task loss')

    test_loader = DataLoader(test_data, batch_size=batch_size_big, shuffle=False, num_workers=4)
    training_loader_2 = DataLoader(training_data, batch_size=batch_size_big, shuffle=False, num_workers=4)

    @torch.no_grad()
    def full_loss(loader, name=None) -> torch.Tensor:
        data_iter = tqdm(iter(loader), name)
        loss_sum = sum(loss_criterion(network(inputs.to(device)), targets.to(device))
                       for inputs, targets in data_iter)
        return loss_sum/len(loader)

    @torch.no_grad()
    def full_accuracy(loader, name="") -> float:
        data_iter = tqdm(iter(loader), name)
        accuracy_sum = sum(accuracy2(targets.to(device), network(inputs.to(device)))
                           for inputs, targets in data_iter)
        return accuracy_sum/len(loader)

    optimizer = get_optimizer(network.parameters())
    training_loop = TrainingLoop(network, optimizer, wrapped_loss)
    epoch_length = len(training_loader)
    loss_logger = LogLoss(log_frequency=epoch_length // 50, epoch_length=epoch_length,
                          average_training_loss=True,
                          additional_losses=wrapped_loss.get_additional_losses(average=True),
                          log_level=logging.DEBUG)
    training_loop.add_post_iteration_hook(loss_logger)
    accuracy_logger = LogLoss(
        log_frequency=epoch_length, epoch_length=epoch_length,
        average_training_loss=False,
        additional_losses=(
            ("training set accuracy", partial(full_accuracy, training_loader_2), False),
            ("test set accuracy", partial(full_accuracy, test_loader), False)
        ),
    )
    wrapped_loss.register_loss_resetting_hook(training_loop)

    training_loop.add_pre_training_hook(ResetOptimizer(optimizer))

    # just train for a few iterations every time
    training_loop.add_termination_criterion(IterationMaximum(training_epochs * epoch_length))
    training_loop.add_termination_criterion(Divergence(network.parameters()))

    experiment_base.register_hooks(training_loop, training_checkpointing_frequency=epoch_length)

    if args.n > 1:
        experiment_name = f'data_points_{spec_start_index}-{spec_end_index}'
    else:
        experiment_name = f'data_point_{spec_start_index}'
    experiment_base.execute(
        network, specification,
        experiment_name=experiment_name,
        repair_backends={
            L1_PENALTY_KEY: lambda: PenaltyFunctionRepairDelegate(
                training_loop, penalty_function=PenaltyFunction.L1, maximum_updates=25,
                device=device,
            ),
        },
        losses=(('training set loss', partial(full_loss, training_loader_2, 'training set loss')),
                ('training set accuracy', partial(full_accuracy, training_loader_2, 'training set accuracy')),
                ('test set loss', partial(full_loss, test_loader, 'test set loss')),
                ('test set accuracy', partial(full_accuracy, test_loader, 'test set accuracy'))),
        repair_network_further_kwargs={
            # do not skip properties here because in other robustness experiments they tended to get violated again
            'do_not_skip_properties': True,
            'abort_on_backend_error': True
        }
    )
