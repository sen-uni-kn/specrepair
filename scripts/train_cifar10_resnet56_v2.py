import logging
from datetime import datetime
import argparse
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nn_repair.training import TrainingLoop, TensorboardLossPlot, LogLoss, IterationMaximum, Checkpointing
from nn_repair.training.loss_functions import accuracy2
from nn_repair.networks.resnet_v2 import ResNetv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FILE = "../resources/cifar10/cifar10_resnet56_v2.pyt"
    CHECKPOINT_DIR = "../resources/cifar10/cifar10_resnet45_v2_checkpoints/"
    try:
        os.mkdir(CHECKPOINT_DIR)
    except FileExistsError:
        pass

    torch.manual_seed(220515)

    print('Loading dataset...')
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    basic_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    training_data = datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        basic_transform
    ]))
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=not args.cpu
    )
    test_data = datasets.CIFAR10(root="../datasets", train=False, download=True, transform=basic_transform)
    # this loader will only be used with @torch.no_grad, which reduces memory demand
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 8, shuffle=False)

    network = ResNetv2(input_shape=(3, 32, 32), num_classes=10, stage_size=6)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    epoch_len = len(training_loader)
    # c.f. https://github.com/Jianbo-Lab/HSJA/blob/master/resnet.py (simplified)
    step_iters = [epoch * epoch_len for epoch in [80, 120, 160, 180]]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_iters, 0.1)

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        network = torch.nn.DataParallel(network)

    network = network.to(device)
    loss_function = loss_function.to(device)

    print('Training...')

    def train_loss():
        batch_inputs, batch_targets = next(iter(training_loader))
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        pred = network(batch_inputs)
        return loss_function(pred, batch_targets)

    @torch.no_grad()
    def test_accuracy() -> float:
        data_iter = tqdm(iter(test_loader), "test set accuracy")
        data_iter = ((inputs.to(device), targets.to(device)) for inputs, targets in data_iter)
        accuracy_sum = sum(accuracy2(targets, network(inputs)) for inputs, targets in data_iter)
        return accuracy_sum/len(test_loader)

    training_loop = TrainingLoop(network, optimizer, train_loss, lr_scheduler=lr_scheduler)
    training_loop.add_termination_criterion(IterationMaximum(200 * epoch_len))

    Checkpointing(
        output_directory=CHECKPOINT_DIR, frequency=epoch_len,
        save_model_state=True, save_optimizer_state=True
    ).register(training_loop)

    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len // 50, epoch_length=epoch_len,
    ))
    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len, epoch_length=epoch_len,
        additional_losses=(("test set accuracy", test_accuracy, False), )
    ))
    if args.tensorboard:
        tensorboard_writer = SummaryWriter('../tensorboard', flush_secs=5)
        training_loop.add_post_iteration_hook(TensorboardLossPlot(
            tensorboard_writer, frequency=epoch_len // 50, average_training_loss=True,
            training_loss_tag='cifar10_resnet56_v2/training',
        ))
    training_loop.execute()

    # save the model
    torch.save(network, OUTPUT_FILE + '.pyt')
