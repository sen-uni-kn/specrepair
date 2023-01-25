import logging
from datetime import datetime
import argparse
import os
from time import time
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nn_repair.networks.resnetcifar10 import ResNetCIFAR10
from nn_repair.training import TrainingLoop, TensorboardLossPlot, LogLoss, IterationMaximum, Checkpointing
from nn_repair.training.loss_functions import accuracy2

if __name__ == "__main__":
    # Training based on CIFAR10 training in He at al. 2015:
    # Deep Residual Learning for Image Recognition
    # https://arxiv.org/pdf/1512.03385.pdf
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--network_size", default=20, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FILE = f"../resources/cifar10/cifar10_resnet20_{log_time}"
    CHECKPOINT_DIR = f"../resources/cifar10/cifar10_resnet20_{log_time}_checkpoints/"
    try:
        os.mkdir(CHECKPOINT_DIR)
    except FileExistsError:
        pass

    torch.manual_seed(638223539954552)
    np_gen = np.random.default_rng(49024731)

    if args.cpu or not torch.cuda.is_available():
        print("Training on CPU")
        device = torch.device('cpu')
        pin_memory = False
        num_workers = 1
    else:
        print("Training on GPU")
        device = torch.device('cuda')
        pin_memory = True
        num_workers = 0

    print('Loading dataset...')
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    basic_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    training_data = datasets.CIFAR10(
        root="../datasets", train=True, download=True,
        transform=transforms.Compose([
            basic_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
        ])
    )
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_data = datasets.CIFAR10(
        root="../datasets", train=False, download=True, transform=basic_transform
    )
    # this loader will only be used with @torch.no_grad, which reduces memory demand
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 8, shuffle=False)

    print('Loading network...')
    resnet_size = (args.network_size - 2) // 6
    network = ResNetCIFAR10(resnet_size)
    network.to(device)

    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=0.1, momentum=0.9, weight_decay=0.0001)

    epoch_len = len(training_loader)
    step_iters = [32000, 48000]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_iters, 0.1)

    print('Training...')
    start_time = time()

    def train_loss():
        batch_inputs, batch_targets = next(iter(training_loader))
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        pred = network(batch_inputs)
        return loss_function(pred, batch_targets)

    test_accs = []

    @torch.no_grad()
    def test_accuracy() -> float:
        data_iter = tqdm(iter(test_loader), "test set accuracy")
        data_iter = ((inputs.to(device), targets.to(device)) for inputs, targets in data_iter)
        accuracy_sum = sum(accuracy2(targets, network(inputs)) for inputs, targets in data_iter)
        test_acc = accuracy_sum/len(test_loader)
        test_accs.append(test_acc.item())
        return test_acc

    training_loop = TrainingLoop(network, optimizer, train_loss, lr_scheduler=lr_scheduler)
    training_loop.add_termination_criterion(IterationMaximum(64000))

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
            tensorboard_writer, frequency=epoch_len // 100, average_training_loss=True,
            training_loss_tag='cifar10_resnet/training',
        ))
    training_loop.execute()
    end_time = time()

    # save the model
    torch.save(network, OUTPUT_FILE + '.pyt')

    duration = end_time - start_time
    print(f"Training finished. Duration: {duration / 3600:.2f}h")
    print(f"Network saved in file {OUTPUT_FILE}.pyt")

    print("====================================================")
    print(f"Testset accuracies / epoch: {test_accs}")
    sys.exit()
