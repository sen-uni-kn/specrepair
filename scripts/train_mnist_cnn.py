import argparse
import logging
import os
from datetime import datetime
from time import time
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from deep_opt import NeuralNetwork
from nn_repair.training import (
    Checkpointing, IterationMaximum, LogLoss,
    TensorboardLossPlot, TrainingLoop
)
from nn_repair.training.loss_functions import accuracy2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FILE = f"../resources/mnist/mnist_cnn_{log_time}"
    # CHECKPOINT_DIR = f"../resources/mnist/mnist_cnn_{log_time}_checkpoints/"
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.manual_seed(933978100148128)
    np.random.seed(148128)

    g = torch.Generator()
    g.manual_seed(144930926225246)

    if args.cpu or not torch.cuda.is_available():
        print("Training on CPU")
        device = torch.device('cpu')
        pin_memory = False
    else:
        print("Training on GPU")
        device = torch.device('cuda')
        pin_memory = True

    print("Loading dataset...")
    train_data = datasets.MNIST(
        root="../datasets", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomRotation(22.5),
            # transforms.RandomCrop(28, 4)
        ])
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory,
    )

    test_data = datasets.MNIST(
        root="../datasets", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    # this loader will only be used with @torch.no_grad, which reduces memory demand
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 8, shuffle=False)

    network = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=3, padding=1),  # 8 x 10 x 10
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(800, 80),
        nn.ReLU(),
        nn.Linear(80, 10)
    )
    network.to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    epoch_len = len(train_loader)
    # lr_schedule = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=epoch_len // 2, gamma=0.1
    # )
    # lr_steps = [epoch_len // 2, epoch_len]
    # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, gamma=0.1)

    print("Training...")
    start_time = time()

    def train_loss():
        batch_inputs, batch_targets = next(iter(train_loader))
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        pred = network(batch_inputs)
        return loss_function(pred, batch_targets)

    def test_accuracy() -> float:
        data_iter = tqdm(iter(test_loader), "test set accuracy")
        data_iter = ((inputs.to(device), targets.to(device)) for inputs, targets in data_iter)
        accuracy_sum = sum(accuracy2(targets, network(inputs)) for inputs, targets in data_iter)
        return accuracy_sum/len(test_loader)

    training_loop = TrainingLoop(network, optimizer, train_loss)
    training_loop.add_termination_criterion(IterationMaximum(2 * epoch_len))

    # Checkpointing(
    #     output_directory=CHECKPOINT_DIR, frequency=epoch_len,
    #     save_model_state=True, save_optimizer_state=True
    # ).register(training_loop)

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
            training_loss_tag='cifar10_resnet/training',
        ))
    training_loop.execute()
    end_time = time()

    network.cpu()
    network = NeuralNetwork(
        mins=0.0, maxes=1.0,
        modules=network,
        inputs_shape=(1, 28, 28),
        outputs_shape=(10,),
    )

    # save the model
    torch.save(network, OUTPUT_FILE + '.pyt')

    duration = end_time - start_time
    print(f"Training finished. Duration: {duration / 3600:.2f}h")
    print(f"Network saved in file {OUTPUT_FILE}.pyt")

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("                                 WARNING                                   ")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Manually merge all batch-normalization layers into the preceeding conv")
    print("layer when using ERAN.")
    print("Not doing so will lead to inconsistent verification results.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
