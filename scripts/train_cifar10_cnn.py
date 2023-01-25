import argparse
import logging
from datetime import datetime
from time import time

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
    OUTPUT_FILE = f"../resources/cifar10/cifar10_cnn_{log_time}"
    # CHECKPOINT_DIR = f"../resources/mnist/mnist_cnn_{log_time}_checkpoints/"
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.manual_seed(462910845716582)
    np.random.seed(7683148)

    if args.cpu or not torch.cuda.is_available():
        print("Training on CPU")
        device = torch.device('cpu')
        pin_memory = False
    else:
        print("Training on GPU")
        device = torch.device('cuda:1')
        pin_memory = True

    print("Loading dataset...")
    train_data = datasets.CIFAR10(
        root="../datasets", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
        ])
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory,
    )

    test_data = datasets.CIFAR10(
        root="../datasets", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    # this loader will only be used with @torch.no_grad, which reduces memory demand
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 32, shuffle=False)
    train_loader_2 = DataLoader(train_data, batch_size=args.batch_size * 32, shuffle=False)

    # based on: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    layers = [
        nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 8 x 32 x 32
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # 8 x 16 x 16

        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # 16 x 8 x 8

        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ]
    per_pixel_mean, per_pixel_std = torch.load(
        "../resources/cifar10/train_set_per_pixel_mean_and_std.pyt"
    )
    network = NeuralNetwork(
        mins=0.0, maxes=1.0,
        modules=layers,
        means_inputs=per_pixel_mean,
        ranges_inputs=per_pixel_std,
        inputs_shape=(3, 32, 32),
        outputs_shape=(10,),
    )
    network.to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    epoch_len = len(train_loader)
    milestones = [
        1 * epoch_len, 10 * epoch_len
    ]
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=0.1
    )
    # lr_schedule = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.01, max_lr=1.0,
    #     step_size_up=epoch_len * 2, step_size_down=epoch_len * 2,
    # )

    print("Training...")
    start_time = time()

    def train_loss():
        batch_inputs, batch_targets = next(iter(train_loader))
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

    @torch.no_grad()
    def training_accuracy() -> float:
        data_iter = tqdm(iter(train_loader_2), "training set accuracy")
        data_iter = ((inputs.to(device), targets.to(device)) for inputs, targets in data_iter)
        accuracy_sum = sum(accuracy2(targets, network(inputs)) for inputs, targets in data_iter)
        return accuracy_sum/len(train_loader_2)

    training_loop = TrainingLoop(network, optimizer, train_loss)
    training_loop.add_termination_criterion(IterationMaximum(100 * epoch_len))

    # Checkpointing(
    #     output_directory=CHECKPOINT_DIR, frequency=epoch_len,
    #     save_model_state=True, save_optimizer_state=True
    # ).register(training_loop)

    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len // 50, epoch_length=epoch_len,
    ))
    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len, epoch_length=epoch_len,
        additional_losses=(
            ("test set accuracy", test_accuracy, False),
            ("training set accuracy", training_accuracy, False)
        )
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
