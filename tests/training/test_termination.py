import unittest
import logging

import numpy as np
import torch

from deep_opt import NeuralNetwork
from deep_opt.models import diff_approx
from nn_repair.training import TrainingLoop, IterationMaximum, TrainingLossValue, TrainingLossChange, GradientValue, \
    ValidationSet, LogLoss

logging.basicConfig(level=logging.DEBUG)


def get_training_loop():
    rng = np.random.default_rng(171459)
    data = torch.as_tensor(rng.random((100, 8), dtype=np.float32))
    train_set = data[:60, :]
    validation_set = data[60:, :]
    train_X = train_set[:, :5]
    train_Y = train_set[:, 5:]
    validation_X = validation_set[:, :5]
    validation_Y = validation_set[:, 5:]

    network = NeuralNetwork(
        modules=[diff_approx.Linear(5, 100), diff_approx.ReLU(), diff_approx.Linear(100, 3)]
    )
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    def loss():
        return criterion(network(train_X), train_Y)

    training_loop = TrainingLoop(network, optimizer, loss, post_iteration_hooks=[LogLoss(log_frequency=1)])
    return training_loop, (train_X, train_Y), (validation_X, validation_Y), network, criterion


class MyTestCase(unittest.TestCase):
    def test_iteration_maximum(self):
        loop, _, _, _, _ = get_training_loop()
        loop.add_termination_criterion(IterationMaximum(10))
        loop.execute()

        self.assertEqual(loop.iteration, 10)

    def test_training_loss_value(self):
        loop, _, _, _, _ = get_training_loop()
        loop.add_termination_criterion(TrainingLossValue(0.3))
        loop.execute()

        self.assertLessEqual(loop.current_training_loss, 0.3)

    def test_training_loss_change(self):
        loop, _, _, _, _ = get_training_loop()
        loop.add_termination_criterion(TrainingLossChange(0.1))
        loop.execute()

    def test_gradient_value(self):
        loop, _, _, network, _ = get_training_loop()
        loop.add_termination_criterion(GradientValue(network.parameters(), 0.001))
        loop.execute()

    def test_validation_set(self):
        loop, _, validation_set, network, criterion = get_training_loop()
        validation_X, validation_Y = validation_set

        def validation_loss():
            return criterion(network(validation_X), validation_Y)
        loop.clear_post_iteration_hooks()
        loop.add_post_iteration_hook(LogLoss(1, additional_losses=[('validation', validation_loss, False)]))
        loop.add_termination_criterion(ValidationSet(validation_loss, 3, 3, 0.0))
        loop.execute()



