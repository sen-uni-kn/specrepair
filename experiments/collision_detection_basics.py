from typing import Tuple

import torch
import pandas

from deep_opt import NeuralNetwork


def get_collision_detection_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    test_data = pandas.read_csv("../resources/collision_detection/CollisionDetection_test_data.csv")
    training_data = pandas.read_csv("../resources/collision_detection/CollisionDetection_train_data.csv")

    train_inputs = torch.tensor(training_data.drop('class', axis=1).values, dtype=torch.float)
    train_targets = torch.tensor(training_data['class'].values, dtype=torch.long)
    test_inputs = torch.tensor(test_data.drop('class', axis=1).values, dtype=torch.float)
    test_targets = torch.tensor(test_data['class'].values, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets


def get_collision_detection_network(name='ReLU_FFNN_1') -> NeuralNetwork:
    return torch.load(f"../resources/collision_detection/CollisionDetection_{name}.pyt")


def get_relu_ffnn_1_non_robust_samples_collision_detection(n=27):
    # take 25 samples that are not locally robust for repair
    # sample 17 is also not robust, but it is not classified correctly by the ReLU_FFNN_1 network
    # if a robustness specification with radius 0.05 is created around these samples, the specification
    # is contradiction free (although some input intervals do overlap, but the outputs for these overlapping
    # intervals agree)
    sample_indices = [
        4, 5, 6, 8, 10, 11, 13, 16, 18, 19, 20, 22, 23, 25, 27, 28, 30, 31, 33, 34,
        36, 37, 39, 42, 44, 47, 49
    ]
    assert n <= len(sample_indices), "Too few non robust samples known"
    return sample_indices[:n]
