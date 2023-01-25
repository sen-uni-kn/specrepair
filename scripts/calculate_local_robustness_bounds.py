import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_opt import RobustnessPropertyFactory
from nn_repair import CounterexampleGenerator
from nn_repair.verifiers import ERAN
from nn_repair.falsifiers import ProjectedGradientDescentAttack
from experiments.datasets import collision_detection


def check_local_robustness(network, loader, robustness_factory: RobustnessPropertyFactory,
                           cx_gen: CounterexampleGenerator, classification_mode='max', progress_bar=True) \
        -> Tuple[float, float]:
    """
    Returns the fraction of certified and falsified samples in the given dataset.

    :param network: The network to check.
    :param loader: A data loader that produces the samples for which to check local robustness.
    :param robustness_factory: The robustness factory for generating properties from data.
      Set the eps.
    :param cx_gen: The verifier or falsifier to use for checking local robustness
    :param classification_mode: Whether the maximal or minimal output of the network provides
     the classification.
    :param progress_bar: Whether to display a progress bar or not.
    :return: The fraction of samples for which robustness was proven and the faction of samples
      for which robustness was disproven.
      For the gap between these values, the result is unknown.
    """
    assert classification_mode in ('min', 'max')

    data_iter = iter(loader)
    if progress_bar:
        data_iter = tqdm(data_iter, total=len(loader))
    verified_counter = 0.0
    falsified_counter = 0.0
    total = 0.0
    for inputs, _ in data_iter:
        network_outputs = network(inputs)
        if classification_mode == 'max':
            targets = torch.argmax(network_outputs, dim=1)
        else:
            targets = torch.argmin(network_outputs, dim=1)
        properties = robustness_factory.get_properties(input_samples=inputs, labels=targets)
        for prop in properties:
            result, _ = cx_gen.find_counterexample(network, prop)
            if result is not None and len(result) > 0:
                falsified_counter += 1
            elif result is not None:  # len == 0 follows
                verified_counter += 1
            total += 1
    return verified_counter/total, falsified_counter/total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Local Robustness bounds.")
    parser.add_argument("--tool", type=str, default="PGD",
                        help="What verifier or falsifier to use. "
                             "Options: PGD (Projected gradient descent with Adam "
                             "and 10 random restarts), PGD100 (same as PGD, but with 100"
                             "random starts, ERAN_complete (ERAN in complete "
                             "mode; DeepPoly + MILP solving).")
    parser.add_argument("--network", type=str, required=True,
                        help="The path to the network to analyse.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="What dataset the network is for. "
                             "Options: CollisionDetection_train, "
                             "CollisionDetection_test.")
    parser.add_argument("--radius", type=float, required=True,
                        help="The radius (eps) of the robustness specification.")
    args = parser.parse_args()

    cx_gen_ = None
    if args.tool.upper() == "ERAN_COMPLETE":
        cx_gen_ = ERAN()
    if args.tool.upper() == "ERAN_PLAIN_COMPLETE":
        cx_gen_ = ERAN(use_acasxu_style=False)
    elif args.tool.upper() == "PGD":
        cx_gen_ = ProjectedGradientDescentAttack(
            'Adam', num_restarts=10, progress_bar=False
        )
    elif args.tool.upper() == "PGD100":
        cx_gen_ = ProjectedGradientDescentAttack(
            'Adam', num_restarts=100, progress_bar=False
        )
    else:
        raise NotImplementedError(f"Unknown verifier/falsifier: {args.tool}.")

    network_ = torch.load(args.network)

    if args.dataset.upper().startswith("COLLISIONDETECTION"):
        classification_mode_ = "max"
        train_set, test_set = collision_detection()
        train_loader = DataLoader(train_set, batch_size=1)
        test_loader = DataLoader(test_set, batch_size=1)

        robustness_factory_ = RobustnessPropertyFactory()\
            .desired_extremum("strict_max").eps(args.radius)

        if args.dataset.upper() == "COLLISIONDETECTION_TRAIN":
            loader_ = train_loader
        elif args.dataset.upper() == "COLLISIONDETECTION_TEST":
            loader_ = test_loader
        else:
            raise NotImplementedError(f"Unknown dataset: {args.dataset}.")
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}.")

    verif_frac, falsif_frac = check_local_robustness(
        network_, loader_, robustness_factory_, cx_gen_, classification_mode_
    )

    print(
        f"Result: verified: {verif_frac*100:.2f}%, falsified: {falsif_frac*100:.2f}%\n"
        f"=========================================================================="
    )
