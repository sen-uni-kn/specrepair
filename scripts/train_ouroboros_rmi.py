import argparse
import logging
from datetime import datetime
from math import log10
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import trange

from deep_opt import NeuralNetwork
from deep_opt.models.property import ConstraintAnd, BoxConstraint, Property
from nn_repair.training import (
    IterationMaximum, LogLoss, TrainingLoop
)

from nn_repair.datasets import IntegerDataset

if __name__ == "__main__":
    # Train a Recursive Model Index. See
    # Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis:
    # The Case for Learned Index Structures. SIGMOD Conference 2018: 489-504
    # https://doi.org/10.1145/3183713.3196909
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser("Train RMI")
    parser.add_argument("--dataset", default="uniform",
                        choices=("uniform", "normal", "log-normal"),
                        help="Which dataset to use. Possible values: "
                             "uniform, normal, log-normal: IntegerDataset with "
                             "the given probability distribution and the default size "
                             "and maximal value. ")
    parser.add_argument("--second_stage_size", default=10, type=int,
                        help="The number of models in the second stage of the RMI.")
    parser.add_argument("--first_stage_tolerance", default=1, type=int,
                        help="The permitted error of the first stage model of the RMI."
                             "The error of the first stage model in picking a model "
                             "from the second stage needs to be bounded by this error.")
    parser.add_argument("--second_stage_tolerance", default=100, type=int,
                        help="The permitted error of a second stage model of the RMI."
                             "This is the maximum permitted distance between the true "
                             "position of a key and the predicted position.")
    parser.add_argument("--dataset_seed", default=0, type=int,
                        help="The seed for generating the dataset.")
    parser.add_argument("--rmi_name", default=None, type=str,
                        help="The name to use for storing the trained RMI.")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--skip_second_stage", action="store_true",
                        help="Only train a stage one model.")
    args = parser.parse_args()

    if args.rmi_name is None:
        log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rmi_name = f"rmi_{log_time}"
    else:
        rmi_name = args.rmi_name
    OUTPUT_DIR = Path("..", "resources", "ouroboros_rmi", rmi_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(117787974261777)
    np.random.seed(804265)

    dataset_distribution = None
    if args.dataset.lower() == "uniform":
        dataset_distribution = "uniform"
    elif args.dataset.lower() == "normal":
        dataset_distribution = "normal"
    elif args.dataset.replace("-", "").replace("_", "").lower() == "lognormal":
        dataset_distribution = "log-normal"
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")
    seed = args.dataset_seed
    dataset = IntegerDataset(
        root="../datasets", distribution=dataset_distribution, seed=seed
    )

    second_stage_size = args.second_stage_size
    first_stage_tolerance = args.first_stage_tolerance
    second_stage_tolerance = args.second_stage_tolerance

    with open(OUTPUT_DIR / "params.yaml", "xt") as file:
        file.write(
            f"second_stage_size: {second_stage_size}\n"
            f"tolerance:\n"
            f"    first_stage: {first_stage_tolerance}\n"
            f"    second_stage: {second_stage_tolerance}\n"
            f"dataset:\n"
            f"    size: {dataset.size}\n"
            f"    distribution: {dataset.distribution}\n"
            f"    seed: {dataset.seed}\n"
            f"    maximum: {dataset.maximum}\n"
        )

    partition_size = len(dataset) // second_stage_size
    partitions = np.split(np.arange(len(dataset)), second_stage_size)

    first_stage_layers = [
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    ]
    first_stage_model = NeuralNetwork(
        mins=dataset.data.amin(),
        maxes=dataset.data.amax(),
        means_inputs=dataset.data.float().mean(),
        ranges_inputs=dataset.data.float().std(),
        # means_outputs=second_stage_size / 2.0,
        # ranges_outputs=second_stage_size / 2.0,
        modules=first_stage_layers,
        inputs_shape=(1,),
        outputs_shape=(1,),
    )

    mse = torch.nn.MSELoss()

    batch_size = args.batch_size
    first_stage_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_first_stage_loader = DataLoader(dataset, batch_size=batch_size * 4, shuffle=True, num_workers=0)
    full_first_stage_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)

    def first_stage_preds_parts(data_loader):
        keys, pos = next(iter(data_loader))
        parts = torch.div(pos, partition_size, rounding_mode='floor')

        keys = keys.unsqueeze(-1)
        parts = parts.unsqueeze(-1).float()

        preds = first_stage_model(keys.float())
        return preds, parts

    def first_stage_loss(data_loader=first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return mse(preds, parts)

    def first_stage_accuracy(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return (parts == preds.round()).float().mean() * 100

    def first_stage_mae(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return torch.abs(preds.round() - parts).mean()

    def first_stage_max_error(data_loader=eval_first_stage_loader):
        preds, parts = first_stage_preds_parts(data_loader)
        return torch.abs(preds.round() - parts).amax()

    print("Training Stage 1")
    optimizer = torch.optim.Adam(first_stage_model.parameters(), lr=0.01)
    epoch_len = len(first_stage_loader)

    training_loop = TrainingLoop(
        first_stage_model, optimizer, first_stage_loss
    )
    training_loop.add_post_iteration_hook(LogLoss(
        log_frequency=epoch_len // 10, epoch_length=epoch_len,
        average_training_loss=True,
        additional_losses=[
            ("accuracy", first_stage_accuracy, False),
            ("mea", first_stage_mae, False),
            ("max error", first_stage_max_error, False),
        ]
    ))
    training_loop.add_termination_criterion(
        IterationMaximum(epoch_len * int(log10(second_stage_size) ** 2))
    )

    start_time = time()
    training_loop.execute()
    end_time = time()

    duration = end_time - start_time
    print(f"Training the first stage finished. Duration: {duration:.1f}s")
    print(f"MSE: {first_stage_loss(full_first_stage_loader)}")
    print(f"Accuracy: {first_stage_accuracy(full_first_stage_loader)}")
    print(f"MAE: {first_stage_mae(full_first_stage_loader)}")
    print(f"Max Error: {first_stage_max_error(full_first_stage_loader)}")
    torch.save(first_stage_model, OUTPUT_DIR / "first_stage.pyt")

    # print("Creating Stage 1 Specification.")
    # first_stage_spec = [
    #     Property(
    #         lower_bounds={0: dataset.data[partitions[i]].amin()},
    #         upper_bounds={0: dataset.data[partitions[i]].amax()},
    #         output_constraint=ConstraintAnd(
    #             BoxConstraint(0, ">=", max(0, i - first_stage_tolerance)),
    #             BoxConstraint(0, "<=", min(second_stage_size - 1, i + first_stage_tolerance))
    #         ),
    #         property_name=f"First Stage Error Bound for Partition {i}"
    #     )
    #     for i in range(second_stage_size)
    # ]
    # with open(OUTPUT_DIR / "first_stage_spec.dill", "xb") as file:
    #     dill.dump(first_stage_spec, file)

    if not args.skip_second_stage:
        print("Training Stage 2")
        for second_stage_i in trange(second_stage_size):
            # take all the data from the partitions within the error tolerance of the
            # first stage model
            data_indices = set()
            for i in range(-first_stage_tolerance, first_stage_tolerance + 1):
                i = min(second_stage_size - 1, max(0, i + second_stage_i))
                data_indices.update(partitions[i])
            data_indices = list(data_indices)

            partition_dataset = Subset(dataset, data_indices)
            loader = DataLoader(
                partition_dataset, batch_size=len(partition_dataset), num_workers=0
            )

            full_keys, full_pos = next(iter(loader))
            model = NeuralNetwork(
                mins=full_keys.amin(),
                maxes=full_keys.amax(),
                # means_inputs=model_data.float().mean(),
                # ranges_inputs=model_data.float().std(),
                # means_outputs=pos.float().mean(),
                # ranges_outputs=pos.float().std(),
                modules=[nn.Linear(1, 1)],
                inputs_shape=(1,),
                outputs_shape=(1,),
            )

            print(f"Training Stage 2, Model {second_stage_i}")

            start_time = time()
            with torch.no_grad():
                 # add a feature full of ones for the bias
                 X = torch.hstack([torch.ones(len(full_keys), 1), full_keys.unsqueeze(-1)])
                 X = X.float()
                 theta = torch.linalg.inv(X.T @ X) @ X.T @ full_pos.float()
                 model[0].bias.set_(theta[0].reshape(1))
                 model[0].weight.set_(theta[1:].reshape(1, 1))
            end_time = time()
            duration = end_time - start_time

            def second_stage_preds_pos():
                keys, pos = next(iter(loader))

                keys = keys.unsqueeze(-1)
                pos = pos.unsqueeze(-1).float()

                preds = model(keys.float())
                return preds, pos

            def second_stage_loss():
                preds, pos = second_stage_preds_pos()
                return mse(preds, pos)

            def second_stage_accuracy():
                preds, pos = second_stage_preds_pos()
                return (pos == preds.round()).float().mean()

            def second_stage_mae():
                preds, pos = second_stage_preds_pos()
                return torch.abs(preds.round() - pos).mean()

            def second_stage_max_error():
                preds, pos = second_stage_preds_pos()
                return torch.abs(preds.round() - pos).amax()

            print(f"Training second stage model {second_stage_i} finished. "
                  f"Duration: {duration:.1f}s")
            print(f"MSE: {second_stage_loss()}")
            print(f"Accuracy: {second_stage_accuracy()}")
            print(f"MAE: {second_stage_mae()}")
            print(f"Max Error: {second_stage_max_error()}")
            torch.save(
                model, OUTPUT_DIR / f"second_stage_{second_stage_i}.pyt"
            )

            # print(f"Creating Stage 2 Specification for Model {second_stage_i}.")
            # full_keys, full_pos = zip(*sorted(zip(full_keys.tolist(), full_pos.tolist()), key=lambda t: t[0]))
            # prev_key = (min(full_keys) - 1,) + full_keys[:-1]
            # next_key = full_keys[1:] + (max(full_keys) + 1,)
            # spec = [
            #     Property(
            #         lower_bounds={0: prev_key[key_i] + 1},
            #         upper_bounds={0: next_key[key_i] - 1},
            #         output_constraint=ConstraintAnd(
            #             BoxConstraint(0, ">=", full_pos[key_i] - second_stage_tolerance),
            #             BoxConstraint(0, "<=", full_pos[key_i] + second_stage_tolerance)
            #         ),
            #         property_name=f"Second Stage Error Bound {key_i} for Model {second_stage_i}"
            #     )
            #     for key_i in range(len(full_keys) - 1)
            # ]
            # with open(OUTPUT_DIR / f"second_stage_{second_stage_i}_spec.dill", "xb") as file:
            #     dill.dump(spec, file)
