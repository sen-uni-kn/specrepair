from typing import Tuple

from datetime import datetime
import os
import re
import argparse

import pandas
import torch
import math

from tqdm import tqdm
import ray

from deep_opt import NeuralNetwork
from nn_repair.training.loss_functions import accuracy
from properties import get_properties

# Calculate the fidelity (in the spirit of Dong et al. 2020: Towards Repairing Neural Networks Correctly)
# of a repaired network for a large grid of the input space.
# Using a grid and the size of the grid come from the HCAS training procedure.
# Fidelity measures the deviation of the repaired network from the predictions of the original network.
# Instead of measuring only the accuracy deviation like Dong et al do, we also measure
# the score deviation using the mean absolute error.

GRID_SPLITS_PER_DIM = 30
BATCH_SIZE = 2 ** 14

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Grid Fidelity')
    parser.add_argument('repaired_networks_directory',
                        help='The directory containing the repaired networks to analyse. '
                             'This script will descend into subdirectories to arbitrary depth.')
    args = parser.parse_args()

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ray.init()

    acasxu_properties = get_properties()
    # we require the following naming scheme of the directories which ultimately contain repaired networks
    # **/property_X_Y_Z_net_X_Y/repaired_network.pyt
    repaired_network_file_regex = re.compile(r'.*property_([1-9]+[_[1-9]*)_net_([1-9])_([1-9])' + os.path.sep +
                                             r'repaired_network\.pyt')

    # tuple of: first: repaired network, second: original network, third: specification
    # fourth: acasxu network index 1, fifth: acasxu network index 2, sixth: property indices
    cases = []
    for root, dirs, files in os.walk(args.repaired_networks_directory):
        for file in files:
            path_to_file = os.path.join(root, file)
            regex_match = repaired_network_file_regex.match(path_to_file)
            if regex_match is not None:
                # repaired network is a pytorch file
                repaired_network_file = path_to_file
                repaired_network = torch.load(repaired_network_file)
                acasxu_i1 = int(regex_match.groups()[1])  # 0 is property index
                acasxu_i2 = int(regex_match.groups()[2])
                original_network = NeuralNetwork.load_from_nnet(f'../resources/acasxu/'
                                                                f'ACASXU_run2a_{acasxu_i1}_{acasxu_i2}_batch_2000.nnet')
                property_indices = tuple(int(i_str) for i_str in regex_match.groups()[0].split('_'))
                specification = tuple(acasxu_properties[i-1][0] for i in property_indices)
                cases.append((repaired_network, original_network, specification,
                              acasxu_i1, acasxu_i2, property_indices))

    fidelity_results = []
    cases_counter = 1
    for repaired, original, spec, i1, i2, spec_is in cases:
        print(
            f'({cases_counter}/{len(cases)}) Calculating grid fidelity for '
            f'repaired network {i1}, {i2} (repaired for properties {spec_is})'
        )
        property_input_bounds = [prop.input_bounds(original) for prop in spec]
        property_lower_bounds = [torch.tensor([lb for lb, _ in bounds]) for bounds in property_input_bounds]
        property_upper_bounds = [torch.tensor([ub for _, ub in bounds]) for bounds in property_input_bounds]

        def satisfies_specification(inputs: torch.Tensor, network: NeuralNetwork) -> torch.Tensor:
            # first dim: property, second dim: batch
            outside_lower_bounds = torch.vstack([
                torch.any(inputs < property_lower_bounds[i], dim=1) for i in range(len(spec))
            ])
            outside_upper_bounds = torch.vstack([
                torch.any(inputs > property_upper_bounds[i], dim=1) for i in range(len(spec))
            ])
            output_constraint_satisfied = torch.vstack(
                [prop.property_satisfied(inputs, network) for prop in spec]
            )
            return torch.all(outside_lower_bounds | outside_upper_bounds | output_constraint_satisfied, dim=0)

        input_bounds = original.get_bounds()

        num_grid_points = GRID_SPLITS_PER_DIM ** len(input_bounds)
        num_batches = math.ceil(num_grid_points / BATCH_SIZE)
        print(f'Processing {num_grid_points} grid points in batches of {BATCH_SIZE} ({num_batches} batches).')

        # these start indices need to be interpreted as base 40 to yield grid indices
        start_grid_indices = torch.arange(start=0, end=num_grid_points, step=BATCH_SIZE)
        grid_elements_per_dim = [torch.linspace(start=lower, end=upper, steps=GRID_SPLITS_PER_DIM)
                                 for lower, upper in input_bounds]

        def increase_counter(counters):
            carry_over = True
            for j, i in enumerate(counters):
                counters[j] = (i + carry_over) % GRID_SPLITS_PER_DIM
                carry_over = counters[j] < i
                if not carry_over:
                    break
            return counters

        @ray.remote
        def process_batch(start_grid_index_) -> Tuple[float, float, int]:
            start_index = start_grid_index_
            counters = []
            for _ in range(len(input_bounds)):
                start_index, dim_index = divmod(start_index, GRID_SPLITS_PER_DIM)
                counters.append(dim_index)

            batch_inputs = torch.empty((BATCH_SIZE, len(input_bounds)), requires_grad=False)
            for k in range(BATCH_SIZE):
                for j, i in enumerate(counters):
                    batch_inputs[k, j] = grid_elements_per_dim[j][i]
                counters = increase_counter(counters)
                if start_grid_index_ + k >= num_grid_points:
                    # remove all remaining empty rows
                    batch_inputs = batch_inputs[:k, :]
                    break

            # drop all grid points for which the original network violates the specification
            spec_satisfied = satisfies_specification(batch_inputs, original)
            batch_inputs_filtered = batch_inputs[spec_satisfied, :]
            orig_scores = original(batch_inputs)
            orig_scores = orig_scores[spec_satisfied, :]

            repaired_scores = repaired(batch_inputs_filtered)

            fidelity_mae_ = torch.abs(orig_scores - repaired_scores).mean()
            fidelity_accuracy_ = accuracy(orig_scores, repaired_scores, classification_mode='min')
            return fidelity_accuracy_.item(), fidelity_mae_.item(), batch_inputs_filtered.shape[0]

        fidelity_accuracy = 0.0
        fidelity_mae = 0.0
        num_data_points = 0  # some grid points may be removed due to specification violations
        result_ids = []
        for start_grid_index in start_grid_indices:
            result_ids.append(process_batch.remote(start_grid_index.item()))

        with tqdm(total=len(result_ids)) as progress_bar:
            while result_ids:
                done, result_ids = ray.wait(result_ids)
                acc, mae, num = ray.get(done[0])
                fidelity_accuracy += acc
                fidelity_mae += mae
                num_data_points += num
                progress_bar.update()

        fidelity_accuracy /= len(start_grid_indices)
        fidelity_mae /= len(start_grid_indices)
        print(f'Repaired network {i1}, {i2} (properties {spec_is}): '
              f'accuracy: {fidelity_accuracy * 100:.2f}%, mae: {fidelity_mae} (n: {num_data_points}/{num_grid_points})')
        fidelity_results.append({
            'specification': spec_is, 'network_i1': i1, 'network_i2': i2,
            'accuracy_fidelity': fidelity_accuracy, 'mae_fidelity': fidelity_mae,
            'n': num_data_points})
        cases_counter += 1

    fidelity_results = pandas.DataFrame(fidelity_results)
    output_file = args.repaired_networks_directory + f'/grid_fidelity_calculation_{time}.csv'
    print(fidelity_results)
    print(f'Saving results in file: {output_file}')
    fidelity_results.sort_values(['network_i1', 'network_i2'], inplace=True)
    fidelity_results.to_csv(output_file, index=False)
