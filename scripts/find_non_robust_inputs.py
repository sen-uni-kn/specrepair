# this script finds non-robust inputs for various datasets
import argparse
from datetime import datetime
import pickle
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import ray
from tqdm import tqdm

from deep_opt import NeuralNetwork, RobustnessPropertyFactory, dump_specification
from nn_repair.falsifiers import FastGradientSignMethod, DeepOpt, ProjectedGradientDescentAttack, \
    DifferentialEvolutionPGDAttack
from nn_repair.verifiers import ERAN

from experiments.datasets import mnist, fashion_mnist, cifar10, collision_detection


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(9604155)
    torch.manual_seed(586119968983443)

    parser = argparse.ArgumentParser('Find non-robust input data points from datasets')
    parser.add_argument('dataset', type=str)
    parser.add_argument('network_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('-e --eps', dest='eps', default=0.01, type=float)
    parser.add_argument('-n --amount', dest='n', default=None, type=int)
    parser.add_argument('--comparison_mode', action='store_true',
                        help='Compare the strengths of the attacks. '
                             'If this flag is set all attacks will be executed for all samples '
                             'to assess their relative strength')
    parser.add_argument('--fgsm', action='store_true')
    parser.add_argument('--pgda', action='store_true')
    parser.add_argument('--pgda_adam', action='store_true')
    parser.add_argument('--dea', action='store_true')
    parser.add_argument('--stronger_dea', action='store_true')
    parser.add_argument('--deep_opt', action='store_true')
    parser.add_argument('--eran', action='store_true')
    parser.add_argument('--use_dill', action='store_true',
                        help='Store the generated specification using dill instead of yaml')
    parser.add_argument('--show_results', action='store_true')
    args = parser.parse_args()

    dataset = None
    if args.dataset.upper() == 'MNIST':
        print("Using MNIST training set")
        dataset = mnist(train_set=True, test_set=False)
    elif args.dataset.upper() == 'FASHION_MNIST' or args.dataset.upper() == 'FASHIONMNIST':
        print("Using FashionMNIST training set")
        dataset = fashion_mnist(train_set=True, test_set=False)
    elif args.dataset.upper() == 'CIFAR10':
        print("Using CIFAR10 training set")
        dataset = cifar10(train_set=True, test_set=False)
    elif args.dataset.upper().replace("-", "_") == 'CIFAR10_NORM':
        print("Using normalized CIFAR10 training set")
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset = datasets.CIFAR10(
            root="../datasets", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), normalize_transform
            ])
        )
    elif args.dataset.replace('-', '').replace('_', '').lower() == 'CollisionDetection'.lower():
        print("Using CollisionDetection training set")
        dataset = collision_detection(train_set=True, test_set=False)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    network = None
    if args.network_file.endswith('.nnet'):
        network = NeuralNetwork.load_from_nnet(args.network_file)
    elif args.network_file.endswith('.onnx'):
        network = NeuralNetwork.load_from_onnx(args.network_file)
    else:
        try:
            network = torch.load(args.network_file)
        except ModuleNotFoundError:
            print("Using legacy load")
            network = NeuralNetwork.load_legacy_pytorch_model(args.network_file)
    print(f'Network loaded: {network}')

    n = args.n if args.n is not None else len(dataset)
    print(f'Looking for {n} non-robust input data points (eps={args.eps}).')
    property_factory = RobustnessPropertyFactory(eps=args.eps)
    attacks = []
    if args.fgsm:
        attacks.append(FastGradientSignMethod())
    if args.pgda:
        attacks.append(ProjectedGradientDescentAttack())
    if args.pgda_adam:
        attacks.append(ProjectedGradientDescentAttack(optimizer='Adam'))
    if args.dea:
        attacks.append(DifferentialEvolutionPGDAttack(optimizer='Adam'))
    if args.stronger_dea:
        attacks.append(DifferentialEvolutionPGDAttack(optimizer='Adam', population_size=50, iterations=20))
    if args.deep_opt:
        attacks.append(DeepOpt())
    if args.eran:
        attacks.append(ERAN())

    violated_properties = []
    counterexamples = []  # indexing does not match violated_properties indexing!
    attack_success_counters = np.zeros((len(attacks), ))

    ray.init()

    @ray.remote
    def execute_attacks(i_, prop_):
        property_violated_ = False
        counterexamples_ = []
        success_counters_ = np.zeros((len(attacks, )))

        for attack_i, attack_ in enumerate(attacks):
            print(f'Data Point {i_}: Running Attack ({attack_.name})')
            cx_, status_ = attack_.find_counterexample(network, prop_)
            print(f'Ran Attack ({attack_.name}; status: {status_})')
            if cx_ is not None and len(cx_) != 0:
                property_violated_ = True
                counterexamples_.extend(cx_)
                success_counters_[attack_i] += 1
                print(f'Adversarial example found for datapoint {i_}')
                if not args.comparison_mode:
                    break
        else:
            if len(counterexamples_) == 0:
                print(f'No adversarial example found for data point {i_}')
        violated_properties_ = [prop_] if property_violated_ else []
        return violated_properties_, counterexamples_, success_counters_

    # schedule tasks in chunks to avoid creating too many tasks and properties
    def results_generator():
        # schedule 1.5 x n tasks, wait for them to complete
        # only then schedule more
        chunk_size = min(len(dataset), 1.5 * n)
        num_chunks = len(dataset) // chunk_size
        indices_permuted = np.random.permutation(range(len(dataset)))

        result_ids = []
        try:
            for chunk in np.array_split(indices_permuted, num_chunks):
                for i in chunk:
                    candidate, true_label = dataset[i]
                    candidate_np = candidate.unsqueeze(0).detach().numpy()
                    prop = next(property_factory.get_properties(input_samples=candidate_np, labels=(true_label,)))
                    prop.property_name = f'robust at data point #{i}'

                    result_id = execute_attacks.remote(i, prop)
                    result_ids.append(result_id)
                while result_ids:
                    done, result_ids = ray.wait(result_ids, num_returns=1)
                    yield ray.get(done[0])
        except GeneratorExit:
            for result_id in result_ids:
                ray.cancel(result_id, force=True)
    results_gen = results_generator()

    progress_bar = tqdm(total=n)
    num_tries = 0  # how many data points were searched overall
    for new_violated_properties, new_counterexamples, attack_success_counters_update in results_gen:
        if args.n is not None:
            progress_bar.update(len(new_violated_properties))
        else:
            progress_bar.update(1)
        violated_properties.extend(new_violated_properties)
        counterexamples.extend(new_counterexamples)
        attack_success_counters += attack_success_counters_update
        num_tries += 1

        if args.n is not None and len(violated_properties) >= args.n:
            break
    results_gen.close()

    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.comparison_mode:
        max_attack_len = max(map(lambda a: len(a.name), attacks))
        max_attack_len = max(max_attack_len, len('Attack'))
        print('Attack' + (' ' * (max_attack_len - len('Attack'))) + ' | % of successfully attacked images')
        for attack, success_counter in zip(attacks, attack_success_counters):
            print(attack.name + (' ' * (max_attack_len - len(attack.name))) +
                  f' | {100*success_counter/num_tries:3.2f}')

    if len(violated_properties) > 0 and args.show_results:
        tensorboard_dir = '../tensorboard'
        print(f'Showing counterexamples in tensorboard (directory: {tensorboard_dir}')
        writer = SummaryWriter(tensorboard_dir)
        writer.add_images(f'counterexamples_{log_time}',
                          np.stack([cx.inputs.reshape(network.inputs_shape) for cx in counterexamples]))

    if len(violated_properties) > 0:
        use_dill = args.use_dill or args.output_file.endswith('.dill')
        print(f'Storing specification in {args.output_file} (as {"dill" if use_dill else "yaml"} file)')
        if use_dill:
            with open(args.output_file, 'w+b') as file:
                pickle.dump(violated_properties, file)
        else:
            with open(args.output_file, 'w+t') as file:
                dump_specification(violated_properties, file, as_multiple_documents=True)
