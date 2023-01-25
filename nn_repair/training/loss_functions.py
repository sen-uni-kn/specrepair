from typing import Callable, Sequence, Optional, Iterator
from logging import debug

import torch
import torch.nn.functional
import numpy as np

from deep_opt import NeuralNetwork, Property
from nn_repair.utils.random_sampling import sample_from_normal_in_range


class ParameterChange(torch.nn.Module):
    def __init__(self, parameters: Optional[Iterator[torch.nn.Parameter]] = None,
                 module: Optional[torch.nn.Module] = None):
        super().__init__()
        assert parameters is not None or module is not None, "Either parameters or module is required as parameter."
        assert parameters is None or module is None, "Only either parameters or module can be passed as parameter."
        if parameters is None:
            self.module = module
            parameters = module.parameters()
        else:
            self.module = None
        self.initial_parameters = tuple(p.detach().clone() for p in parameters)

    def forward(self, module=None):
        if module is None:
            assert self.module is not None, "No module available. If none is passed at initialization, " \
                "a module has to be supplied when ParameterChange is called."
            module = self.module
        param_diff = (old - new for new, old in zip(module.parameters(), self.initial_parameters))
        param_diff = (x ** 2 for x in param_diff)
        param_diff = [torch.sum(x) for x in param_diff]
        return torch.stack(param_diff).sum()


def get_fidelity_data(network: NeuralNetwork, spec: Sequence[Property], n=10000, seed=2703,
                      rng: Optional[torch.Generator] = None,
                      show_progress=False) -> torch.Tensor:
    """
    Returns a randomly generated dataset for measuring the fidelity [c.f. Dong et al. 2020].

    [Dong et al. 2020]: Towards Repairing Neural Networks Correctly. CoRR, Vol. abs/2012.01872

    :param network: The original network which provides the ground truth predictions.
    :param spec: The specification for which the fidelity data is generated. The specification is used to
     remove samples that violate the specification from the generated dataset.
    :param n: The number of data points of the generated dataset.
    :param seed: The seed for the random number generator. This parameter is ignored if rng is not None.
    :param rng: The random number generator for generating the data. If this is None, a new random number generator
     is created with the seed that can be passed as parameter `seed`.
    :param show_progress: Whether to display a progress bar, that shows how many data points have already been obtained.
    :return: A dataset of n data points, for which the given network does not violate the specification.
    """
    debug(f'Generating data... (n={n}, seed={seed})')
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    # property_input_bounds contains the lower bounds as first column and the upper bounds as second
    property_input_bounds = [torch.tensor(prop.input_bounds(network)) for prop in spec]

    def satisfies_specification(data: torch.Tensor) -> torch.Tensor:
        spec_satisfied = torch.tensor([True] * len(data))
        for i in range(len(spec)):
            in_input_bounds = torch.all(data >= property_input_bounds[i][:, 0], dim=-1)
            in_input_bounds &= torch.all(data <= property_input_bounds[i][:, 1], dim=-1)
            prop_satisfied = spec[i].property_satisfied(data, network)
            prop_satisfied |= ~in_input_bounds
            spec_satisfied &= prop_satisfied
        return spec_satisfied

    test_data = sample_from_normal_in_range(
        network.means_inputs.flatten().detach(), network.ranges_inputs.flatten().detach(),
        network.mins.detach(), network.maxes.detach(),
        num_rows=n, further_filter=satisfies_specification,
        rng=rng, show_progress=show_progress
    )
    return test_data


class FidelityLoss:
    def __init__(self, network, criterion, test_data, original_predictions):
        self._network = network
        self._criterion = criterion
        self._test_data = test_data
        self._original_predictions = original_predictions

    @property
    def criterion(self):
        return self._criterion

    @property
    def test_data(self):
        return self._test_data

    @property
    def original_predictions(self):
        return self._original_predictions

    def __call__(self, nn: Optional[NeuralNetwork] = None):
        if nn is None:
            nn = self._network
        return self._criterion(nn(self._test_data), self._original_predictions)


def get_fidelity_loss_cross_entropy(network: NeuralNetwork, spec: Sequence[Property], seed=2703,
                                    classification_mode='min') -> FidelityLoss:
    """
    Returns a cross-entropy loss function for a randomly generated dataset.
    based on the fidelity measure from [Dong et al. 2020].

    [Dong et al. 2020]: Towards Repairing Neural Networks Correctly. CoRR, Vol. abs/2012.01872
    :param network: The original network which provides the ground truth predictions
    :param spec: The specification for which the loss function is generated. The specification is used to
     remove samples that violate it from the generated dataset.
    :param seed: The seed for the random number generator.
    :param classification_mode: Whether the output with the maximal or minimal score
     provides the predicted label. Default is that the minimum output provides the label, since
     this loss function is intended for use with ACAS Xu.
     Allowed values are 'min' and 'max'.
    :return: A loss function that measures how much the predictions of a network differ from the
     predictions of the network that is passed to this function.
    """
    # generate a random test set
    test_data = get_fidelity_data(network, spec, n=10000, seed=seed, show_progress=True)
    # obtain the predictions of the original network
    if classification_mode == 'max':
        original_predictions = torch.argmax(network(test_data), dim=1)
    else:
        original_predictions = torch.argmin(network(test_data), dim=1)

    criterion = torch.nn.CrossEntropyLoss()
    return FidelityLoss(network, criterion, test_data, original_predictions)


def get_fidelity_loss_hcas_loss(network: NeuralNetwork, spec: Sequence[Property],
                                seed=2703, num_samples=10000, classification_mode='min') -> FidelityLoss:
    """
    Returns a cross-entropy loss function for a randomly generated dataset.
    based on the fidelity measure from [Dong et al. 2020].

    [Dong et al. 2020]: Towards Repairing Neural Networks Correctly. CoRR, Vol. abs/2012.01872
    :param network: The original network which provides the ground truth predictions
    :param spec: The specification for which the loss function is generated. The specification is used to
     remove samples that violate it from the generated dataset.
    :param seed: The seed for the random number generator.
    :param classification_mode: Whether the output with the maximal or minimal score
     provides the predicted label. Default is that the minimum output provides the label, since
     this loss function is intended for use with ACAS Xu.
     Allowed values are 'min' and 'max'.
    :return: A loss function that measures how much the predictions of a network differ from the
     predictions of the network that is passed to this function.
    """
    # generate a random test set
    test_data = get_fidelity_data(network, spec, n=num_samples, seed=seed, show_progress=True)
    # obtain the scores of the original network
    original_scores = network(test_data).detach().clone().requires_grad_(False)

    criterion = HCASLoss(classification_mode=classification_mode)
    return FidelityLoss(network, criterion, test_data, original_scores)



class HCASLoss(torch.nn.Module):
    """
    Asymmetric MSE loss function for training HCAS networks.
    Compare with [Julian2018]_ and https://github.com/sisl/HorizontalCAS/blob/master/GenerateNetworks/trainHCAS.py

    By default, this loss function takes the output with the maximum score as the label
    (as opposed to the minimum output).
    To use the loss function with the ACAS Xu networks from [Kath2017]_ ``classification_mode='min'``
    has to be passed to the initializer

    .. [Julian2018] Kyle D. Julian, Mykel J. Kochenderfer, and Michael P. Owen. “Deep Neural Network Compression for Aircraft Collision
        Avoidance Systems”. In: CoRR abs/1810.04240 (2018).
    .. [Katz 2017] Guy Katz, Clark W. Barrett, David L. Dill, Kyle Julian, Mykel J. Kochenderfer:
        Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. CAV (1) 2017: 97-117
    """

    def __init__(self, classification_mode='max'):
        """
        Initialize a HCAS loss function.

        :param classification_mode: Whether the output with the maximal or minimal score
         provides the predicted label. Default is that the maximum output provides the label.
         Allowed values are 'min' and 'max'.
        """
        super().__init__()
        assert classification_mode in ['min', 'max']
        self.classification_mode = classification_mode

    def forward(self, y_pred, y_true):
        def get_labels(scores):
            if self.classification_mode == 'max':
                return torch.argmax(scores, dim=1)
            else:
                return torch.argmin(scores, dim=1)

        # implementation based on: https://github.com/sisl/HorizontalCAS/blob/master/GenerateNetworks/trainHCAS.py
        loss_factor = 40.0
        num_out = 5

        error = y_pred - y_true
        true_labels = get_labels(y_true)
        one_hot = torch.nn.functional.one_hot(true_labels, num_classes=num_out)
        error_optimal = error * one_hot  # e(s,a) if a = pi(s) (c.f. Julian et al. 2018)
        error_suboptimal = error * -1 * (one_hot - 1)  # e(s,a) if a != pi(s)
        a = loss_factor * (num_out - 1) * (torch.square(error_optimal) + torch.abs(error_optimal))
        b = torch.square(error_optimal)
        c = loss_factor * (torch.square(error_suboptimal) + torch.abs(error_suboptimal))
        d = torch.square(error_suboptimal)
        loss = torch.where(error_suboptimal > 0, c, d) + torch.where(error_optimal < 0, a, b)
        return torch.mean(loss)


def accuracy(y_true, y_pred, classification_mode='max'):
    """
    Ordinary accuracy calculation for ground truth scores.

    :param classification_mode: Whether the output with the maximal or minimal score
     provides the predicted label. Default is that the maximum output provides the label.
     Allowed values are 'min' and 'max'.
    """
    assert classification_mode in ['min', 'max']
    if classification_mode == 'max':
        true_labels = torch.argmax(y_true, dim=1)
    else:
        true_labels = torch.argmin(y_true, dim=1)
    return accuracy2(true_labels, y_pred, classification_mode=classification_mode)


def accuracy2(true_labels, y_pred, classification_mode='max'):
    """
    Ordinary accuracy calculation for ground truth labels.

    :param classification_mode: Whether the output with the maximal or minimal score
     provides the predicted label. Default is that the maximum output provides the label.
     Allowed values are 'min' and 'max'.
    """
    if classification_mode == 'max':
        pred_labels = torch.argmax(y_pred, dim=1)
    else:
        pred_labels = torch.argmin(y_pred, dim=1)
    return (true_labels == pred_labels).float().mean()
