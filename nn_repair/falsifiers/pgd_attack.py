from typing import Tuple, Optional, Sequence

import torch
from torch.optim import SGD, Adam, RMSprop

from tqdm import tqdm

from deep_opt import NeuralNetwork, Property
from nn_repair.counterexamples import CounterexampleGenerator, Counterexample
from nn_repair.training.optim import ProjectedOptimizer


class ProjectedGradientDescentAttack(CounterexampleGenerator):
    """
    Generates counterexamples/adversarial examples through projected gradient descent, as in [Madry2018]_.

    .. [Madry2018] Madry, Aleksander / Makelov, Aleksandar / Schmidt, Ludwig / Tsipras, Dimitris / Vladu, Adrian
     Towards Deep Learning Models Resistant to Adversarial Attacks
     2018
     ICLR (Poster)
    """

    def __init__(self, optimizer='SGD', steps=10, num_restarts=10, single_counterexample=False, progress_bar=True):
        """
        Creates a new ProjectedGradientDescentAttack (PGDA) for generating counterexamples.

        :param optimizer: The optimizer to use for finding counterexamples. Options: SGD, Adam, RMSprop
        :param steps: The number of optimisation steps to perform.
        :param num_restarts: How many random restarts to perform
        :param single_counterexample: Whether to return only a single counterexample if counterexamples were found.
            This does not speed up falsification in any case, but is required for some repair backends.
        :param progress_bar: Whether to display a progress bar while falsifying
        """
        assert optimizer in ['SGD', 'Adam', 'RMSprop'], "Unknown optimizer."
        self.optimizer_name = optimizer
        self.steps = steps
        self.num_restarts = num_restarts
        self.single_counterexample = single_counterexample
        self.progress_bar = progress_bar

    @property
    def name(self) -> str:
        return 'PGDA+' + self.optimizer_name + f' [steps={self.steps}, restarts={self.num_restarts}]'

    def _get_optimizer(self, x, lr):
        if self.optimizer_name == 'SGD':
            return SGD([x], lr=lr)
        elif self.optimizer_name == 'Adam':
            return Adam([x], lr=lr)
        elif self.optimizer_name == 'RMSprop':
            return RMSprop([x], lr=lr)
        else:
            assert False, "Unknown optimizer"

    def find_counterexample(self, target_network: NeuralNetwork, target_property: Property) \
            -> Tuple[Optional[Sequence[Counterexample]], str]:
        bounds = target_property.input_bounds(target_network)
        lower, upper = zip(*bounds)
        lower = torch.tensor(lower)
        upper = torch.tensor(upper)
        # calculate the largest radius
        max_radius = max(u - l for l, u in bounds)
        lr = max_radius/self.steps * 2

        counterexamples = []
        for _ in tqdm(range(self.num_restarts), disable=not self.progress_bar):
            # generate flat inputs
            x_val = lower + (upper - lower) * torch.rand(lower.shape)
            x_val.unsqueeze_(0)
            x_val = target_property.input_constraint_projection(x_val)

            x = x_val.clone().detach().float().requires_grad_(True)
            optimizer = self._get_optimizer(x, lr)
            optimizer = ProjectedOptimizer(optimizer, [{
                'params': [x],
                'lower_bounds': lower,
                'upper_bounds': upper,
                'projection_function': target_property.input_constraint_projection
            }])

            for _ in range(self.steps):
                optimizer.zero_grad()
                violation = target_property.satisfaction_function(x, target_network)
                violation.backward(inputs=[x])
                optimizer.step()

            _, network_outputs, sat_value, is_sat = \
                target_property.full_witness(x, target_network)
            if not is_sat:
                cx = Counterexample(inputs=x[0].detach().numpy(), network_outputs=network_outputs,
                                    property_satisfaction=sat_value.item(), property=target_property)
                counterexamples.append(cx)

        counterexamples.sort(key=lambda c: c.property_satisfaction)
        if self.single_counterexample and len(counterexamples) > 0:
            counterexamples = [counterexamples[0]]  # counterexamples were already sorted by their violation
        return counterexamples, 'Violated' if len(counterexamples) > 0 else 'Unknown'
