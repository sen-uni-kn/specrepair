from typing import Tuple, Optional, Sequence

from tqdm import tqdm

import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop

from deep_opt import NeuralNetwork, Property
from nn_repair.counterexamples import CounterexampleGenerator, Counterexample
from nn_repair.training.optim import ProjectedOptimizer


class DifferentialEvolutionPGDAttack(CounterexampleGenerator):
    """
    Generates counterexamples/adversarial examples through projected gradient descent, as in [Madry2018]_
    and differential evolution.

    In difference to [Madry2018]_, who simply perform a number of random restarts,
    this implementation applies differential evolution on the results of the optimisation procedure.

    .. [Madry2018] Madry, Aleksander / Makelov, Aleksandar / Schmidt, Ludwig / Tsipras, Dimitris / Vladu, Adrian
     Towards Deep Learning Models Resistant to Adversarial Attacks
     2018
     ICLR (Poster)
    """

    def __init__(self, optimizer='SGD', population_size=10, iterations=10, optimizer_steps=10,
                 cr=0.9, dw=0.8, progress_bar=True):
        """
        Creates a new DifferentialEvolutionProjectedGradientDescentAttack (DEA) for generating counterexamples.

        :param optimizer: The optimizer to use for finding counterexamples. Options: SGD, Adam, RMSprop
        :param population_size: The number of points to consider.
        :param iterations: The number of differential evolution recombination steps to perform.
        :param optimizer_steps: The number of optimisation steps using the optimizer to perform between
         differential evolution recombination.
        :param cr: Crossover probability :math:`cr \\in [0, 1]`. See [wikipedia]_.
        :param dw: Differential weight :math:`dw \\in [0, 2]`. See [wikipedia]_ (F).
        :param progress_bar: Whether to display a progress bar while falsifying

        .. [wikipedia] https://en.wikipedia.org/wiki/Differential_evolution
        """
        assert optimizer in ['SGD', 'Adam', 'RMSprop'], "Unknown optimizer."
        assert population_size >= 4
        assert 0 <= cr <= 1
        assert 0 <= dw <= 2
        self.optimizer_name = optimizer
        self.population_size = population_size
        self.diff_evol_steps = iterations
        self.local_steps = optimizer_steps
        self.cr = cr
        self.dw = dw
        self.progress_bar = progress_bar

    @property
    def name(self) -> str:
        return 'DEA+PGDA+' + self.optimizer_name + f'[population_size={self.population_size}, ' \
                                                   f'iterations={self.diff_evol_steps}, ' \
                                                   f'optimizer_steps={self.local_steps}]'

    def _get_optimiser(self, x, lr):
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
        lr = max_radius/self.local_steps * 2

        rng = np.random.default_rng()

        starting_points = [lower + (upper - lower) * rng.random(lower.shape) for _ in range(self.population_size)]
        population = []
        optimizers = []
        for point in starting_points:
            x = point.clone().detach().float().unsqueeze(0).requires_grad_(True)
            optim = self._get_optimiser(x, lr)
            optim = ProjectedOptimizer(optim, [{
                'params': [x],
                'lower_bounds': lower,
                'upper_bounds': upper,
                'projection_function': target_property.input_constraint_projection
            }])
            population.append(x)
            optimizers.append(optim)

        def calc_violation(x_):
            return target_property.satisfaction_function(x_, target_network)

        with tqdm(
                total=self.diff_evol_steps*self.population_size*(self.local_steps + 1),
                disable=not self.progress_bar
        ) as progress:
            for _ in range(self.diff_evol_steps):
                for x, optim in zip(population, optimizers):
                    for _ in range(self.local_steps):
                        optim.zero_grad()
                        violation = calc_violation(x)
                        violation.backward(inputs=[x])
                        optim.step()
                        progress.update()

                # differential evolution recombination step
                for i, x in enumerate(population):
                    # get three random indices of the population that are all different and all different from i
                    js = np.random.permutation([j for j in range(self.population_size) if j != i])[:3]
                    a, b, c = [population[j] for j in js]
                    r = rng.integers(0, len(lower))  # get a random dimension that will certainly be mutated
                    candidate = torch.where(
                        (torch.tensor(range(len(lower))) == r) | (torch.as_tensor(rng.random(x.shape)) < self.cr),
                        a + self.dw * (b - c),
                        x
                    )
                    # project candidate
                    candidate = torch.where(torch.lt(candidate, lower), lower, candidate)
                    candidate = torch.where(torch.gt(candidate, upper), upper, candidate)
                    if calc_violation(candidate) <= calc_violation(x):
                        # debug("Differential Evolution Recombination yielded better point.")
                        x.data = candidate.data
                    progress.update()

        full_infos = [target_property.full_witness(x, target_network) for x in population]
        counterexamples = [Counterexample(inputs=x[0].detach().numpy(),  # x in full_infos is batched, don't want that
                                          network_outputs=outputs,
                                          property_satisfaction=satisfaction.item(),
                                          property=target_property)
                           for x, outputs, satisfaction, is_sat in full_infos
                           if not is_sat]
        counterexamples.sort(
            key=lambda cx: cx.property_satisfaction
        )
        return counterexamples, 'Violated' if len(counterexamples) > 0 else 'Unknown'
