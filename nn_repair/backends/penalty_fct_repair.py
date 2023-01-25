from typing import Callable, List, Tuple, Optional, Union, Sequence
from abc import ABC, abstractmethod
from logging import info, warning, debug

import torch

from deep_opt import Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory, PenaltyFunction
from nn_repair.training import TrainingLoop


class PenaltyFunctionRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    with the penalty function constrained optimisation method.
    """

    def __init__(
        self,
        training_loop: TrainingLoop,
        loss: Optional[Callable[[], torch.Tensor]] = None,
        penalty_function: PenaltyFunction = PenaltyFunction.L1,
        maximum_updates: int = 10,
        initial_penalty_weight: float = 1/16,
        penalty_increase: Callable[[float], float] = lambda x: 2 * x,
        satisfaction_eps: float = 0.0001,
        keep_all_counterexamples: bool = True,
        penalty_checkpoint_handler: Optional[Callable[[int, Sequence[Tuple[Property, torch.Tensor, float]]], None]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Creates a new RepairNetworkDelegate that uses the penalty function method for repair.

        :param training_loop: The training loop to use for training.
         The loss function of this training loop
         will be updated during execution of this repair backend with a penalized loss function.
         If the parameter loss is not supplied / supplied as None, the loss function of the training loop is used
         as a task loss function.
        :param loss: The loss function that measures the performance of a network on the original machine learning task.
         This loss function needs to support gradient computation with torch.
         If None (default value) is passed for this parameter, the repair backend uses the current loss function of the
         training_loop as task loss function.
        :param penalty_function: The type of penalty function to use.
        :param maximum_updates: The maximum number of penalty weight updates
         that may be performed to fix the counterexamples.
        :param initial_penalty_weight: The initial weights.
        :param penalty_increase: The strategy for increasing individual penalty weights if a constraint is not
         yet satisfied.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method.
         This parameter gives this constant.
         Large values may lead to faster termination compared to smaller values, but larger values will also restrain
         the parameter search space for the original task more.
        :param keep_all_counterexamples: Whether only the most up-to-date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        :param penalty_checkpoint_handler: Provide a function for storing intermediate penalty weights for
         the individual counterexamples.
         The first argument is the current iteration.
         The second is a sequence of tuples, each of which contains as first element the property
         for which the counterexample was discovered.
         The second element of the tuples in the sequence
         is the counterexample, and the third is the corresponding penalty weight.
        :param device: The device to use for training.
         Can be a string that can be converted using :code:`torch.device`.
         When None, this backend doesn't move data between devices.
        """
        super().__init__(keep_all_counterexamples)

        self.training_loop = training_loop
        assert loss is not None or training_loop.loss_function is not None, "No task loss function available"
        if loss is not None:
            self.task_loss = loss
        else:
            self.task_loss = training_loop.loss_function
        self.penalty_function = penalty_function
        self.max_iter = maximum_updates
        self.initial_lambda = initial_penalty_weight
        self.lambda_increase = penalty_increase
        self.sat_eps = satisfaction_eps

        self.penalty_checkpoint_handler = penalty_checkpoint_handler
        self.device = device

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.

        :return: The final status of the repair.
         Return SUCCESS if all counterexamples no longer
         violate the specification.
         Return FAILURE if this could not be archived.
         Return ERROR if any other error occurred from which recovery is possible, otherwise raise
         an exception.
        """
        constr_factory = ConstraintFunctionFactory()\
            .with_network(self.network)\
            .with_satisfaction_eps(self.sat_eps)

        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = self.unfolded_counterexamples

        info(f'Using {self.penalty_function} penalty functions.')
        # `count_violations` is overwritten in this method to
        # count which counterexamples are satisfied >= sat_eps
        # adding some slack avoids property satisfaction floating point issues
        num_violations = self.count_violations(unfolded_counterexamples)

        counterexamples = self.counterexamples
        if self.device is not None:
            counterexamples = {
                prop: [cx.to(self.device) for cx in cxs]
                for prop, cxs in counterexamples.items()
            }

        # a vector of penalties (hence plural)
        penalty_functions = constr_factory.create_vector_penalty_function(
            counterexamples, self.penalty_function
        )
        lambdas: torch.Tensor = torch.full(
            (len(unfolded_counterexamples),), self.initial_lambda
        )
        if self.device is not None:
            lambdas = lambdas.to(self.device)

        cx_art_scale = None  # logging utility (counterexample art).
        for i in range(self.max_iter):
            cx_art, cx_art_scale = self.counterexample_art(unfolded_counterexamples, cx_art_scale)
            info(f"Penalty Function Iteration {i}: "
                 f"{num_violations} remaining counterexample{'s' if num_violations != 1 else ''}\n\n"
                 f"{cx_art}")
            # check if we need to repair at all
            if num_violations == 0:
                break

            # training
            info(f"Training network with penalized loss function. "
                 f"Penalty weights: {self._get_penalty_weight_overview(lambdas)}")
            if self.penalty_checkpoint_handler is not None:
                penalty_checkpoint = tuple((p, cx, l) for (p, cx), l in zip(unfolded_counterexamples, lambdas))
                self.penalty_checkpoint_handler(i, penalty_checkpoint)

            def penalized_loss():
                return self.task_loss() + torch.sum(lambdas * penalty_functions())

            self.training_loop.loss_function = penalized_loss
            self.training_loop.execute()
            self.network = self.training_loop.model

            # increase weights
            num_violations = 0
            for j in range(len(unfolded_counterexamples)):
                p, c = unfolded_counterexamples[j]
                if p.satisfaction_function(c.unsqueeze(0), self.network).item() < self.sat_eps:
                    lambdas[j] = self.lambda_increase(lambdas[j])
                    num_violations += 1
            debug(f"Penalty weight updated.\n"
                  f"New weights: {self._get_penalty_weight_overview(lambdas)}")

        else:
            warning("Maximum number of iterations exhausted: Repair failed")
            return RepairStatus.FAILURE
        # loop exiting with break
        return RepairStatus.SUCCESS

    @staticmethod
    def _get_penalty_weight_overview(lambdas):
        if len(lambdas) < 10:
            return ', '.join(str(v.item()) for v in lambdas)
        else:
            # count value frequencies
            values, frequencies = torch.unique(lambdas, sorted=True, return_counts=True)
            combined = list(zip(values, frequencies))
            combined.reverse()
            return '[' + ', '.join([f'{v}: {f}' for v, f in combined]) + ']'

    def count_violations(self, unfolded_counterexamples=None) -> int:
        """
        Counts the number of stored (potential) counterexamples
        that currently do not satisfy the specification with this instances satisfaction eps (or even violate it).

        :param unfolded_counterexamples: The unfolded counterexamples
         of which the violations should be counted.
         If None the value of the unfolded_counterexamples property is used.
        :return: The number of recorded violations.
        """
        if unfolded_counterexamples is None:
            unfolded_counterexamples = self.unfolded_counterexamples
        # uses True ~> 1, False ~> 0
        return sum(p.satisfaction_function(c.unsqueeze(0), self.network).item() < self.sat_eps
                   for p, c in unfolded_counterexamples)

    def _calc_violation_for_stats(self, prop: Property, cx: torch.Tensor,
                                  constraint_factory: ConstraintFunctionFactory) -> float:
        constraint_factory.with_satisfaction_eps(self.sat_eps)
        return constraint_factory.create_penalty_function(cx, self.penalty_function)().item()
