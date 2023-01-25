from math import prod
from typing import Callable, List, Tuple, Optional, Union, Sequence
from logging import info, warning, debug

import torch

from deep_opt import BoxConstraint, NeuralNetwork, Property
from nn_repair.backends import RepairNetworkDelegateBase
from nn_repair.repair import RepairNetworkDelegate, RepairStatus
from nn_repair.backends.constraints import ConstraintFunctionFactory, PenaltyFunction
from nn_repair.utils.shift_property import ShiftedOutputConstraint


class LinearModelDatasetAugmentationRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate for linear regressors that tries to
    fix counterexamples by adding them to the training set.
    It uses the analytic solution for training the linear regressor.

    This backend only supports repairing properties with BoxConstraint
    output constraints.
    The task loss function must be the mean square error loss function.
    """

    def __init__(
        self,
        training_inputs: torch.Tensor,
        training_targets: torch.Tensor,
        target_oracle: Callable[[torch.Tensor], torch.Tensor],
        satisfaction_eps: float = 0.0001,
        keep_all_counterexamples: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Creates a new RepairNetworkDelegate for linear regressors that uses
        dataset augmentation.

        :param training_inputs: The inputs from the training set.
         Used to compute the task loss function (MSE to training targets).
         The order needs to match the order of training_targets.
        :param training_targets: The targets from the training set.
         Used to compute the task loss function.
         The order needs to match the order of training_inputs.
        :param target_oracle: A function that computes the true output for a
         counterexample.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method.
         This parameter gives this constant.
         Large values may lead to faster termination compared to smaller values, but larger values will also restrain
         the parameter search space for the original task more.
        :param keep_all_counterexamples: Whether only the most up-to-date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        :param device: The device to use for training.
         Can be a string that can be converted using :code:`torch.device`.
         When None, this backend doesn't move data between devices.
        """
        super().__init__(keep_all_counterexamples)

        if training_inputs.ndim == 1:
            training_inputs = training_inputs.unsqueeze(-1)
        # add ones for the bias, so that we can treat bias and weights as one vector
        x = torch.hstack([torch.ones(len(training_inputs), 1), training_inputs])
        self._x = x.float()
        self._y = training_targets.float()
        self._oracle = target_oracle

        self.sat_eps = satisfaction_eps

        self.device = device

    def with_specification(self, specification: Sequence[Property]):
        for prop in specification:
            output_constraint = prop.output_constraint
            if isinstance(output_constraint, ShiftedOutputConstraint):
                output_constraint = output_constraint.original
            if not isinstance(output_constraint, BoxConstraint):
                raise ValueError("LinearModelDatasetAugmentationRepairDelegate only supports "
                                 f"properties with BoxConstraints. "
                                 f"Got: {output_constraint}.")
        super().with_specification(specification)

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for
        the previously registered counterexamples.

        :return: The final status of the repair.
         Return SUCCESS if all counterexamples no longer
         violate the specification.
         Return FAILURE if this could not be archived.
         Return ERROR if any other error occurred from which recovery is possible, otherwise raise
         an exception.
        """
        if not isinstance(self.network[0], torch.nn.Linear) or len(self.network) != 1:
            raise ValueError(f"{self.network} is not a linear regression model.")

        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = self.unfolded_counterexamples
        counterexample_outputs = torch.tensor([
            self._oracle(cx).to(self.device)
            for _, cx in unfolded_counterexamples
        ])
        counterexamples_tensor = torch.stack([
            # potentially remove the batch dimension
            cx.reshape(self.network.inputs_shape).to(self.device)
            for _, cx in unfolded_counterexamples
        ])
        # add ones for the bias, so that we can treat bias and weights as one vector
        counterexamples_tensor = torch.hstack([
            torch.ones(len(counterexamples_tensor), 1),
            counterexamples_tensor.float()
        ])
        x = torch.cat([self._x, counterexamples_tensor])
        y = torch.cat([self._y, counterexample_outputs])

        # training
        info(f"Training network with augmented dataset.")

        with torch.no_grad():
            theta = torch.linalg.inv(x.T @ x) @ x.T @ y
            self.network[0].bias.set_(theta[0].reshape(1))
            self.network[0].weight.set_(theta[1:].reshape(1, 1))

        num_violations = 0
        for j in range(len(unfolded_counterexamples)):
            p, c = unfolded_counterexamples[j]
            if p.satisfaction_function(c.unsqueeze(0), self.network).item() < self.sat_eps:
                num_violations += 1

        info(
            f"Training with augmented dataset finished. "
            f"{num_violations} remaining counterexample{'s' if num_violations != 1 else ''}"
        )
        if num_violations > 0:
            warning("Fixing counterexamples failed.")
            return RepairStatus.FAILURE
        else:
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
        return constraint_factory.create_penalty_function(cx, PenaltyFunction.L1)().item()
