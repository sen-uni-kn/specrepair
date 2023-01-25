from typing import Sequence, Callable, List, Dict, Tuple, Optional
from logging import info, warning, debug

import numpy as np
import torch

from deep_opt import NeuralNetwork, Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory
from nn_repair.training import TrainingLoop


class AugmentedLagrangianRepairDelegate2(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    with the augmented lagrangian constrained optimisation method.

    This class implements the algorithm presented in [Bazaraa2006]_.
    The implementation uses a training algorithm directly, without any modifications of the
    training loop (e.g. no projections to bound constraints).
    The main difference to AugmentedLagrangianRepairDelegate is how the algorithm decides whether
    to update the penalty weights or the lagrange multipliers.

    .. [Bazaraa2006] Mokhtar S. Bazaraa, Hanif D. Sherali, and C. M. Shetty. Nonlinear Programming - Theory and Algorithms.
     3rd ed. Wiley, 2006.
    """

    def __init__(self, training_loop: TrainingLoop, loss: Optional[Callable[[], torch.Tensor]] = None,
                 maximum_updates: int = 10, initial_lagrange_multipliers: float = 1, initial_penalty_weight: float = 8,
                 penalty_increase: Callable[[float], float] = lambda x: 128 * x,
                 constraint_satisfaction_progress_factor=0.25, satisfaction_eps: float = 0.0001,
                 keep_all_counterexamples: bool = True):
        """
        Creates a new RepairNetworkDelegate that uses the augmented lagrangian method for repair.
        The augmented lagrangian L is defined as:

        .. math::
            L(x) = f(x) - \\lambda * c(x) + \\mu/2 * c(x)^2
        where f is the task loss function, lambda is an estimated lagrange multiplier,
        c is the constraint function and mu is the penalty weight.

        :param training_loop: The training loop to use for training. The loss function of this training loop
         will be updated during execution of this repair backend with the augmented lagrangian.
         If the parameter loss is not supplied / supplied as None, the loss function of the training loop is used
         as task loss function.
        :param loss: The loss function that measures the performance of a network on the original machine learning
         task. This loss function needs to support gradient computation with torch.
         If None (default value) is passed for this parameter, the repair backend uses the current loss function of the
         training_loop as task loss function.
        :param maximum_updates: The maximum number of lagrange multiplier and penalty weight updates
         that may be performed to fix the counterexamples.
        :param initial_lagrange_multipliers: The initial lagrange multiplier estimates
        :param initial_penalty_weight: The initial weights.
        :param penalty_increase: The strategy for increasing individual penalty weights if a constraint is not
         yet satisfied.
        :param constraint_satisfaction_progress_factor: If constraint violation decreases by less than this
         fraction of the previous constraint violation, penalty weights are updated, otherwise lagrange multipliers
         are updated. Default value is 0.25, from Mokhtar S. Bazaraa, Hanif D. Sherali, and C. M. Shetty. Nonlinear
         Programming - Theory and Algorithms. 3rd ed. Wiley, 2006.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method. This parameter gives this constant.
        :param keep_all_counterexamples: Whether only the most up to date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        """
        super().__init__(keep_all_counterexamples)

        assert 0 < constraint_satisfaction_progress_factor < 1
        assert initial_lagrange_multipliers > 0
        assert initial_penalty_weight > 0

        self.training_loop = training_loop
        assert loss is not None or training_loop.loss_function is not None, "No task loss function available"
        if loss is not None:
            self.task_loss = loss
        else:
            self.task_loss = training_loop.loss_function
        self.max_iter = maximum_updates
        self.init_lambda = initial_lagrange_multipliers
        self.init_mu = initial_penalty_weight
        self.mu_increase = penalty_increase
        self.sat_progress_factor = constraint_satisfaction_progress_factor
        self.sat_eps = satisfaction_eps

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.
        :return: The final status of the repair. Return SUCCESS if all counterexamples no longer
        violate the specification. Return FAILURE if this could not be archived.
        Return ERROR if any other error occurred, from which recovery is possible, otherwise raise
        an exception.
        """
        constr_factory = ConstraintFunctionFactory()\
            .with_network(self.network)\
            .with_satisfaction_eps(self.sat_eps)

        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = self.unfolded_counterexamples

        constraints: List[Callable[[], torch.Tensor]] = []
        violations: List[Callable[[], torch.Tensor]] = []
        lambdas: List[float] = [self.init_lambda] * len(unfolded_counterexamples)
        mus: List[float] = [self.init_mu] * len(unfolded_counterexamples)
        num_violations = 0
        for p, c in unfolded_counterexamples:
            constr_factory.with_property(p)
            constraints.append(constr_factory.create_constraint_function(c))
            violations.append(constr_factory.create_l1_penalty(c))

            if not p.property_satisfied(c.unsqueeze(0), self.network):
                num_violations += 1
        max_violation_before_update = max(violation().item() for violation in violations)

        cx_art_scale = None
        for i in range(self.max_iter):
            cx_art, cx_art_scale = self.counterexample_art(unfolded_counterexamples, cx_art_scale)
            info(f"Augmented Lagrangian Iteration {i}: "
                 f"{num_violations} remaining counterexample{'s' if num_violations != 1 else ''}. "
                 f"Largest violation: {max_violation_before_update}\n\n"
                 f"{cx_art}")
            # check if we need to repair at all
            if num_violations == 0:
                break

            # training
            info(f"Training network with augmented lagrangian.\n"
                 f"Lagrange Multiplier Estimates: {lambdas}\n"
                 f"Penalty Weights: {mus}")

            def augmented_lagrangian():
                loss = self.task_loss()
                for j_ in range(len(constraints)):
                    loss += -lambdas[j_] * constraints[j_]()[0] + mus[j_]/2 * torch.square(constraints[j_]()[0])
                return loss

            self.training_loop.loss_function = augmented_lagrangian
            self.training_loop.execute()

            # update lambda or mu, depending on constraint satisfaction progress
            max_violation_after_update = max(violation().item() for violation in violations)
            num_violations = 0
            if max_violation_after_update <= self.sat_progress_factor * max_violation_before_update:
                # update lagrange multiplier estimates (lambda)
                for j in range(len(unfolded_counterexamples)):
                    if constraints[j]().item() >= lambdas[j]/mus[j]:
                        # constraint not tight up to tolerance
                        # see Jorge Nocedal and Stephen J. Wright. Numerical Optimization. 2nd ed.
                        # Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
                        lambdas[j] = 0
                    else:
                        # see Jorge Nocedal and Stephen J. Wright. Numerical Optimization. 2nd ed.
                        # Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
                        lambdas[j] = lambdas[j] - mus[j]*constraints[j]().item()
                    p, c = unfolded_counterexamples[j]
                    if not p.property_satisfied(c.unsqueeze(0), self.network):
                        num_violations += 1
            else:
                # increase penalties
                for j in range(len(unfolded_counterexamples)):
                    p, c = unfolded_counterexamples[j]
                    if not p.property_satisfied(c.unsqueeze(0), self.network):
                        mus[j] = self.mu_increase(mus[j])
                        num_violations += 1
            # store for next iteration
            max_violation_before_update = max_violation_after_update
            debug(f"Parameters updated.\nNew lagrange multipliers: {lambdas}\nNew penalty weights: {mus}")

        else:
            warning("Maximum number of iterations exhausted: Repair failed")
            return RepairStatus.FAILURE
        # loop exiting with break
        return RepairStatus.SUCCESS
