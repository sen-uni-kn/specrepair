from typing import Sequence, Callable, List, Dict, Tuple, Optional
from logging import info, warning, debug

import numpy as np
import torch

from deep_opt import NeuralNetwork, Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory
from nn_repair.training import TrainingLoop, GradientValue


class AugmentedLagrangianRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    with the augmented lagrangian constrained optimisation method.

    This class implements the algorithm presented in [Nocedal2006]_.
    The implementation uses a training algorithm directly, without any modifications of the
    training loop (e.g. no projections to bound constraints).

    .. [Nocedal2006] Jorge Nocedal and Stephen J. Wright. Numerical Optimization.
     2nd ed. Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
    """

    def __init__(self, training_loop: TrainingLoop, loss: Optional[Callable[[], torch.Tensor]] = None,
                 maximum_updates: int = 10, initial_lagrange_multipliers: float = 1, initial_penalty_weight: float = 2,
                 penalty_increase: Callable[[float], float] = lambda x: 2 * x,
                 constraint_satisfaction_tightening_factor=0.25, satisfaction_eps: float = 0.0001,
                 keep_all_counterexamples: bool = True,
                 checkpoint_handler: Optional[
                     Callable[[int, float, float, float, Sequence[Tuple[Property, torch.Tensor, float]]], None]
                 ] = None):
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
        :param constraint_satisfaction_tightening_factor: A constant used to calculate constraint satisfaction
         progress constants from the penalty weights. When penalty weights are increased
         the expected progress in constraint satisfaction is calculated as :math:`\frac{1}{\\mu^{c}}`
         where c is this parameter. If the lagrange multipliers are updated the expected progress is calculated
         as :math:`\frac{1}{\\mu^{1-c}}`.
         [Nocedal2006]_ use 0.1, but this value is too small for the gradient descent algorithms used here.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method. This parameter gives this constant.
        :param keep_all_counterexamples: Whether only the most up to date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        :param checkpoint_handler: Provide a function for storing intermediate penalty weights
         and Lagrange multipliers for the individual counterexamples as well as the convergence constants.
         The first argument is the current iteration. The second argument is the current value of mu,
         the collective penalty weight. The third argument is the omega which is an upper bound
         for the gradient values used to terminate training. The forth is the eta constant, which controls
         how large satisfaction progress should be. This value determines whether the penalty weights or
         the lagrange multipliers are updated.
         The fifth argument finally is a sequence of tuples, each of which contains as first element the property
         for which the counterexample was discovered. The second element of the tuples in the sequence
         is the counterexample and the third is the corresponding Lagrange multiplier

        .. [Nocedal2006] Jorge Nocedal and Stephen J. Wright. Numerical Optimization.
           2nd ed Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
        """
        super().__init__(keep_all_counterexamples)

        assert 0 < constraint_satisfaction_tightening_factor < 0.5
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
        self.sat_tightening_factor = constraint_satisfaction_tightening_factor
        self.sat_eps = satisfaction_eps

        self.checkpoint_handler = checkpoint_handler

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
        # mus: List[float] = [self.init_mu] * len(unfolded_counterexamples)
        mu = self.init_mu
        omega = 1/mu  # controls convergence of training
        eta = 1/(mu ** self.sat_tightening_factor)

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
                 f"Penalty Weight: {mu}\n"
                 f"Omega: {omega}, Eta: {eta}")

            if self.checkpoint_handler is not None:
                lagrange_checkpoint = tuple((p, cx, l) for (p, cx), l in zip(unfolded_counterexamples, lambdas))
                self.checkpoint_handler(i, mu, omega, eta, lagrange_checkpoint)

            def augmented_lagrangian():
                loss = self.task_loss()
                for j_ in range(len(constraints)):
                    loss += -lambdas[j_] * constraints[j_]()[0] + mu/2 * torch.square(constraints[j_]()[0])
                return loss

            orig_termination_criterion = self.training_loop.termination_criterion
            omega_criterion = GradientValue(parameters=self.network.parameters(), gradient_threshold=omega)
            self.training_loop.termination_criterion = orig_termination_criterion | omega_criterion

            self.training_loop.loss_function = augmented_lagrangian
            self.training_loop.execute()

            # restore termination criterion
            self.training_loop.termination_criterion = orig_termination_criterion

            # update lambda or mu, depending on constraint satisfaction progress
            max_violation_after_update = max(violation().item() for violation in violations)
            if max_violation_after_update <= eta:
                # update lagrange multiplier estimates (lambda)
                for j in range(len(unfolded_counterexamples)):
                    if constraints[j]().item() >= lambdas[j]/mu:
                        # constraint not tight up to tolerance
                        # see Jorge Nocedal and Stephen J. Wright. Numerical Optimization. 2nd ed.
                        # Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
                        lambdas[j] = 0
                    else:
                        # see Jorge Nocedal and Stephen J. Wright. Numerical Optimization. 2nd ed.
                        # Springer, 2006 (Chapter 17.4/Unconstrained Formulation)
                        lambdas[j] = lambdas[j] - mu*constraints[j]().item()
                eta = eta / (mu ** (1 - self.sat_tightening_factor))
                omega = omega / mu
            else:
                # increase penalties
                mu = self.mu_increase(mu)
                eta = 1 / (mu ** self.sat_tightening_factor)
                omega = 1 / mu

            num_violations = self.count_violations()
            # store for next iteration
            max_violation_before_update = max_violation_after_update
            debug(f"Parameters updated.\nNew lagrange multipliers: {lambdas}\nNew penalty weight: {mu}")

        else:
            warning("Maximum number of iterations exhausted: Repair failed")
            return RepairStatus.FAILURE
        # loop exiting with break
        return RepairStatus.SUCCESS
