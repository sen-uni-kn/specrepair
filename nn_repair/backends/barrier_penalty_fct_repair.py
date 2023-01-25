from typing import Callable, List, Tuple, Optional
from logging import info, warning, debug

import torch

from deep_opt import Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory, PenaltyFunction, BarrierFunction
from nn_repair.training import TrainingLoop


class BarrierAndPenaltyFunctionRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    with the penalty function constrained optimisation method.
    Once a counterexample has been fixed, a barrier function is used for this counterexample,
    to avoid leaving the feasible region again and penalising being close to the feasibility boundary.
    """

    def __init__(self, training_loop: TrainingLoop, loss: Optional[Callable[[], torch.Tensor]] = None,
                 penalty_function: PenaltyFunction = PenaltyFunction.L1,
                 barrier_function: BarrierFunction = BarrierFunction.RECIPROCAL,
                 initial_penalty_weight: float = 1,
                 penalty_increase: Callable[[float], float] = lambda x: 2 * x, barrier_factor: float = 0.25,
                 maximum_updates: int = 10, satisfaction_eps: float = 0.0001, keep_all_counterexamples: bool = True):
        """
        Creates a new BarrierAndRepairNetworkDelegate that uses the penalty function method to archive
        feasibility and the barrier function method to maintain feasibility.

        :param training_loop: The training loop to use for training. The loss function of this training loop
         will be updated during execution of this repair backend with a penalized loss function.
         If the parameter loss is not supplied / supplied as None, the loss function of the training loop is used
         as task loss function.
        :param loss: The loss function that measures the performance of a network on the original machine learning
         task. This loss function needs to support gradient computation with torch.
         If None (default value) is passed for this parameter, the repair backend uses the current loss function of the
         training_loop as task loss function.
        :param penalty_function: The type of penalty function to use.
        :param barrier_function: The type of barrier function to use.
        :param initial_penalty_weight: The initial weights of the penalty function.
        :param barrier_factor: The multiplier before the barrier term.
        :param penalty_increase: The strategy for increasing individual penalty weights if a constraint is not
         yet satisfied.
        :param maximum_updates: The maximum number of penalty weight updates
         that may be performed to fix the counterexamples.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount. This parameter gives this constant.
        :param keep_all_counterexamples: Whether only the most up to date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        """
        super().__init__(keep_all_counterexamples)

        self.training_loop = training_loop
        assert loss is not None or training_loop.loss_function is not None, "No task loss function available"
        if loss is not None:
            self.task_loss = loss
        else:
            self.task_loss = training_loop.loss_function
        self.penalty_function = penalty_function
        self.barrier_function = barrier_function
        self.lambda_init = initial_penalty_weight
        self.lambda_increase = penalty_increase
        self.barrier_factor = barrier_factor
        self.max_iter = maximum_updates
        self.sat_eps = satisfaction_eps

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.

        :return: The final status of the repair. Return SUCCESS if all counterexamples no longer
         violate the specification. Return FAILURE if this could not be archived.
         Return ERROR if any other error occurred from which recovery is possible, otherwise raise
         an exception.
        """
        constr_factory = ConstraintFunctionFactory()\
            .with_network(self.network)\
            .with_satisfaction_eps(self.sat_eps)

        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = self.unfolded_counterexamples

        info(f'Using {self.penalty_function} penalty functions and {self.barrier_function} barrier functions.')
        penalty_functions: List[Callable[[], torch.Tensor]] = []
        barrier_functions: List[Callable[[], torch.Tensor]] = []
        # count which counterexamples are satisfied >= sat_eps
        # adding some slack avoids property satisfaction floating point issues
        num_violations = 0
        lambdas: List[float] = []
        mus: List[float] = []
        for p, c in unfolded_counterexamples:
            constr_factory.with_property(p)
            penalty_functions.append(constr_factory.create_penalty_function(c, self.penalty_function))
            barrier_functions.append(constr_factory.create_barrier_function(c, self.barrier_function))

            if p.satisfaction_function(c.unsqueeze(0), self.network).item() < self.sat_eps:
                num_violations += 1
                # use penalty function only to archive feasibility
                lambdas.append(self.lambda_init)
                mus.append(0)
            else:
                # use barrier function only to maintain feasibility
                lambdas.append(0)
                mus.append(self.barrier_factor)

        cx_art_scale = None  # logging utility (counterexample art).
        for i in range(self.max_iter):
            cx_art, cx_art_scale = self.counterexample_art(unfolded_counterexamples, cx_art_scale)
            info(f"Barrier and Penalty Function Iteration {i}: "
                 f"{num_violations} remaining counterexample{'s' if num_violations != 1 else ''}\n\n"
                 f"{cx_art}")
            # check if we need to repair at all
            if num_violations == 0:
                break

            # training
            info(f"Training network with penalized loss function. "
                 f"Penalty weights: {lambdas}. Barrier coefficients: {mus}.")

            def penalized_loss():
                loss = self.task_loss()
                for j_ in range(len(penalty_functions)):
                    loss += lambdas[j_] * penalty_functions[j_]()
                    if mus[j_] > 0:  # barrier functions will be infinity f mu = 0, adding will lead to nan
                        loss += mus[j_] * barrier_functions[j_]()
                return loss

            self.training_loop.loss_function = penalized_loss
            self.training_loop.execute()

            # increase weights
            num_violations = 0
            for j in range(len(unfolded_counterexamples)):
                p, c = unfolded_counterexamples[j]
                if p.satisfaction_function(c.unsqueeze(0), self.network).item() < self.sat_eps:
                    lambdas[j] = self.lambda_increase(lambdas[j])
                    num_violations += 1
                else:
                    lambdas[j] = 0
                    mus[j] = self.barrier_factor
            debug(f"Penalty weight updated.\nNew penalty weights: {lambdas}\nNew barrier coefficients: {mus}")

        else:
            warning("Maximum number of iterations exhausted: Repair failed")
            return RepairStatus.FAILURE
        # loop exiting with break
        return RepairStatus.SUCCESS

    def _calc_violation_for_stats(self, prop: Property, cx: torch.Tensor,
                                  constraint_factory: ConstraintFunctionFactory) -> float:
        constraint_factory.with_satisfaction_eps(self.sat_eps)
        return constraint_factory.create_penalty_function(cx, self.penalty_function)().item()
