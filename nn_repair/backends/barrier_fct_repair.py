from typing import Callable, List, Tuple, Optional
from logging import info, warning, error

import torch

from deep_opt import Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory, BarrierFunction
from nn_repair.training import TrainingLoop, TrainingLossValue


class BarrierFunctionRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    by trying to archive feasibility first and then optimising the task loss with barriers
    that hinder the optimiser from leaving the feasible region.

    The algorithm tries to archive feasibility by minimising counterexample violation alone.
    """

    def __init__(self, training_loop: TrainingLoop, loss: Optional[Callable[[], torch.Tensor]] = None,
                 barrier_function: BarrierFunction = BarrierFunction.RECIPROCAL, barrier_factor: float = 0.25,
                 satisfaction_eps: float = 1, keep_all_counterexamples: bool = True):
        """
        Creates a new RepairNetworkDelegate that uses the barrier function method for repair.

        :param training_loop: The training loop to use for training. The loss function of this training loop
         will be updated during execution of this repair backend first with a loss function that
         only minimises property violation for the counterexamples and
         then with the task loss function with additional barrier terms.
         If the parameter loss is not supplied / supplied as None, the loss function of the training loop is used
         as task loss function.
        :param loss: The loss function that measures the performance of a network on the original machine learning
         task. This loss function needs to support gradient computation with torch.
         If None (default value) is passed for this parameter, the repair backend uses the current loss function of the
         training_loop as task loss function.
        :param barrier_function: The barrier function to use.
        :param barrier_factor: A factor for all barrier terms. Smaller values lead to better task loss results, but
         too small values can lead to numerical instability.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method. This parameter gives this constant.
         This value assures that in the feasibility step, enough space to the feasibility boundary is created.
         Otherwise the training algorithm might quickly leave the feasible region.
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
        self.barrier_function = barrier_function
        self.barrier_factor = barrier_factor
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
            .with_satisfaction_eps(self.sat_eps)  # the l1 penalty used for feasibility uses the satisfaction eps

        unfolded_counterexamples: List[Tuple[Property, torch.Tensor]] = self.unfolded_counterexamples

        info(f'Find feasible point by minimising violation.')
        # a vector function of violations
        violation_functions = constr_factory.create_vector_l1_penalty(self.counterexamples)

        def violation_loss():
            return torch.sum(violation_functions())

        orig_termination_criterion = self.training_loop.termination_criterion
        # we also want to make sure counterexamples are satisfied with sufficient slack (satisfaction eps)
        self.training_loop.termination_criterion = orig_termination_criterion | TrainingLossValue(0.0)

        self.training_loop.loss_function = violation_loss
        self.training_loop.execute()

        # restore original termination criterion
        self.training_loop.termination_criterion = orig_termination_criterion

        num_violations = self.count_violations(unfolded_counterexamples)
        if num_violations > 0:
            warning(f"Feasibility step failed: {num_violations} remaining.")
            return RepairStatus.FAILURE
        info("Feasibility step successful.")

        info(f'Using {self.barrier_function} barrier functions.')
        barrier_functions = constr_factory.create_vector_barrier_function(
            unfolded_counterexamples,
            self.barrier_function
        )

        def barrier_loss():
            return self.task_loss() + self.barrier_factor * torch.sum(barrier_functions())

        self.training_loop.loss_function = barrier_loss
        self.training_loop.execute()

        num_violations = self.count_violations(unfolded_counterexamples)
        if num_violations > 0:
            error("Counterexamples re-introduced after training with barrier objective")
            return RepairStatus.ERROR

        return RepairStatus.SUCCESS
