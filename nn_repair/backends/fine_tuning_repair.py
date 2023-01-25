from typing import List, Tuple
from logging import info, warning

import torch

from deep_opt import Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.backends.constraints import ConstraintFunctionFactory
from nn_repair.training import TrainingLoop, TrainingLossValue


class FineTuningRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to fix counterexamples of a network
    only by trying to archive feasibility. This means that only the counterexample
    violation is minimized. No task loss function is used.

    This backend should be used with very small learning rates (fine-tuning).
    """

    def __init__(self, training_loop: TrainingLoop, satisfaction_eps: float = 1e-4,
                 keep_all_counterexamples: bool = True):
        """
        Creates a new RepairNetworkDelegate that minimises the counterexample violation alone
        doing small steps (fine-tuning) to fix counterexamples.

        :param training_loop: The training loop to use for fine-tuning. The loss function of this training loop
         will be updated during execution of this repair backend with a loss function that
         only minimises property violation for the counterexamples.

         The optimizer of this training loop should have a very small learning rate to enable fine-tuning.
         The repair delegate itself does not ensure in any way that fine-tuning is happening otherwise.
        :param satisfaction_eps: The actual constant of an inequality constraint for a counterexamples is shifted by a
         certain amount to guarantee termination of the penalty function method. This parameter gives this constant.
         This value assures that in the feasibility step, enough space to the feasibility boundary is created.
         Otherwise the training algorithm might quickly leave the feasible region.
        :param keep_all_counterexamples: Whether only the most up to date counterexample for each property should be
         used during repair or if all previously obtained counterexamples should be used.
        """
        super().__init__(keep_all_counterexamples)

        self.training_loop = training_loop
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

        info(f'Fixing counterexamples by minimising violation trough fine-tuning.')
        # a vector function of violations
        violation_functions = constr_factory.create_vector_l1_penalty(self.counterexamples)

        def violation_loss():
            return torch.sum(violation_functions())

        orig_termination_criterion = self.training_loop.termination_criterion
        # if the training loss value (of the violation loss) is zero, all counterexamples have been fixed with
        # sufficient slack (satisfaction_eps)
        self.training_loop.termination_criterion = orig_termination_criterion | TrainingLossValue(0.0)

        self.training_loop.loss_function = violation_loss
        self.training_loop.execute()

        # restore original termination criterion
        self.training_loop.termination_criterion = orig_termination_criterion

        num_violations = self.count_violations(unfolded_counterexamples)
        if num_violations > 0:
            warning(f"Fine-tuning failed: {num_violations} remaining.")
            return RepairStatus.FAILURE
        else:
            info("Fine-tuning was successful.")
            return RepairStatus.SUCCESS
