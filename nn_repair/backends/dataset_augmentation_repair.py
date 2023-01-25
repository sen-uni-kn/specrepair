from typing import Callable, List, Mapping
from logging import info, warning

import torch

from deep_opt import Property
from nn_repair.repair import RepairStatus
from nn_repair.backends.repair_network_delegate_base import RepairNetworkDelegateBase
from nn_repair.training import TrainingLoop, TerminationCriterion


class DatasetAugmentationRepairDelegate(RepairNetworkDelegateBase):
    """
    A RepairNetworkDelegate that tries to repair a network by adding counterexamples to the dataset.
    """
    def __init__(self, get_loss: Callable[[Mapping[Property, List[torch.Tensor]]], Callable[[], torch.Tensor]],
                 training_loop: TrainingLoop, keep_all_counterexamples: bool = True):
        """
        Creates a new RepairNetworkDelegate that augments the dataset for repair.
        :param get_loss: Used to obtain a loss function that measures the performance of a network
        on the original machine learning task for an augmented dataset. The get_loss function
        needs to take care of adding the counterexamples to the dataset.
        At each call it returns a loss function which measures the performance of the network
        on the augmented training dataset.
        This loss function needs to support gradient computation with torch.
        :param training_loop: The training loop used to train the the network to minimise the loss function
        returned by get_loss.
        :param keep_all_counterexamples: Whether only the most up to date counterexample for each property should be
        used during repair or if all previously obtained counterexamples should be used.
        """
        super().__init__(keep_all_counterexamples)

        self.get_loss = get_loss
        self.training_loop = training_loop

    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.
        :return: The final status of the repair. Return SUCCESS if all counterexamples no longer
        violate the specification. Return FAILURE if this could not be archived.
        Return ERROR if any other error occurred, from which recovery is possible, otherwise raise
        an exception.
        """

        outer = self

        class CounterexamplesSatisfied(TerminationCriterion):
            def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
                num_violations = outer.count_violations()
                info(f"Iteration {training_loop.iteration}: "
                     f"{num_violations} remaining counterexample{'s' if num_violations != 1 else ''}")
                return num_violations == 0

        self.training_loop.loss_function = self.get_loss(self.counterexamples)
        self.training_loop.add_termination_criterion(CounterexamplesSatisfied())
        self.training_loop.execute()

        remaining_violations = self.count_violations()
        if remaining_violations > 0:
            warning("Training stopped without fixing counterexamples: Repair failed")
            return RepairStatus.FAILURE
        else:
            return RepairStatus.SUCCESS

    def _calc_violation_for_stats(self, prop: Property, cx: torch.Tensor, constraint_factory) -> float:
        return 0 if prop.property_satisfied( cx.unsqueeze(0), self.network) else 1
