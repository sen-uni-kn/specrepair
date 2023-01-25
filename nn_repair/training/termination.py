from typing import Callable, Union
from abc import ABC, abstractmethod
from logging import warning, info
from collections import deque

import torch
import math


class TerminationCriterion(ABC):
    """
    An object which determines whether training should terminate.

    Instances of this class are callable and when called,
    they return whether training should stop (True) or continue (False)
    """
    @abstractmethod
    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        """
        Determines whether training should stop.

        :param training_loop: The training loop that is determining if training should stop
         via this call. Supplies information on the current training iteration, the current model,
         the last training loss and potentially further information.
        :param args: Additional arguments that the calling training loop may supply.
        :param kwargs: Additional arguments that the calling training loop may supply.
        :return: Whether training should terminate (True) or continue (False)
        """
        raise NotImplementedError()

    def reset(self, training_loop: 'TrainingLoop'):
        """
        Reset internal state of the termination criterion if training has ended.

        In it's default implementation, this method does nothing.

        :param training_loop: The training loop that triggers the reset.
        """
        pass

    def __and__(self, other):
        return TerminationCriterionConjunction(self, other)

    def __or__(self, other):
        return TerminationCriterionDisjunction(self, other)

    def __invert__(self):
        return TerminationCriterionNegation(self)


class TerminationCriterionConjunction(TerminationCriterion):
    """
    A conjunction of termination criteria. Training should only stop if
    all termination criteria of this conjunction are met (all criteria return True when called).
    """
    def __init__(self, *criteria: TerminationCriterion):
        self.criteria = criteria

    def __call__(self, *args, **kwargs):
        return all(criterion(*args, **kwargs) for criterion in self.criteria)

    def reset(self, training_loop: 'TrainingLoop'):
        for criterion in self.criteria:
            criterion.reset(training_loop)


class TerminationCriterionDisjunction(TerminationCriterion):
    """
    A disjunction of termination criteria. Training should stop if
    any termination criterion of this disjunction is met (some criterion returns True when called).
    """
    def __init__(self, *criteria: TerminationCriterion):
        self.criteria = criteria

    def __call__(self, *args, **kwargs):
        return any(criterion(*args, **kwargs) for criterion in self.criteria)

    def reset(self, training_loop: 'TrainingLoop'):
        for criterion in self.criteria:
            criterion.reset(training_loop)


class TerminationCriterionNegation(TerminationCriterion):
    """
    The negation of a termination criterion. Training should continue exactly then
    if the original criterion says it should stop (the criterion returns True when called).
    """

    def __init__(self, criterion: TerminationCriterion):
        self.criterion = criterion

    def __call__(self, *args, **kwargs):
        return not self.criterion(*args, **kwargs)

    def reset(self, training_loop: 'TrainingLoop'):
        self.criterion.reset(training_loop)


class IterationMaximum(TerminationCriterion):
    """
    Terminates training if a maximum number of iterations is reached.

    The training is terminated when the maximum number of iterations have been performed.
    E.g. for a maximum value of 10, the training is terminated after ten training steps have been performed.
    """
    def __init__(self, iterations_maximum: int):
        assert iterations_maximum >= 0, 'iterations_maximum may not be negative'
        self.max_iterations = iterations_maximum

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        # iteration counter starts at 0 for the first iteration
        return training_loop.iteration + 1 >= self.max_iterations


class TrainingLossValue(TerminationCriterion):
    """
    Terminates training if the training loss reaches a certain value.
    """
    def __init__(self, acceptable_loss: float):
        self.loss_value = acceptable_loss

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        return training_loop.current_training_loss <= self.loss_value


class TrainingLossChange(TerminationCriterion):
    """
    Terminates training if the change in training loss in the last few number of iterations
    falls below a certain threshold.

    The loss values are averaged over a number of iterations to smooth out
    the random fluctuations of the loss that occur when working with mini-batches.
    The iterations over which the loss is averaged is called an iteration block.

    The change is measured by the different between the largest loss value and the smallest loss
    value between the average loss of two blocks for a certain number of past blocks.
    If this change falls below a certain threshold, training is stopped.

    Training can be stopped in every iteration,
    but not in the first iterations in which not enough loss values have been recorded.
    In other words, termination can occur only if loss values for the specified number of blocks are
    available.
    The blocks are formed anew in every iteration. The oldest loss value is no longer taken into account
    and the block boundaries move forward one iteration.
    """
    def __init__(self, change_threshold: float = 1e-4, iteration_block_size=10, num_blocks=10):
        assert change_threshold >= 0, 'Loss change threshold can not be negative.'
        assert iteration_block_size > 0, 'Blocks need to consists of at least one iteration'
        assert num_blocks > 0, 'At least one block necessary'
        self.change_threshold = change_threshold
        self.block_size = iteration_block_size
        self.num_blocks = num_blocks

        self.losses = deque(maxlen=iteration_block_size * num_blocks)

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        self.losses.append(training_loop.current_training_loss)

        if len(self.losses) < self.block_size * self.num_blocks:
            # too few iterations performed yet
            return False

        max_avg_loss = -float('inf')
        min_avg_loss = float('inf')
        losses_iter = iter(self.losses)
        for _ in range(self.num_blocks):
            avg_loss = 1/self.block_size * sum(next(losses_iter) for _ in range(self.block_size))
            max_avg_loss = max(max_avg_loss, avg_loss)
            min_avg_loss = min(min_avg_loss, avg_loss)

        return (max_avg_loss - min_avg_loss) <= self.change_threshold

    def reset(self, training_loop: 'TrainingLoop'):
        self.losses = deque(maxlen=self.block_size * self.num_blocks)


class GradientValue(TerminationCriterion):
    """
    Terminates training if the mean absolute gradient of a set of parameters
    with respect to the training loss
    falls below a certain threshold.

    This termination criterion relies on the values of the ``.grad`` attributes
    of the parameters that should be available in the training loop when this termination
    criterion is called. If these are not available, a warning is logged.
    """
    def __init__(self, parameters, gradient_threshold: float = 1e-2):
        assert gradient_threshold >= 0, 'Gradient threshold may not be negative.'
        self.parameters = tuple(parameters)
        self.gradient_threshold = gradient_threshold

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        gradients = []
        for parameter in self.parameters:
            if parameter.grad is None:
                warning(f"gradient not available for parameter: {parameter}")
            else:
                gradients.append(parameter.grad.abs().flatten())
        avg_abs_gradient = torch.hstack(gradients).mean()
        return avg_abs_gradient <= self.gradient_threshold


class ValidationSet(TerminationCriterion):
    """
    Terminates training if the validation loss (e.g. on a held-out validation set)
    increases for a certain number of validation evaluations.

    Validation evaluations can be performed every iteration or after only every few iterations.
    """
    def __init__(self, validation_loss: Union[Callable[[], float], Callable[[], torch.Tensor]],
                 iterations_between_validations=0, acceptable_increase_length=5, tolerance_fraction=0.01,
                 reset_parameters=True):
        """
        Initializes a ValidationSet termination criterion.
        :param validation_loss: The loss function for measuring the validation loss.
        :param iterations_between_validations: The iterations gap between two validation events.
         A value of 0 means that validation is performed in every iteration.
         A value of 1 means in every second iteration, etc.
        :param acceptable_increase_length: How often validation performance may consecutively increase before
         training is stopped.
        :param tolerance_fraction: Controls fluctuations of what magnitude are not counted as increases.
         If the increase between two validation evaluations is smaller than this value times the previous loss value,
         this increase is not counted as an increase.
         Similarly if the decrease is smaller than this fraction, this decrease will also not be counted as a decrease.
         The default value allows fluctuations of 1%.
        :param reset_parameters: Whether to reset the parameters of the trained model to the parameters
         of the iteration that caused the first validation loss increase that lead to termination.
        """
        assert iterations_between_validations >= 0, "Gap between validation evaluations can not be negative."
        self.validation_loss = validation_loss
        self.validation_frequency = iterations_between_validations + 1
        self.num_increases = acceptable_increase_length
        self.tolerance = tolerance_fraction
        self.reset_parameters = reset_parameters

        self.last_validation_loss = float('inf')
        self.increases_counter = 0
        self.model_state_dict_at_first_increase = None
        self.criterion_lead_to_termination = False

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        if training_loop.iteration % self.validation_frequency != 0:
            return False

        loss = self.validation_loss()
        if self.last_validation_loss * (1 + self.tolerance) < loss:
            if self.increases_counter == 0 and self.reset_parameters:
                self.model_state_dict_at_first_increase = training_loop.model.state_dict()
            self.increases_counter += 1
        elif self.last_validation_loss * (1 - self.tolerance) > loss:
            self.increases_counter = 0
            self.model_state_dict_at_first_increase = None

        self.last_validation_loss = loss
        stop = self.increases_counter >= self.num_increases
        self.criterion_lead_to_termination = stop
        return stop

    def reset(self, training_loop: 'TrainingLoop'):
        if self.criterion_lead_to_termination and self.reset_parameters:
            assert self.model_state_dict_at_first_increase is not None
            training_loop.model.load_state_dict(self.model_state_dict_at_first_increase)

        self.last_validation_loss = float('inf')
        self.increases_counter = 0
        self.model_state_dict_at_first_increase = None
        self.criterion_lead_to_termination = False


class Divergence(TerminationCriterion):
    """
    Stops training if it diverges. This is detected via the loss value becoming infinity or nan.
    Optionally also the gradient values can be chacked.
    """
    def __init__(self, parameters=()):
        self.parameters = tuple(parameters)

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        if not math.isfinite(training_loop.current_training_loss):
            return True
        for parameter in self.parameters:
            if parameter.grad is not None and any(not math.isfinite(p) for p in parameter.grad.flatten()):
                return True
        return False

