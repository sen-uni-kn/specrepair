from typing import Optional, Tuple, Callable, Sequence
from abc import ABC, abstractmethod
import logging

import torch.utils.tensorboard as tensorboard

from nn_repair.training.training_loop import TrainingLoop, TrainingLoopHook, TrainingLoopHookCallLocation


class LossLogger(TrainingLoopHook, ABC):
    """
    Abstract base class for classes that somehow display loss information as training hooks.
    """
    def __init__(self, frequency: int, average_training_loss: bool,
                 training_loss_name: str = 'training',
                 additional_losses: Sequence[Tuple[str, Callable, bool]] = ()):
        """
        Initialize the LossLogger hook.

        :param frequency: After how many iterations a new log action is performed.
        :param average_training_loss: Whether to use the average loss over all iterations since the last log (True, default)
         or only the loss of the iteration in which the log is cast. This option only applies for the
         loss recorded in the training loop. Averaging of the additional losses is controlled for each additional
         loss individually.
        :param additional_losses: Additional callables that supply losses to log. The first element of the tuple
         if a name for the loss supplied by the second element, which calculates the loss.
         The callable needs to return a single value which needs to be addable to a float value.
         The callables may also return None to indicate that a value is not available.
         The third element specifies whether the callable should be calculated every iteration and averaged
         if average_losses is True (True) or only calculated if a log action is performed (False).
         None values are ignored in averaging. If all recorded values were None when averaging, this loss
         value is not logged in the current iteration.
        """
        self.frequency = frequency
        self.average_losses = average_training_loss
        self.training_loss_name = training_loss_name

        every_iter_losses = []
        log_only_losses = []
        # makes sure the order in which the additional losses are passed to us are preserved when logging
        logger_order = []
        for additional_loss in additional_losses:
            _, _, every_iter = additional_loss
            if every_iter:
                every_iter_losses.append(additional_loss)
                logger_order.append((True, len(every_iter_losses) - 1))
            else:
                log_only_losses.append(additional_loss)
                logger_order.append((False, len(log_only_losses) - 1))
        self.every_iter_losses = tuple(every_iter_losses)
        self.log_only_losses = tuple(log_only_losses)
        self.logger_order = tuple(logger_order)

        self._reset_running_losses()

    def _reset_running_losses(self):
        self.running_loss = 0.0
        # there are various occasions in which less than frequency many values are recorded
        # the counters count how many values were recorded to allow averaging.
        self.running_loss_counter = 0
        self.additional_running_losses = [0.0] * len(self.every_iter_losses)
        self.additional_counters = [0] * len(self.every_iter_losses)

    def __call__(self, training_loop: TrainingLoop, call_location: TrainingLoopHookCallLocation, *args, **kwargs):
        if self.average_losses:
            training_loss = training_loop.current_training_loss
            if training_loss is not None:
                self.running_loss += training_loss.item()
                self.running_loss_counter += 1
            for i, (_, calc_loss, _) in enumerate(self.every_iter_losses):
                additional_loss = calc_loss()
                if additional_loss is not None:
                    self.additional_running_losses[i] += additional_loss
                    self.additional_counters[i] += 1

        if training_loop.iteration % self.frequency == 0 or training_loop.terminated:
            # logging iteration
            if self.average_losses:
                training_loss = self.running_loss / self.running_loss_counter \
                    if self.running_loss_counter > 0 else None
            else:
                training_loss = training_loop.current_training_loss
                if training_loss is not None:
                    training_loss = training_loss.item()

            losses = [(self.training_loss_name, training_loss)]
            for every_iter_loss, i in self.logger_order:
                if not every_iter_loss:
                    name, calc_loss, _ = self.log_only_losses[i]
                    losses.append((name, calc_loss()))
                else:
                    name, _, _ = self.every_iter_losses[i]
                    avg_loss = self.additional_running_losses[i] / self.additional_counters[i] \
                        if self.additional_counters[i] > 0 else None
                    losses.append((name, avg_loss))

            self.log_action(training_loop, call_location, losses)
            self._reset_running_losses()

    def register(self, training_loop: 'TrainingLoop'):
        """
        Registers this logging hook as POST_ITERATION hook at the given TrainingLoop.
        """
        training_loop.add_post_iteration_hook(self)

    @abstractmethod
    def log_action(self, training_loop: TrainingLoop, call_location: TrainingLoopHookCallLocation,
                   losses: Sequence[Tuple[str, float]]):
        raise NotImplementedError()


class LogLoss(LossLogger):
    """
    A training hook which logs the training set loss and potentially losses on other datasets.
    """
    def __init__(self, log_frequency: int = 10, epoch_length: Optional[int] = None,
                 average_training_loss: bool = True, additional_losses: Sequence[Tuple[str, Callable, bool]] = (),
                 log_level: int = logging.INFO, loss_format_string: str = '.4f'):
        """
        Initialize the LogLoss hook.

        :param log_frequency: After how many iterations a new log message should be displayed.
        :param epoch_length: The length of one epoch. If specified splits the iteration counter
            into an epoch counter and an per epoch iteration counter.
        :param average_training_loss: Whether to log the average loss over all iterations since the last log (True, default)
            or only the loss of the iteration in which the log is cast.
        :param additional_losses: Additional callables that supply losses to log. The first element of the tuple
            is a name for the loss supplied by the second element, which calculates the loss.
            The callable needs to return a single value, which needs to be printable in a float format string.
            The callable may optimally return None, which will disable the log message for this iteration.
            The third element specifies whether the callable should be calculated every iteration and averaged
            if average_losses is True (True) or calculated only if a message is logged (False).
        :param log_level: The logging level to use, e.g. logging.INFO (default), logging.DEBUG, etc.
        :param loss_format_string: The format string to use for printing the losses.
        """
        super().__init__(log_frequency, average_training_loss,
                         training_loss_name='training loss', additional_losses=additional_losses)
        self.epoch_length = epoch_length
        self.level = log_level
        self.loss_format_string = loss_format_string

    def log_action(self, training_loop: TrainingLoop, call_location: TrainingLoopHookCallLocation,
                   losses: Sequence[Tuple[str, float]]):
        log_string = ''
        if self.epoch_length is not None:
            epoch = training_loop.iteration // self.epoch_length
            epoch_iteration = training_loop.iteration % self.epoch_length
            log_string += f'epoch: {epoch} ' \
                          f'(iteration: {epoch_iteration}, {(epoch_iteration / self.epoch_length) * 100:3.0f}%)'
        else:
            log_string += f'iteration: {training_loop.iteration}'

        log_string += ' | '
        log_string += ', '.join([
            ('{}: {:' + self.loss_format_string + '}').format(name, loss)
            for name, loss in losses
            if loss is not None  # None indicates "do not log"
        ])

        logging.log(self.level, log_string)


class TensorboardLossPlot(LossLogger):
    """
    A training hook which plot the training set loss and potentially losses on other datasets
    using tensorboard
    """
    def __init__(self, summary_writer: tensorboard.SummaryWriter, frequency: int = 1,
                 training_loss_tag: str = 'training loss', average_training_loss: bool = True,
                 additional_losses: Sequence[Tuple[str, Callable, bool]] = ()):
        """
        Initialize the LogLoss hook.

        :param summary_writer: The torch.utils.tensorboard.SummaryWriter for sending data to tensorboard.
        :param frequency: After how many iterations a new data point should be displayed.
        :param training_loss_tag: Which tag to use for the data series of the training loss.
        :param average_training_loss: Whether to average loss over all iterations since the last update (True, default)
         or only the loss of the iteration in which the data point is send to tensorboard.
        :param additional_losses: Additional callables that supply losses to show. The first element of the tuple
         is the tag of the loss supplied by the second element, which calculates the loss.
         The callable needs to return a single value
         which needs to be acceptable for summary_writer.add_scalar as scalar_value.
         The third element specifies whether the callable should be calculated every iteration and averaged
         if average_losses is True (True) or calculated only if a data point is send (False).
        """
        super().__init__(frequency, average_training_loss, training_loss_tag, additional_losses)
        self.writer = summary_writer

    def log_action(self, training_loop: TrainingLoop, call_location: TrainingLoopHookCallLocation,
                   losses: Sequence[Tuple[str, float]]):
        for tag, loss in losses:
            self.writer.add_scalar(tag, loss, training_loop.iteration)
