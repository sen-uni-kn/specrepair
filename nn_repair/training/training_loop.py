from typing import Optional, Callable, Sequence
from enum import Enum, auto
from abc import ABC, abstractmethod

import torch
from nn_repair.training.termination import TerminationCriterion
from nn_repair.utils.eval_mode import Train


class TrainingLoopHookCallLocation(Enum):
    """
    The different locations in a TrainingLoop a TrainingLoopHook can be called in.
    """
    PRE_ITERATION = auto()
    POST_ITERATION = auto()
    PRE_TRAINING = auto()
    POST_TRAINING = auto()


class TrainingLoopHook(ABC):
    """
    A callable, which is invoked at certain locations in the training loop.

    Hooks are for example useful for logging and saving checkpoints.
    """
    @abstractmethod
    def __call__(self, training_loop: 'TrainingLoop', call_location: TrainingLoopHookCallLocation, *args, **kwargs):
        """
        Determines whether training should stop.

        :param training_loop: The training loop that is calling the hook.
        :param call_location: The current location in the training loops execution.
             * PRE_ITERATION: just before an optimizer step is performed.
             * POST_ITERATION: after an optimizer step has been performed and after the termination criterion
               has been evaluated. The iteration counter will not yet have been updated.
               Hence the value will be the same as for the PRE_ITERATION hooks called before the last optimizer step.
             * PRE_TRAINING: just before the training loop is started. Iteration will be zero.
             * POST_TRAINING: after the training loop was terminated, before the termination criterion is reset.
        :param args: Additional arguments that the calling training loop may supply.
        :param kwargs: Additional arguments that the calling training loop may supply.
        """
        raise NotImplementedError()

    @abstractmethod
    def register(self, training_loop: 'TrainingLoop'):
        """
        Properly register this training hook to the given training loop.

        Adds this hook to the training_loop, at the proper place(s).
        :param training_loop: The training loop at which this hook should be registered.
        """
        raise NotImplementedError()


class TrainingLoop:
    """
    Executes a training loop using a training algorithm and loss function.

    Supplies a number of statistics about training.
    Can plot training progress using tensorboard.

    Training is executed when the `execute` method is called. When this method is called,
    the model to train, the training algorithm, the loss function and a termination criterion
    need to have been set either via the respective properties or via the initializer.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: Optional[Callable[[], torch.Tensor]] = None,
                 termination_criterion: Optional[TerminationCriterion] = None,
                 lr_scheduler: Optional = None,
                 pre_iteration_hooks: Sequence[TrainingLoopHook] = (),
                 post_iteration_hooks: Sequence[TrainingLoopHook] = (),
                 pre_training_hooks: Sequence[TrainingLoopHook] = (),
                 post_training_hooks: Sequence[TrainingLoopHook] = ()):
        """
        Initialize the TrainingLoop.

        :param model: The model which parameters will be optimized in the training loop.
        :param optimizer: The `torch.optim.Optimizer` to use for updating the parameters.
         The optimizer needs to be initialized with the parameters of the model that should be trained already.
        :param loss_function: The loss function that is minimized during training.
         The loss function can also be specified or updated later by setting the loss_function property.
         Pass None to supply the loss function later.

         The loss function should not call backward itself. It should only calculate the loss.
         Backward will be called independently by the training loop.
        :param termination_criterion: The termination criterion which determines when training stops.
         The termination criterion can also be specified or updated later by setting the termination_criterion property
         or using the add_termination_criterion method. Pass None to supply the termination criterion later.
        :param lr_scheduler: An optional learning rate scheduler. A learning rate scheduler needs to provide
         a `step()` method.
        :param pre_iteration_hooks: A collection of callables, which are called before a training step is performed.
        :param post_iteration_hooks: A collection of callables,
         which are called after a training step has been performed.
        :param pre_training_hooks: A collection of callables, which are called before the training loop is executed.
        :param post_training_hooks: A collection of callables, which are called after the training loop has stopped.
        """
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._loss = loss_function
        self._terminate = termination_criterion
        self._model = model
        self._pre_iteration_hooks = list(pre_iteration_hooks)
        self._post_iteration_hooks = list(post_iteration_hooks)
        self._pre_training_hooks = list(pre_training_hooks)
        self._post_training_hooks = list(post_training_hooks)

        self._i = 0  # iteration counter
        self._training_loss = None  # the last training loss
        self._stop_training = False

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> Optional:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, new_lr_scheduler: Optional):
        """
        Sets a learning rate scheduler.
        Set this to None for not using a scheduler.

        :param new_lr_scheduler: the new learning rate scheduler or None
        """
        self._lr_scheduler = new_lr_scheduler

    @property
    def loss_function(self) -> Optional[Callable[[], torch.Tensor]]:
        return self._loss

    @loss_function.setter
    def loss_function(self, new_loss_function: Callable[[], torch.Tensor]):
        """
        Sets the loss function that is minimized during training.

        The loss function should not call backward itself. It should only calculate the loss.
        Backward will be called independently by the training loop.

        :param new_loss_function: the new loss function to minimize during training.
        """
        self._loss = new_loss_function

    @property
    def termination_criterion(self) -> TerminationCriterion:
        return self._terminate

    @termination_criterion.setter
    def termination_criterion(self, new_termination_criterion):
        self._terminate = new_termination_criterion

    def add_termination_criterion(self, additional_termination_criterion):
        """
        Adds another termination criterion.
        If no other termination criterion is present, then the additional_termination_criterion is
        set as the termination criterion.
        If a termination criterion has already been set, this termination criterion
        is combined with the additional one using |.
        This means that training will be terminated if any termination criterion instructs the training loop
        to do so.

        :param additional_termination_criterion: An additional termination criterion.
         Training will be terminated if this or any of the already added termination criteria instructs the training
         loop to stop.
        """
        if self._terminate is None:
            self._terminate = additional_termination_criterion
        else:
            self._terminate = self._terminate | additional_termination_criterion

    @property
    def iteration(self) -> int:
        """
        The current iteration. The first iteration is iteration 0.
        """
        return self._i

    @property
    def current_training_loss(self) -> torch.Tensor:
        """
        The training loss in the last iteration returned by `optimizer.step`.
        None if no training step has yet been performed.
        """
        return self._training_loss

    @property
    def terminated(self) -> bool:
        """
        Whether training was stopped in the current iteration.
        """
        return self._stop_training

    def execute(self) -> torch.nn.Module:
        """
        Execute the training loop.

        The properties optimizer, loss_function, model and termination_criterion need to be set.
        If any of these is missing, an assertion error is raised.

        :return: The trained module. The return value is identical to the model property of this training loop.
        """
        assert self._optimizer is not None
        assert self._loss is not None
        assert self._terminate is not None
        assert self._model is not None

        def loss_closure():
            if torch.is_grad_enabled():
                self._optimizer.zero_grad()
            loss = self._loss()
            if loss.requires_grad:
                loss.backward()
            return loss

        self._i = 0
        self._stop_training = False
        self._training_loss = None

        with Train(self._model):
            for hook in self._pre_training_hooks:
                hook(self, TrainingLoopHookCallLocation.PRE_TRAINING)

            while not self._stop_training:
                for hook in self._pre_iteration_hooks:
                    hook(self, TrainingLoopHookCallLocation.PRE_ITERATION)

                self._training_loss = self._optimizer.step(closure=loss_closure).float()
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()
                self._stop_training = self._terminate(training_loop=self)

                for hook in self._post_iteration_hooks:
                    hook(self, TrainingLoopHookCallLocation.POST_ITERATION)

                if not self._stop_training:
                    self._i += 1

            for hook in self._post_training_hooks:
                hook(self, TrainingLoopHookCallLocation.POST_TRAINING)
            # reset state
            self._terminate.reset(training_loop=self)
            return self._model

    def clear_pre_training_hooks(self):
        """
        Removes all pre training callables.
        """
        self._pre_training_hooks = []

    def clear_post_training_hooks(self):
        """
        Removes all post training callables.
        """
        self._post_training_hooks = []

    def clear_pre_iteration_hooks(self):
        """
        Removes all pre training iteration callables.
        """
        self._pre_iteration_hooks = []

    def clear_post_iteration_hooks(self):
        """
        Removes all post training iteration callables.
        """
        self._post_iteration_hooks = []

    def add_pre_training_hook(self, hook: TrainingLoopHook):
        """
        Adds a callable which is invoked before the training loop starts.
        :param hook: The callable to invoke before the training loop starts.
        """
        self._pre_training_hooks.append(hook)

    def add_post_training_hook(self, hook: TrainingLoopHook):
        """
        Adds a callable which is invoked after the training loop has finished.
        :param hook: The callable to invoke after the training loop has finished.
        """
        self._post_training_hooks.append(hook)

    def add_pre_iteration_hook(self, hook: TrainingLoopHook):
        """
        Adds a callable which is invoked before each training step.
        :param hook: The callable to invoke before each training step.
        """
        self._pre_iteration_hooks.append(hook)

    def add_post_iteration_hook(self, hook: TrainingLoopHook):
        """
        Adds a callable which is invoked after each training step,
        after it has been determined whether training should terminate.
        :param hook: The callable to invoke before each training step.
        """
        self._post_iteration_hooks.append(hook)

    def remove_pre_training_hook(self, hook: TrainingLoopHook):
        """
        Remove one of the callables which are invoked before the training loop starts.
        :param hook: The callable to remove
        """
        self._pre_training_hooks.remove(hook)

    def remove_post_training_hook(self, hook: TrainingLoopHook):
        """
        Remove one of the callables which are invoked after the training loop has finished.
        :param hook: The callable to remove
        """
        self._post_training_hooks.remove(hook)

    def remove_pre_iteration_hook(self, hook: TrainingLoopHook):
        """
        Remove one of the callables which are invoked before each training iteration.
        :param hook: The callable to remove
        """
        self._pre_iteration_hooks.remove(hook)

    def remove_post_iteration_hook(self, hook: TrainingLoopHook):
        """
        Remove one of the callables which are invoked after each training iteration.
        :param hook: The callable to remove
        """
        self._post_iteration_hooks.remove(hook)
