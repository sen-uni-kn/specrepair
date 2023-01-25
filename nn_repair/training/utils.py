from logging import debug
import torch

from nn_repair.training.training_loop import TrainingLoop, TrainingLoopHook, TrainingLoopHookCallLocation


class ResetOptimizer(TrainingLoopHook):
    """
    A training hook to reset the optimizer state.
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Initialize the ResetOptimizer hook.
        :param optimizer: The optimizer that will be reset using this hook.
        """
        self.optimizer = optimizer
        self.initial_state_dict = optimizer.state_dict()

    def __call__(self, training_loop: TrainingLoop, call_location: TrainingLoopHookCallLocation, *args, **kwargs):
        debug(f"Resetting optimizer.")
        self.optimizer.load_state_dict(self.initial_state_dict)

    def register(self, training_loop: 'TrainingLoop'):
        """
        Registers this ResetOptimizer hook as PRE_TRAINING hook at the given TrainingLoop.
        """
        training_loop.add_pre_training_hook(self)

    def update_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.initial_state_dict = optimizer.state_dict()
