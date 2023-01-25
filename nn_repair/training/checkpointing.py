from typing import Sequence, Tuple, Any, Callable, Optional, Union
from os import PathLike
from copy import deepcopy
from pathlib import Path
from queue import Queue

import torch

from nn_repair.utils.background_file_writer import BackgroundFileWriter
from nn_repair.training.training_loop import TrainingLoop, TrainingLoopHook, TrainingLoopHookCallLocation


class Checkpointing(TrainingLoopHook):
    """
    A TrainingLoopHook, that stores model and optimizer checkpoints at fixed intervals.
    Intended for use as POST_ITERATION hook.
    Checkpoints are stored using ``torch.save``. File writing is performed in a background thread.
    The stored files are named "run_j_iteration_i.pyt", where i is the training loop iteration and j
    counts how often the TrainingLoop was already executed.

    The states are stored in a dictionary with the keys ``'model'`` for the model's state
    and ``'optimizer'`` for the optimizers state.
    The stored dictionaries also contain the current iteration for the ``'iteration'`` key.
    Additional states can be saved by providing keys and callables that produce the respective state in the
    constructor.

    The static ``restore`` method allows restoring the state saved in checkpoints.
    """

    def __init__(self, output_directory: Optional[Union[str, PathLike]] = None, frequency: int = 10,
                 save_model_state: bool = True, save_optimizer_state: bool = True,
                 additional_states: Sequence[Tuple[str, Callable[[], Any]]] = ()):
        """
        Initializes a Checkpointing TrainingLoopHook.

        :param output_directory: The directory in which the checkpoint files should be placed.
         May be None. In this case the output directory needs to be specified later on using the
         set_output_directory method. If no output directory is supplied when the first checkpoint should be saved,
         an assertion error will be produced.
        :param frequency: After how many iterations the next checkpoint is to be saved.
         If frequency is 10, the first checkpoint is saved in iteration 9
         (which is the 10th iteration since iterations count from 0). The second one is saved in iteration 19
         (20th iteration), and so on.
        :param save_model_state: Whether to checkpoint the model state (using the key ``'model'``).
        :param save_optimizer_state: Whether to checkpoint the optimizer state (using the key ``'optimizer'``).
        :param additional_states: Additional states that can be stored. The first element of each tuple is
         the key used in the stored dictionary. The second element is a callable that produces the state to be saved.
        """
        self.output_dir = output_directory
        self.frequency = frequency
        self.save_model_state = save_model_state
        self.save_optimizer_state = save_optimizer_state
        self.additional_states = additional_states

        self.run_counter = 0
        self.object_queue = Queue()
        self.writer = BackgroundFileWriter(self.object_queue, writing_function='torch')
        self.writer.start()

    def set_output_directory(self, output_directory: Union[str, PathLike]):
        """
        Sets the directory where checkpoints are stored.
        """
        self.output_dir = output_directory

    def __call__(self, training_loop: 'TrainingLoop', call_location: TrainingLoopHookCallLocation, *args, **kwargs):
        if training_loop.iteration % self.frequency == self.frequency - 1:
            assert self.output_dir is not None, "Output directory was not set when trying to checkpoint."
            states = {'iteration': training_loop.iteration}
            if self.save_model_state:
                states['model'] = deepcopy(training_loop.model.state_dict())
            if self.save_optimizer_state:
                states['optimizer'] = deepcopy(training_loop.optimizer.state_dict())
            for key, get_state in self.additional_states:
                states[key] = deepcopy(get_state())

            output_file = Path(self.output_dir, f'run_{self.run_counter}_iteration_{training_loop.iteration}.pyt')
            self.object_queue.put((states, output_file))

        if training_loop.terminated:
            self.run_counter += 1

    def register(self, training_loop: 'TrainingLoop'):
        """
        Registers this Checkpointing hook as POST_ITERATION hook
        at the given training_loop.
        """
        training_loop.add_post_iteration_hook(self)

    @staticmethod
    def restore(filename: str, model: Optional[torch.nn.Module] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                additional_restores: Sequence[Tuple[str, Callable[[Any], None]]] = ()):
        """
        Restores state from a file.

        :param filename: The checkpoint file created by Checkpointing.
        :param model: The model whose state_dict (parameters and buffers) to restore.
         If None, no model information is restored.
        :param optimizer: The optimizer whose state_dict (e.g. learning rate, etc.) to restore.
         If None, no optimizer information is restored.
        :param additional_restores: Additional callables for restoring state.
         Analogue to the additional_states parameter of the init function.
        """
        states = torch.load(filename)
        if model is not None:
            model.load_state_dict(states['model'])
        if optimizer is not None:
            optimizer.load_state_dict(states['optimizer'])

        for key, restore_state in additional_restores:
            restore_state(states[key])

    def close(self):
        """
        Waits until all files have been written to disk and then stops the background file saving thread.
        """
        self.object_queue.put(None)
        self.object_queue.join()
