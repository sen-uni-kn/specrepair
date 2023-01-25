# this file contains classes which reset the networks parameters
# to some value that was found during training.
# these classes should usually be used as post iteration hooks or termination criteria.

from logging import warning

from copy import deepcopy

import torch
from nn_repair.training.termination import TerminationCriterion


class LBFGSFixNanParameters(TerminationCriterion):
    """
    Fixes the misbehaviour of torch.optim.LBFGS
    to produce nan values documented here: https://github.com/pytorch/pytorch/issues/5953.

    This reset hook/termination criterion restores the last state_dict of the trained module
    before any value in the state dict became nan.
    """
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.last_state = deepcopy(module.state_dict())

    def __call__(self, training_loop: 'TrainingLoop', *args, **kwargs) -> bool:
        new_state = self.module.state_dict()
        if any(torch.any(torch.isnan(tensor)) for tensor in new_state.values()):
            warning('LBFGSFixNanParameters: nan detected, resetting parameters to previous.')
            # nan encountered
            self.module.load_state_dict(self.last_state)
            return True
        else:
            self.last_state = deepcopy(new_state)
            return False
