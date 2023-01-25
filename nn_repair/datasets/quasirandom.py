from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset
from torch.quasirandom import SobolEngine


class SobolDataset(Dataset):
    """
    A dataset of sobol quasi-random samples.
    """

    def __init__(self, number_of_samples: int, dimension: int,
                 target_fn: Callable[[int, torch.Tensor], torch.Tensor],
                 seed=92417172022):
        self.target_fn = target_fn
        self.length = number_of_samples
        self.eng = SobolEngine(dimension, scramble=True, seed=seed)
        self.prev_index = -1

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prev_index != index - 1:
            self.eng.reset()
            self.eng.fast_forward(index - 1)
        self.prev_index = index
        sample = self.eng.draw().squeeze(0)
        return sample, self.target_fn(index, sample)

    def __len__(self):
        return self.length
