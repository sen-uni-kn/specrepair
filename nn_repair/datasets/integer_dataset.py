import os
from logging import info
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class IntegerDataset(Dataset):
    """
    A randomly generated integer dataset.
    The dataset consists of a sorted list of randomly sampled integers.
    The goal is to map an integer value to its position in the sorted list.
    IntegerDataset is based on [KraskaEtAl2018]_.

    IntegerDataset supports sampling integers from different distributions:
    uniform, normal, and log-normal.

    An IntegerDataset can be stored to avoid having to regenerate it.
    The dataset is identified by the random distribution and the random key
    used to generate the dataset.
    To turn off saving a dataset, pass None for the dataset root directory
    to the initializer.

    .. [KraskaEtAl2018] Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean,
       Neoklis Polyzotis: The Case for Learned Index Structures.
       SIGMOD Conference 2018: 489-504 https://dl.acm.org/doi/10.1145/3183713.3196909
    """
    files_dir = "IntegerDatasets"

    def __init__(self, root: Optional[Union[os.PathLike, str]], size: int = 190000,
                 distribution: str = "uniform", maximum=1e6, seed=668509836114062):
        """
        Create or load an IntegerDataset.

        :param root: The root directory where IntegerDatasets are stored.
         When an IntegerDataset with the right size, random distribution, scale,
         and random seed
         is stored in this directory, this dataset is loaded instead of generating
         it again.
         When no such dataset is already stored, the newly generated dataset is
         stored in this directory.
         To turn loading and storing the dataset off, pass None for this argument.
        :param size: The number of samples in the IntegerDataset.
        :param distribution: The random distribution to use for generating the dataset.
         Possible values are: uniform, normal, and log-normal.
        :param maximum: The sampled values are re-scaled up to a certain limit.
         This value is the largest possible value in the generated dataset.
        :param seed: The random seed to use for generating the dataset.
        """
        if distribution not in ("uniform", "normal", "log-normal"):
            raise ValueError(f"Unknown distribution: {distribution}. "
                             f"Choose one of 'uniform', 'normal', and "
                             f"'log-normal'.")
        self.size = size
        self.distribution = distribution
        self.seed = seed
        self.maximum = maximum

        if root is None:
            self.data = self._generate()
            return

        file_name = f"{distribution}_{size}_{maximum:.0f}_{seed}.pyt"
        file_path = Path(root, self.files_dir, file_name)
        if not file_path.exists():
            self.data = self._generate()
            file_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.data, file_path)
        else:
            self.data = torch.load(file_path)

    def _generate(self) -> torch.Tensor:
        info(f"Generating IntegerDataset (n={self.size}, "
             f"distribution={self.distribution}, seed={self.seed})")
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        data = torch.empty((self.size,), dtype=torch.float)
        if self.distribution == "uniform":
            data.uniform_(0.0, 1.0, generator=rng)
        elif self.distribution == "normal":
            data.normal_(generator=rng)
        elif self.distribution == "log-normal":
            data.log_normal_(generator=rng)

        # rescale
        min_value = data.amin()
        total_range = data.amax() - min_value
        data -= min_value  # shift minimum to 0
        data /= total_range  # normalize to [0, 1]
        data *= self.maximum  # now scale to [0, maximum]

        data = data.type(dtype=torch.int)

        return data.msort()

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], torch.tensor(index, dtype=torch.long)

    def __len__(self):
        return self.size
