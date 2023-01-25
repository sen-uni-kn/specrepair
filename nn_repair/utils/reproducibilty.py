import torch
from torch.utils.data import DataLoader
import numpy as np
import random

# Code based on: https://pytorch.org/docs/stable/notes/randomness.html


def torch_data_loading_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_data_loader(seed, **kwargs):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        **kwargs,
        worker_init_fn=torch_data_loading_seed_worker,
        generator=g,
    )
