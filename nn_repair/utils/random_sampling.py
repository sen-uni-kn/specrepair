from typing import Callable, Optional

import numpy as np
import torch
from itertools import islice

from tqdm import tqdm


def sample_from_normal_in_range(locs: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor, maxes: torch.Tensor,
                                num_rows: int = 1,
                                further_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                                rng: torch.Generator = torch.Generator(),
                                dtype: np.dtype = np.float32, show_progress=False) -> torch.Tensor:
    """
    Produces random samples from a normal distribution, but only keeping samples
    that are within a certain interval.

    :param locs: The location of the normal distribution from which data is generated.
     May have multiple values to generate data with multiple dimension, but needs to be a vectors.
     The number of elements needs to agree with the number of elements of the std, mins and maxes vectors.
    :param scales: The scale of the normal distribution from which data is generated.
     May have multiple values to generate data with multiple dimension, but needs to be a vectors.
     The number of elements needs to agree with the number of elements of the means, std and maxes vectors.
    :param mins: The lower bound of the interval to consider. May have multiple values to generate
     data with multiple dimension, but needs to be a vectors. The number of elements needs to agree with
     the number of elements of the means, std and maxes vectors.
    :param maxes: The upper bound of the interval to consider. May have multiple values to generate
     data with multiple dimension, but needs to be a vectors. The number of elements needs to agree with
     the number of elements of the means, std and mins vectors.
    :param num_rows: The number of rows of the generated array. The number of columns is the number of elements of the
     locs vectors.
    :param further_filter: This parameter allows to specify another filtering function that determines if a sample
     should be contained in the output array. With this parameter you can further refine the values you want to obtain.
     The argument to the filter function is a row of the final array. Return True if this row should be kept.
    :param rng: The random number generator to use.
    :param dtype: The dtype of the generated array. Defaults to single precision float.
    :param show_progress: Whether to display a progress bar of the sampling process.
    :return: A tensor with num_rows rows containing random samples drawn from a normal distribution that lie
     within the interval specified via mins and maxes.
    """
    batch_size = max(1, (5 * 1024) // locs.numel())
    batch_size = min(batch_size, num_rows)

    # generates an infinite sequence of normal distributed samples
    def samples_from_normal():
        locs_ = locs.expand((batch_size,) + locs.shape)
        scales_ = scales.expand((batch_size,) + locs.shape)
        while True:
            data = torch.normal(locs_, scales_, generator=rng)
            yield data

    def filter_out_of_range(data):
        select_idx = torch.all(
            torch.flatten((data >= mins) & (data <= maxes), start_dim=1),
            dim=-1
        )
        return data[select_idx]

    def apply_further_filter(data):
        select_idx = further_filter(data)
        return data[select_idx]

    filtered = map(filter_out_of_range, samples_from_normal())
    if further_filter is not None:
        filtered = map(apply_further_filter, filtered)

    if show_progress:
        progress_bar = tqdm(range(num_rows))
    else:
        progress_bar = None

    first_batch = next(filtered)
    result = first_batch
    result = result[:num_rows]
    if progress_bar is not None:
        progress_bar.update(result.size(0))
    while result.size(0) < num_rows:
        next_batch = next(filtered)
        if progress_bar is not None:
            progress_bar.update(next_batch.size(0))
        result = torch.cat([result, next_batch])
        result = result[:num_rows]

    return result
