from typing import List

from tqdm import tqdm

from deep_opt.utils.base_converter import decimal_to_base, decode


def sample_inputs(mins: List, maxes: List, num_samples: int, num_inputs: int) -> List:
    """
    Samples inputs with a number of samples per inputs dimension.
    :param mins: minimum list
    :param maxes: maximum list
    :param num_samples: number of samples
    :param num_inputs: inputs dimension
    :return: list of sampled inputs tuples (cartesian product of all sampled inputs per inputs dimension)
    """
    sampled_inputs = []
    cartesian_product_size = num_samples ** num_inputs

    for s in tqdm(range(cartesian_product_size), 'Sampling'):
        s_base = decimal_to_base(s, num_samples)

        s_base = s_base.zfill(num_inputs)

        sampled_input = [mins[i] + ((maxes[i] - mins[i]) / num_samples) * decode(s_base[i])
                         for i in range(num_inputs)]

        sampled_inputs.append(sampled_input)

    return sampled_inputs
