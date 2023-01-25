from typing import List, Tuple, Sequence, Dict

from deep_opt import NeuralNetwork, Property
from properties import property_1, property_2, property_3, property_4, property_7, property_8


def acasxu_properties() -> Dict[int, Property]:
    """
    Returns the properties retrievable by property number (indexed from 1)
    used in the ACAS Xu repair experiments.

    :return: A dictionary of ACAS Xu properties.
    """
    return {
        1: property_1(), 2: property_2(), 3: property_3(), 4: property_4(),
        7: property_7(), 8: property_8()
    }


def acasxu_repair_1_repair_cases() -> List[Tuple[Tuple[int, ...], int, int]]:
    return (
        # property 2 repair cases (individual)
        [((2, ), i, j) for i in range(2, 6) for j in range(1, 10) if (i, j) not in ((3, 3), (4, 2))] +
        [((7, ), 1, 9)] + [((8, ), 2, 9)] +  # property 7 and 8 individual repair cases
        # [((1, 7), 1, 9)] +  # this one is not interesting (same as only p7)
        [((1, 2, 3, 4, 8), 2, 9)]
    )


def load_acasxu_network(net_i0, net_i1):
    return NeuralNetwork.load_from_nnet(f'../resources/acasxu/ACASXU_run2a_{net_i0}_{net_i1}_batch_2000.nnet')


def acasxu_repair_case_dir_name(property_indices, net_i0, net_i1):
    return 'property_' + '_'.join(str(i) for i in property_indices) + f'_net_{net_i0}_{net_i1}'
