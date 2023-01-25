from pathlib import Path
from typing import Optional, Union, Tuple
from enum import Enum, auto
import os

import numpy as np
import pandas
import torch
import torch.utils.data as torch_data
from torchvision import datasets
from torchvision.transforms import ToTensor

from nn_repair.datasets import DREBIN, Adult, CMAPSS, IntegerDataset


class Datasets(Enum):
    HCAS_PRA1_TAU00 = auto()
    COLLISION_DETECTION = auto()
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    PIMPLE_BOWL = auto()
    DREBIN = auto()
    ADULT = auto()
    CMAPSS20 = auto()
    CMAPSS40 = auto()
    OPTIDOSE_EASY = auto()
    OPTIDOSE_ONCO = auto()
    PK_1_CMP_PO = auto()

    @staticmethod
    def from_str(descriptor: str) -> 'Datasets':
        if descriptor.upper() == 'MNIST':
            return Datasets.MNIST
        elif descriptor.upper() == 'CIFAR10':
            return Datasets.CIFAR10
        elif descriptor.upper() == 'FASHIONMNIST' or descriptor.upper() == 'FASHION_MNIST':
            return Datasets.FASHION_MNIST
        elif descriptor.upper() == 'DREBIN':
            return Datasets.DREBIN
        elif descriptor.upper() == 'ADULT':
            return Datasets.ADULT
        elif descriptor.replace("_", "").replace(" ", "").replace("-", "").upper() == 'CMAPSS20':
            return Datasets.CMAPSS20
        elif descriptor.replace("_", "").replace(" ", "").replace("-", "").upper() == 'CMAPSS40':
            return Datasets.CMAPSS40
        elif descriptor.upper() == 'COLLISIONDETECTION' or descriptor.upper() == 'COLLISION_DETECTION':
            return Datasets.COLLISION_DETECTION
        elif descriptor.upper() == 'PIMPLEBOWL' or descriptor.upper() == 'PIMPLE_BOWL':
            return Datasets.PIMPLE_BOWL
        elif descriptor.upper() == 'HCAS' or descriptor.upper() == 'HCAS_PRA1_TAU00':
            return Datasets.HCAS_PRA1_TAU00
        elif descriptor.upper() == "OPTIDOSEEASY" or descriptor.upper() == "OPTIDOSE_EASY":
            return Datasets.OPTIDOSE_EASY
        elif descriptor.upper() == "OPTIDOSEONCO" or descriptor.upper() == "OPTIDOSE_ONCO":
            return Datasets.OPTIDOSE_ONCO
        elif descriptor.replace(" ", "_").replace("-", "_").upper() == "PK_1_CMP_PO":
            return Datasets.PK_1_CMP_PO
        else:
            raise ValueError(f'Unknown dataset: {descriptor}')


def get_dataset(dataset: Datasets, project_root_path='..'):
    if dataset == Datasets.HCAS_PRA1_TAU00:
        return hcas_pra1_tau00(project_root_path=project_root_path)
    elif dataset == Datasets.COLLISION_DETECTION:
        return collision_detection(project_root_path=project_root_path)
    elif dataset == Datasets.MNIST:
        return mnist(project_root_path=project_root_path)
    elif dataset == Datasets.CIFAR10:
        return cifar10(project_root_path=project_root_path)
    elif dataset == Datasets.FASHION_MNIST:
        return fashion_mnist(project_root_path=project_root_path)
    elif dataset == Datasets.PIMPLE_BOWL:
        return pimple_bowl(project_root_path=project_root_path)
    elif dataset == Datasets.DREBIN:
        return drebin(project_root_path=project_root_path)
    elif dataset == Datasets.ADULT:
        return adult(project_root_path=project_root_path)
    elif dataset == Datasets.CMAPSS20:
        return cmapss(project_root_path=project_root_path, window_size=20)
    elif dataset == Datasets.CMAPSS40:
        return cmapss(project_root_path=project_root_path, window_size=40)
    elif dataset == Datasets.OPTIDOSE_EASY:
        return optidose_easy(project_root_path=project_root_path)
    elif dataset == Datasets.OPTIDOSE_ONCO:
        return optidose_onco(project_root_path=project_root_path)
    elif dataset == Datasets.PK_1_CMP_PO:
        return pk_1_cmp_po(project_root_path=project_root_path)
    else:
        raise ValueError("Unknown dataset")


def get_training_set(dataset: Datasets, project_root_path='..'):
    if dataset == Datasets.HCAS_PRA1_TAU00:
        return hcas_pra1_tau00(project_root_path=project_root_path)
    elif dataset == Datasets.COLLISION_DETECTION:
        return collision_detection(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.MNIST:
        return mnist(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.CIFAR10:
        return cifar10(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.FASHION_MNIST:
        return fashion_mnist(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.PIMPLE_BOWL:
        return pimple_bowl(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.DREBIN:
        return drebin(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.ADULT:
        return adult(train_set=True, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.CMAPSS20:
        return cmapss(train_set=True, test_set=False, project_root_path=project_root_path, window_size=20)
    elif dataset == Datasets.CMAPSS40:
        return cmapss(train_set=True, test_set=False, project_root_path=project_root_path, window_size=40)
    elif dataset == Datasets.OPTIDOSE_EASY:
        return optidose_easy(train_set=True, validation_set=False, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.OPTIDOSE_ONCO:
        return optidose_onco(train_set=True, validation_set=False, test_set=False, project_root_path=project_root_path)
    elif dataset == Datasets.PK_1_CMP_PO:
        return pk_1_cmp_po(train_set=True, test_set=False, project_root_path=project_root_path)
    else:
        raise ValueError("Unknown dataset")


def get_test_set(dataset: Datasets, project_root_path='..'):
    if dataset == Datasets.HCAS_PRA1_TAU00:
        raise ValueError('HCAS datasets do not have a test set')
    elif dataset == Datasets.COLLISION_DETECTION:
        return collision_detection(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.MNIST:
        return mnist(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.CIFAR10:
        return cifar10(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.FASHION_MNIST:
        return fashion_mnist(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.PIMPLE_BOWL:
        return pimple_bowl(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.DREBIN:
        return drebin(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.ADULT:
        return adult(train_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.CMAPSS20:
        return cmapss(train_set=False, test_set=True, project_root_path=project_root_path, window_size=20)
    elif dataset == Datasets.CMAPSS40:
        return cmapss(train_set=False, test_set=True, project_root_path=project_root_path, window_size=40)
    elif dataset == Datasets.OPTIDOSE_EASY:
        return optidose_easy(train_set=False, validation_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.OPTIDOSE_ONCO:
        return optidose_onco(train_set=False, validation_set=False, test_set=True, project_root_path=project_root_path)
    elif dataset == Datasets.PK_1_CMP_PO:
        return pk_1_cmp_po(train_set=False, test_set=True, project_root_path=project_root_path)
    else:
        raise ValueError("Unknown dataset")


def mnist(train_set: bool = True, test_set: bool = True, project_root_path='..') \
        -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the MNIST dataset.
    All pixels are normalized to 0, 1.

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if train_set:
        train_data = datasets.MNIST(
            root=os.path.join(project_root_path, 'datasets'),
            train=True,
            download=True, transform=ToTensor()
        )
    if test_set:
        test_data = datasets.MNIST(
            root=os.path.join(project_root_path, 'datasets'),
            train=False,
            download=True, transform=ToTensor()
        )

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def fashion_mnist(train_set: bool = True, test_set: bool = True, project_root_path='..') \
        -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the Fashion MNIST dataset.
    All pixels are normalized to 0, 1.

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if train_set:
        train_data = datasets.FashionMNIST(
            root=os.path.join(project_root_path, 'datasets'),
            train=True,
            download=True, transform=ToTensor()
        )
    if test_set:
        test_data = datasets.FashionMNIST(
            root=os.path.join(project_root_path, 'datasets'),
            train=False,
            download=True, transform=ToTensor()
        )

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def cifar10(train_set: bool = True, test_set: bool = True, project_root_path='..') \
        -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the CIFAR10 dataset.

    All pixels are normalized to 0, 1.
    Image shape is channel, height, weight.

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if train_set:
        train_data = datasets.CIFAR10(
            root=os.path.join(project_root_path, 'datasets'),
            train=True,
            download=True, transform=ToTensor()
        )
    if test_set:
        test_data = datasets.CIFAR10(
            root=os.path.join(project_root_path, 'datasets'),
            train=False,
            download=True, transform=ToTensor()
        )

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def collision_detection(train_set: bool = True, test_set: bool = True, project_root_path='..') \
        -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads a CollisionDetection dataset (resources/collision_detection/CollisionDetection_test_data and _train_data).

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    def pandas_data_to_dataset(pandas_data):
        inputs = torch.tensor(pandas_data.drop('class', axis=1).values, dtype=torch.float)
        targets = torch.tensor(pandas_data['class'].values, dtype=torch.long)
        return torch_data.TensorDataset(inputs, targets)

    train_data = None
    test_data = None
    if test_set:
        test_data = pandas.read_csv(os.path.join(
            project_root_path, "resources", "collision_detection", "CollisionDetection_test_data.csv"))
        test_data = pandas_data_to_dataset(test_data)
    if train_set:
        train_data = pandas.read_csv(os.path.join(
            project_root_path, "resources", "collision_detection", "CollisionDetection_train_data.csv"))
        train_data = pandas_data_to_dataset(train_data)

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def hcas_pra1_tau00(project_root_path='..') -> torch_data.Dataset:
    """
    Loads the HCAS (pra=1 tau=00) dataset (resources/hcas/unnormalized_HCAS_polar_TrainingData_v6_pra1_tau00)

    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    """
    dataset = pandas.read_csv(os.path.join(
        project_root_path, "resources", "hcas", "unnormalized_HCAS_polar_TrainingData_v6_pra1_tau00.csv")).to_numpy()
    data = torch.as_tensor(dataset[:, :5], dtype=torch.float)
    targets = torch.as_tensor(dataset[:, 5:], dtype=torch.float)

    return torch_data.TensorDataset(data, targets)


def pimple_bowl(train_set: bool = True, test_set: bool = True, project_root_path='..') \
        -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the pimple_bowl dataset (resources/pimple_bowl/pimple_bowl_test_data.pyt and _train_data.pyt).

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if test_set:
        test_data = torch.load(os.path.join(project_root_path, 'resources', 'pimple_bowl', 'pimple_bowl_test_data.pyt'))
    if train_set:
        train_data = torch.load(os.path.join(
            project_root_path, 'resources', 'pimple_bowl', 'pimple_bowl_train_data.pyt'))

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def drebin(train_set: bool = True, test_set: bool = True, project_root_path='..') -> Union[DREBIN, Tuple[DREBIN, DREBIN]]:
    """
    Loads the DREBIN dataset.
    If the data has not yet been extracted, it is extracted, which can take a long time.

    Have a look at the DREBIN class or the Datasets section in the README file for more information
    on obtaining and using the dataset.

    :param train_set: Whether to load the training set.
    :param test_set: Whether to load the test set.
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if test_set:
        test_data = DREBIN(root_dir=os.path.join(project_root_path, 'datasets'), train=False, extract=True)
    if train_set:
        train_data = DREBIN(root_dir=os.path.join(project_root_path, 'datasets'), train=True, extract=True)

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def adult(train_set: bool = True, test_set: bool = True, project_root_path='..') -> Union[Adult, Tuple[Adult, Adult]]:
    """
    Loads the Adult dataset.

    :param train_set: Whether to load the training set.
    :param test_set: Whether to load the test set.
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if test_set:
        test_data = Adult(root=os.path.join(project_root_path, 'datasets'), train=False, download=True)
    if train_set:
        train_data = Adult(root=os.path.join(project_root_path, 'datasets'), train=True, download=True)

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def cmapss(train_set: bool = True, test_set: bool = True, project_root_path='..', window_size: int = 40) -> Union[CMAPSS, Tuple[CMAPSS, CMAPSS]]:
    """
    Loads the CMAPSS dataset.

    :param train_set: Whether to load the training set.
    :param test_set: Whether to load the test set.
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If both train_set and test_set are true (default) returns
     two datasets, first the training set, then the test set.
     If only one is true, only the respective dataset is returned.
    """
    train_data = None
    test_data = None
    if test_set:
        test_data = CMAPSS(root=os.path.join(project_root_path, 'datasets'), train=False, download=True, window_size=window_size)
    if train_set:
        train_data = CMAPSS(root=os.path.join(project_root_path, 'datasets'), train=True, download=True, window_size=window_size)

    if train_data and test_data:
        return train_data, test_data
    else:
        return train_data if train_data else test_data


def optidose_easy(train_set: bool = True, validation_set: bool = True, test_set: bool = True, project_root_path='..') \
    -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset], Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the Optidose easy dataset (datasets/OptiDose/data_2020_ß6_03_n4984.csv).
    This dataset is not included in the git repository.
    It needs to be added manually.

    The dataset has three inputs and five outputs.

    :param train_set: Whether to load the training set
    :param validation_set: Whether to load the validation set.
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If train_set, validation_set, and test_set are true (default) returns
     three datasets, first the training set, then the validation set, and then the test set.
     If only one is true, only the respective dataset is returned.
     When two are true, the respective datasets are returned in the same order as for
     all three datasets.
    """
    # we don't need the full dataset
    return optidose_dataset(
        "data_2020_06_03_n4984.csv",
        num_inputs=3,
        num_rows_in_csv=1000,
        split=(0.6, 0.2, 0.2),
        train_set=train_set, validation_set=validation_set, test_set=test_set,
        project_root_path=project_root_path
    )


def optidose_onco(train_set: bool = True, validation_set: bool = True, test_set: bool = True, project_root_path='..',
                  random_noise: float = 0.0) \
    -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset], Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the Optidose onco dataset (datasets/OptiDose/results_onco_2021_03_11.csv).
    This dataset is not included in the git repository.
    It needs to be added manually.

    The dataset has three inputs and five outputs.

    :param train_set: Whether to load the training set
    :param validation_set: Whether to load the validation set.
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory.
    :param random_noise: The absolute magnitude of the random noise to add to the
     dataset outputs.
     Adding random noise is turned off by default with a magnitude of `0.0`.
     Setting random noise to a non-zero value :math:`\\epsilon` leads to adding uniform
     noise in the range :math:`[-\\epsilon, \\epsilon]` to the outputs.
     Make sure to initialize the torch random seed.
    :return: If train_set, validation_set, and test_set are true (default) returns
     three datasets, first the training set, then the validation set, and then the test set.
     If only one is true, only the respective dataset is returned.
     When two are true, the respective datasets are returned in the same order as for
     all three datasets.
    """
    return optidose_dataset(
        "results_onco_2021_03_11.csv",
        num_inputs=5,
        split=(0.6, 0.2, 0.2),
        random_noise=random_noise,
        train_set=train_set, validation_set=validation_set, test_set=test_set,
        project_root_path=project_root_path
    )


def optidose_dataset(
    dataset_filename: str, num_inputs: int, num_rows_in_csv: Optional[int] = None,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    random_noise: float = 0.0,
    train_set: bool = True, validation_set: bool = True, test_set: bool = True,
    project_root_path='..') \
    -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset], Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads a Optidose dataset from `datasets/OptiDose`.
    OptiDose datasets are not included in the git repository.
    They need to be added manually to the `datasets` directory.

    :param dataset_filename: The csv file containing the OptiDose data to read
    :param num_inputs: The number of input columns of the dataset.
    :param num_rows_in_csv: The number of rows from the csv file to read.
     By default (`None`), read all rows.
    :param split: How to split the dataset into training set, validation set
     and test set.
     The first entry of the split tuple of the fraction of data to use for the
     test set, the second entry is the faction for the validation set, and the
     last entry is the fraction for the test set.
    :param random_noise: The absolute magnitude of the random noise to add to the
     dataset outputs.
     Adding random noise is turned off by default with a magnitude of `0.0`.
     Setting random noise to a non-zero value :math:`\\epsilon` leads to adding uniform
     noise in the range :math:`[-\\epsilon, \\epsilon]` to the outputs.
     Make sure to initialize the torch random seed.
    :param train_set: Whether to load the training set
    :param validation_set: Whether to load the validation set.
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory
    :return: If train_set, validation_set, and test_set are true (default) returns
     three datasets, first the training set, then the validation set, and then the test set.
     If only one is true, only the respective dataset is returned.
     When two are true, the respective datasets are returned in the same order as for
     all three datasets.
    """
    raw_data = pandas.read_csv(
        os.path.join(project_root_path, "datasets", "OptiDose", dataset_filename),
        header=0,
        nrows=num_rows_in_csv
    )
    data_n = len(raw_data)
    raw_data = torch.from_numpy(raw_data.to_numpy(dtype=np.float32))
    inputs_full = raw_data[:, :num_inputs]
    outputs_full = raw_data[:, num_inputs:]

    if random_noise != 0.0:
        outputs_full += torch.rand(*outputs_full.shape) * 2 * random_noise - random_noise

    def get_subset(slice_):
        inputs = inputs_full[slice_, :]
        outputs = outputs_full[slice_, :]
        return torch_data.TensorDataset(inputs, outputs)

    train_frac, val_frac, test_frac = split
    assert np.isclose(train_frac + val_frac + test_frac, 1.0)
    train_end = int(train_frac * data_n)
    val_end = train_end + int(val_frac * data_n)

    datasets = []
    if train_set:
        train_data = get_subset(slice(train_end))
        datasets.append(train_data)
    if validation_set:
        val_data = get_subset(slice(train_end, val_end))
        datasets.append(val_data)
    if test_set:
        test_data = get_subset(slice(val_end, None))
        datasets.append(test_data)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return tuple(datasets)


def pk_1_cmp_po(train_set: bool = True, test_set: bool = True, project_root_path='..') \
    -> Union[torch_data.Dataset, Tuple[torch_data.Dataset, torch_data.Dataset]]:
    """
    Loads the PK 1-cmp PO Model dataset (datasets/OptiDose/PK_1-cmp_PO.csv).
    This dataset is not included in the git repository.
    It needs to be added manually.

    The dataset has five inputs and 24 outputs.

    :param train_set: Whether to load the training set
    :param test_set: Whether to load the test set
    :param project_root_path: Path to the project root directory which contains the datasets
     directory and the resources directory.
    :return: If train_set and test_set are true (default) returns
     two datasets, the training set and then the test set.
     If only one is true, only the respective dataset is returned.
    """
    raw_data = pandas.read_csv(
        os.path.join(project_root_path, "datasets", "OptiDose", "PK_1-cmp_PO.csv"),
        header=0,
        nrows=1000
    )
    data_n = len(raw_data)
    raw_data = torch.from_numpy(raw_data.to_numpy(dtype=np.float32))
    inputs_full = raw_data[:, :5]
    outputs_full = raw_data[:, 5:]

    def get_subset(slice_):
        inputs = inputs_full[slice_, :]
        outputs = outputs_full[slice_, :]
        return torch_data.TensorDataset(inputs, outputs)

    train_end = int(0.7 * data_n)

    datasets = []
    if train_set:
        train_data = get_subset(slice(train_end))
        datasets.append(train_data)
    if test_set:
        test_data = get_subset(slice(train_end, None))
        datasets.append(test_data)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return tuple(datasets)


def integer_dataset(
    size: Optional[int] = None,
    distribution: Optional[str] = None,
    maximum: Optional[int] = None,
    seed: Optional[int] = None,
    project_root_path='..'
) -> IntegerDataset:
    """
    Create or load an IntegerDataset.
    The dataset is stored or loaded from datasets/IntegerDatasets.

    An IntegerDataset has one input and one output.

    :param size: The size of the IntegerDataset.
     See IntegerDataset.__init__ for more details.
     When size is None, the default value from IntegerDataset.__init__ is used.
    :param distribution: The random distribution the IntegerDataset is drawn from.
     See IntegerDataset.__init__ for more details.
     When distribution is None, the default value from IntegerDataset.__init__ is used.
    :param maximum: The maximal value of an integer in the dataset.
     See IntegerDataset.__init__ for more details.
     When maximum is None, the default value from IntegerDataset.__init__ is used.
    :param seed: The random seed used for generating the dataset.
     See IntegerDataset.__init__ for more details.
     When seed is None, the default value from IntegerDataset.__init__ is used.
    :param project_root_path: Path to the project root directory which
     contains the datasets directory and the resources directory.
    :return: An IntegerDataset.
    """
    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if distribution is not None:
        kwargs["distribution"] = distribution
    if maximum is not None:
        kwargs["maximum"] = maximum
    if seed is not None:
        kwargs["seed"] = seed
    return IntegerDataset(root=Path(project_root_path, "datasets"), **kwargs)
