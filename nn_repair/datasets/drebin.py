import math
from typing import Optional, Callable, Tuple, List

import torch
import numpy as np
import pandas
from torch.utils.data import Dataset, Sampler
import os
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm


class DREBIN(Dataset):
    """
    The DREBIN Malware detection dataset.

    In difference to the original DREBIN dataset this dataset only contains information
    whether an app is malware or not, not to which class of malware it belongs.

    Therefore, this dataset is a binary classification dataset. The features contained in the dataset are binary.

    The training/test split provided by this class is split_1 from the DREBIN dataset.
    The validation and training set are merged to form the training set provided by this class.
    The test set is not modified.

    Uses the contents of the family labels, features and dataset splits from
    https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html.
    These files have to be downloaded separately. You have to request permission from the DREBIN
    owners for accessing the dataset. Follow the instructions on
    https://www.sec.cs.tu-bs.de/~danarp/drebin/index.html to gain access.

    The DREBIN dataset was introduced in [Arp2014]_ and [Spreitzenbarth2013]_:

    .. [Arp2014] Daniel Arp, Michael Spreitzenbarth, Malte Huebner, Hugo Gascon, and Konrad Rieck
        "Drebin: Efficient and Explainable Detection of Android Malware in Your Pocket",
        21th Annual Network and Distributed System Security Symposium (NDSS), February 2014

    .. [Spreitzenbarth2013] Michael Spreitzenbarth, Florian Echtler, Thomas Schreck, Felix C. Freling,
        Johannes Hoffmann, "MobileSandbox: Looking Deeper into Android Applications",
        28th International ACM Symposium on Applied Computing (SAC), March 2013
    """

    feature_vectors_dir_name = 'feature_vectors'
    app_info_file_name = 'app_information.csv'
    feature_names_file_name = 'feature_names'

    def __init__(
            self,
            root_dir: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            extract: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'DREBIN')
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        if extract:
            self.extract()

        if not self._extracted_dataset_exists():
            raise RuntimeError(f'Extracted DREBIN dataset could not be found at {self.root_dir}. '
                               f'Use extract=True to extract the dataset from a DREBIN feature_vectors.zip'
                               f'archive, a sha256_family.csv file and a dataset_splits.zip archive.'
                               f'These files can be obtained after requesting permission at '
                               f'https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html.')

        with open(Path(self.data_dir, self.feature_names_file_name)) as feature_names_file:
            self.feature_names = [line.strip() for line in feature_names_file.readlines()]

        app_information = pandas.read_csv(Path(self.data_dir, self.app_info_file_name))
        subset = app_information[app_information['training_set'] == train]
        subset.reset_index(drop=True, inplace=True)
        self.apps = subset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the feature vector of an app and whether this app is malware or not.
        The data point is returned as a pair of an boolean tensor (the features)
        and a boolean scalar tensor (the label).

        The returned label (malware or not) is True if the app is malware and false otherwise.
        """
        app = self.apps.iloc[index]
        checksum = app['sha256']
        label = app['malware']

        features_file = Path(self.data_dir, self.feature_vectors_dir_name, checksum)
        with open(features_file, 'rt') as file:
            features = [line.strip() for line in file.readlines()]
        feature_tensor = torch.tensor(
            [feat_name in features for feat_name in self.feature_names],
            dtype=torch.bool
        )

        return feature_tensor, torch.tensor(label, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.apps)

    def malware_indices(self) -> List[int]:
        return list(self.apps.index[self.apps['malware']])

    def benign_apps_indices(self) -> List[int]:
        return list(self.apps.index[~self.apps['malware']])

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def _extracted_dataset_exists(self):
        feature_vectors_dir = Path(self.data_dir, self.feature_vectors_dir_name)
        info_file = Path(self.data_dir, self.app_info_file_name)
        feature_names_file = Path(self.data_dir, self.feature_names_file_name)
        return feature_vectors_dir.exists() and feature_vectors_dir.is_dir() \
            and info_file.exists() and feature_names_file.exists()

    def extract(self):
        """
        Extracts the DREBIN dataset from a manually downloaded zip file and a csv file.
        """
        if self._extracted_dataset_exists():
            return

        os.makedirs(self.data_dir, exist_ok=True)

        # A zip file which contains the feature vectors.
        # The archive contains one test file for each app.
        # The name of the file is the sha256 checksum of the app.
        # The file contains the features that were extracted for this app as string identifiers.
        # (one identifier per line)
        data_file = Path(self.root_dir, self.feature_vectors_dir_name + '.zip')
        # A csv file which lists which apps belong to which category of malware.
        # Hence sha256 checksums contained in this file are malware apps, checksums that are not contained
        # are legitimate software.
        classes_file = Path(self.root_dir, 'sha256_family.csv')
        # A zip file containing some spits of the dataset.
        # The archive contains a directory named 'all' which contains a subdirectory 'split_1'
        # The split_1 directory contains train_cs, test_cs and validate_cs files, which
        # contain sha256 checksums of the apps contained in the corresponding sets of this split.
        # We are using split_1, but the other splits could be used as well.
        # For a proper use case all ten splits should be used to perform 10-fold cross validation.
        splits_file = Path(self.root_dir, 'dataset_splits.zip')

        if not data_file.exists() or not classes_file.exists() or not splits_file.exists():
            raise RuntimeError(f"Could not find dataset files. Please place the following files"
                               f"in provided root directory ({self.root_dir}): {self.feature_vectors_dir_name}.zip, "
                               f"sha256_family.csv and dataset_splits.zip.\n"
                               f"These files can be downloaded from "
                               f"https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html after permission "
                               f"has been requested.")

        print(f"Extracting archives to {self.data_dir}. This may take a minute.")
        with ZipFile(data_file, 'r') as zip_file:
            print(f"Extracting {data_file}...")
            zip_file.extractall(self.data_dir)
        with ZipFile(splits_file, 'r') as zip_file:
            print(f"Extracting {splits_file}...")
            zip_file.extractall(self.data_dir)

        print("Reading app checksums...")
        # list all the file names contain in data_dir/feature_vectors
        # this directory contains the features of all apps in the dataset
        features_dir = Path(self.data_dir, self.feature_vectors_dir_name)
        if not features_dir.exists() or not features_dir.is_dir():
            raise RuntimeError(f"Could not find extracted feature vectors directory: {features_dir}.")
        app_checksums = [path.name for path in features_dir.iterdir()]

        feature_names_file_path = Path(self.data_dir, self.feature_names_file_name)
        if not feature_names_file_path.exists():
            print("Extracting feature names... (Involves reading all feature vector files once)")
            feature_names = set()
            # TODO: extremely inefficient
            for features_file_path in tqdm(features_dir.iterdir(), total=len(app_checksums)):
                with open(features_file_path, 'rt') as features_file:
                    feature_names = feature_names.union(features_file.readlines())

            feature_names = sorted(feature_names)
            with open(feature_names_file_path, 'wt') as feature_names_file:
                feature_names_file.writelines(feature_names)

        print("Reading malware app list...")
        # the data_dir now contains a subdirectory containing the feature vectors and one containing the splits
        with open(classes_file, 'rt') as csv_file:
            malware_apps = pandas.read_csv(csv_file)
        # we don't care about the concrete class here, only whether an app is malware or not
        malware_apps = set(malware_apps['sha256'])

        print("Reading train/test split...")
        split_dir = Path(self.data_dir, 'dataset_splits', 'all', 'split_1')
        test_set_file = Path(split_dir, 'test_cs')
        # We actually don't need to load those files.
        # Everything that is not in the test set belongs to the training set.
        # train_set_file = Path(split_dir, 'train_cs')
        # validation_set_file = Path(split_dir, 'validate_cs')

        if not test_set_file.exists():  # or not validation_set_file.exists() or not train_set_file.exists():
            raise RuntimeError(f"Could not find split files in extracted dataset_splits directory:"
                               # f"{train_set_file} (exists: {train_set_file.exists()}), "
                               # f"{validation_set_file} (exists: {validation_set_file.exists()}), "
                               f"{test_set_file} (exists: {test_set_file.exists()}). "
                               f"The provided DREBIN dataset_splits archive appears to be incompatible.")
        # with open(train_set_file, 'rt') as train_file:
        #     with open(validation_set_file, 'rt') as validation_file:
        #         train_set_checksums = set(itertools.chain(train_file.readlines(), validation_file.readlines()))
        with open(test_set_file, 'rt') as test_file:
            test_set_checksums = set(line.strip() for line in test_file.readlines())

        # create an overview file which stores whether an app (identified by sha256 checksum)
        # is malware and if it is contained in the training or test set.
        app_information = {
            'sha256': app_checksums,
            'malware': [checksum in malware_apps for checksum in app_checksums],
            'training_set': [checksum not in test_set_checksums for checksum in app_checksums]
        }
        app_info_file = Path(self.data_dir, self.app_info_file_name)
        pandas.DataFrame(app_information).to_csv(app_info_file)


class DREBINBalancedSampler(Sampler):
    """
    Samples batches of indices from a DREBIN dataset instance, such that
    a certain fraction of samples in the produced batches are malware samples.

    This sampler uses each malware sample once before shuffling batches anew.
    """
    def __init__(self, dataset: DREBIN, batch_size, malware_ratio, rng: np.random.Generator = np.random.default_rng()):
        super().__init__(dataset)
        self.dataset = dataset
        self.rng = rng
        self.malware_indices = dataset.malware_indices()
        self.benign_indices = dataset.benign_apps_indices()

        self.batch_size = batch_size
        self.num_malware_samples_per_batch = math.ceil(batch_size * malware_ratio)

    def __iter__(self):
        def iterator():
            malware_indices = list(self.malware_indices)
            benign_indices = list(self.benign_indices)
            self.rng.shuffle(malware_indices)
            self.rng.shuffle(benign_indices)

            next_malware_id = 0
            next_benign_id = 0
            num_benign_samples = self.batch_size - self.num_malware_samples_per_batch
            while next_malware_id < len(malware_indices) and next_benign_id < len(benign_indices):
                end_malware_ids = min(len(malware_indices), next_malware_id + self.num_malware_samples_per_batch)
                end_benign_ids = min(len(benign_indices), next_benign_id + num_benign_samples)
                malware_ids = malware_indices[next_malware_id:end_malware_ids]
                benign_ids = benign_indices[next_benign_id:end_benign_ids]
                next_malware_id = end_malware_ids
                next_benign_id = end_benign_ids

                malware_ids.extend(benign_ids)
                yield malware_ids
        return iterator()

    def __len__(self):
        return math.ceil(len(self.malware_indices) / self.num_malware_samples_per_batch)
