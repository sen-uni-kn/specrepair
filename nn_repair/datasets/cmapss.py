import os
from logging import info
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from nn_repair.utils.checksums import sha256sum


class CMAPSS(Dataset):
    """
    The `Turbofan Engine Degradation Dataset (C-MAPSS)
    <https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository>`_
    [SaxenaGoebel2008]_.

    Specifically, the FD001 dataset from the C-MAPSS dataset.
    The time series data is sliced into windows of a certain length to allow
    processing with a CNN.
    Each window corresponds to one data point in the resulting training or test set.
    The windows are overlapping with a stride of 1.

    The features of this data set are 2D tensors with shape `20, window_size`.
    The 20 channels are the setting and sensor features of the dataset with
    setting_3, sensor_1, sensor_18, and sensor_19 removed as they're constant
    in the dataset.
    The window size is set in the initializer.

    The targets are the remaining useful file (RUL) clipped to a value of 150, following
    [MathworksRULPred]_.

    .. [SaxenaGoebel2008] A. Saxena and K. Goebel (2008).
       "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics
       Data Repository, NASA Ames Research Center, Moffett Field, CA

    .. [MathworksRULPred] https://de.mathworks.com/help/releases/R2021a/predmaint/ug/remaining-useful-life-estimation-using-convolutional-neural-network.html#mw_rtc_RULEstimationUsingCNNExample_98C35430

    """
    zip_url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    zip_checksum = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
    subset = "FD001"
    train_inputs_file = "train_inputs.pyt"
    train_targets_file = "train_targets.pyt"
    test_inputs_file = "test_inputs.pyt"
    test_targets_file = "test_targets.pyt"

    def __init__(self, root: str, train: bool = True, window_size: int = 40, download: bool = False):
        """
        Loads the `Turbofan Engine Degradation Dataset (C-MAPSS)
        <https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository>`_
        [SaxenaGoebel2008]_.

        :param root: The root directory where the C-MAPSS folder is placed or
          is to be downloaded to if download is set to True.
        :param train: Whether to get the training set or test set of
          the dataset.
        :param window_size: The size of the window into which the time series data
         is split.
         This is the last dimension
        :param download: Whether to download the C-MAPSS dataset from
          https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
          if it isn't present in the root directory.
        """
        self.files_dir = Path(root, f"C-MAPSS_window_{window_size}")
        self.window_size = window_size
        if not self.files_dir.exists():
            if not download:
                raise RuntimeError("Dataset not found. "
                                   "Download it by passing download=True.")
            os.mkdir(self.files_dir)
            self._download()

        if train:
            inputs_file = self.train_inputs_file
            targets_file = self.train_targets_file
        else:
            inputs_file = self.test_inputs_file
            targets_file = self.test_targets_file
        self.data = torch.load(Path(self.files_dir, inputs_file))
        self.targets = torch.load(Path(self.files_dir, targets_file))

    def _download(self):
        info("Downloading C-MAPSS dataset file...")
        # the dataset is in a zip file in a zip file
        zip1_path = Path(self.files_dir, "Turbofan.zip")
        temp_extracted_path = Path(
            self.files_dir, "6. Turbofan Engine Degradation Simulation Data Set"
        )
        zip2_path = Path(temp_extracted_path, "CMAPSSData.zip")

        result = requests.get(self.zip_url)
        with open(zip1_path, "xb") as file:
            file.write(result.content)

        zip1_checksum = sha256sum(zip1_path)
        if zip1_checksum != zip1_checksum:
            raise RuntimeError(
                f"Downloaded dataset file has different checksum than "
                f"expected: {zip1_path}. "
                f"Expected sha256 checksum: {zip1_checksum}"
            )

        info("Extracting data...")
        with ZipFile(zip1_path) as zip_obj:
            zip_obj.extractall(self.files_dir)

        with ZipFile(zip2_path) as zip_obj:
            with zip_obj.open(f"train_{self.subset}.txt") as file:
                train_raw = pd.read_csv(file, sep=" ", header=None)
            with zip_obj.open(f"test_{self.subset}.txt") as file:
                test_raw = pd.read_csv(file, sep=" ", header=None)
            with zip_obj.open(f"RUL_{self.subset}.txt") as file:
                test_targets_raw = pd.read_csv(file, sep=" ", header=None)

        zip1_path.unlink()
        zip2_path.unlink()
        temp_extracted_path.rmdir()

        info("Preprocessing data...")
        # the train and test files contain two training spaces that lead
        # to two extra columns
        train_raw.drop([26, 27], axis=1, inplace=True)
        test_raw.drop([26, 27], axis=1, inplace=True)
        # the RUL file has only one training space
        test_targets_raw.drop([1], axis=1, inplace=True)

        # split the data by ids => each result is a time series
        train_time_series = train_raw.groupby(0)  # column 0 is the ID
        test_time_series = test_raw.groupby(0)

        # compute the number of cycles until failure for the training set
        # column 1 is the cycle number
        train_failure_cycle = {id_: df[1].max() for id_, df in train_time_series}

        # 0, 1 are id and cycle number
        # the remaining ones are constant in the dataset (zero variance)
        cols_to_drop = [0, 1, 4, 5, 9, 14, 20, 22, 23]

        # make windows of the training data
        train_inputs = []
        train_targets = []
        for id_, df in train_time_series:
            for i in range(len(df) - self.window_size + 1):
                window = df.iloc[i:i+self.window_size]
                # predict the remaining useful life at the end of the
                # time series
                target = train_failure_cycle[id_] - window[1].max()
                # drop id, cycle number and constant features
                window = window.drop(cols_to_drop, axis=1)
                train_inputs.append(torch.as_tensor(window.to_numpy(), dtype=torch.float))
                train_targets.append(target)
        train_inputs = torch.stack(train_inputs)
        train_targets = torch.tensor(train_targets, dtype=torch.int)

        test_inputs = []
        test_targets = []
        for id_, df in test_time_series:
            final_target = test_targets_raw.iloc[id_ - 1]  # ids start from 1
            if len(df) < self.window_size:
                data = df.drop(cols_to_drop, axis=1).to_numpy()
                window = torch.full((self.window_size, data.shape[1]), fill_value=0.0)
                window[:len(df), :] = torch.as_tensor(data)
                test_inputs.append(window)
                test_targets.append(final_target)
            else:
                final_target += df[1].max()  # RUL at the start of the time series
                for i in range(len(df) - self.window_size + 1):
                    window = df.iloc[i:i+self.window_size]
                    # predict the remaining useful life at the end of the
                    # time series
                    target = final_target - window[1].max()
                    window = window.drop(cols_to_drop, axis=1)
                    test_inputs.append(torch.as_tensor(window.to_numpy(), dtype=torch.float))
                    test_targets.append(target)
        test_inputs = torch.stack(test_inputs)
        test_targets = torch.tensor(test_targets, dtype=torch.int)

        # make channel x window_size tensors
        train_inputs.transpose_(-2, -1)
        test_inputs.transpose_(-2, -1)

        # clip RULs to 150
        train_targets.clip_(max=150)
        test_targets.clip_(max=150)

        torch.save(train_inputs, Path(self.files_dir, self.train_inputs_file))
        torch.save(train_targets, Path(self.files_dir, self.train_targets_file))
        torch.save(test_inputs, Path(self.files_dir, self.test_inputs_file))
        torch.save(test_targets, Path(self.files_dir, self.test_targets_file))
        info("Download finished.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)
