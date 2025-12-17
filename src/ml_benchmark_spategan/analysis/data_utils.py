"""Utilities for data loading and preparation."""

from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def load_cordex_data(
    domain: str,
    training_experiment: str,
    var_target: str = "tasmax",
    data_path: str = "/bg/fast/aihydromet/cordexbench/",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load CORDEX benchmark data.

    Args:
        domain: Domain name ('SA', 'NZ', 'ALPS')
        training_experiment: Experiment name
        var_target: Target variable
        data_path: Path to data directory

    Returns:
        Tuple of (predictor, predictand) datasets
    """
    # Determine period and GCM
    if training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError(f"Invalid experiment: {training_experiment}")

    if domain == "ALPS":
        gcm_name = "CNRM-CM5"
    elif domain in ["NZ", "SA"]:
        gcm_name = "ACCESS-CM2"
    else:
        raise ValueError(f"Invalid domain: {domain}")

    # Load predictor
    predictor_filename = (
        f"{data_path}/{domain}/{domain}_domain/train/{training_experiment}/"
        f"predictors/{gcm_name}_{period_training}.nc"
    )
    predictor = xr.open_dataset(predictor_filename)

    if domain == "SA":
        predictor = predictor.drop_vars("time_bnds", errors="ignore")

    # Load predictand
    predictand_filename = (
        f"{data_path}/{domain}/{domain}_domain/train/{training_experiment}/"
        f"target/pr_tasmax_{gcm_name}_{period_training}.nc"
    )
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[var_target]]

    return predictor, predictand


def split_train_test(
    predictor: xr.Dataset, predictand: xr.Dataset, training_experiment: str
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Split data into train and test sets.

    Args:
        predictor: Predictor dataset
        predictand: Predictand dataset
        training_experiment: Experiment name

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    if training_experiment == "ESD_pseudo_reality":
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1981))
    elif training_experiment == "Emulator_hist_future":
        years_train = list(range(1961, 1981)) + list(range(2080, 2090))
        years_test = list(range(2090, 2100))
    else:
        raise ValueError(f"Invalid experiment: {training_experiment}")

    x_train = predictor.sel(time=np.isin(predictor["time"].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand["time"].dt.year, years_train))
    x_test = predictor.sel(time=np.isin(predictor["time"].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand["time"].dt.year, years_test))

    return x_train, y_train, x_test, y_test


def standardize_predictors(
    x_train: xr.Dataset, x_test: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Standardize predictors using training statistics.

    Args:
        x_train: Training predictors
        x_test: Test predictors

    Returns:
        Tuple of (x_train_stand, x_test_stand)
    """
    mean_train = x_train.mean("time")
    std_train = x_train.std("time")

    x_train_stand = (x_train - mean_train) / std_train
    x_test_stand = (x_test - mean_train) / std_train

    return x_train_stand, x_test_stand


def normalize_predictors(
    x_train: xr.Dataset, x_test: xr.Dataset, normalization: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Normalize predictors according to specified method.

    Args:
        x_train: Training predictors
        x_test: Test predictors
        normalization: Normalization method ('standardization', 'minmax', 'minus1_to_plus1',
                      'mp1p1_input_m1p1log_target', 'm1p1_log_target', etc.)

    Returns:
        Tuple of (x_train_normalized, x_test_normalized)
    """
    if normalization == "standardization":
        mean_train = x_train.mean("time")
        std_train = x_train.std("time")
        x_train_norm = (x_train - mean_train) / std_train
        x_test_norm = (x_test - mean_train) / std_train

    elif normalization in [
        "minmax",
        "minus1_to_plus1",
        "mp1p1_input_m1p1log_target",
        "m1p1_log_target",
    ]:
        # All these use min-max scaling for inputs
        min_train = x_train.min("time")
        max_train = x_train.max("time")

        if normalization == "minmax":
            # Scale to [0, 1]
            x_train_norm = (x_train - min_train) / (max_train - min_train)
            x_test_norm = (x_test - min_train) / (max_train - min_train)
        else:
            # Scale to [-1, 1]
            x_train_norm = (x_train - min_train) / (max_train - min_train)
            x_train_norm = x_train_norm * 2 - 1
            x_test_norm = (x_test - min_train) / (max_train - min_train)
            x_test_norm = x_test_norm * 2 - 1

    else:
        # No normalization
        x_train_norm = x_train
        x_test_norm = x_test

    return x_train_norm, x_test_norm


def prepare_torch_data(
    x_data: xr.Dataset, y_data: xr.Dataset, domain: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert xarray datasets to PyTorch tensors.

    Args:
        x_data: Predictor dataset
        y_data: Predictand dataset
        domain: Domain name for determining spatial dimensions

    Returns:
        Tuple of (x_tensor, y_tensor)
    """
    # Determine spatial dimensions
    if domain == "ALPS":
        spatial_dims = ("x", "y")
    elif domain in ["NZ", "SA"]:
        spatial_dims = ("lat", "lon")
    else:
        raise ValueError(f"Invalid domain: {domain}")

    # Convert to arrays
    x_array = (
        x_data.to_array()
        .transpose("time", "variable", spatial_dims[0], spatial_dims[1])
        .values
    )
    y_stack = y_data.stack(gridpoint=spatial_dims)
    y_array = y_stack.to_array()[0, :].values

    # Convert to tensors
    x_tensor = torch.from_numpy(x_array).float()
    y_tensor = torch.from_numpy(y_array).float()

    return x_tensor, y_tensor


class TestDataset(Dataset):
    """Dataset for test data (predictors only)."""

    def __init__(self, x_data: torch.Tensor):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx]
