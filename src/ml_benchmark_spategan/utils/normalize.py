"""Utilities for data normalization and denormalization."""

from typing import Tuple

import numpy as np
import torch
import xarray as xr


def normalize_predictors(
    x_train: xr.Dataset,
    x_test: xr.Dataset,
    y_train: xr.Dataset,
    y_test: xr.Dataset,
    normalization: str,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, dict]:
    """
    Normalize predictors and predictands according to specified method.

    Args:
        x_train: Training predictors
        x_test: Test predictors
        y_train: Training predictands
        y_test: Test predictands
        normalization: Normalization method ('standardization', 'minmax', 'minus1_to_plus1',
                      'mp1p1_input_m1p1log_target', 'm1p1_log_target', 'std_log_target', etc.)

    Returns:
        Tuple of (x_train_norm, x_test_norm, y_train_norm, y_test_norm, norm_params)
        where norm_params is a dict containing normalization statistics for denormalization
        and climatology statistics (min, max, mean, std) for score computation
    """
    norm_params = {"normalization": normalization}

    # Always compute climatology statistics for score computation
    norm_params["x_min"] = x_train.min("time")
    norm_params["x_max"] = x_train.max("time")
    norm_params["x_mean"] = x_train.mean("time")
    norm_params["x_std"] = x_train.std("time")

    norm_params["y_min"] = y_train.min("time")
    norm_params["y_max"] = y_train.max("time")
    norm_params["y_mean"] = y_train.mean("time")
    norm_params["y_std"] = y_train.std("time")

    if normalization == "standardization":
        x_train_norm = (x_train - norm_params["x_mean"]) / norm_params["x_std"]
        x_test_norm = (x_test - norm_params["x_mean"]) / norm_params["x_std"]
        y_train_norm = y_train
        y_test_norm = y_test

    elif normalization == "minmax":
        x_train_norm = (x_train - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_test_norm = (x_test - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        y_train_norm = y_train
        y_test_norm = y_test

    elif normalization == "std_log_target":
        x_train_norm = (x_train - norm_params["x_mean"]) / norm_params["x_std"]
        x_test_norm = (x_test - norm_params["x_mean"]) / norm_params["x_std"]

        # log transform y
        y_train_norm = np.log1p(y_train + 1e-6)
        y_test_norm = np.log1p(y_test + 1e-6)

    elif normalization == "mp1p1_input_m1p1log_target":
        # x sample normalization to [-1, 1]
        x_train_norm = (x_train - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_train_norm = x_train_norm * 2 - 1
        x_test_norm = (x_test - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_test_norm = x_test_norm * 2 - 1

        # y sample normalization: log transform then scale to [-1, 1]
        y_train_norm = np.log1p(y_train + 1e-6) - np.log1p(1e-6)
        y_test_norm = np.log1p(y_test + 1e-6) - np.log1p(1e-6)

        y_min_train = y_train_norm.min("time")
        y_max_train = y_train_norm.max("time")
        norm_params["y_min_log"] = y_min_train
        norm_params["y_max_log"] = y_max_train

        y_train_norm = (y_train_norm - y_min_train) / (y_max_train - y_min_train)
        y_train_norm = y_train_norm * 2 - 1
        y_test_norm = (y_test_norm - y_min_train) / (y_max_train - y_min_train)
        y_test_norm = y_test_norm * 2 - 1

        # Convert to float32
        y_train_norm = y_train_norm.astype(np.float32)
        y_test_norm = y_test_norm.astype(np.float32)

    elif normalization == "m1p1_log_target":
        x_train_norm = (x_train - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_train_norm = x_train_norm * 2 - 1
        x_test_norm = (x_test - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_test_norm = x_test_norm * 2 - 1

        # log transform y
        y_train_norm = np.log1p(y_train + 1e-6)
        y_test_norm = np.log1p(y_test + 1e-6)

    elif normalization == "minus1_to_plus1":
        x_train_norm = (x_train - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_train_norm = x_train_norm * 2 - 1
        x_test_norm = (x_test - norm_params["x_min"]) / (
            norm_params["x_max"] - norm_params["x_min"]
        )
        x_test_norm = x_test_norm * 2 - 1

        y_train_norm = (y_train - norm_params["y_min"]) / (
            norm_params["y_max"] - norm_params["y_min"]
        )
        y_train_norm = y_train_norm * 2 - 1
        y_test_norm = (y_test - norm_params["y_min"]) / (
            norm_params["y_max"] - norm_params["y_min"]
        )
        y_test_norm = y_test_norm * 2 - 1

    else:
        # No normalization
        x_train_norm = x_train
        x_test_norm = x_test
        y_train_norm = y_train
        y_test_norm = y_test

    return x_train_norm, x_test_norm, y_train_norm, y_test_norm, norm_params


def denormalize_predictions(y_pred: torch.Tensor, norm_params: dict) -> torch.Tensor:
    """
    Denormalize predictions based on normalization method.

    Args:
        y_pred: Normalized predictions, shape (batch, H*W) or (batch, 1, H, W)
        norm_params: Dictionary containing normalization parameters

    Returns:
        Denormalized predictions
    """
    normalization = norm_params.get("normalization")

    if normalization is None or normalization == "none":
        return y_pred

    elif normalization == "standardization":
        # No y normalization in this method
        return y_pred

    elif normalization == "minmax":
        # No y normalization in this method
        return y_pred

    elif normalization == "std_log_target":
        # Reverse: y_norm = log1p(y + 1e-6)
        # So: y = expm1(y_norm) - 1e-6
        return torch.expm1(y_pred) - 1e-6

    elif normalization == "m1p1_log_target":
        # Reverse: y_norm = log1p(y + 1e-6)
        # So: y = expm1(y_norm) - 1e-6
        return torch.expm1(y_pred) - 1e-6

    elif normalization == "minus1_to_plus1":
        # Reverse: y_norm = (y - y_min) / (y_max - y_min) * 2 - 1
        # So: y = ((y_norm + 1) / 2) * (y_max - y_min) + y_min
        y_min = norm_params["y_min"]
        y_max = norm_params["y_max"]

        # Convert xarray to tensor if needed
        if isinstance(y_min, xr.Dataset):
            y_min = torch.from_numpy(y_min.to_array()[0].values).float()
            y_max = torch.from_numpy(y_max.to_array()[0].values).float()

        # Move to same device as predictions
        y_min = y_min.to(y_pred.device)
        y_max = y_max.to(y_pred.device)

        # Reshape based on prediction shape
        if y_pred.dim() == 2:  # (batch, H*W)
            y_min_flat = y_min.flatten()
            y_max_flat = y_max.flatten()
            return ((y_pred + 1) / 2) * (y_max_flat - y_min_flat) + y_min_flat
        elif y_pred.dim() == 4:  # (batch, 1, H, W)
            y_min_exp = y_min.unsqueeze(0).unsqueeze(0)
            y_max_exp = y_max.unsqueeze(0).unsqueeze(0)
            return ((y_pred + 1) / 2) * (y_max_exp - y_min_exp) + y_min_exp
        else:
            raise ValueError(f"Unexpected prediction shape: {y_pred.shape}")

    elif normalization == "mp1p1_input_m1p1log_target":
        # Step 1: Reverse [-1, 1] scaling to log space
        y_min_log = norm_params["y_min_log"]
        y_max_log = norm_params["y_max_log"]

        # Convert xarray to tensor if needed
        if isinstance(y_min_log, xr.Dataset):
            y_min_log = torch.from_numpy(y_min_log.to_array()[0].values).float()
            y_max_log = torch.from_numpy(y_max_log.to_array()[0].values).float()

        # Move to same device as predictions
        y_min_log = y_min_log.to(y_pred.device)
        y_max_log = y_max_log.to(y_pred.device)

        # Reshape and reverse [-1, 1] scaling
        if y_pred.dim() == 2:  # (batch, H*W)
            y_min_flat = y_min_log.flatten()
            y_max_flat = y_max_log.flatten()
            y_log = ((y_pred + 1) / 2) * (y_max_flat - y_min_flat) + y_min_flat
        elif y_pred.dim() == 4:  # (batch, 1, H, W)
            y_min_exp = y_min_log.unsqueeze(0).unsqueeze(0)
            y_max_exp = y_max_log.unsqueeze(0).unsqueeze(0)
            y_log = ((y_pred + 1) / 2) * (y_max_exp - y_min_exp) + y_min_exp
        else:
            raise ValueError(f"Unexpected prediction shape: {y_pred.shape}")

        # Step 2: Reverse log transform
        # Forward was: log1p(y + 1e-6) - log1p(1e-6)
        # So reverse is: expm1(y_log + log1p(1e-6)) - 1e-6
        return (
            torch.expm1(y_log + torch.log1p(torch.tensor(1e-6, device=y_pred.device)))
            - 1e-6
        )

    else:
        raise NotImplementedError(
            f"Denormalization for '{normalization}' not implemented"
        )
