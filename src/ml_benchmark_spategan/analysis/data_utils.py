"""Utilities for data preparation specific to analysis tasks."""

from typing import Tuple

import torch
import xarray as xr


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
