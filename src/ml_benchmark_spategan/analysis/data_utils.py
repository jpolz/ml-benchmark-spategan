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
    # Determine spatial dimensions for x_data
    if "lat" in x_data.dims and "lon" in x_data.dims:
        x_spatial_dims = ("lat", "lon")
    elif "x" in x_data.dims and "y" in x_data.dims:
        x_spatial_dims = ("x", "y")
    else:
        raise ValueError(
            f"Could not determine spatial dimensions from x_data. Available dims: {x_data.dims}"
        )

    # Convert x_data to arrays
    x_array = (
        x_data.to_array()
        .transpose("time", "variable", x_spatial_dims[0], x_spatial_dims[1])
        .values
    )

    # For y_data, detect spatial dimensions separately (may differ from x_data!)
    y_dataarray = y_data.to_array()[0]  # Get the first variable
    if "lat" in y_dataarray.dims and "lon" in y_dataarray.dims:
        y_spatial_dims = ("lat", "lon")
    elif "x" in y_dataarray.dims and "y" in y_dataarray.dims:
        y_spatial_dims = ("x", "y")
    else:
        raise ValueError(
            f"Could not determine spatial dimensions from y_data. Available dims: {y_dataarray.dims}"
        )

    # Check if y_data has 2D coordinates (meshgrid) or 1D
    coord_is_2d = (
        y_dataarray[y_spatial_dims[0]].ndim > 1
        or y_dataarray[y_spatial_dims[1]].ndim > 1
    )

    if coord_is_2d:
        # For 2D coordinates (like ALPS rotated pole), flatten directly
        y_transposed = y_dataarray.transpose(
            "time", y_spatial_dims[0], y_spatial_dims[1]
        )
        y_array = y_transposed.values.reshape(len(y_transposed.time), -1)
    else:
        # For 1D coordinates, use the standard stack method
        y_stack = y_data.stack(gridpoint=y_spatial_dims)
        y_array = y_stack.to_array()[0, :].values

    # Convert to tensors
    x_tensor = torch.from_numpy(x_array).float()
    y_tensor = torch.from_numpy(y_array).float()

    return x_tensor, y_tensor
