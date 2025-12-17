"""
Utility functions for denormalizing predictions and converting them to xarray format.
"""

import numpy as np
import xarray as xr

from .normalize import denormalize_predictions


def predictions_to_xarray(y_pred, y_true, norm_params, var_name="tasmax"):
    """
    Convert predictions and ground truth to xarray Datasets for diagnostics.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions, shape (batch, H*W) or (batch, 1, H, W)
    y_true : torch.Tensor
        Ground truth, shape (batch, H*W)
    norm_params : dict
        Dictionary containing normalization parameters and coordinates
    var_name : str
        Name of the target variable

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Denormalized predictions and ground truth as xarray Datasets
    """
    # # Denormalize
    y_pred_denorm = denormalize_predictions(y_pred, norm_params)
    y_true_denorm = denormalize_predictions(y_true, norm_params)
    # y_pred_denorm = y_pred
    # y_true_denorm = y_true

    # Get spatial dimensions info
    spatial_dims = norm_params["spatial_dims"]
    coords = norm_params["y_test_coords"]
    H, W = norm_params["spatial_shape"]  # Get from stored shape (H, W)

    # Convert to numpy and reshape to 3D (batch, H, W)
    if y_pred_denorm.dim() == 4:  # (batch, 1, H, W)
        y_pred_np = y_pred_denorm.squeeze(1).cpu().numpy()
    elif y_pred_denorm.dim() == 2:  # (batch, H*W)
        batch_size = y_pred_denorm.shape[0]
        y_pred_np = y_pred_denorm.cpu().numpy().reshape(batch_size, H, W)
    else:
        y_pred_np = y_pred_denorm.cpu().numpy()

    if y_true_denorm.dim() == 2:  # (batch, H*W)
        batch_size = y_true_denorm.shape[0]
        y_true_np = y_true_denorm.cpu().numpy().reshape(batch_size, H, W)
    elif y_true_denorm.dim() == 4:  # (batch, 1, H, W)
        y_true_np = y_true_denorm.squeeze(1).cpu().numpy()
    else:
        y_true_np = y_true_denorm.cpu().numpy()

    # Get coordinates
    spatial_dims = norm_params["spatial_dims"]
    coords = norm_params["y_test_coords"]

    # Create xarray DataArrays
    time_coords = np.arange(y_pred_np.shape[0])

    pred_da = xr.DataArray(
        y_pred_np,
        dims=["time", spatial_dims[0], spatial_dims[1]],
        coords={
            "time": time_coords,
            spatial_dims[0]: coords[spatial_dims[0]],
            spatial_dims[1]: coords[spatial_dims[1]],
        },
        name=var_name,
    )

    true_da = xr.DataArray(
        y_true_np,
        dims=["time", spatial_dims[0], spatial_dims[1]],
        coords={
            "time": time_coords,
            spatial_dims[0]: coords[spatial_dims[0]],
            spatial_dims[1]: coords[spatial_dims[1]],
        },
        name=var_name,
    )

    # Convert to Datasets
    pred_ds = pred_da.to_dataset()
    true_ds = true_da.to_dataset()

    return pred_ds, true_ds
