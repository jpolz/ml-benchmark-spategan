from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr

from numpy import fft

def _filter_by_season(data: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season label.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to filter.
    season : str | None
        Season name (winter, summer, spring, autumn) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """
    if season is None:
        return data
    if season == "winter":
        return data.where(data["time.season"] == "DJF", drop=True)
    if season == "summer":
        return data.where(data["time.season"] == "JJA", drop=True)
    if season == "spring":
        return data.where(data["time.season"] == "MAM", drop=True)
    if season == "autumn":
        return data.where(data["time.season"] == "SON", drop=True)
    return data

def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.
    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

def bias_index( 
    target: xr.Dataset,
    pred: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    relative: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the bias between target and prediction for a supplied index function.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth dataset.
    pred : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    relative : bool, default False
        If True, return the relative bias.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Absolute or relative bias of the selected index.
    """
    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    target_index = index_fn(target, **call_kwargs)
    pred_index = index_fn(pred, **call_kwargs)
    bias = pred_index - target_index

    if relative:
        return bias / target_index
    else:
        return bias

def ratio_index(
    target: xr.Dataset,
    pred: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the ratio between prediction and target for a supplied index function.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth dataset.
    pred : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Ratio of the selected index (pred / target).
    """
    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    target_index = index_fn(target, **call_kwargs)
    pred_index = index_fn(pred, **call_kwargs)
    ratio = pred_index / target_index
    return ratio


def bias_multivariable_correlation(
    target: xr.Dataset,
    pred: xr.Dataset,
    var_x: str,
    var_y: str,
    season: str | None = None,
) -> xr.Dataset:
    """
    Compute correlations between two variables for target and prediction datasets
    and return the correlation bias.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth dataset containing ``var_x`` and ``var_y``.
    pred : xr.Dataset
        Predicted dataset containing ``var_x`` and ``var_y``.
    var_x : str
        Name of the first variable used in the correlation computation.
    var_y : str
        Name of the second variable used in the correlation computation.
    season : str | None, optional
        Season to filter before computing the correlations.

    Returns
    -------
    xr.Dataset
        Dataset containing the target correlation, prediction correlation, signed
        correlation difference, and the requested correlation bias.
    """

    target_filtered = _filter_by_season(target, season)
    pred_filtered = _filter_by_season(pred, season)

    target_corr = xr.corr(target_filtered[var_x], target_filtered[var_y], dim='time')
    pred_corr = xr.corr(pred_filtered[var_x], pred_filtered[var_y], dim='time')

    correlation_bias = pred_corr - target_corr

    return xr.Dataset(
        {
            "correlation_target": target_corr,
            "correlation_pred": pred_corr,
            "correlation_bias": correlation_bias,
        }
    )


def rmse( 
    target: xr.Dataset,
    pred: xr.Dataset,
    var: str,
    season: str | None = None,
    dim: str | None = None,
) -> xr.Dataset:
    """
    Compute the root mean square error between target and prediction datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth dataset.
    pred : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the RMSE.
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE. If None, computes over all dimensions.

    Returns
    -------
    xr.Dataset
        Root mean square error between target and prediction.
    """
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target_da = target[[var]]
    pred_da = pred[[var]]

    squared_diff = (pred_da - target_da) ** 2

    if dim is None:
        mse = squared_diff.mean()
    else:
        mse = squared_diff.mean(dim=dim)

    rmse_result = np.sqrt(mse)
    
    return rmse_result

def psd(
    target: xr.Dataset,
    pred: xr.Dataset,
    var: str,
    season: str | None = None,
):
    """
    Compute the power spectral density for target and prediction datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth dataset.
    pred : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the PSD.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for target and prediction.
    """
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target_da = target[var]
    pred_da = pred[var]

    target_np = np.nan_to_num(target_da.values, nan=0.0)
    pred_np = np.nan_to_num(pred_da.values, nan=0.0)

    fft_target = fft.fftshift(fft.fft2(target_np, axes=(-2, -1)), axes=(-2, -1))
    fft_pred = fft.fftshift(fft.fft2(pred_np, axes=(-2, -1)), axes=(-2, -1))

    power_target = np.abs(fft_target) ** 2
    power_pred = np.abs(fft_pred) ** 2

    psd_target_list = [_radial_average(p) for p in power_target]
    psd_pred_list = [_radial_average(p) for p in power_pred]

    avg_psd_target = np.mean(psd_target_list, axis=0)
    avg_psd_pred = np.mean(psd_pred_list, axis=0)

    psd_target_da = xr.DataArray(avg_psd_target, dims=["wavenumber"], name="PSD_target")
    psd_pred_da = xr.DataArray(avg_psd_pred, dims=["wavenumber"], name="PSD_pred")

    return psd_target_da, psd_pred_da