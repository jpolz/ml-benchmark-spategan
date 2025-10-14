import numpy as np
import xarray as xr


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

def _quantile(data: xr.Dataset, var: str, q: float) -> xr.Dataset:
    """
    Compute the specified time quantile of a variable in a dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].

    Returns
    -------
    xr.Dataset
        Quantile of the input data.
    """
    da = data[[var]]
    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result

def _mean(data: xr.Dataset, var: str) -> xr.Dataset:
    """
    Compute the mean of a variable in a dataset over the time dimension.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the mean for.

    Returns
    -------
    xr.Dataset
        Mean of the input data over time.
    """
    da = data[[var]]
    result = da.mean(dim="time", skipna=True)
    return result

def _max_spell_length(condition: np.ndarray) -> float:
    """
    Compute the maximum consecutive spell length in a boolean sequence.

    Parameters
    ----------
    condition : np.ndarray
        Boolean sequence describing the spell condition.

    Returns
    -------
    float
        Maximum consecutive spell length.
    """
    max_run = 0
    run = 0
    for flag in condition:
        if flag:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return float(max_run)

def _resample_max_spell(condition: xr.DataArray, freq: str = "1YE") -> xr.DataArray:
    """
    Aggregate maximum spell length over the resampled period and return the mean annual spell length.

    Parameters
    ----------
    condition : xr.DataArray
        Boolean array describing the spell condition over time.
    freq : str, default "1YE"
        Resampling frequency following pandas offset aliases.

    Returns
    -------
    xr.DataArray
        Resampled maximum spell length.
    """
    def _compute(arr: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _max_spell_length,
            arr,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    return condition.resample(time=freq).map(_compute).mean('time')

def su(data: xr.Dataset, var: str, season: str | None = None, threshold: float =  300.0) -> xr.Dataset:
    """
    Count summer days defined by daily maximum temperature above a threshold. First the summer days
    are counted for each year, then the mean is computed.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.
    threshold : float, default 25.0
        Temperature threshold in degrees Celsius.

    Returns
    -------
    xr.Dataset
        Annual count of summer days.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    return (da > threshold).groupby('time.year').sum('time').mean('year')

def txx(data: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual maximum of daily maximum temperature.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual maximum values of daily maximum temperature.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    annual_max = da.resample(time="1YE").max()
    return annual_max.mean(dim="time")

def txn(data: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual minimum of daily maximum temperature.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual minimum values of daily maximum temperature.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    annual_min = da.resample(time="1YE").min()
    return annual_min.mean(dim="time")

def rx1day(data: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual maximum 1-day precipitation.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual maximum daily precipitation.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    annual_max = da.resample(time="1YE").max()
    return annual_max.mean(dim="time")

def sdii(
    data: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the simple precipitation intensity index.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    wet_threshold : float, default 1.0
        Minimum precipitation to consider a wet day.

    Returns
    -------
    xr.Dataset
        Simple precipitation intensity index.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    wet = da.where(da >= wet_threshold)
    return wet.mean(dim="time")

def cdd(
    data: xr.Dataset,
    var: str,
    season: str | None = None,
    dry_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the maximum length of dry spells with precipitation below a threshold.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    dry_threshold : float, default 1.0
        Precipitation threshold to qualify as a dry day.

    Returns
    -------
    xr.Dataset
        Annual maximum dry spell length.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    condition = da < dry_threshold
    return _resample_max_spell(condition)

def cwd(
    data: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the maximum length of wet spells with precipitation above a threshold.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    wet_threshold : float, default 1.0
        Precipitation threshold to qualify as a wet day.

    Returns
    -------
    xr.Dataset
        Annual maximum wet spell length.
    """
    data = _filter_by_season(data, season)
    da = data[[var]]
    condition = da >= wet_threshold
    return _resample_max_spell(condition)

def lag1_corr(data: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the lag-1 autocorrelation of the target variable along time.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Lag-1 autocorrelation computed along the time dimension.
    """

    data = _filter_by_season(data, season)
    da = data[var]
    shifted = da.shift(time=1)
    lag1_corr = xr.corr(da, shifted, dim="time")
    return xr.Dataset({var: lag1_corr})

def interannual_var(data: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the interannual variability of the target variable along time.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Interannual variability computed along the time dimension.
    """

    data = _filter_by_season(data, season)
    da = data[[var]]
    return da.groupby('time.year').mean(dim='time').std('year')