import logging

import numpy as np
import xarray as xr

from ml_benchmark_spategan.evaluate import scores
from ml_benchmark_spategan.evaluate.model_selection_score import (
    DEFAULT_WEIGHTS as SCORE_WEIGHTS,
)
from ml_benchmark_spategan.train.normalize import predictions_to_xarray


def compute_model_selection_score(
    diagnostics_dict: dict, weights: dict = None
) -> float:
    """
    Compute the weighted composite model selection score from diagnostic metrics.

    Lower scores indicate better models. This score is used to determine the
    best model checkpoint during training.

    Args:
        diagnostics_dict: Dictionary of diagnostic metrics (single epoch, not history)
        weights: Optional custom weights dict. Uses DEFAULT_WEIGHTS if None.

    Returns:
        Composite score (lower is better)
    """
    if weights is None:
        weights = SCORE_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for metric_name, weight in weights.items():
        if metric_name not in diagnostics_dict:
            continue

        value = diagnostics_dict[metric_name]

        # Skip NaN values or non-numeric
        if not isinstance(value, (int, float)) or np.isnan(value):
            continue

        # Metrics where ideal value is 1.0
        if metric_name in ["std_ratio", "correlation", "anomaly_correlation"]:
            value = abs(value - 1.0)
        else:
            # For metrics where higher is better (negative weight), negate first
            if weight < 0:
                value = -value
            # Then take absolute value so all contributions are positive
            value = abs(value)

        # Apply weight
        weighted_value = abs(weight) * value
        weighted_sum += weighted_value
        total_weight += abs(weight)

    if total_weight > 0:
        return weighted_sum / total_weight
    return float("inf")


def compute_diagnostics(
    y_pred_all,
    y_true_all,
    norm_params,
    cf,
    mean_fss_test,
    epoch,
    ensemble_preds_all=None,
):
    """
    Compute diagnostic metrics from predictions and ground truth.

    Args:
        y_pred_all: Concatenated predictions tensor (all test batches)
        y_true_all: Concatenated ground truth tensor (all test batches)
        norm_params: Normalization parameters dictionary
        cf: Configuration object
        mean_fss_test: Mean FSS test loss for this epoch
        epoch: Current epoch number
        ensemble_preds_all: Optional ensemble predictions (B, N_ensemble, H, W) for variability computation

    Returns:
        dict: Dictionary of diagnostic metrics
    """
    logger = logging.getLogger(__name__)

    # Convert to xarray with denormalization
    pred_ds, true_ds = predictions_to_xarray(
        y_pred_all, y_true_all, norm_params, var_name=cf.data.var_target
    )

    var_target = cf.data.var_target

    # Compute standard diagnostics
    rmse = scores.rmse(true_ds, pred_ds, var=var_target, dim="time")
    bias_mean = scores.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].mean("time"),
    )
    bias_q95 = scores.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].quantile(0.95, dim="time"),
    )
    bias_q98 = scores.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].quantile(0.98, dim="time"),
    )
    std_ratio = scores.ratio_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].std("time"),
    )

    # Mean Absolute Error
    mae = np.abs(pred_ds[var_target] - true_ds[var_target]).mean("time")

    # Pearson correlation
    correlation = xr.corr(
        pred_ds[var_target],
        true_ds[var_target],
        dim="time",
    )

    spatial_dims = norm_params["spatial_dims"]
    # Anomaly correlation (after removing climatology)
    pred_anomaly = pred_ds[var_target] - pred_ds[var_target].mean("time")
    true_anomaly = true_ds[var_target] - true_ds[var_target].mean("time")
    anomaly_correlation = xr.corr(pred_anomaly, true_anomaly, dim=spatial_dims)

    # Power Spectral Density and distance metric
    psd_true, psd_pred = scores.psd(x0=true_ds, x1=pred_ds, var=var_target)

    # Compute PSD distance (RMSE in log space)
    wavenumber_min = 1
    wavenumber_max = min(60, len(psd_true) - 1)
    wavenumber = psd_true["wavenumber"].values
    mask = (wavenumber >= wavenumber_min) & (wavenumber <= wavenumber_max)
    eps = 1e-10
    log_psd_true = np.log10(psd_true.values[mask] + eps)
    log_psd_pred = np.log10(psd_pred.values[mask] + eps)
    psd_distance = float(np.sqrt(np.mean((log_psd_true - log_psd_pred) ** 2)))

    # Build diagnostics dictionary with spatially-averaged values
    diagnostics_dict = {
        "rmse": rmse[var_target].mean().values.item(),
        "bias_mean": bias_mean.mean().values.item(),
        "bias_q95": bias_q95.mean().values.item(),
        "bias_q98": bias_q98.mean().values.item(),
        "std_ratio": std_ratio.mean().values.item(),
        "mae": mae.mean().values.item(),
        "correlation": correlation.mean().values.item(),
        "anomaly_correlation": anomaly_correlation.mean().values.item(),
        "psd_distance": psd_distance,
        "fss": mean_fss_test,
        "epoch": epoch,
    }

    # Ensemble variability (if ensemble predictions provided)
    if ensemble_preds_all is not None:
        # ensemble_preds_all shape: (B, N_ensemble, H*W)
        # Compute std along ensemble dimension, then spatial mean
        ensemble_std = ensemble_preds_all.std(dim=1).mean().item()
        diagnostics_dict["ensemble_std"] = float(ensemble_std)
        logger.info(f"  Ensemble Variability (std): {ensemble_std:.4f}")

    # Variable-specific climate indices
    if var_target == "tasmax":
        # Temperature-specific indices
        # Summer days (days > 25°C, threshold=298.15K for data in Kelvin)
        su_true = scores.su(true_ds, var_target, threshold=298.15)
        su_pred = scores.su(pred_ds, var_target, threshold=298.15)
        su_bias = (su_pred[var_target] - su_true[var_target]).mean().values.item()
        diagnostics_dict["su_bias"] = float(su_bias)

        # Mean annual maximum temperature
        txx_true = scores.txx(true_ds, var_target)
        txx_pred = scores.txx(pred_ds, var_target)
        txx_bias = (txx_pred[var_target] - txx_true[var_target]).mean().values.item()
        diagnostics_dict["txx_bias"] = float(txx_bias)

        # Mean annual minimum temperature
        txn_true = scores.txn(true_ds, var_target)
        txn_pred = scores.txn(pred_ds, var_target)
        txn_bias = (txn_pred[var_target] - txn_true[var_target]).mean().values.item()
        diagnostics_dict["txn_bias"] = float(txn_bias)

        logger.info(f"  Summer Days Bias: {su_bias:.4f}")
        logger.info(f"  TXx (Annual Max) Bias: {txx_bias:.4f}")
        logger.info(f"  TXn (Annual Min) Bias: {txn_bias:.4f}")

    elif var_target == "pr":
        # Precipitation-specific indices
        # Maximum 1-day precipitation
        rx1day_true = scores.rx1day(true_ds, var_target)
        rx1day_pred = scores.rx1day(pred_ds, var_target)
        rx1day_bias = (
            (rx1day_pred[var_target] - rx1day_true[var_target]).mean().values.item()
        )
        diagnostics_dict["rx1day_bias"] = float(rx1day_bias)

        # Simple precipitation intensity (mean precip on wet days)
        sdii_true = scores.sdii(true_ds, var_target, wet_threshold=1.0)
        sdii_pred = scores.sdii(pred_ds, var_target, wet_threshold=1.0)
        sdii_bias = (sdii_pred[var_target] - sdii_true[var_target]).mean().values.item()
        diagnostics_dict["sdii_bias"] = float(sdii_bias)

        # Consecutive dry days
        cdd_true = scores.cdd(true_ds, var_target, dry_threshold=1.0)
        cdd_pred = scores.cdd(pred_ds, var_target, dry_threshold=1.0)
        cdd_bias = (cdd_pred[var_target] - cdd_true[var_target]).mean().values.item()
        diagnostics_dict["cdd_bias"] = float(cdd_bias)

        # Consecutive wet days
        cwd_true = scores.cwd(true_ds, var_target, wet_threshold=1.0)
        cwd_pred = scores.cwd(pred_ds, var_target, wet_threshold=1.0)
        cwd_bias = (cwd_pred[var_target] - cwd_true[var_target]).mean().values.item()
        diagnostics_dict["cwd_bias"] = float(cwd_bias)

        logger.info(f"  Rx1day (Max 1-day Precip) Bias: {rx1day_bias:.4f}")
        logger.info(f"  SDII (Precip Intensity) Bias: {sdii_bias:.4f}")
        logger.info(f"  CDD (Max Dry Spell) Bias: {cdd_bias:.4f}")
        logger.info(f"  CWD (Max Wet Spell) Bias: {cwd_bias:.4f}")

    # Universal indices (applicable to both variables)
    # Lag-1 autocorrelation
    lag1_true = scores.lag1_corr(true_ds, var_target)
    lag1_pred = scores.lag1_corr(pred_ds, var_target)
    lag1_bias = (lag1_pred[var_target] - lag1_true[var_target]).mean().values.item()
    diagnostics_dict["lag1_corr_bias"] = float(lag1_bias)

    # Interannual variability
    interann_true = scores.interannual_var(true_ds, var_target)
    interann_pred = scores.interannual_var(pred_ds, var_target)
    interann_bias = (
        (interann_pred[var_target] - interann_true[var_target]).mean().values.item()
    )
    diagnostics_dict["interannual_var_bias"] = float(interann_bias)

    # Log key diagnostics
    logger.info(f"  RMSE (spatial mean): {diagnostics_dict['rmse']:.4f}")
    logger.info(f"  Bias Mean (spatial mean): {diagnostics_dict['bias_mean']:.4f}")
    logger.info(f"  Bias Q95 (spatial mean): {diagnostics_dict['bias_q95']:.4f}")
    logger.info(f"  Std Ratio (spatial mean): {diagnostics_dict['std_ratio']:.4f}")
    logger.info(f"  Correlation (spatial mean): {diagnostics_dict['correlation']:.4f}")
    logger.info(f"  PSD Distance (log RMSE): {diagnostics_dict['psd_distance']:.4f}")
    logger.info(f"  Lag-1 Autocorr Bias: {lag1_bias:.4f}")
    logger.info(f"  Interannual Var Bias: {interann_bias:.4f}")

    return diagnostics_dict
