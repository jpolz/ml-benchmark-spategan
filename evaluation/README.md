# Evaluation Metrics

This module provides the setf of evaluation metrics recommended by CORDEX-ML-Bench for evaluating AI-based regional climate downscaling. It supports filtering by season and includes both standard statistical metrics and spatial analysis via power spectral density (PSD).

| Metric           | Description                                                                 | Input Parameters |
|-----------------|-----------------------------------------------------------------------------|-----------------|
| rmse          | Root Mean Square Error over the time dimension                              | target, pred, var_target, season |
| bias_mean     | Mean bias between prediction and target                                     | target, pred, var_target, season |
| bias_quantile | Bias for a specific quantile over time                                      | target, pred, quantile, var_target, season |
| ratio_std     | Ratio of standard deviations (predicted / target) over time                | target, pred, var_target, season |
| psd           | Power Spectral Density (includes radial averaging for 2D spatial fields) | target, pred, var_target, season |

*Example Usage*

```python
import xarray as xr
from metrics import rmse, bias_mean, psd

# Load your datasets
target_ds = xr.open_dataset("target.nc")
pred_ds = xr.open_dataset("prediction.nc")

# Compute RMSE for temperature
rmse_val = rmse(target_ds, pred_ds, var_target="tasmax", season="winter")

# Compute mean bias
bias = bias_mean(target_ds, pred_ds, var_target="tasmax")

# Compute Power Spectral Density
psd_target, psd_pred = psd(target_ds, pred_ds, var_target="tasmax")
```
