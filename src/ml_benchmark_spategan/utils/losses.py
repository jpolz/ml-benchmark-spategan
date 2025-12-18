from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSSCalculator:
    """
    Differentiable Fraction Skill Score (FSS) calculator.

    Can be used both as evaluation metric and as training loss backend.

    Shape conventions (ML style):
    - 4D: (Batch, Channel, Height, Width)
    - 5D: (Batch, Channel, Time, Height, Width)

    Ensemble detection:
    - If obs has 1 channel and fcst has >1 channels, treat fcst channels as
      ensemble members and average them before computing FSS.
    """

    def __init__(
        self,
        thresholds,
        scales,
        device="cuda",
        default_batch_size=32,
        sharpness=10.0,
        config: dict = None,
        norm_params: dict = None,
    ):
        self.device = device
        self.default_batch_size = default_batch_size
        self.kernel_cache = {}
        self.scales = scales
        self.sharpness = sharpness

        # Convert thresholds to tensor for normalization
        if isinstance(thresholds, (list, np.ndarray)):
            thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32)
        elif isinstance(thresholds, torch.Tensor):
            thresholds_tensor = thresholds.float().detach().clone()
        else:
            raise ValueError(f"Unsupported thresholds type: {type(thresholds)}")

        print("FSS thresholds before normalization:", thresholds_tensor)

        # Apply normalization if configured
        if config is not None and config.data.normalization is not None:
            thresholds_tensor = self._normalize_threshold(
                thresholds_tensor, config, norm_params
            )

        print("FSS thresholds after normalization:", thresholds_tensor)

        # Convert to Python list
        self.thresholds: List[float] = thresholds_tensor.tolist()

    def _normalize_threshold(
        self, th: torch.Tensor, config, norm_params
    ) -> torch.Tensor:
        if config.data.normalization == "mp1p1_input_m1p1log_target":
            y_min = norm_params["y_min"][config.data.var_target].mean().values
            y_max = norm_params["y_max"][config.data.var_target].mean().values

            th = torch.log1p(th + 1e-6) - torch.log1p(torch.tensor(1e-6))
            th = (th - y_min) / (y_max - y_min)
            th = th * 2 - 1
            return th

        elif config.data.normalization == "m1p1_log_target":
            return torch.log1p(th) - 1e-6

        elif config.data.normalization == "minus1_to_plus1":
            y_min = norm_params["y_min"][config.data.var_target].mean().values
            y_max = norm_params["y_max"][config.data.var_target].mean().values

            th = (th - y_min) / (y_max - y_min)
            th = th * 2 - 1
            return th

        elif config.data.normalization == "standardization":
            # For standardization, y is not normalized (stays in original units)
            # So thresholds should also stay in original units
            return th

        elif config.data.normalization == "minmax":
            # For minmax, y is not normalized (stays in original units)
            # So thresholds should also stay in original units
            return th

        elif config.data.normalization == "log":
            # Apply log transform to thresholds
            return torch.log1p(th)

        else:
            # No normalization or unknown - return thresholds as-is
            return th

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply soft thresholding using sigmoid for differentiability."""
        return torch.sigmoid(self.sharpness * (x - threshold))

    def _get_kernel(self, window_size: int, dtype: torch.dtype) -> torch.Tensor:
        """Get averaging kernel matching the input dtype for mixed precision compatibility."""
        cache_key = (window_size, dtype)
        if cache_key not in self.kernel_cache:
            with torch.no_grad():
                kernel = torch.ones(
                    (1, 1, window_size, window_size),
                    device=self.device,
                    dtype=dtype,
                )
                kernel = kernel / (window_size * window_size)
                self.kernel_cache[cache_key] = kernel
        return self.kernel_cache[cache_key]

    def compute(
        self,
        fcst: Union[torch.Tensor, np.ndarray],
        obs: Union[torch.Tensor, np.ndarray],
        threshold: float,
        window_size: int,
        batch_size: int = None,
    ) -> torch.Tensor:
        """
        Compute FSS for a single threshold and window size.

        Parameters
        ----------
        fcst : torch.Tensor
            Forecast tensor. Shape: (B, C, H, W) or (B, C, T, H, W)
            If C > 1 and obs has C = 1, channels are treated as ensemble members.
        obs : torch.Tensor
            Observation tensor. Shape: (B, C, H, W) or (B, C, T, H, W)
        threshold : float
            Threshold value for FSS computation
        window_size : int
            Spatial window size for neighborhood averaging
        batch_size : int, optional
            Batch size for processing convolutions

        Returns
        -------
        torch.Tensor
            FSS score (scalar)
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Convert to tensor if needed
        if not isinstance(fcst, torch.Tensor):
            fcst = torch.tensor(fcst, dtype=torch.float32, device=self.device)
        else:
            fcst = fcst.to(self.device)

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        # Detach observations
        obs = obs.detach()

        fcst_shape = fcst.shape
        obs_shape = obs.shape

        # Validate dimensions
        if len(fcst_shape) != len(obs_shape):
            raise ValueError(
                f"Forecast and observation must have same number of dimensions. "
                f"Got fcst: {len(fcst_shape)}D, obs: {len(obs_shape)}D"
            )

        if len(fcst_shape) == 4:
            # 4D: (Batch, Channel, Height, Width)
            B, C_fcst, H, W = fcst_shape
            _, C_obs, _, _ = obs_shape

            # Ensemble mode: fcst has multiple channels, obs has 1 channel
            ensemble = C_obs == 1 and C_fcst > 1

            if ensemble:
                # Average soft-thresholded forecast over channel/ensemble dimension
                fcst_proc = self._soft_threshold(fcst, threshold).mean(
                    dim=1, keepdim=True
                )
                with torch.no_grad():
                    obs_proc = (obs > threshold).to(
                        fcst_proc.dtype
                    )  # Already (B, 1, H, W)
            else:
                # No ensemble: process each channel independently
                # Reshape to (B*C, 1, H, W) for convolution
                fcst_proc = self._soft_threshold(fcst, threshold).reshape(
                    B * C_fcst, 1, H, W
                )
                with torch.no_grad():
                    obs_proc = (
                        (obs > threshold)
                        .to(fcst_proc.dtype)
                        .reshape(B * C_obs, 1, H, W)
                    )

        elif len(fcst_shape) == 5:
            # 5D: (Batch, Channel, Time, Height, Width)
            B, C_fcst, T, H, W = fcst_shape
            _, C_obs, _, _, _ = obs_shape

            # Ensemble mode: fcst has multiple channels, obs has 1 channel
            ensemble = C_obs == 1 and C_fcst > 1

            if ensemble:
                # Average soft-thresholded forecast over channel/ensemble dimension
                # Result: (B, 1, T, H, W)
                fcst_soft = self._soft_threshold(fcst, threshold).mean(
                    dim=1, keepdim=True
                )
                # Reshape to (B*T, 1, H, W) for 2D convolution
                fcst_proc = fcst_soft.reshape(B * T, 1, H, W)
                with torch.no_grad():
                    obs_proc = (
                        (obs > threshold).to(fcst_proc.dtype).reshape(B * T, 1, H, W)
                    )
            else:
                # No ensemble: process each channel and time independently
                # Reshape to (B*C*T, 1, H, W) for convolution
                fcst_proc = self._soft_threshold(fcst, threshold).reshape(
                    B * C_fcst * T, 1, H, W
                )
                with torch.no_grad():
                    obs_proc = (
                        (obs > threshold)
                        .to(fcst_proc.dtype)
                        .reshape(B * C_obs * T, 1, H, W)
                    )
        else:
            raise ValueError(
                f"Unsupported tensor shape. Expected 4D (B,C,H,W) or 5D (B,C,T,H,W). "
                f"Got {len(fcst_shape)}D with shape {fcst_shape}"
            )

        # Get kernel matching the dtype of fcst_proc (handles mixed precision)
        kernel = self._get_kernel(window_size, fcst_proc.dtype)
        pad = window_size // 2

        fss_values = []
        N = fcst_proc.shape[0]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            F_frac = F.conv2d(fcst_proc[start:end], kernel, padding=pad)

            with torch.no_grad():
                O_frac = F.conv2d(obs_proc[start:end], kernel, padding=pad)

            mse = torch.mean((F_frac - O_frac) ** 2, dim=(2, 3))
            mse_ref = torch.mean(F_frac**2 + O_frac**2, dim=(2, 3))

            fss = 1.0 - mse / (mse_ref + 1e-8)
            fss_values.append(fss)

        fss = torch.cat(fss_values).squeeze(1)

        valid = torch.isfinite(fss)

        if valid.any():
            return fss[valid].mean()

        return torch.tensor(
            1.0, device=self.device, dtype=fcst.dtype, requires_grad=True
        )

    def compute_multi(
        self,
        fcst: Union[torch.Tensor, np.ndarray],
        obs: Union[torch.Tensor, np.ndarray],
        batch_size: int = None,
    ) -> Dict[float, Dict[int, torch.Tensor]]:
        """
        Compute FSS for multiple thresholds and scales.
        """
        results = {}

        for th in self.thresholds:
            results[th] = {}
            for scale in self.scales:
                results[th][scale] = self.compute(
                    fcst, obs, threshold=th, window_size=scale, batch_size=batch_size
                )
        return results


class FSSLoss(nn.Module):
    """
    Training-ready FSS loss: loss = 1 - mean(FSS)

    Shape conventions (ML style):
    - 4D: (Batch, Channel, Height, Width)
    - 5D: (Batch, Channel, Time, Height, Width)

    Ensemble detection:
    - If obs has 1 channel and fcst has >1 channels, treat fcst channels as
      ensemble members and average them before computing FSS.

    """

    def __init__(
        self,
        thresholds,
        scales,
        device="cuda",
        sharpness=10.0,
        batch_size: int = 8,
        config: dict = None,
        norm_params: dict = None,
    ):
        super().__init__()
        self.calculator = FSSCalculator(
            thresholds=thresholds,
            scales=scales,
            device=device,
            sharpness=sharpness,
            default_batch_size=batch_size,
            config=config,
            norm_params=norm_params,
        )

    def forward(self, fcst: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute FSS loss.

        Parameters
        ----------
        fcst : torch.Tensor
            Forecast tensor. Shape: (B, C, H, W) or (B, C, T, H, W)
        obs : torch.Tensor
            Observation/target tensor. Shape: (B, C, H, W) or (B, C, T, H, W)

        Returns
        -------
        torch.Tensor
            Loss value (1 - mean FSS)
        """
        # Clone to create independent computational graph branch
        fcst_clone = fcst.clone()

        fss_dict = self.calculator.compute_multi(fcst_clone, obs)

        fss_values = []
        for th_dict in fss_dict.values():
            for fss in th_dict.values():
                fss_values.append(fss)

        fss_stack = torch.stack(fss_values)

        loss = 1.0 - fss_stack.mean()

        return loss
