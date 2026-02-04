"""Learnable noise modules for stochastic generation."""

import torch
import torch.nn as nn


class LearnableNoiseScale(nn.Module):
    """
    Learnable global noise scale parameter.

    This module learns a single scalar that scales the noise added to the generator.
    The scale is constrained to a valid range using sigmoid and affine transformation.

    Args:
        init_value: Initial noise scale (e.g., 0.05 for tasmax, 0.1 for pr)
        min_value: Minimum allowed noise scale
        max_value: Maximum allowed noise scale
    """

    def __init__(
        self, init_value: float = 0.05, min_value: float = 0.0, max_value: float = 1.0
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

        # Initialize with inverse sigmoid to get init_value after transformation
        # sigmoid(logit) = (init - min) / (max - min)
        # logit = log((init - min) / (max - min - init + min))
        normalized_init = (init_value - min_value) / (max_value - min_value)
        normalized_init = torch.clamp(
            torch.tensor(normalized_init), 0.01, 0.99
        )  # Avoid extremes
        logit_init = torch.log(normalized_init / (1.0 - normalized_init))

        self.logit_scale = nn.Parameter(logit_init)

    def forward(self) -> torch.Tensor:
        """
        Get the current noise scale.

        Returns:
            Scalar tensor with noise scale in [min_value, max_value]
        """
        # Transform logit to valid range
        normalized = torch.sigmoid(self.logit_scale)
        scale = self.min_value + normalized * (self.max_value - self.min_value)
        return scale

    def get_scale(self) -> float:
        """Get current scale as Python float for logging."""
        with torch.no_grad():
            return self.forward().item()


class SpatialNoiseScale(nn.Module):
    """
    Learnable per-pixel noise scale map.

    This module learns a spatial map of noise scales, allowing different regions
    to have different amounts of stochastic variability.

    Args:
        height: Spatial height of the noise map
        width: Spatial width of the noise map
        init_value: Initial noise scale (e.g., 0.05 for tasmax, 0.1 for pr)
        min_value: Minimum allowed noise scale
        max_value: Maximum allowed noise scale
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        init_value: float = 0.05,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.min_value = min_value
        self.max_value = max_value

        # Initialize spatial map with inverse sigmoid
        normalized_init = (init_value - min_value) / (max_value - min_value)
        normalized_init = torch.clamp(torch.tensor(normalized_init), 0.01, 0.99)
        logit_init = torch.log(normalized_init / (1.0 - normalized_init))

        # Spatial map: (1, 1, H, W) for broadcasting
        self.logit_scale_map = nn.Parameter(
            torch.full((1, 1, height, width), logit_init.item())
        )

    def forward(self) -> torch.Tensor:
        """
        Get the current noise scale map.

        Returns:
            Tensor of shape (1, 1, H, W) with noise scales in [min_value, max_value]
        """
        # Transform logit map to valid range
        normalized = torch.sigmoid(self.logit_scale_map)
        scale_map = self.min_value + normalized * (self.max_value - self.min_value)
        return scale_map

    def get_mean_scale(self) -> float:
        """Get mean scale across spatial dimensions for logging."""
        with torch.no_grad():
            return self.forward().mean().item()

    def get_scale_stats(self) -> dict:
        """Get statistics of the learned scale map."""
        with torch.no_grad():
            scale_map = self.forward()
            return {
                "mean": scale_map.mean().item(),
                "std": scale_map.std().item(),
                "min": scale_map.min().item(),
                "max": scale_map.max().item(),
            }
