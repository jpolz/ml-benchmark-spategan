"""Interpolation utilities for upsampling low-resolution inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def upscale_bilinear(x: torch.Tensor, target_size: tuple = (128, 128)) -> torch.Tensor:
    """
    Upscale input using bilinear interpolation.

    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        target_size: Target spatial dimensions (H_out, W_out)

    Returns:
        Upscaled tensor of shape (B, C, H_out, W_out) or (C, H_out, W_out)
    """
    return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


def add_noise_channel(x: torch.Tensor, noise_std: float = 0.2) -> torch.Tensor:
    """
    Add a random noise channel to the input tensor.

    Args:
        x: Input tensor of shape (B, C, H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Tensor of shape (B, C+1, H, W) with noise channel concatenated
    """
    noise = (
        torch.randn(
            x.size(0),  # batch
            1,  # 1 noise channel
            x.size(2),  # height
            x.size(3),  # width
            device=x.device,
        )
        * noise_std
    )
    return torch.cat([x, noise], dim=1)
