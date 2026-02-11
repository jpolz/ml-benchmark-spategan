"""Interpolation utilities for upsampling low-resolution inputs."""

import torch
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

    Handles both 4D (spatial) and 5D (temporal) tensors:
    - 4D: (B, C, H, W) → (B, C+1, H, W)
    - 5D: (B, C, T, H, W) → (B, C+1, T, H, W)

    Args:
        x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Tensor with noise channel concatenated along channel dimension
    """
    if x.ndim == 4:
        # 4D case: (B, C, H, W)
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
    elif x.ndim == 5:
        # 5D case: (B, C, T, H, W)
        noise = (
            torch.randn(
                x.size(0),  # batch
                1,  # 1 noise channel
                x.size(2),  # time
                x.size(3),  # height
                x.size(4),  # width
                device=x.device,
            )
            * noise_std
        )
    else:
        raise ValueError(
            f"add_noise_channel expects 4D (B,C,H,W) or 5D (B,C,T,H,W) tensor, "
            f"got {x.ndim}D tensor with shape {x.shape}"
        )

    return torch.cat([x, noise], dim=1)
