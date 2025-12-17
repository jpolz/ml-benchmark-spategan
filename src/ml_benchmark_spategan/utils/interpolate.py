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


class LearnableUpsampler(nn.Module):
    """
    Learnable upsampling module using convolutional layers.

    Upsamples from 16x16 to 128x128 through three 2x upsampling stages
    with convolutional layers and activation functions.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels in intermediate layers
    """

    def __init__(self, in_channels: int = 15, hidden_channels: int = 64):
        super().__init__()
        # 16x16 -> 32x32 -> 64x64 -> 128x128
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv4 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of learnable upsampler.

        Args:
            x: Input tensor of shape (B, in_channels, 16, 16)

        Returns:
            Upsampled tensor of shape (B, in_channels, 128, 128)
        """
        # 16x16 -> 32x32
        x = self.activation(self.conv1(x))
        x = self.up1(x)

        # 32x32 -> 64x64
        x = self.activation(self.conv2(x))
        x = self.up2(x)

        # 64x64 -> 128x128
        x = self.activation(self.conv3(x))
        x = self.up3(x)

        # Final conv without activation
        x = self.conv4(x)
        return x
