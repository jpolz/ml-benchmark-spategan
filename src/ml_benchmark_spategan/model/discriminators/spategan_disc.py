"""Spatial GAN discriminator architecture."""

import torch
import torch.nn as nn

from ml_benchmark_spategan.model.layers import ResidualBlock2D

###########################################################################
### DISCRIMINATOR
###########################################################################


class Discriminator(nn.Module):
    """
    Spatial GAN discriminator with dual-path architecture.

    Processes high-resolution (fine) and low-resolution (coarse) inputs separately,
    then combines them for classification.
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.apply(self._init_weights)
        self.n_coarse_channels = config.model.n_input_channels
        self.n_fine_channels = config.model.n_output_channels

        # Get discriminator config with defaults
        disc_config = getattr(config.model, "discriminator", None)
        if disc_config is None:
            # Default channel progression - REDUCED capacity to prevent overfitting
            self.hr_channels = [64, 64, 32]  # Was [128, 128, 128, 64, 64]
            self.lr_channels = [32]  # Was [64, 32]
            self.combined_channels = [32]  # Was [64]
            self.output_channels = [32, 1]  # Was [64, 1]
            self.dropout = 0.2  # Add dropout by default
            self.use_spectral_norm = True  # Enable spectral norm for stability
            self.use_lr_path = True  # Use LR input by default
        else:
            self.hr_channels = getattr(disc_config, "hr_channels", [64, 64, 32])
            self.lr_channels = getattr(disc_config, "lr_channels", [32])
            self.combined_channels = getattr(disc_config, "combined_channels", [32])
            self.output_channels = getattr(disc_config, "output_channels", [32, 1])
            self.dropout = getattr(disc_config, "dropout", 0.2)
            self.use_spectral_norm = getattr(disc_config, "spectral_norm", True)
            self.use_lr_path = getattr(disc_config, "use_lr_path", True)

        # HIGH RESOLUTION path
        hr_layers = []
        in_ch = self.n_fine_channels
        for i, out_ch in enumerate(self.hr_channels):
            block = ResidualBlock2D(
                in_ch,
                out_ch,
                use_layer_norm=(i > 0),  # No norm on first layer
                stride=(1, 1) if i == 0 else (2, 2),
                dropout=self.dropout,
            )
            # Apply spectral norm to conv layers
            if self.use_spectral_norm:
                block.conv1 = nn.utils.spectral_norm(block.conv1)
                block.conv2 = nn.utils.spectral_norm(block.conv2)
                if block.shortcut is not None:
                    block.shortcut = nn.utils.spectral_norm(block.shortcut)
            hr_layers.append(block)
            in_ch = out_ch

        self.hr_path = nn.ModuleList(hr_layers)

        # LOW RESOLUTION path (optional for ablation studies)
        if self.use_lr_path:
            lr_layers = []
            in_ch = self.n_coarse_channels
            for i, out_ch in enumerate(self.lr_channels):
                block = ResidualBlock2D(
                    in_ch,
                    out_ch,
                    use_layer_norm=(i > 0),
                    stride=(1, 1) if i == 0 else (2, 2),
                    dropout=self.dropout,
                )
                # Apply spectral norm to conv layers
                if self.use_spectral_norm:
                    block.conv1 = nn.utils.spectral_norm(block.conv1)
                    block.conv2 = nn.utils.spectral_norm(block.conv2)
                    if block.shortcut is not None:
                        block.shortcut = nn.utils.spectral_norm(block.shortcut)
                lr_layers.append(block)
                in_ch = out_ch

            self.lr_path = nn.ModuleList(lr_layers)

            # Adaptive pooling to match spatial dimensions before concatenation
            # This ensures hr_out and lr_out have the same spatial size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Will be set dynamically
        else:
            self.lr_path = None
            self.adaptive_pool = None

        # Combined path
        combined_layers = []
        # Input channels: HR only or HR + LR depending on use_lr_path
        in_ch = self.hr_channels[-1] + (self.lr_channels[-1] if self.use_lr_path else 0)
        for out_ch in self.combined_channels:
            block = ResidualBlock2D(
                in_ch,
                out_ch,
                use_layer_norm=True,
                stride=(2, 2),
                dropout=self.dropout,
            )
            # Apply spectral norm to conv layers
            if self.use_spectral_norm:
                block.conv1 = nn.utils.spectral_norm(block.conv1)
                block.conv2 = nn.utils.spectral_norm(block.conv2)
                if block.shortcut is not None:
                    block.shortcut = nn.utils.spectral_norm(block.shortcut)
            combined_layers.append(block)
            in_ch = out_ch

        self.combined_path = nn.ModuleList(combined_layers)

        # Output head
        output_layers = []
        in_ch = (
            self.combined_channels[-1]
            if self.combined_channels
            else (self.hr_channels[-1] + self.lr_channels[-1])
        )
        for i, out_ch in enumerate(self.output_channels):
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            if self.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)

            if i < len(self.output_channels) - 1:
                output_layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
            else:
                output_layers.append(conv)
            in_ch = out_ch

        self.output_conv = nn.Sequential(*output_layers)

    def _init_weights(self, m):
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv3d)
        ):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        """
        Forward pass through discriminator.

        Args:
            x: High-resolution (fine) input, shape (batch, n_fine_channels, H, W)
            y: Low-resolution (coarse) input, shape (batch, n_coarse_channels, h, w)

        Returns:
            Discriminator output (logits), shape (batch, 1, H', W')
        """
        # Note: Noise is now added in training loop for consistency
        # High resolution path
        hr_out = x
        for layer in self.hr_path:
            hr_out = layer(hr_out)

        # Low resolution path (optional)
        if self.use_lr_path:
            lr_out = y
            for layer in self.lr_path:
                lr_out = layer(lr_out)

            # Match spatial dimensions using adaptive pooling
            # Pool LR to match HR spatial size (in case they differ)
            target_size = (hr_out.shape[2], hr_out.shape[3])
            if lr_out.shape[2:] != hr_out.shape[2:]:
                lr_out = nn.functional.interpolate(
                    lr_out, size=target_size, mode="bilinear", align_corners=False
                )

            # Concatenate HR and LR features
            combined = torch.cat((hr_out, lr_out), dim=1)
        else:
            # Use HR features only
            combined = hr_out

        # Combined path
        for layer in self.combined_path:
            combined = layer(combined)

        # Output
        out = self.output_conv(combined)

        return out
