"""
UNet3D generator with a true temporal (depth) dimension.

Instead of flattening time steps into channels (as unet3d.py does), this model
keeps time as a separate spatial dimension and uses 3D convolutions via the
HuggingFace diffusers UNet3DConditionModel.

The UNet3DConditionModel expects input of shape (B, C, T, H, W) where T is the
temporal/depth dimension.  After the UNet processes the full temporal volume we
extract one or more time steps as predictions (by default: centre step = current day;
optionally: centre + future steps for multi-step forecasting).

Key differences from the channel-stacking approach (unet3d.py):
  * Temporal convolutions learn short-range temporal patterns explicitly.
  * Channel count stays constant regardless of temporal window size.
  * The model can potentially generalise better to different window lengths.

IMPORTANT - Minimum temporal length requirement:
  The UNet3D architecture uses temporal pooling (typically 2× per down block).
  With N down blocks, you need at least T = 2^N time steps.
  Example: 4 down blocks → minimum T = 16 → need t_past + t_future >= 15

Usage (via config):
  model:
    architecture: "diffusion_unet_3d"   # triggers this generator
  data:
    t_past: 2
    t_future: 2
"""

import torch
import torch.nn as nn

from ml_benchmark_spategan.train.model.base import BaseModel

# ---------------------------------------------------------------------------
# Robust import for UNet3DConditionModel (API moved across diffusers versions)
# ---------------------------------------------------------------------------
try:
    from diffusers.models.unet_3d_condition import UNet3DConditionModel
except ImportError:
    try:
        from diffusers import UNet3DConditionModel
    except ImportError as exc:
        raise ImportError(
            "Could not import UNet3DConditionModel from diffusers. "
            "Please install a diffusers version that ships this model "
            "(e.g. pip install diffusers>=0.25)."
        ) from exc


# ────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ────────────────────────────────────────────────────────────────────────────


class UNet3DTemporalGenerator(BaseModel):
    """
    Generator that wraps UNet3DConditionModel with true temporal convolutions.

    Input shapes
    ------------
    * sample : (B, C, T, H, W)
        - C = number of predictor channels (+ optional orography + noise)
        - T = temporal window length (t_past + 1 + t_future)
        - H, W = spatial dimensions (128, 128 after upscaling)
    * timestep : (B,)
        Scalar conditioning (e.g. sinusoidal DOY encoding).  Passed to the
        UNet3DConditionModel as the ``timestep`` argument.

    Output
    ------
    * prediction : (B, n_pred_steps, H, W)
        Predictions for n_pred_steps time steps starting from the centre.
        For n_pred_steps=1 (default): only the current day.
        For n_pred_steps>1: current day + (n_pred_steps-1) future days.

    Parameters
    ----------
    unet : UNet3DConditionModel
        The wrapped diffusers model.
    activation : nn.Module
        Applied element-wise after extraction (e.g. Identity or Softplus).
    num_timesteps : int
        Total number of time steps T in the input window (used to compute the
        centre index).
    n_pred_steps : int
        Number of consecutive time steps to predict (starting from centre).
        Default 1 = current day only.
    n_context_steps : int or None
        Number of past time steps to use as context (before centre).
        If None, uses all available past steps (t_past).
        Useful for testing model with different context lengths.
    """

    def __init__(
        self,
        unet: UNet3DConditionModel,
        activation: nn.Module,
        num_timesteps: int,
        n_pred_steps: int = 1,
        n_context_steps: int = None,
    ):
        super().__init__()
        self.unet = unet
        self.activation = activation
        self.num_timesteps = num_timesteps
        self.n_pred_steps = n_pred_steps
        self.centre_idx = num_timesteps // 2  # centre of (t_past, current, t_future)

        # Validate and set context window
        if n_context_steps is None:
            self.n_context_steps = self.centre_idx  # Use all past steps
        else:
            if n_context_steps > self.centre_idx:
                raise ValueError(
                    f"n_context_steps ({n_context_steps}) cannot exceed available past steps ({self.centre_idx})."
                )
            self.n_context_steps = n_context_steps

        # Compute start index for temporal slicing
        self.start_idx = self.centre_idx - self.n_context_steps

        # Validate prediction window fits in the temporal input
        if self.centre_idx + n_pred_steps > num_timesteps:
            raise ValueError(
                f"Cannot predict {n_pred_steps} steps starting from centre (idx {self.centre_idx}). "
                f"Total temporal window is only {num_timesteps} steps. "
                f"Reduce n_pred_steps or increase t_future."
            )

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        sample : (B, C, T, H, W) or (B, C*T, H, W)
            If 4D tensor is provided, it will be automatically reshaped to 5D.
        timestep : (B,)

        Returns
        -------
        (B, n_pred_steps, H, W) or (B, 1, H, W) if n_pred_steps=1
        """
        # Auto-reshape 4D input to 5D if needed (from channel-stacked format)
        if sample.ndim == 4:
            sample = self._reshape_channels_to_temporal(sample)

        # Slice temporal dimension if using reduced context
        if self.n_context_steps < self.centre_idx:
            # Extract: [start_idx : centre_idx + n_pred_steps]
            # This gives us: n_context_steps past + current + (n_pred_steps-1) future
            end_idx = self.centre_idx + self.n_pred_steps
            sample = sample[:, :, self.start_idx : end_idx, :, :]
            # Update centre index relative to sliced input
            relative_centre = self.n_context_steps
        else:
            relative_centre = self.centre_idx

        # UNet3DConditionModel requires encoder_hidden_states even if unused.
        # We pass a dummy (B, 1, 1) tensor of zeros.
        encoder_hidden_states = torch.zeros(
            sample.shape[0], 1, 1, device=sample.device, dtype=sample.dtype
        )

        unet_out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        # diffusers returns a dataclass with .sample
        out = unet_out.sample if hasattr(unet_out, "sample") else unet_out

        # out shape: (B, out_channels, T, H, W) – extract prediction time steps
        if self.n_pred_steps == 1:
            # Single step: (B, 1, H, W)
            out = out[:, :, relative_centre, :, :]  # (B, out_channels, H, W)
        else:
            # Multi-step: (B, n_pred_steps, H, W)
            end_idx = relative_centre + self.n_pred_steps
            out = out[
                :, :, relative_centre:end_idx, :, :
            ]  # (B, out_channels, n_pred_steps, H, W)
            # Reshape to (B, n_pred_steps, H, W) by flattening out_channels into time
            # Assumes out_channels == 1 (single variable prediction)
            out = out.squeeze(1)  # (B, n_pred_steps, H, W)

        return self.activation(out)

    # ── interface stubs (training handled by GAN step) ───────────────────

    def train_step(self, batch, optimizers, criterion, scaler, config, **kwargs):
        raise NotImplementedError(
            "UNet3D training is handled by train_gan_step in the GAN training module."
        )

    def predict_step(
        self, x: torch.Tensor, timesteps: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Inference helper (no grad).

        Parameters
        ----------
        x : (B, C, T, H, W) or (B, C*T, H, W)
            If 4-D, it is assumed the temporal frames are stacked along the
            channel dimension and will be reshaped automatically.
        timesteps : (B,) or None

        Returns
        -------
        (B, n_pred_steps, H, W) or (B, 1, H, W) if n_pred_steps=1
        """
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], device=x.device)

        # If the caller passes a 4-D tensor, try to reshape to 5-D
        if x.ndim == 4:
            x = self._reshape_channels_to_temporal(x)

        with torch.no_grad():
            return self(x, timesteps)

    # ── utilities ────────────────────────────────────────────────────────

    def _reshape_channels_to_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, C*T+extra, H, W) → (B, C+extra, T, H, W).

        Handles the case where noise/orography channels were added after
        temporal flattening. Extra non-temporal channels are broadcast
        across all timesteps.

        For example:
        - Input: (B, 15*16+1, H, W) = (B, 241, H, W)
          where 15*16=240 are temporal channels, +1 is noise
        - Output: (B, 16, 16, H, W) where last channel is noise repeated
        """
        B, CT, H, W = x.shape
        T = self.num_timesteps

        # Check if we have extra channels beyond temporal
        extra_channels = CT % T
        if extra_channels == 0:
            # Perfect division - standard case
            C = CT // T
            # Reshape to (B, T, C, H, W) then permute to (B, C, T, H, W)
            return x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        else:
            # We have extra channels (e.g., noise added after temporal flattening)
            temporal_channels = CT - extra_channels
            C_base = temporal_channels // T

            # Split into temporal and extra parts
            x_temporal = x[:, :temporal_channels, :, :]  # (B, C_base*T, H, W)
            x_extra = x[:, temporal_channels:, :, :]  # (B, extra_channels, H, W)

            # Reshape temporal part
            x_temporal = x_temporal.view(B, T, C_base, H, W).permute(
                0, 2, 1, 3, 4
            )  # (B, C_base, T, H, W)

            # Expand extra channels across temporal dimension
            x_extra = x_extra.unsqueeze(2).expand(
                -1, -1, T, -1, -1
            )  # (B, extra_channels, T, H, W)

            # Concatenate
            return torch.cat(
                [x_temporal, x_extra], dim=1
            )  # (B, C_base+extra_channels, T, H, W)


# ────────────────────────────────────────────────────────────────────────────
# Factory function
# ────────────────────────────────────────────────────────────────────────────


def create_unet3d_temporal_generator(
    unet_cfg,
    normalization: str = "minus1_to_plus1",
    t_past: int = 0,
    t_future: int = 0,
    n_pred_steps: int = 1,
    n_context_steps: int = None,
):
    """
    Build a UNet3DTemporalGenerator from config.

    Parameters
    ----------
    unet_cfg : OmegaConf / dict-like
        Must contain at least:
        - in_channels : int  (per-timestep channels, including noise)
        - out_channels : int
        - sample_size : [H, W]
        - layers_per_block : int
        - block_out_channels : list[int]
        - down_block_types : list[str]  (2D names, auto-converted to 3D)
        - up_block_types : list[str]    (2D names, auto-converted to 3D)
    normalization : str
        Determines the final activation.
    t_past, t_future : int
        Temporal window extents.
    n_pred_steps : int
        Number of consecutive time steps to predict (starting from current day).
        Default 1 = current day only. Set to 2+ for multi-step forecasting.
    n_context_steps : int or None
        Number of past time steps to use as context. If None, uses all (t_past).

    Returns
    -------
    UNet3DTemporalGenerator
    """
    num_timesteps = t_past + 1 + t_future

    # Validate minimum temporal length based on network depth
    # Each down block typically does 2× temporal pooling
    num_down_blocks = len(unet_cfg.down_block_types)
    min_temporal_length = 2**num_down_blocks

    if num_timesteps < min_temporal_length:
        raise ValueError(
            f"UNet3D with {num_down_blocks} down blocks requires at least "
            f"T={min_temporal_length} time steps (due to temporal pooling 2^{num_down_blocks}). "
            f"Got T={num_timesteps} (t_past={t_past}, t_future={t_future}). \n"
            f"Minimum configuration: t_past + t_future >= {min_temporal_length - 1}"
        )

    if num_timesteps < 2:
        raise ValueError(
            f"UNet3D temporal generator requires at least 2 time steps, "
            f"got t_past={t_past}, t_future={t_future} (total T={num_timesteps})."
        )

    in_channels = unet_cfg.in_channels  # per-timestep (e.g. 16 = 15 vars + 1 noise)
    out_channels = unet_cfg.out_channels

    # Build down/up block types for 3D – mirror the 2D config structure
    # Replace "DownBlock2D" → "DownBlock3D", "AttnDownBlock2D" → "CrossAttnDownBlock3D"
    def _to_3d_block(block_name_2d: str) -> str:
        mapping = {
            "DownBlock2D": "DownBlock3D",
            "AttnDownBlock2D": "CrossAttnDownBlock3D",
            "UpBlock2D": "UpBlock3D",
            "AttnUpBlock2D": "CrossAttnUpBlock3D",
        }
        if block_name_2d in mapping:
            return mapping[block_name_2d]
        raise ValueError(
            f"No 3D equivalent known for block type '{block_name_2d}'. "
            f"Known mappings: {list(mapping.keys())}"
        )

    down_block_types = tuple(_to_3d_block(b) for b in unet_cfg.down_block_types)
    up_block_types = tuple(_to_3d_block(b) for b in unet_cfg.up_block_types)

    print(
        f"Creating UNet3D temporal generator: "
        f"t_past={t_past}, t_future={t_future}, T={num_timesteps}, n_pred_steps={n_pred_steps}"
    )
    print(f"  in_channels={in_channels} (per time step), out_channels={out_channels}")
    print(f"  block_out_channels={list(unet_cfg.block_out_channels)}")
    print(f"  down_blocks={list(down_block_types)} (requires T>={min_temporal_length})")
    print(f"  up_blocks={list(up_block_types)}")

    # Find appropriate norm_num_groups (must divide in_channels evenly)
    # For 15 channels: divisors are 1, 3, 5, 15
    # Use largest divisor <= 32 for better normalization
    norm_num_groups = 1
    for divisor in [32, 16, 15, 8, 5, 3, 1]:
        if in_channels % divisor == 0:
            norm_num_groups = divisor
            break

    # Set attention_head_dim based on smallest block_out_channels
    # This ensures attention heads are properly configured
    min_channels = min(unet_cfg.block_out_channels)
    attention_head_dim = min(
        8, min_channels
    )  # Use 8 or smaller if channels are limited

    unet = UNet3DConditionModel(
        sample_size=tuple(unet_cfg.sample_size),
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=unet_cfg.layers_per_block,
        block_out_channels=tuple(unet_cfg.block_out_channels),
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        cross_attention_dim=1,  # minimal cross-attention dim (not really used)
        norm_num_groups=norm_num_groups,
        attention_head_dim=attention_head_dim,  # Required for attention blocks
    )

    # Choose activation based on normalization
    if normalization in ["m1p1_log_target", "mp1p1_input_m1p1log_target", "mp1p1_input_m1p1log_target"]:
        activation = nn.Softplus()
    else:
        activation = nn.Identity()

    return UNet3DTemporalGenerator(
        unet=unet,
        activation=activation,
        num_timesteps=num_timesteps,
        n_pred_steps=n_pred_steps,
        n_context_steps=n_context_steps,
    )


# ────────────────────────────────────────────────────────────────────────────
# Inference wrapper
# ────────────────────────────────────────────────────────────────────────────


class UNet3DWrapper:
    """
    Inference wrapper for UNet3D temporal models.

    Handles model loading, checkpoint management, normalization, upsampling,
    and prediction with proper preprocessing and denormalization for 5D tensors.

    This wrapper is specifically for UNet3D-based generators and handles:
    - Bilinear upsampling from 16x16 to 128x128 with temporal dimension
    - Noise channel addition for diffusion conditioning (5D)
    - Optional orography concatenation (5D)
    - Timestep conditioning for diffusion models
    - Temporal data reshaping from (B, C*T, H, W) to (B, C, T, H, W)

    Parameters
    ----------
    run_dir : str
        Directory containing the trained model.
    config : OmegaConf
        Model configuration object.
    checkpoint_epoch : int, str, or None
        Specific epoch to load ('best', epoch number, or None for final).
    device : torch.device or None
        Device to run model on.
    orography : torch.Tensor or None
        Optional orography tensor for conditioning (2D: H x W).
    n_pred_steps : int
        Number of future steps to predict (default 1).
    n_context_steps : int or None
        Number of past steps to use as context (default None = use all).
    """

    def __init__(
        self,
        run_dir: str,
        config,
        checkpoint_epoch=None,
        device=None,
        orography=None,
        n_pred_steps: int = 1,
        n_context_steps: int = None,
    ):
        from pathlib import Path

        # Determine checkpoint name
        if checkpoint_epoch == "best":
            checkpoint_name = "checkpoints/best_model.pt"
        elif checkpoint_epoch is not None:
            checkpoint_name = f"checkpoints/checkpoint_epoch_{checkpoint_epoch}.pt"
        else:
            checkpoint_name = "checkpoints/final_models.pt"

        self.run_dir = Path(run_dir)
        self.checkpoint_name = checkpoint_name
        self.config = config
        self.orography = orography
        self.checkpoint_epoch = checkpoint_epoch
        self.n_pred_steps = n_pred_steps
        self.n_context_steps = n_context_steps

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Extract temporal config
        self.t_past = getattr(config.data, "t_past", 0)
        self.t_future = getattr(config.data, "t_future", 0)
        self.num_timesteps = self.t_past + 1 + self.t_future

        # Load model and normalization
        self._load_model()
        self._load_normalization()

    def _load_model(self):
        """Load UNet3D temporal generator architecture and weights."""
        unet_cfg = self.config.model.generator.diffusion_unet
        normalization = self.config.data.get("normalization", "minus1_to_plus1")

        self.model = create_unet3d_temporal_generator(
            unet_cfg,
            normalization=normalization,
            t_past=self.t_past,
            t_future=self.t_future,
            n_pred_steps=self.n_pred_steps,
            n_context_steps=self.n_context_steps,
        )

        # Load checkpoint
        checkpoint_path = self.run_dir / self.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if self.checkpoint_epoch is None:
            self.checkpoint_epoch = checkpoint.get("epoch", None)

    def _load_normalization(self):
        """Load normalization parameters from checkpoint."""
        norm_path = self.run_dir / "norm_params.pt"
        if norm_path.exists():
            self.norm_params = torch.load(
                norm_path, map_location=self.device, weights_only=False
            )
        else:
            # Fallback: try to load from checkpoint
            checkpoint_path = self.run_dir / self.checkpoint_name
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.norm_params = checkpoint.get("norm_params", {})

    def _upscale_temporal_bilinear(
        self, x: torch.Tensor, target_size=(128, 128)
    ) -> torch.Tensor:
        """
        Upscale 5D temporal input using bilinear interpolation.

        Parameters
        ----------
        x : (B, C, T, H, W) or (B, C*T, H, W)

        Returns
        -------
        (B, C, T, H_out, W_out)
        """
        import torch.nn.functional as F

        # Handle 4D input (channel-stacked)
        if x.ndim == 4:
            B, CT, H, W = x.shape
            T = self.num_timesteps
            C = CT // T
            # Reshape to (B, T, C, H, W) then permute to (B, C, T, H, W)
            x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        B, C, T, H, W = x.shape

        # Reshape to (B*T, C, H, W) for 2D interpolation
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x_flat = x_flat.view(B * T, C, H, W)

        # Upsample
        x_up = F.interpolate(
            x_flat, size=target_size, mode="bilinear", align_corners=False
        )

        # Reshape back to (B, C, T, H_out, W_out)
        _, _, H_out, W_out = x_up.shape
        x_up = x_up.view(B, T, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()

        return x_up

    def _add_noise_channel_5d(
        self, x: torch.Tensor, noise_std: float = 0.2
    ) -> torch.Tensor:
        """
        Add random noise channel to 5D tensor.

        Parameters
        ----------
        x : (B, C, T, H, W)
        noise_std : float

        Returns
        -------
        (B, C+1, T, H, W)
        """
        B, C, T, H, W = x.shape
        noise = torch.randn(B, 1, T, H, W, device=x.device) * noise_std
        return torch.cat([x, noise], dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input with UNet3D-specific preprocessing.

        Performs the following steps:
        1. Reshape from (B, C*T, H, W) to (B, C, T, H, W) if needed
        2. Upscale from 16x16 to 128x128 (per time step)
        3. Concatenate orography if configured (repeated across T)
        4. Add noise channel for diffusion conditioning
        5. Generate with timestep conditioning
        6. Denormalize output

        Parameters
        ----------
        x : (B, C*T, 16, 16) or (B, C, T, 16, 16)

        Returns
        -------
        (B, n_pred_steps, 128, 128) - denormalized predictions
        """
        from ml_benchmark_spategan.train.normalize import denormalize_predictions

        x = x.to(self.device)

        with torch.no_grad():
            # Upscale with temporal dimension
            x_hr = self._upscale_temporal_bilinear(x, target_size=(128, 128))
            # Now: (B, C, T, 128, 128)

            # Concatenate orography if configured
            if (
                self.config.data.get("use_orography", False)
                and self.orography is not None
            ):
                B, C, T, H, W = x_hr.shape
                # Expand orography to (B, 1, T, H, W)
                orography_batch = (
                    self.orography.unsqueeze(0)  # (1, H, W)
                    .unsqueeze(0)  # (1, 1, H, W)
                    .unsqueeze(2)  # (1, 1, 1, H, W)
                    .repeat(B, 1, T, 1, 1)  # (B, 1, T, H, W)
                    .to(self.device)
                )
                x_hr = torch.cat([x_hr, orography_batch], dim=1)

            # Add noise channel for diffusion conditioning
            x_with_noise = self._add_noise_channel_5d(
                x_hr, noise_std=self.config.training.get("noise_std_gen", 0.05)
            )

            # Generate with timestep conditioning (timestep=0 for inference)
            timesteps = torch.zeros(x.shape[0], device=self.device)
            output = self.model(x_with_noise, timesteps)

            # Denormalize predictions
            # output shape: (B, n_pred_steps, 128, 128) or (B, 1, 128, 128)
            if output.ndim == 3:
                # Add channel dim for denormalization
                output = output.unsqueeze(1)  # (B, 1, 128, 128)
            elif output.ndim == 4 and self.n_pred_steps > 1:
                # (B, n_pred_steps, 128, 128) -> treat each step separately
                # Reshape to (B*n_pred_steps, 1, 128, 128)
                B, T_pred, H, W = output.shape
                output = output.unsqueeze(2)  # (B, T_pred, 1, H, W)
                output = output.view(B * T_pred, 1, H, W)
                output = denormalize_predictions(output, self.norm_params)
                # Reshape back
                output = output.view(B, T_pred, H, W)
                return output

            output = denormalize_predictions(output, self.norm_params)

            return output

    def to(self, device: torch.device):
        """Move model and orography to specified device."""
        self.device = device
        self.model = self.model.to(device)
        if self.orography is not None:
            self.orography = self.orography.to(device)
        return self
