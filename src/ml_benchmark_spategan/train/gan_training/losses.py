"""Configurable loss manager for GAN training."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class GANLossManager:
    """
    Manages multiple loss functions with configurable weights for GAN training.

    This class encapsulates all loss computation logic, making it easy to:
    - Configure loss weights from config
    - Add new loss functions
    - Compute weighted combined losses
    - Track individual loss components

    Args:
        loss_weights: Dictionary mapping loss names to weights
        gan_criterion: Loss function for GAN training (e.g., BCEWithLogitsLoss)
        fss_criterion: FSS loss function (optional)
        use_fss: Whether to use FSS loss
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        gan_criterion: nn.Module,
        fss_criterion: Optional[nn.Module] = None,
        use_fss: bool = False,
    ):
        self.weights = loss_weights
        self.gan_criterion = gan_criterion
        self.fss_criterion = fss_criterion
        self.use_fss = use_fss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def compute_generator_loss(
        self,
        gen_ensemble: torch.Tensor,
        target: torch.Tensor,
        disc_fake_output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined generator loss.

        Args:
            gen_ensemble: Generator ensemble outputs (B, N_ensemble, H, W)
            target: Ground truth (B, 1, H, W)
            disc_fake_output: Discriminator output on fake samples (optional)

        Returns:
            Combined loss tensor and dictionary of individual loss components
        """
        losses = {}
        total_loss = 0.0

        # Compute ensemble mean
        gen_ensemble_mean = gen_ensemble.mean(dim=1, keepdim=True)

        # L1 loss
        if self.weights.get("l1", 0.0) > 0.0:
            l1_loss = self.l1_loss(gen_ensemble_mean, target)
            losses["l1"] = l1_loss.item()
            total_loss = total_loss + self.weights["l1"] * l1_loss

        # MSE loss
        if self.weights.get("mse", 0.0) > 0.0:
            mse_loss = self.mse_loss(gen_ensemble_mean, target)
            losses["mse"] = mse_loss.item()
            total_loss = total_loss + self.weights["mse"] * mse_loss

        # GAN loss
        if self.weights.get("gan", 0.0) > 0.0 and disc_fake_output is not None:
            # BCE loss: fool discriminator
            gan_loss = self.gan_criterion(
                disc_fake_output, torch.ones_like(disc_fake_output)
            )
            losses["gan"] = gan_loss.item()
            total_loss = total_loss + self.weights["gan"] * gan_loss

        # Diversity bonus (maximize variance across ensemble members)
        if self.weights.get("diversity", 0.0) > 0.0 and gen_ensemble.shape[1] > 1:
            # Compute variance across ensemble dimension
            # Negative sign: we want to MAXIMIZE variance (minimize negative variance)
            ensemble_variance = gen_ensemble.var(dim=1, unbiased=False).mean()
            diversity_loss = -ensemble_variance
            losses["diversity"] = diversity_loss.item()
            losses["ensemble_variance"] = ensemble_variance.item()  # For monitoring
            total_loss = total_loss + self.weights["diversity"] * diversity_loss
        else:
            losses["diversity"] = 0.0
            losses["ensemble_variance"] = 0.0

        # FSS - always compute as a diagnostic metric
        if self.fss_criterion is not None:
            fss_loss = self.fss_criterion(gen_ensemble, target)
            losses["fss"] = fss_loss.item()
            # Only add to total loss if using FSS as a loss function
            if self.use_fss and self.weights.get("fss", 0.0) > 0.0:
                total_loss = total_loss + self.weights["fss"] * fss_loss
        else:
            # If no FSS criterion provided, set to 0
            losses["fss"] = 0.0

        return total_loss, losses

    def compute_discriminator_loss(
        self,
        disc_real_output: torch.Tensor,
        disc_fake_output: torch.Tensor,
        use_label_smoothing: bool = True,
        gradient_penalty: Optional[torch.Tensor] = None,
        gp_weight: float = 10.0,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss (real + fake + gradient penalty).

        Args:
            disc_real_output: Discriminator output on real samples
            disc_fake_output: Discriminator output on fake samples
            use_label_smoothing: Whether to apply label smoothing to real labels
            gradient_penalty: Precomputed gradient penalty (optional)
            gp_weight: Weight for gradient penalty term

        Returns:
            Combined loss tensor and dictionary of loss components
        """
        # Real loss with optional label smoothing
        if use_label_smoothing:
            real_labels = 0.8 + 0.2 * torch.rand_like(disc_real_output)
        else:
            real_labels = torch.ones_like(disc_real_output)

        disc_real_loss = self.gan_criterion(disc_real_output, real_labels)

        # Fake loss
        disc_fake_loss = self.gan_criterion(
            disc_fake_output, torch.zeros_like(disc_fake_output)
        )

        total_loss = disc_real_loss + disc_fake_loss

        losses = {
            "real": disc_real_loss.item(),
            "fake": disc_fake_loss.item(),
        }

        # Add gradient penalty if provided
        if gradient_penalty is not None:
            gp_loss = gradient_penalty * gp_weight
            total_loss = total_loss + gp_loss
            losses["gp"] = gp_loss.item()

        return total_loss, losses
