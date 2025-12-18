"""Base classes for model interface consistency."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import xarray as xr


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.

    All models should inherit from this class and implement:
    - forward: standard PyTorch forward pass
    - train_step: single training iteration
    - predict_step: inference without denormalization
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Standard PyTorch forward pass."""
        pass

    @abstractmethod
    def train_step(
        self,
        batch,
        optimizers,
        criterion,
        scaler,
        config,
        **kwargs,
    ) -> dict:
        """
        Perform a single training step.

        Args:
            batch: Tuple of (input, target) tensors
            optimizers: Dict of optimizers (e.g., {'generator': opt_g, 'discriminator': opt_d})
            criterion: Loss function(s)
            scaler: Gradient scaler for mixed precision
            config: Configuration object
            **kwargs: Additional model-specific arguments

        Returns:
            Dict with loss values (e.g., {'loss': 0.5, 'gen_loss': 0.3, 'disc_loss': 0.2})
        """
        pass

    @abstractmethod
    def predict_step(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform prediction without denormalization.

        Args:
            x: Input tensor
            **kwargs: Additional model-specific arguments

        Returns:
            Raw model output (normalized)
        """
        pass


class BaseWrapper(ABC):
    """
    Abstract base class for model wrappers.

    Handles model loading, normalization parameters, and prediction with denormalization.
    """

    def __init__(
        self,
        run_dir: Path,
        checkpoint_name: str = "final_model.pth",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize wrapper.

        Args:
            run_dir: Directory containing model checkpoint and normalization params
            checkpoint_name: Name of checkpoint file
            device: Device to run model on
        """
        self.run_dir = Path(run_dir)
        self.checkpoint_name = checkpoint_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Will be set by subclasses
        self.model = None
        self.config = None
        self.y_min = None
        self.y_max = None
        self.y_min_log = None
        self.y_max_log = None

    @abstractmethod
    def _load_model(self):
        """Load model architecture and weights."""
        pass

    def _load_normalization(self):
        """Load normalization parameters from run directory."""
        ymin_path = self.run_dir / "y_min.nc"
        ymax_path = self.run_dir / "y_max.nc"
        ymin_log_path = self.run_dir / "y_min_log.nc"
        ymax_log_path = self.run_dir / "y_max_log.nc"

        if ymin_path.exists() and ymax_path.exists():
            self.y_min = xr.open_dataarray(ymin_path)
            self.y_max = xr.open_dataarray(ymax_path)

        if ymin_log_path.exists() and ymax_log_path.exists():
            self.y_min_log = xr.open_dataarray(ymin_log_path)
            self.y_max_log = xr.open_dataarray(ymax_log_path)

    def _build_norm_params(self) -> dict:
        """Build norm_params dict for denormalization."""
        norm_params = {
            "normalization": self.config.data.get("normalization", "minus1_to_plus1")
        }

        if self.y_min is not None:
            norm_params["y_min"] = self.y_min
        if self.y_max is not None:
            norm_params["y_max"] = self.y_max
        if self.y_min_log is not None:
            norm_params["y_min_log"] = self.y_min_log
        if self.y_max_log is not None:
            norm_params["y_max_log"] = self.y_max_log

        return norm_params

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions with denormalization.

        Args:
            x: Input tensor

        Returns:
            Denormalized predictions
        """
        pass

    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
