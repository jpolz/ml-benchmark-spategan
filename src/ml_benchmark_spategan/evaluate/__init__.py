"""
Evaluation module for ML Benchmark SpatialGAN.

This module contains evaluation metrics, diagnostics, and model comparison tools
for assessing the performance of climate downscaling models.
"""

from ml_benchmark_spategan.evaluate import scores
from ml_benchmark_spategan.evaluate.diagnostics import (
    compute_diagnostics,
    compute_model_selection_score,
)

__all__ = [
    "compute_diagnostics",
    "compute_model_selection_score",
    "scores",
]
