"""Generator models for downscaling."""

from .deepesd import DeepESD, DeepESDWrapper
from .spategan import Generator as SpatialGANGenerator
from .spategan import SpaGANWrapper
from .unet2d import UNetWithActivation, UNetWrapper, create_unet_generator

__all__ = [
    "SpatialGANGenerator",
    "SpaGANWrapper",
    "DeepESD",
    "DeepESDWrapper",
    "UNetWithActivation",
    "UNetWrapper",
    "create_unet_generator",
]
