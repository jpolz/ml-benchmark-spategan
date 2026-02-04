"""Common utilities shared across SpaGAN packages."""

__version__ = "0.1.0"

# Domain configurations
DOMAINS = {
    "SA": {"grid_size": (128, 128), "coord_type": "latlon"},
    "NZ": {"grid_size": (128, 128), "coord_type": "latlon"},
    "ALPS": {"grid_size": (128, 128), "coord_type": "xy"},
}

# Experiment configurations
EXPERIMENTS = {
    "ESD_pseudo_reality": {
        "training_years": (1961, 1980),
        "test_years": (1975, 1980),
    },
    "Emulator_hist_future": {
        "training_years": [(1961, 1980), (2080, 2099)],
        "test_years": (2090, 2099),
    },
}

# Variable configurations
VARIABLES = ["tasmax", "pr"]
