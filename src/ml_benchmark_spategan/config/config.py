import logging
import os
import random
import string
from datetime import datetime

import yaml
from omegaconf import DictConfig, OmegaConf

# Initialize logger
logger = logging.getLogger(__name__)


class Config:
    """
    Wrapper class for OmegaConf configuration with convenient methods.
    """

    def __init__(self, config: DictConfig, config_path: str = None):
        self._config = config
        self._config_path = config_path

    def __getattr__(self, name):
        """Delegate attribute access to the underlying OmegaConf object."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to the underlying OmegaConf object."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._config, name, value)

    def save(self):
        """
        Save configuration to a YAML file.

        Args:
            file_path: Path to save the config. Required to prevent accidental overwrites.
        """
        save_config_to_yaml(self._config, self._config.config_path)
        logger.info(f"Configuration saved to: {self._config.config_path}")

    def to_dict(self):
        """Convert configuration to a regular Python dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)

    def to_yaml(self):
        """Convert configuration to YAML string."""
        return OmegaConf.to_yaml(self._config)

    def __repr__(self):
        return f"Config({OmegaConf.to_yaml(self._config)})"


def generate_run_id(length=8):
    """
    Generate a run ID in format YYYYMMDD_HHMM_<random_string>.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    letters = string.ascii_lowercase + string.digits
    random_suffix = "".join(random.choice(letters) for i in range(length))
    return f"{timestamp}_{random_suffix}"


def setup_logging(config: dict = None):
    """
    Set up logging configuration based on config.
    """
    if config and "logging" in config:
        level = getattr(logging, config.logging.get("level", "INFO").upper())
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config_from_yaml(file_path: str) -> Config:
    """
    Load configuration parameters from a YAML file. Convert to omega config.

    Returns:
        Config: A Config object wrapping the OmegaConf configuration.
    """
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    omega_config = OmegaConf.create(config_dict)
    setup_logging(omega_config)
    return Config(omega_config, file_path)


def save_config_to_yaml(config: dict, file_path: str):
    """
    Save configuration parameters to a YAML file.
    """
    # Convert OmegaConf to regular dict if needed
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)

    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    return None


def setup_experiment_directory(base_dir: str, run_id: str) -> str:
    """
    Set up a directory for the experiment.
    """
    experiment_dir = os.path.join(base_dir / "runs", run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    # sample plots subdirectory
    os.makedirs(os.path.join(experiment_dir, "sample_plots"), exist_ok=True)
    # checkpoints subdirectory
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    return experiment_dir


def print_config(config):
    """
    Log the configuration in a readable format.
    """
    if isinstance(config, Config):
        logger.info("Configuration:\n" + config.to_yaml())
    else:
        logger.info("Configuration:\n" + OmegaConf.to_yaml(config))


def set_up_run(project_base: str) -> Config:
    """
    Set up the run directory and save the configuration.

    Returns:
        Config: The configuration object with run_id and run_dir added.
    """
    cf = load_config_from_yaml(os.path.join(project_base, "config.yml"))
    print_config(cf)
    run_id = generate_run_id()
    logger.info(f"Run ID: {run_id}")
    # add run_id to the config
    cf.logging.run_id = run_id
    run_dir = setup_experiment_directory(project_base, run_id)
    # add run_dir to the config
    cf.logging.run_dir = run_dir
    # Save the configuration to the run directory
    cf.config_path = os.path.join(run_dir, "config.yaml")
    cf.save()

    logger.info(f"Run directory set up at: {run_dir}")

    return cf


# Removed: save_norm_params - functionality moved to utils.normalize.save_normalization_params
