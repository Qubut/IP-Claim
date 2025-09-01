from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf
from returns.io import impure


def load_config(path: Path) -> DictConfig | ListConfig:
    """Load and merge configuration from CLI and config file."""
    user_config = OmegaConf.from_cli()
    return OmegaConf.merge(
        OmegaConf.load(user_config.get('config', path)),
        user_config,
    )
