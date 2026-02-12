"""
Utility functions for experiment infrastructure.
Seed management, device detection, logging, and disk checks.
"""

import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Get best available device with MPS preference and CPU fallback.

    Sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to avoid OOM on Apple Silicon.

    Returns:
        torch.device("mps") if available, else torch.device("cpu")
    """
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info(f"Using device: {device} (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info(f"Using device: {device} (MPS not available)")

    return device


def setup_logging(log_file: Path) -> None:
    """
    Setup logging to file and console.

    Args:
        log_file: Path to log file (parent directory created if needed)
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Logging to {log_file}")


def check_disk_space(min_gb: float = 5.0) -> None:
    """
    Check available disk space and raise if insufficient.

    Args:
        min_gb: Minimum required free space in GB

    Raises:
        RuntimeError: If available disk space is below min_gb
    """
    usage = shutil.disk_usage(Path.home())
    free_gb = usage.free / (1024**3)

    if free_gb < min_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.1f} GB free, {min_gb:.1f} GB required"
        )
    logger.info(f"Disk space OK: {free_gb:.1f} GB free")


def load_config(config_path: Path) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config
