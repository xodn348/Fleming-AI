"""Google Drive synchronization module using rclone."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def check_rclone_available() -> bool:
    """
    Check if rclone is available in the system.

    Returns:
        bool: True if rclone is installed and accessible, False otherwise.
    """
    return shutil.which("rclone") is not None


def sync_to_drive(
    local_path: str,
    remote_path: str,
    config_name: str = "gdrive",
    dry_run: bool = False,
    use_copy: bool = True,
) -> Optional[dict[str, str]]:
    """
    Sync local directory to Google Drive using rclone.

    Args:
        local_path: Local directory path to sync from
        remote_path: Remote path on Google Drive (e.g., "gdrive:/path/to/folder")
        config_name: rclone config name (default: "gdrive")
        dry_run: If True, perform a dry run without actual sync
        use_copy: If True, use copy (safer, doesn't delete), if False use sync (mirrors)

    Returns:
        dict with sync results or None if rclone is not available

    Raises:
        subprocess.CalledProcessError: If rclone command fails
    """
    if not check_rclone_available():
        logger.warning("rclone is not installed. Skipping sync to Google Drive.")
        return None

    local_path_obj = Path(local_path).resolve()
    if not local_path_obj.exists():
        raise ValueError(f"Local path does not exist: {local_path_obj}")

    operation = "copy" if use_copy else "sync"
    cmd = [
        "rclone",
        operation,
        str(local_path_obj),
        remote_path,
        "--verbose",
    ]

    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Successfully {operation}ed {local_path_obj} to {remote_path}")
        return {
            "status": "success",
            "local_path": str(local_path_obj),
            "remote_path": remote_path,
            "stdout": result.stdout,
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone {operation} failed: {e.stderr}")
        raise


def sync_from_drive(
    remote_path: str,
    local_path: str,
    config_name: str = "gdrive",
    dry_run: bool = False,
) -> Optional[dict[str, str]]:
    """
    Sync from Google Drive to local directory using rclone.

    Args:
        remote_path: Remote path on Google Drive (e.g., "gdrive:/path/to/folder")
        local_path: Local directory path to sync to
        config_name: rclone config name (default: "gdrive")
        dry_run: If True, perform a dry run without actual sync

    Returns:
        dict with sync results or None if rclone is not available

    Raises:
        subprocess.CalledProcessError: If rclone command fails
    """
    if not check_rclone_available():
        logger.warning("rclone is not installed. Skipping sync from Google Drive.")
        return None

    local_path_obj = Path(local_path)
    local_path_obj.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone",
        "sync",
        remote_path,
        str(local_path_obj),
        "--verbose",
    ]

    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Successfully synced {remote_path} to {local_path_obj}")
        return {
            "status": "success",
            "remote_path": remote_path,
            "local_path": str(local_path_obj),
            "stdout": result.stdout,
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone sync failed: {e.stderr}")
        raise
