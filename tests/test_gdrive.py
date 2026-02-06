"""Tests for Google Drive synchronization module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from src.storage.gdrive import (
    check_rclone_available,
    sync_to_drive,
    sync_from_drive,
)


class TestRcloneAvailable:
    """Tests for rclone availability check."""

    def test_check_rclone_available_when_installed(self):
        """Test that check_rclone_available returns True when rclone is installed."""
        with patch("shutil.which", return_value="/usr/local/bin/rclone"):
            assert check_rclone_available() is True

    def test_check_rclone_available_when_not_installed(self):
        """Test that check_rclone_available returns False when rclone is not installed."""
        with patch("shutil.which", return_value=None):
            assert check_rclone_available() is False


class TestSyncToDrive:
    """Tests for sync_to_drive function."""

    def test_sync_to_drive_success(self):
        """Test successful sync to Google Drive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir)
            (local_path / "test.txt").write_text("test content")

            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Sync completed",
                        returncode=0,
                    )

                    result = sync_to_drive(
                        str(local_path),
                        "gdrive:/backup",
                    )

                    assert result is not None
                    assert result["status"] == "success"
                    # Compare resolved paths to handle macOS /private symlink
                    assert Path(result["local_path"]).resolve() == local_path.resolve()
                    assert result["remote_path"] == "gdrive:/backup"

    def test_sync_to_drive_rclone_not_available(self):
        """Test sync_to_drive when rclone is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=False):
                result = sync_to_drive(
                    tmpdir,
                    "gdrive:/backup",
                )
                assert result is None

    def test_sync_to_drive_local_path_not_exists(self):
        """Test sync_to_drive with non-existent local path."""
        with patch("src.storage.gdrive.check_rclone_available", return_value=True):
            with pytest.raises(ValueError, match="Local path does not exist"):
                sync_to_drive(
                    "/nonexistent/path",
                    "gdrive:/backup",
                )

    def test_sync_to_drive_rclone_fails(self):
        """Test sync_to_drive when rclone command fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.CalledProcessError(
                        1,
                        "rclone",
                        stderr="Authentication failed",
                    )

                    with pytest.raises(subprocess.CalledProcessError):
                        sync_to_drive(tmpdir, "gdrive:/backup")

    def test_sync_to_drive_dry_run(self):
        """Test sync_to_drive with dry_run flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Dry run completed",
                        returncode=0,
                    )

                    sync_to_drive(tmpdir, "gdrive:/backup", dry_run=True)

                    # Verify --dry-run flag was passed
                    call_args = mock_run.call_args[0][0]
                    assert "--dry-run" in call_args


class TestSyncFromDrive:
    """Tests for sync_from_drive function."""

    def test_sync_from_drive_success(self):
        """Test successful sync from Google Drive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Sync completed",
                        returncode=0,
                    )

                    result = sync_from_drive(
                        "gdrive:/backup",
                        tmpdir,
                    )

                    assert result is not None
                    assert result["status"] == "success"
                    assert result["remote_path"] == "gdrive:/backup"
                    assert result["local_path"] == tmpdir

    def test_sync_from_drive_rclone_not_available(self):
        """Test sync_from_drive when rclone is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=False):
                result = sync_from_drive(
                    "gdrive:/backup",
                    tmpdir,
                )
                assert result is None

    def test_sync_from_drive_creates_local_path(self):
        """Test that sync_from_drive creates local path if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "new_dir" / "nested"

            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Sync completed",
                        returncode=0,
                    )

                    sync_from_drive(
                        "gdrive:/backup",
                        str(local_path),
                    )

                    # Verify directory was created
                    assert local_path.exists()

    def test_sync_from_drive_rclone_fails(self):
        """Test sync_from_drive when rclone command fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.CalledProcessError(
                        1,
                        "rclone",
                        stderr="Remote path not found",
                    )

                    with pytest.raises(subprocess.CalledProcessError):
                        sync_from_drive("gdrive:/backup", tmpdir)

    def test_sync_from_drive_dry_run(self):
        """Test sync_from_drive with dry_run flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.storage.gdrive.check_rclone_available", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout="Dry run completed",
                        returncode=0,
                    )

                    sync_from_drive(
                        "gdrive:/backup",
                        tmpdir,
                        dry_run=True,
                    )

                    # Verify --dry-run flag was passed
                    call_args = mock_run.call_args[0][0]
                    assert "--dry-run" in call_args
