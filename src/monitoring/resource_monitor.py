import logging
import os
import psutil

logger = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def get_system_stats(self) -> dict:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "process_memory_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        }

    def is_system_healthy(self) -> tuple[bool, str]:
        stats = self.get_system_stats()

        if stats["cpu_percent"] > self.cpu_threshold:
            return False, f"CPU usage {stats['cpu_percent']:.1f}% exceeds {self.cpu_threshold}%"

        if stats["memory_percent"] > self.memory_threshold:
            return (
                False,
                f"Memory usage {stats['memory_percent']:.1f}% exceeds {self.memory_threshold}%",
            )

        if stats["disk_percent"] > self.disk_threshold:
            return False, f"Disk usage {stats['disk_percent']:.1f}% exceeds {self.disk_threshold}%"

        return True, "System healthy"

    def log_stats(self):
        stats = self.get_system_stats()
        logger.info(
            f"Resource usage: CPU {stats['cpu_percent']:.1f}%, "
            f"Memory {stats['memory_percent']:.1f}%, "
            f"Disk {stats['disk_percent']:.1f}%, "
            f"Process {stats['process_memory_mb']:.1f}MB"
        )

    def should_backoff(self) -> bool:
        healthy, reason = self.is_system_healthy()
        if not healthy:
            logger.warning(f"Resource constraint detected: {reason}")
            return True
        return False
