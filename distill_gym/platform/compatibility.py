import subprocess
import time
import logging

from distill_gym.platform.detection import PlatformInfo, detect

logger = logging.getLogger(__name__)


def ensure_podman_ready(info: PlatformInfo | None = None) -> bool:
    if info is None:
        info = detect()

    if not info.has_podman:
        logger.error("Podman binary not found. Install Podman: https://podman.io/docs/installation")
        return False

    if info.is_windows or info.is_macos:
        if not info.podman_machine_running:
            logger.info("Starting Podman machine...")
            try:
                subprocess.run(
                    ["podman", "machine", "start"],
                    capture_output=True, text=True, timeout=120,
                )
                time.sleep(3)
                return True
            except subprocess.TimeoutExpired:
                logger.error("Podman machine start timed out")
                return False
            except OSError as e:
                logger.error(f"Failed to start Podman machine: {e}")
                return False
    return True


def resolve_container_host(info: PlatformInfo | None = None) -> str:
    if info is None:
        info = detect()

    if info.is_linux:
        return "host.containers.internal"
    if info.is_windows:
        return "host.containers.internal"
    if info.is_macos:
        return "host.containers.internal"
    return "host.containers.internal"
