import json
import logging
import re
import subprocess

from distill_gym.config.schema import Config
from distill_gym.platform.detection import PlatformInfo


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_FALLBACK_PODMAN_HOST = "host.containers.internal"

logger = logging.getLogger(__name__)


def proxy_listen_host_for_sandbox(config: Config, platform: PlatformInfo) -> str:
    """Return a bind host reachable from sandbox runtimes on the current host."""
    listen_host = config.logging_proxy.listen_host
    if platform.is_windows and listen_host.lower() in _LOOPBACK_HOSTS:
        return "0.0.0.0"
    return listen_host


def proxy_connect_host_for_sandbox(config: Config, platform: PlatformInfo) -> str:
    configured = config.logging_proxy.sandbox_host
    if configured != "auto":
        return configured

    if platform.is_windows and config.sandbox.engine.value == "docker":
        return "host.docker.internal"
    if platform.is_windows and config.sandbox.engine.value == "podman":
        return _windows_podman_connect_host()
    return "host.containers.internal"


def proxy_base_url_for_sandbox(config: Config, platform: PlatformInfo) -> str:
    host = proxy_connect_host_for_sandbox(config, platform)
    return f"http://{host}:{config.logging_proxy.listen_port}/v1"


def _windows_podman_connect_host() -> str:
    if not _podman_machine_uses_wsl():
        return _FALLBACK_PODMAN_HOST

    gateway = _podman_machine_default_gateway()
    if gateway:
        return gateway

    logger.warning(
        "Could not resolve the WSL default gateway for Podman machine; "
        "falling back to %s. If the sandbox cannot reach the logging proxy, "
        "set logging_proxy.sandbox_host explicitly.",
        _FALLBACK_PODMAN_HOST,
    )
    return _FALLBACK_PODMAN_HOST


def _podman_machine_uses_wsl() -> bool:
    try:
        result = subprocess.run(
            ["podman", "machine", "list", "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0 or not result.stdout.strip():
        return False

    try:
        machines = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False

    if not isinstance(machines, list):
        return False

    for machine in machines:
        if not isinstance(machine, dict):
            continue
        vm_type = str(machine.get("VMType", "")).lower()
        if vm_type == "wsl" and machine.get("Running", True):
            return True
    return False


def _podman_machine_default_gateway() -> str:
    try:
        result = subprocess.run(
            ["podman", "machine", "ssh", "ip route show default"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    if result.returncode != 0:
        return ""

    match = re.search(r"\bdefault\s+via\s+([0-9a-fA-F:.]+)\b", result.stdout)
    if not match:
        return ""
    return match.group(1)
