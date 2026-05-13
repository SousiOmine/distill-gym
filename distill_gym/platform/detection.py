import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


_ENCODING = "utf-8"
_ERRORS = "replace"


@dataclass
class PlatformInfo:
    system: str
    arch: str
    has_podman: bool = False
    has_docker: bool = False
    podman_machine_running: bool = False
    wsl_available: bool = False
    podman_version: str = ""
    docker_version: str = ""

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_macos(self) -> bool:
        return self.system == "darwin"

    @property
    def is_linux(self) -> bool:
        return self.system == "linux"


def _check_binary(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if not path:
        return False, ""
    try:
        result = subprocess.run(
            [name, "--version"],
            capture_output=True, text=True, timeout=10,
            encoding=_ENCODING, errors=_ERRORS,
        )
        version = result.stdout.strip() or result.stderr.strip()
        return True, version
    except (subprocess.TimeoutExpired, OSError):
        return True, ""


def _check_podman_machine() -> bool:
    try:
        result = subprocess.run(
            ["podman", "machine", "list", "--format", "json"],
            capture_output=True, text=True, timeout=10,
            encoding=_ENCODING, errors=_ERRORS,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False
        machines = json.loads(result.stdout)
        return any(m.get("Running", False) for m in machines)
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return False


def _check_wsl() -> bool:
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            capture_output=True, text=True, timeout=10,
            encoding=_ENCODING, errors=_ERRORS,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False


def detect() -> PlatformInfo:
    system_map = {"win32": "windows", "darwin": "darwin"}
    system = system_map.get(sys.platform, "linux")

    arch = "x86_64"
    if system == "windows":
        raw_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "")
        if raw_arch.lower() in ("arm64", "aarch64"):
            arch = "aarch64"
    else:
        import platform as _platform
        machine = _platform.machine()
        if machine.lower() in ("aarch64", "arm64"):
            arch = "aarch64"

    has_podman, podman_ver = _check_binary("podman")
    has_docker, docker_ver = _check_binary("docker")

    info = PlatformInfo(
        system=system,
        arch=arch,
        has_podman=has_podman,
        has_docker=has_docker,
        podman_version=podman_ver,
        docker_version=docker_ver,
    )

    if has_podman and system in ("windows", "darwin"):
        info.podman_machine_running = _check_podman_machine()

    if system == "windows":
        info.wsl_available = _check_wsl()

    return info
