import pytest
from distill_gym.platform.detection import detect, PlatformInfo
from distill_gym.platform.compatibility import resolve_container_host


class TestPlatformDetection:
    def test_detect_returns_platform_info(self):
        info = detect()
        assert isinstance(info, PlatformInfo)
        assert info.system in ("windows", "darwin", "linux")
        assert info.arch in ("x86_64", "aarch64")

    def test_platform_info_properties(self):
        linux = PlatformInfo(system="linux", arch="x86_64")
        assert linux.is_linux
        assert not linux.is_windows
        assert not linux.is_macos

        macos = PlatformInfo(system="darwin", arch="aarch64")
        assert macos.is_macos
        assert not macos.is_linux

        win = PlatformInfo(system="windows", arch="x86_64")
        assert win.is_windows
        assert not win.is_linux
        assert not win.is_macos

    def test_resolve_container_host(self):
        host = resolve_container_host()
        assert isinstance(host, str)
        assert host == "host.containers.internal"

    def test_podman_version_default_empty(self):
        info = PlatformInfo(system="linux", arch="x86_64")
        assert info.podman_version == ""
        assert info.docker_version == ""

    def test_has_runtime_defaults_false(self):
        info = PlatformInfo(system="linux", arch="x86_64")
        assert not info.has_podman
        assert not info.has_docker
        assert not info.podman_machine_running
        assert not info.wsl_available
