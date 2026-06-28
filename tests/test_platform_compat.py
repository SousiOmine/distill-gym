import subprocess

import pytest
from unittest import mock

from distill_gym.platform.compatibility import ensure_podman_ready, resolve_container_host
from distill_gym.platform.detection import PlatformInfo


class TestEnsurePodmanReady:
    def test_no_podman_returns_false(self):
        info = PlatformInfo(system="linux", arch="x86_64", has_podman=False)
        assert ensure_podman_ready(info) is False

    def test_linux_always_true(self):
        info = PlatformInfo(system="linux", arch="x86_64", has_podman=True)
        assert ensure_podman_ready(info) is True

    @mock.patch("distill_gym.platform.compatibility.subprocess.run")
    def test_windows_machine_not_running_starts_machine(self, mock_run):
        info = PlatformInfo(system="windows", arch="x86_64", has_podman=True, podman_machine_running=False)
        result = ensure_podman_ready(info)
        assert result is True
        mock_run.assert_called_once_with(
            ["podman", "machine", "start"],
            capture_output=True, text=True, timeout=120,
        )

    @mock.patch("distill_gym.platform.compatibility.subprocess.run")
    def test_windows_machine_running_no_start(self, mock_run):
        info = PlatformInfo(system="windows", arch="x86_64", has_podman=True, podman_machine_running=True)
        result = ensure_podman_ready(info)
        assert result is True
        mock_run.assert_not_called()

    @mock.patch("distill_gym.platform.compatibility.subprocess.run")
    def test_windows_machine_start_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="podman machine start", timeout=120)
        info = PlatformInfo(system="windows", arch="x86_64", has_podman=True, podman_machine_running=False)
        result = ensure_podman_ready(info)
        assert result is False

    @mock.patch("distill_gym.platform.compatibility.subprocess.run")
    def test_windows_machine_start_oserror(self, mock_run):
        mock_run.side_effect = OSError("mock error")
        info = PlatformInfo(system="windows", arch="x86_64", has_podman=True, podman_machine_running=False)
        result = ensure_podman_ready(info)
        assert result is False

    @mock.patch("distill_gym.platform.compatibility.detect")
    def test_info_none_calls_detect(self, mock_detect):
        mock_detect.return_value = PlatformInfo(system="linux", arch="x86_64", has_podman=True)
        ensure_podman_ready()
        mock_detect.assert_called_once()


class TestResolveContainerHost:
    def test_linux(self):
        info = PlatformInfo(system="linux", arch="x86_64")
        assert resolve_container_host(info) == "host.containers.internal"

    def test_windows(self):
        info = PlatformInfo(system="windows", arch="x86_64")
        assert resolve_container_host(info) == "host.containers.internal"

    def test_macos(self):
        info = PlatformInfo(system="darwin", arch="aarch64")
        assert resolve_container_host(info) == "host.containers.internal"

    @mock.patch("distill_gym.platform.compatibility.detect")
    def test_info_none_calls_detect(self, mock_detect):
        mock_detect.return_value = PlatformInfo(system="linux", arch="x86_64")
        resolve_container_host()
        mock_detect.assert_called_once()
