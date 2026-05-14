import subprocess
from types import SimpleNamespace

from distill_gym.config.schema import Config, SandboxEngine
from distill_gym.platform.detection import PlatformInfo
from distill_gym.proxy.addressing import (
    proxy_base_url_for_sandbox,
    proxy_connect_host_for_sandbox,
    proxy_listen_host_for_sandbox,
)


def _platform(system: str) -> PlatformInfo:
    return PlatformInfo(system=system, arch="x86_64")


def test_windows_loopback_proxy_binds_all_interfaces_for_sandbox():
    config = Config()

    assert proxy_listen_host_for_sandbox(config, _platform("windows")) == "0.0.0.0"


def test_non_windows_keeps_configured_listen_host():
    config = Config()

    assert proxy_listen_host_for_sandbox(config, _platform("linux")) == "127.0.0.1"


def test_windows_docker_uses_docker_host_alias():
    config = Config()
    config.sandbox.engine = SandboxEngine.docker

    assert proxy_connect_host_for_sandbox(config, _platform("windows")) == "host.docker.internal"
    assert proxy_base_url_for_sandbox(config, _platform("windows")) == "http://host.docker.internal:5002/v1"


def test_windows_podman_wsl_machine_uses_wsl_default_gateway(monkeypatch):
    def fake_run(args, **kwargs):
        if args == ["podman", "machine", "list", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Name":"podman-machine-default","Running":true,"VMType":"wsl"}]',
                stderr="",
            )
        if args == ["podman", "machine", "ssh", "ip route show default"]:
            return SimpleNamespace(
                returncode=0,
                stdout="default via 172.22.224.1 dev eth0 proto kernel\n",
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    config = Config()
    config.sandbox.engine = SandboxEngine.podman

    assert proxy_connect_host_for_sandbox(config, _platform("windows")) == "172.22.224.1"
    assert proxy_base_url_for_sandbox(config, _platform("windows")) == "http://172.22.224.1:5002/v1"


def test_windows_podman_wsl_machine_falls_back_when_gateway_is_unavailable(monkeypatch):
    def fake_run(args, **kwargs):
        if args == ["podman", "machine", "list", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Name":"podman-machine-default","Running":true,"VMType":"wsl"}]',
                stderr="",
            )
        if args == ["podman", "machine", "ssh", "ip route show default"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="failed")
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    config = Config()
    config.sandbox.engine = SandboxEngine.podman

    assert proxy_connect_host_for_sandbox(config, _platform("windows")) == "host.containers.internal"


def test_windows_podman_non_wsl_machine_uses_podman_host_alias(monkeypatch):
    def fake_run(args, **kwargs):
        if args == ["podman", "machine", "list", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Name":"podman-machine-default","Running":true,"VMType":"qemu"}]',
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", fake_run)

    config = Config()
    config.sandbox.engine = SandboxEngine.podman

    assert proxy_connect_host_for_sandbox(config, _platform("windows")) == "host.containers.internal"


def test_configured_sandbox_host_overrides_auto_detection():
    config = Config()
    config.logging_proxy.sandbox_host = "172.20.16.1"

    assert proxy_base_url_for_sandbox(config, _platform("windows")) == "http://172.20.16.1:5002/v1"
