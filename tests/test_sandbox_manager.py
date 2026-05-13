import pytest

from distill_gym.config.schema import SandboxConfig
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager


class FakePodman:
    def __init__(self):
        self.exec_calls = []
        self.copy_to_calls = []

    def container_run(self, spec: SandboxSpec) -> str:
        return "container-id"

    def container_exec(self, container_id, command, timeout=300, workdir=None):
        self.exec_calls.append(
            {
                "container_id": container_id,
                "command": command,
                "timeout": timeout,
                "workdir": workdir,
            }
        )
        return 0, "", ""

    def container_cp_to(self, container_id, source, target):
        self.copy_to_calls.append((container_id, source, target))


@pytest.mark.asyncio
async def test_prepare_git_repository_clones_and_runs_setup():
    podman = FakePodman()
    manager = SandboxManager(podman=podman)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace/repo"))

    await manager.prepare_git_repository(
        SandboxConfig(
            repo_url="https://github.com/example/test.git",
            ref="main",
            use_git_cache=False,
            workdir="/workspace/repo",
            setup=["echo setup"],
        )
    )

    commands = [call["command"] for call in podman.exec_calls]
    assert commands[0] == "mkdir -p /workspace/repo"
    assert "command -v git" in commands[1]
    assert commands[2] == (
        "git clone --branch main --depth 1 "
        "https://github.com/example/test.git /workspace/repo"
    )
    assert commands[3] == "echo setup"
    assert podman.exec_calls[3]["workdir"] == "/workspace/repo"


@pytest.mark.asyncio
async def test_prepare_git_repository_uses_host_git_cache(monkeypatch, tmp_path):
    podman = FakePodman()
    manager = SandboxManager(podman=podman)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace/repo"))

    def fake_clone_from_mirror(repo_url, target, ref="main"):
        target.mkdir(parents=True)
        (target / "README.md").write_text("cached")

    monkeypatch.setattr("distill_gym.sandbox.manager.clone_from_mirror", fake_clone_from_mirror)

    await manager.prepare_git_repository(
        SandboxConfig(
            repo_url="https://github.com/example/test.git",
            ref="main",
            workdir="/workspace/repo",
            setup=["echo setup"],
        )
    )

    commands = [call["command"] for call in podman.exec_calls]
    assert commands == ["mkdir -p /workspace", "echo setup"]
    assert podman.copy_to_calls[0][2] == "/workspace"
