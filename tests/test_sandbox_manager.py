import pytest

from distill_gym.config.schema import SandboxConfig
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager


class FakePodman:
    def __init__(self):
        self.exec_calls = []

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


@pytest.mark.asyncio
async def test_prepare_git_repository_clones_and_runs_setup():
    podman = FakePodman()
    manager = SandboxManager(podman=podman)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace/repo"))

    await manager.prepare_git_repository(
        SandboxConfig(
            repo_url="https://github.com/example/test.git",
            ref="main",
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
