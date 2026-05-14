import pytest

from distill_gym.config.schema import SandboxConfig
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.sandbox.runtime import SandboxRuntime


class FakeRuntime(SandboxRuntime):
    def __init__(self):
        self.exec_calls = []
        self.copy_to_calls = []

    def start(self, spec: SandboxSpec) -> str:
        return "container-id"

    def exec(self, container_id, command, timeout=300, workdir=None):
        self.exec_calls.append(
            {
                "container_id": container_id,
                "command": command,
                "timeout": timeout,
                "workdir": workdir,
            }
        )
        return 0, "", ""

    def copy_to(self, container_id, source, target):
        self.copy_to_calls.append((container_id, source, target))

    def copy_from(self, container_id, source, target):
        pass

    def stop(self, container_id):
        pass

    def remove(self, container_id):
        pass

    def cleanup_resources(self, label=""):
        return {"containers": 0, "volumes": 0, "networks": 0}


@pytest.mark.asyncio
async def test_prepare_git_repository_clones_and_runs_setup():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
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

    commands = [call["command"] for call in runtime.exec_calls]
    assert commands[0] == "mkdir -p /workspace/repo"
    assert "command -v git" in commands[1]
    assert commands[2] == (
        "git clone --branch main --depth 1 "
        "https://github.com/example/test.git /workspace/repo"
    )
    assert commands[3] == "echo setup"
    assert runtime.exec_calls[3]["workdir"] == "/workspace/repo"


@pytest.mark.asyncio
async def test_prepare_git_repository_uses_host_git_cache(monkeypatch, tmp_path):
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
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

    commands = [call["command"] for call in runtime.exec_calls]
    assert commands == ["mkdir -p /workspace", "echo setup"]
    assert runtime.copy_to_calls[0][2] == "/workspace"
