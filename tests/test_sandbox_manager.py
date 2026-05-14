import pytest

from distill_gym.config.schema import SandboxConfig
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.sandbox.runtime import SandboxRuntime


class FakeRuntime(SandboxRuntime):
    def __init__(self):
        self.exec_calls = []
        self.copy_to_calls = []

    async def start(self, spec: SandboxSpec) -> str:
        return "container-id"

    async def exec(self, container_id, command, timeout=300, workdir=None):
        self.exec_calls.append(
            {
                "container_id": container_id,
                "command": command,
                "timeout": timeout,
                "workdir": workdir,
            }
        )
        return 0, "", ""

    async def copy_to(self, container_id, source, target):
        self.copy_to_calls.append((container_id, source, target))

    async def copy_from(self, container_id, source, target):
        pass

    async def stop(self, container_id):
        pass

    async def remove(self, container_id):
        pass

    async def cleanup_resources(self, label=""):
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
    prepare_commands = [c for c in commands if c != f"mkdir -p /workspace/repo"]
    assert "mkdir -p /workspace/repo" in commands[0] or commands[1] == "mkdir -p /workspace/repo"
    assert "command -v git" in prepare_commands[0]
    assert prepare_commands[1] == (
        "git clone --branch main --depth 1 "
        "https://github.com/example/test.git /workspace/repo"
    )
    assert prepare_commands[2] == "echo setup"


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
    workdir_mkdirs = [c for c in commands if c.startswith("mkdir -p /workspace")]
    assert any("mkdir -p /workspace" in c for c in workdir_mkdirs)
    assert "echo setup" in commands
    assert runtime.copy_to_calls[0][2] == "/workspace"
