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
async def test_start_executes_git_clone_and_setup_steps():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace/repo",
        steps=[
            {"type": "git_clone", "args": {"repo": "https://github.com/example/test.git", "ref": "main", "dest": "/workspace/repo", "use_cache": False}},
            {"type": "run", "args": {"command": "echo setup"}},
        ],
    )
    await manager.start(spec)

    commands = [call["command"] for call in runtime.exec_calls]
    assert "mkdir -p /workspace/repo" in commands[0]
    cmd_text = " ".join(commands)
    assert "command -v git" in cmd_text
    assert "git clone --branch main --depth 1 https://github.com/example/test.git /workspace/repo" in cmd_text
    assert "echo setup" in cmd_text


@pytest.mark.asyncio
async def test_start_executes_cached_git_clone(monkeypatch, tmp_path):
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)

    def fake_clone_from_mirror(repo_url, target, ref="main"):
        target.mkdir(parents=True)
        (target / "README.md").write_text("cached")

    monkeypatch.setattr("distill_gym.sandbox.manager.clone_from_mirror", fake_clone_from_mirror)

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace/repo",
        steps=[
            {"type": "git_clone", "args": {"repo": "https://github.com/example/test.git", "ref": "main", "dest": "/workspace/repo", "use_cache": True}},
            {"type": "run", "args": {"command": "echo setup"}},
        ],
    )
    await manager.start(spec)

    commands = [call["command"] for call in runtime.exec_calls]
    assert "mkdir -p /workspace/repo" in commands[0]
    assert "echo setup" in " ".join(commands)
    assert len(runtime.copy_to_calls) > 0
    assert runtime.copy_to_calls[0][2] == "/workspace"
