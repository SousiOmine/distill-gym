import pytest
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.config.schema import SandboxConfig


class FakeExecClient:
    def __init__(self):
        self.commands = []

    async def container_run(self, spec):
        return "container-id"

    async def container_exec(self, cid, cmd, timeout=300, workdir=None):
        self.commands.append((cid, cmd, timeout, workdir))
        return 0, "", ""

    async def container_cp_to(self, cid, src, dst):
        pass

    async def container_cp_from(self, cid, src, dst):
        pass

    async def container_stop(self, cid):
        pass

    async def container_rm(self, cid):
        pass

    async def list_containers(self, label=""):
        return []

    async def list_volumes(self, label=""):
        return []

    async def list_networks(self, label=""):
        return []

    async def volume_rm(self, name):
        pass

    async def network_create(self, name, driver="bridge"):
        pass

    async def network_rm(self, name):
        pass


class FakeRuntime:
    def __init__(self):
        self.client = FakeExecClient()
        self.commands = []

    async def start(self, spec):
        return "container-id"

    async def exec(self, cid, cmd, timeout=300, workdir=None):
        self.commands.append((cid, cmd, timeout, workdir))
        return 0, "", ""

    async def copy_to(self, cid, src, dst):
        pass

    async def copy_from(self, cid, src, dst):
        pass

    async def stop(self, cid):
        pass

    async def remove(self, cid):
        pass

    async def cleanup_resources(self, label=""):
        return {"containers": 0, "volumes": 0, "networks": 0}


@pytest.mark.asyncio
async def test_execute_git_clone_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "git_clone", "args": {"repo": "https://github.com/example/test.git", "ref": "main", "dest": "/workspace/repo"}}],
    )
    await manager.execute_steps(spec)

    git_commands = [c for c in runtime.commands if "git clone" in c[1]]
    assert len(git_commands) > 0
    assert "test.git" in git_commands[0][1]


@pytest.mark.asyncio
async def test_execute_pip_install_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "pip_install", "args": {"packages": ["pytest", "ruff"], "timeout": 300}}],
    )
    await manager.execute_steps(spec)

    pip_commands = [c for c in runtime.commands if "pip install" in c[1]]
    assert len(pip_commands) > 0
    assert "pytest" in pip_commands[0][1]
    assert "ruff" in pip_commands[0][1]


@pytest.mark.asyncio
async def test_execute_apt_get_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "apt_get", "args": {"packages": ["git", "curl"], "update_first": True}}],
    )
    await manager.execute_steps(spec)

    apt_commands = [c for c in runtime.commands if "apt-get" in c[1]]
    assert len(apt_commands) > 0
    assert "apt-get install" in apt_commands[0][1]


@pytest.mark.asyncio
async def test_execute_mkdir_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "mkdir", "args": {"path": "/workspace/data"}}],
    )
    await manager.execute_steps(spec)

    mkdir_commands = [c for c in runtime.commands if "mkdir" in c[1]]
    assert len(mkdir_commands) > 0
    assert "/workspace/data" in mkdir_commands[-1][1]


@pytest.mark.asyncio
async def test_execute_unknown_step_raises():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "nonexistent", "args": {}}],
    )
    with pytest.raises(RuntimeError, match="Unknown build step type"):
        await manager.execute_steps(spec)


@pytest.mark.asyncio
async def test_execute_npm_install_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "npm_install", "args": {"packages": ["opencode-ai"], "global": True}}],
    )
    await manager.execute_steps(spec)

    npm_cmds = [c for c in runtime.commands if "npm install" in c[1]]
    assert len(npm_cmds) > 0
    assert "opencode-ai" in npm_cmds[0][1]



@pytest.mark.asyncio
async def test_execute_run_step():
    runtime = FakeRuntime()
    manager = SandboxManager(runtime=runtime)
    await manager.start(SandboxSpec(image="python:3.12", workdir="/workspace"))

    spec = SandboxSpec(
        image="python:3.12",
        workdir="/workspace",
        steps=[{"type": "run", "args": {"command": "echo hello", "timeout": 30}}],
    )
    await manager.execute_steps(spec)

    run_cmds = [c for c in runtime.commands if "echo hello" in c[1]]
    assert len(run_cmds) > 0
