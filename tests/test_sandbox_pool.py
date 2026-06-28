import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.pool import SandboxPool
from distill_gym.sandbox.isolation import ContainerIsolation, ContainerSession


@pytest.fixture
def spec():
    return SandboxSpec(
        image="python:3.12",
        container_name="test-container",
        workdir="/workspace",
    )


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.start = AsyncMock(return_value="container-id")
    runtime.exec = AsyncMock(return_value=(0, "", ""))
    runtime.copy_to = AsyncMock()
    runtime.copy_from = AsyncMock()
    runtime.stop = AsyncMock()
    runtime.remove = AsyncMock()
    runtime.cleanup_resources = AsyncMock()
    return runtime


@pytest.fixture
def mock_sandbox_manager():
    sb = MagicMock()
    sb.container_id = "container-id"
    sb.workdir = "/workspace"
    sb.exec = AsyncMock(return_value=(0, "", ""))
    sb.start = AsyncMock(return_value="container-id")
    sb.destroy = AsyncMock()
    sb.stop = AsyncMock()
    sb.checked_exec = AsyncMock(return_value=("", ""))
    sb.copy_to = AsyncMock()
    sb.copy_from = AsyncMock()
    return sb


# ============================================================
# SandboxPool tests
# ============================================================


@pytest.mark.asyncio
async def test_acquire_creates_sandbox_and_runs_init_hook(spec, mock_runtime, mock_sandbox_manager):
    init_hook = AsyncMock()
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2, init_hook=init_hook)
        sb = await pool.acquire()

        assert sb is mock_sandbox_manager
        assert len(pool._all_sandboxes) == 1
        mock_sandbox_manager.start.assert_awaited_once()
        init_hook.assert_awaited_once_with(mock_sandbox_manager)
        args, _ = mock_sandbox_manager.start.call_args
        assert args[0].container_name == "test-container-pool-0"


@pytest.mark.asyncio
async def test_acquire_reuses_idle_sandbox(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        sb = await pool.acquire()
        await pool.release(sb)
        mock_sandbox_manager.exec.reset_mock()

        sb2 = await pool.acquire()

        assert sb2 is mock_sandbox_manager
        assert len(pool._all_sandboxes) == 1
        mock_sandbox_manager.exec.assert_awaited_once_with(
            "git checkout -- . && git clean -fd 2>/dev/null || true"
        )


@pytest.mark.asyncio
async def test_acquire_respects_max_size(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        sb1 = await pool.acquire()
        sb2 = await pool.acquire()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pool.acquire(), timeout=0.05)

        await pool.release(sb1)
        sb3 = await asyncio.wait_for(pool.acquire(), timeout=0.05)
        assert sb3 is not None


@pytest.mark.asyncio
async def test_release_returns_to_idle_and_releases_semaphore(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=1)
        sb = await pool.acquire()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pool.acquire(), timeout=0.05)

        await pool.release(sb)

        assert not pool._idle.empty()
        sb2 = await asyncio.wait_for(pool.acquire(), timeout=0.05)
        assert sb2 is sb


@pytest.mark.asyncio
async def test_exec_on_all_calls_exec_on_each_sandbox(spec, mock_runtime):
    sb1 = MagicMock()
    sb1.exec = AsyncMock(return_value=(0, "", ""))
    sb1.start = AsyncMock(return_value="id1")
    sb2 = MagicMock()
    sb2.exec = AsyncMock(return_value=(0, "", ""))
    sb2.start = AsyncMock(return_value="id2")

    with patch("distill_gym.sandbox.pool.SandboxManager", side_effect=[sb1, sb2]):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        await pool.acquire()
        await pool.acquire()

        await pool.exec_on_all("echo hello", timeout=60)

        sb1.exec.assert_awaited_once_with("echo hello", timeout=60)
        sb2.exec.assert_awaited_once_with("echo hello", timeout=60)


@pytest.mark.asyncio
async def test_create_sandbox_applies_pool_name_suffix(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        await pool.acquire()
        await pool.acquire()

        start_calls = mock_sandbox_manager.start.call_args_list
        assert start_calls[0][0][0].container_name == "test-container-pool-0"
        assert start_calls[1][0][0].container_name == "test-container-pool-1"


@pytest.mark.asyncio
async def test_destroy_all_destroys_and_clears_list(spec, mock_runtime):
    sb1 = MagicMock()
    sb1.exec = AsyncMock()
    sb1.start = AsyncMock(return_value="id1")
    sb1.destroy = AsyncMock()
    sb2 = MagicMock()
    sb2.exec = AsyncMock()
    sb2.start = AsyncMock(return_value="id2")
    sb2.destroy = AsyncMock()

    with patch("distill_gym.sandbox.pool.SandboxManager", side_effect=[sb1, sb2]):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        await pool.acquire()
        await pool.acquire()

        assert len(pool._all_sandboxes) == 2
        await pool.destroy_all()

        sb1.destroy.assert_awaited_once()
        sb2.destroy.assert_awaited_once()
        assert len(pool._all_sandboxes) == 0


@pytest.mark.asyncio
async def test_destroy_all_continues_on_exception(spec, mock_runtime):
    sb1 = MagicMock()
    sb1.exec = AsyncMock()
    sb1.start = AsyncMock(return_value="id1")
    sb1.destroy = AsyncMock(side_effect=RuntimeError("destroy failed"))
    sb2 = MagicMock()
    sb2.exec = AsyncMock()
    sb2.start = AsyncMock(return_value="id2")
    sb2.destroy = AsyncMock()

    with patch("distill_gym.sandbox.pool.SandboxManager", side_effect=[sb1, sb2]):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        await pool.acquire()
        await pool.acquire()

        await pool.destroy_all()

        sb2.destroy.assert_awaited_once()
        assert len(pool._all_sandboxes) == 0


@pytest.mark.asyncio
async def test_acquire_releases_semaphore_on_exception(spec, mock_runtime, mock_sandbox_manager):
    mock_sandbox_manager.start = AsyncMock(side_effect=[RuntimeError("start failed"), "container-id"])

    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=1)

        with pytest.raises(RuntimeError, match="start failed"):
            await pool.acquire()

        assert len(pool._all_sandboxes) == 1
        sb = await asyncio.wait_for(pool.acquire(), timeout=0.05)
        assert sb is not None


# ============================================================
# ContainerIsolation tests
# ============================================================


@pytest.mark.asyncio
async def test_container_isolation_acquire_returns_session(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        isolation = ContainerIsolation(pool)

        session = await isolation.acquire("task-1")

        assert isinstance(session, ContainerSession)
        assert session.sandbox is mock_sandbox_manager


@pytest.mark.asyncio
async def test_container_isolation_release_calls_cleanup_and_returns_to_pool(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=1)
        isolation = ContainerIsolation(pool)

        session = await isolation.acquire("task-1")
        mock_sandbox_manager.exec.reset_mock()

        await isolation.release(session)

        mock_sandbox_manager.exec.assert_awaited_once_with(
            "git checkout -- . && git clean -fd 2>/dev/null || true"
        )
        assert not pool._idle.empty()


@pytest.mark.asyncio
async def test_container_isolation_destroy_all(spec, mock_runtime, mock_sandbox_manager):
    with patch("distill_gym.sandbox.pool.SandboxManager", return_value=mock_sandbox_manager):
        pool = SandboxPool(runtime=mock_runtime, spec_template=spec, max_size=2)
        isolation = ContainerIsolation(pool)

        pool.destroy_all = AsyncMock()
        await isolation.destroy_all()
        pool.destroy_all.assert_awaited_once()


# ============================================================
# ContainerSession tests
# ============================================================


@pytest.mark.asyncio
async def test_container_session_reset(mock_sandbox_manager):
    session = ContainerSession(mock_sandbox_manager)
    await session.reset()
    mock_sandbox_manager.exec.assert_awaited_once_with(
        "git checkout -- . && git clean -fd 2>/dev/null || true"
    )


@pytest.mark.asyncio
async def test_container_session_cleanup(mock_sandbox_manager):
    session = ContainerSession(mock_sandbox_manager)
    await session.cleanup()
    mock_sandbox_manager.exec.assert_awaited_once_with(
        "git checkout -- . && git clean -fd 2>/dev/null || true"
    )
