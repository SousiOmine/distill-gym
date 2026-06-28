import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from distill_gym.config.schema import (
    Config, HarnessConfig, ProviderConfig, SandboxNetworkMode, TaskItem,
)
from distill_gym.harness.base import HarnessResult
from distill_gym.orchestrator.events import get_event_bus, reset_event_bus
from distill_gym.orchestrator.orchestrator import (
    _effective_concurrency,
    _ensure_proxy_reachable_from_sandbox,
    _execute_run,
    _make_harness,
    _make_harness_from_config,
    _needs_harness_taskgen,
    _proxy_health_url,
    _run_tasks_concurrent,
    _run_tasks_mock,
    _setup_run,
    _start_proxy,
    _stop_proxy,
    cleanup,
    run,
)
from distill_gym.orchestrator.task_worker import TaskResult
from distill_gym.orchestrator.run_plan import RunPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEventBus:
    async def emit(self, *args, **kwargs):
        pass


class _FakeWorkerStore:
    async def close(self):
        pass


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """1–7: Helper function tests"""

    def test_make_harness_from_config(self):
        hc = HarnessConfig(type="mock")
        prov = ProviderConfig()
        with patch(
            "distill_gym.orchestrator.orchestrator.HarnessRegistry.create"
        ) as mock_create:
            result = _make_harness_from_config(
                hc, provider=prov, proxy_base_url="http://p:5002"
            )
            mock_create.assert_called_once_with(
                "mock", hc, provider=prov, proxy_base_url="http://p:5002"
            )

    def test_make_harness(self):
        config = Config()
        config.harness.type = "generic_cli"
        with patch(
            "distill_gym.orchestrator.orchestrator._make_harness_from_config"
        ) as mock_sub:
            _make_harness(config, provider=config.provider)
            mock_sub.assert_called_once_with(
                config.harness, provider=config.provider
            )

    def test_needs_harness_taskgen_true(self):
        config = Config()
        config.taskgen.type = "harness"
        config.taskgen.tasks = []
        assert _needs_harness_taskgen(config) is True

    def test_needs_harness_taskgen_false_by_type(self):
        config = Config()
        config.taskgen.type = "repo_auto"
        config.taskgen.tasks = []
        assert _needs_harness_taskgen(config) is False

    def test_needs_harness_taskgen_false_by_tasks(self):
        config = Config()
        config.taskgen.type = "harness"
        config.taskgen.tasks = [TaskItem(id="t1", title="T", prompt="p")]
        assert _needs_harness_taskgen(config) is False

    def test_proxy_health_url(self):
        result = _proxy_health_url("http://host.containers.internal:5002/v1")
        assert result == "http://host.containers.internal:5002/health"

    def test_effective_concurrency(self):
        config = Config()
        config.run.concurrency = 4
        assert _effective_concurrency(config) == 4

    def test_effective_concurrency_zero(self):
        config = Config()
        config.run.concurrency = 0
        assert _effective_concurrency(config) == 1

    def test_effective_concurrency_negative(self):
        config = Config()
        config.run.concurrency = -1
        assert _effective_concurrency(config) == 1


# ---------------------------------------------------------------------------
# _start_proxy / _stop_proxy
# ---------------------------------------------------------------------------

class TestProxyLifecycle:
    """8–11: Proxy start / stop"""

    @pytest.mark.asyncio
    async def test_start_proxy_success(self):
        mock_app = MagicMock()
        mock_server = MagicMock()
        mock_server.started = True
        mock_server.serve = AsyncMock()

        with (
            patch("distill_gym.orchestrator.orchestrator.create_proxy_app",
                  return_value=mock_app),
            patch("uvicorn.Config") as mock_uv_config,
        ):
            mock_uv_config.return_value = MagicMock()
            with patch("uvicorn.Server", return_value=mock_server):
                config = Config()
                recorder = MagicMock()
                server, task = await _start_proxy(config, recorder)

        assert server is mock_server
        assert task is not None
        assert not task.done() or task.cancelled()

    @pytest.mark.asyncio
    async def test_start_proxy_timeout(self):
        mock_app = MagicMock()
        mock_server = MagicMock()
        mock_server.started = False  # never starts
        async def never_complete():
            await asyncio.Event().wait()
        mock_server.serve = never_complete

        with (
            patch("distill_gym.orchestrator.orchestrator.create_proxy_app",
                  return_value=mock_app),
            patch("uvicorn.Config") as mock_uv_config,
            patch("asyncio.sleep", AsyncMock()),
        ):
            mock_uv_config.return_value = MagicMock()
            with patch("uvicorn.Server", return_value=mock_server):
                config = Config()
                recorder = MagicMock()
                with pytest.raises(RuntimeError, match="logging proxy did not start"):
                    await _start_proxy(config, recorder)
        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_proxy(self):
        server = MagicMock()
        task = asyncio.create_task(asyncio.sleep(9999))
        await _stop_proxy(server, task)
        assert server.should_exit is True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_proxy_none(self):
        # Should not raise
        await _stop_proxy(None, None)

    @pytest.mark.asyncio
    async def test_stop_proxy_none_server_only(self):
        task = asyncio.create_task(asyncio.sleep(9999))
        await _stop_proxy(None, task)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_proxy_none_task_only(self):
        server = MagicMock()
        should_exit_before = server.should_exit
        await _stop_proxy(server, None)
        # server.should_exit should still be the default
        assert server.should_exit == should_exit_before


# ---------------------------------------------------------------------------
# _setup_run
# ---------------------------------------------------------------------------

class TestSetupRun:
    """12: _setup_run"""

    @pytest.mark.asyncio
    async def test_setup_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "setup-test"
        config.taskgen.tasks = [TaskItem(id="t1", title="T", prompt="p")]

        plan, store = await _setup_run(config)
        try:
            assert plan.run_id is not None
            assert plan.run_id.startswith("run_")
            assert len(plan.tasks) == 1
            run_rec = await store.get_run(plan.run_id)
            assert run_rec is not None
            assert run_rec.status == "pending"
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# _execute_run – dry-run path
# ---------------------------------------------------------------------------

class TestExecuteRunDryRun:
    """13: dry_run=True"""

    @pytest.mark.asyncio
    async def test_execute_run_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "dry-test"
        config.run.task_count = 1
        config.harness.type = "mock"
        config.taskgen.tasks = [TaskItem(id="t1", title="T", prompt="p")]

        plan, store = await _setup_run(config)
        try:
            result = await _execute_run(config, plan, store, dry_run=True)
            assert result == plan.run_id
            run_rec = await store.get_run(plan.run_id)
            assert run_rec.status == "completed"
            assert run_rec.success is True
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# _execute_run – mock harness (success / failure / exception)
# ---------------------------------------------------------------------------

class TestExecuteRunMockHarness:
    """14: mock harness, all tasks succeed"""

    @pytest.mark.asyncio
    async def test_execute_run_mock_harness(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "mock-test"
        config.run.task_count = 2
        config.harness.type = "mock"
        config.taskgen.tasks = [
            TaskItem(id="t1", title="T1", prompt="p1"),
            TaskItem(id="t2", title="T2", prompt="p2"),
        ]

        plan, store = await _setup_run(config)
        try:
            result = await _execute_run(config, plan, store)
            assert result == plan.run_id
            run_rec = await store.get_run(plan.run_id)
            assert run_rec.status == "completed"
            assert run_rec.success is True
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_execute_run_task_failure(self, tmp_path, monkeypatch):
        """16: a task returns failure → run marked failed"""
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        async def fake_mock_run(plan, store, event_bus):
            return [TaskResult("t1", False, 1, error_message="mock failure")]

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._run_tasks_mock",
            fake_mock_run,
        )

        config = Config()
        config.run.name = "fail-test"
        config.harness.type = "mock"
        config.taskgen.tasks = [TaskItem(id="t1", title="T", prompt="p")]

        plan, store = await _setup_run(config)
        try:
            result = await _execute_run(config, plan, store)
            assert result == plan.run_id
            run_rec = await store.get_run(plan.run_id)
            assert run_rec.status == "failed"
            assert run_rec.success is False
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_execute_run_exception(self, tmp_path, monkeypatch):
        """17: exception during execution → run marked failed + RuntimeError"""
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        async def fake_mock_run_exc(plan, store, event_bus):
            raise ValueError("internal failure")

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._run_tasks_mock",
            fake_mock_run_exc,
        )

        config = Config()
        config.run.name = "exc-test"
        config.harness.type = "mock"
        config.taskgen.tasks = [TaskItem(id="t1", title="T", prompt="p")]

        plan, store = await _setup_run(config)
        try:
            with pytest.raises(RuntimeError, match="internal failure"):
                await _execute_run(config, plan, store)
            run_rec = await store.get_run(plan.run_id)
            assert run_rec.status == "failed"
            assert run_rec.success is False
            assert "internal failure" in (run_rec.error_message or "")
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_execute_run_mock_with_taskgen(
        self, tmp_path, monkeypatch
    ):
        """15: taskgen.type=="harness" & taskgen.harness.type!="mock"
        → needs_proxy=True, so _start_proxy is called."""
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        # Mute all external side-effects
        from distill_gym.platform.detection import PlatformInfo

        fake_platform = PlatformInfo(system="linux", arch="x86_64")

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.detect",
            lambda: fake_platform,
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.proxy_base_url_for_sandbox",
            lambda c, p: "http://sandbox.test:5002/v1",
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.proxy_listen_host_for_sandbox",
            lambda c, p: "0.0.0.0",
        )

        start_proxy_called = False

        async def fake_start_proxy(config, recorder, listen_host=None):
            nonlocal start_proxy_called
            start_proxy_called = True
            server = MagicMock()
            server.started = True
            server.serve = AsyncMock()
            task = asyncio.create_task(asyncio.sleep(9999))
            return server, task

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._start_proxy",
            fake_start_proxy,
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._ensure_proxy_reachable_from_sandbox",
            AsyncMock(),
        )

        # Mocks for sandbox
        mock_builder = MagicMock()
        mock_builder.validate.return_value = []
        mock_builder.build.return_value = MagicMock()
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.BuilderRegistry.create",
            lambda name: mock_builder,
        )

        mock_runtime = MagicMock()
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.create_runtime",
            lambda engine: mock_runtime,
        )

        mock_pool = MagicMock()
        mock_sandbox = AsyncMock()
        mock_sandbox.exec = AsyncMock(return_value=(0, "abc123", ""))
        mock_pool.acquire = AsyncMock(return_value=mock_sandbox)
        mock_pool.release = AsyncMock()
        mock_pool.destroy_all = AsyncMock()
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.SandboxPool",
            lambda **kwargs: mock_pool,
        )

        # HarnessTaskGenerator
        mock_taskgen = MagicMock()
        mock_taskgen.generate = AsyncMock(
            return_value=[TaskItem(id="g1", title="gen", prompt="do")]
        )
        mock_taskgen.cleanup_output_file = AsyncMock()
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.HarnessTaskGenerator",
            lambda cfg, harness, sandbox: mock_taskgen,
        )

        # Harness
        mock_harness = MagicMock()
        mock_harness.install = AsyncMock()
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.HarnessRegistry.create",
            lambda name, config, **kwargs: mock_harness,
        )

        # Ensure mock run tasks succeeds
        async def fake_run_mock(plan, store, event_bus):
            return [TaskResult("g1", True, 0)]

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._run_tasks_mock",
            fake_run_mock,
        )

        config = Config()
        config.run.name = "mock-tg"
        config.run.task_count = 2
        config.harness.type = "mock"
        config.taskgen.type = "harness"
        config.taskgen.tasks = []
        config.taskgen.harness.type = "generic_cli"
        # Avoid network_create code path
        config.sandbox.network.mode = SandboxNetworkMode.bridge

        plan, store = await _setup_run(config)
        try:
            result = await _execute_run(config, plan, store)
            assert result == plan.run_id
            assert start_proxy_called, "_start_proxy should have been called"
            run_rec = await store.get_run(plan.run_id)
            assert run_rec.status == "completed"
        finally:
            await store.close()
            # Cancel any lingering tasks
            if start_proxy_called:
                for t in asyncio.all_tasks():
                    if t.get_name() == "Task-2":  # rough; cancel all sleep tasks
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, RuntimeError):
                            pass


# ---------------------------------------------------------------------------
# _run_tasks_mock
# ---------------------------------------------------------------------------

class TestRunTasksMock:
    """18–19: _run_tasks_mock"""

    @pytest.mark.asyncio
    async def test_run_tasks_mock(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "mock-run"
        config.taskgen.tasks = [
            TaskItem(id="m1", title="M1", prompt="do1"),
        ]
        plan = await RunPlan.from_config(config)
        store = __import__("distill_gym").storage.run_store.RunStore()

        try:
            await store.create_run(plan.to_run_record())
            results = await _run_tasks_mock(plan, store, get_event_bus())
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].task_id == "m1"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_run_tasks_mock_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "art-test"
        config.taskgen.tasks = [
            TaskItem(id="a1", title="A1", prompt="do2"),
        ]
        plan = await RunPlan.from_config(config)
        store = __import__("distill_gym").storage.run_store.RunStore()

        try:
            await store.create_run(plan.to_run_record())
            # Pre-register the task (normally done by _execute_run)
            for tr in plan.to_task_records():
                await store.create_task(tr)
            await _run_tasks_mock(plan, store, get_event_bus())

            tasks = await store.list_tasks(plan.run_id)
            assert len(tasks) == 1
            assert tasks[0].status == "completed"
            assert tasks[0].success is True

            # _run_tasks_mock saves artifacts to disk via ArtifactCollector
            artifacts_dir = tmp_path / "artifacts" / plan.run_id / "a1"
            assert (artifacts_dir / "stdout.txt").exists()
            assert (artifacts_dir / "stderr.txt").exists()
            assert (artifacts_dir / "metadata.json").exists()

            # Check file content
            metadata = __import__("json").loads(
                (artifacts_dir / "metadata.json").read_text()
            )
            assert metadata["harness_type"] == "mock"
            assert metadata["success"] is True
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# _run_tasks_concurrent
# ---------------------------------------------------------------------------

class TestRunTasksConcurrent:
    """20–21: _run_tasks_concurrent"""

    @pytest.mark.asyncio
    async def test_run_tasks_concurrent_success(self, monkeypatch):
        seen = []

        class FakeWorker:
            def __init__(self, harness, store, trace_pipeline, run_id):
                seen.append((harness, run_id))

            async def execute(self, task, session):
                return TaskResult(task.id, True, 0)

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.TaskWorker",
            FakeWorker,
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.RunStore.create_worker",
            AsyncMock(return_value=_FakeWorkerStore()),
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._make_harness",
            lambda config, **kwargs: MagicMock(),
        )

        plan = MagicMock()
        plan.run_id = "run_conc_001"
        plan.tasks = [
            TaskItem(id="c1", title="C1", prompt="p1"),
            TaskItem(id="c2", title="C2", prompt="p2"),
        ]

        class FakeIsolation:
            async def acquire(self, task_id):
                return MagicMock()
            async def release(self, session):
                pass

        config = Config()
        config.harness.type = "generic_cli"

        results = await _run_tasks_concurrent(
            config, plan, MagicMock(), FakeIsolation(),
            trace_pipeline=None, concurrency=2,
            event_bus=_FakeEventBus(),
            proxy_base_url="http://proxy:5002/v1",
        )
        assert len(results) == 2
        assert all(r.success for r in results)
        assert [r.task_id for r in results] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_run_tasks_concurrent_exception(self, monkeypatch):
        class FailingWorker:
            def __init__(self, harness, store, trace_pipeline, run_id):
                pass

            async def execute(self, task, session):
                raise RuntimeError("worker crashed")

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.TaskWorker",
            FailingWorker,
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator.RunStore.create_worker",
            AsyncMock(return_value=_FakeWorkerStore()),
        )
        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._make_harness",
            lambda config, **kwargs: MagicMock(),
        )

        plan = MagicMock()
        plan.run_id = "run_conc_002"
        plan.tasks = [TaskItem(id="e1", title="E1", prompt="p1")]

        class FakeIsolation:
            async def acquire(self, task_id):
                return MagicMock()
            async def release(self, session):
                pass

        config = Config()
        config.harness.type = "generic_cli"

        results = await _run_tasks_concurrent(
            config, plan, MagicMock(), FakeIsolation(),
            trace_pipeline=None, concurrency=1,
            event_bus=_FakeEventBus(),
            proxy_base_url="http://proxy:5002/v1",
        )
        assert len(results) == 1
        assert results[0].success is False
        assert "worker crashed" in results[0].error_message


# ---------------------------------------------------------------------------
# _ensure_proxy_reachable_from_sandbox
# ---------------------------------------------------------------------------

class TestEnsureProxyReachable:
    """22–24: reachability check"""

    @pytest.mark.asyncio
    async def test_exit_0(self):
        mgr = MagicMock()
        mgr.exec = AsyncMock(return_value=(0, "", ""))
        await _ensure_proxy_reachable_from_sandbox(mgr, "http://h:5002/v1")
        # no exception → success

    @pytest.mark.asyncio
    async def test_exit_77(self, caplog):
        caplog.set_level("INFO")
        mgr = MagicMock()
        mgr.exec = AsyncMock(return_value=(77, "", ""))
        await _ensure_proxy_reachable_from_sandbox(mgr, "http://h:5002/v1")
        assert "Skipping" in caplog.text

    @pytest.mark.asyncio
    async def test_exit_failure(self):
        mgr = MagicMock()
        mgr.exec = AsyncMock(return_value=(1, "", "connection refused"))
        with pytest.raises(RuntimeError, match="connection refused"):
            await _ensure_proxy_reachable_from_sandbox(mgr, "http://h:5002/v1")


# ---------------------------------------------------------------------------
# run 関数
# ---------------------------------------------------------------------------

class TestRunFunction:
    """25–26: run()"""

    @pytest.mark.asyncio
    async def test_run_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        config = Config()
        config.run.name = "run-dry"
        config.run.task_count = 1
        config.harness.type = "mock"
        config.taskgen.tasks = [TaskItem(id="r1", title="R", prompt="p")]

        run_id = await run(config, dry_run=True)
        assert run_id.startswith("run_")

    @pytest.mark.asyncio
    async def test_run_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))

        async def fake_exec(config, plan, store, dry_run=False):
            raise RuntimeError("oops")

        monkeypatch.setattr(
            "distill_gym.orchestrator.orchestrator._execute_run",
            fake_exec,
        )

        config = Config()
        config.run.name = "run-exc"
        config.run.task_count = 1
        config.harness.type = "mock"
        config.taskgen.tasks = [TaskItem(id="r2", title="R", prompt="p")]

        with pytest.raises(RuntimeError, match="oops"):
            await run(config)


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """27: cleanup()"""

    @pytest.mark.asyncio
    async def test_cleanup(self):
        mock_manager = MagicMock()
        mock_manager.cleanup_resources = AsyncMock(return_value={"removed": 3})
        with patch(
            "distill_gym.sandbox.manager.SandboxManager",
            return_value=mock_manager,
        ):
            result = await cleanup(label="test-label")
            assert result == {"removed": 3}
            mock_manager.cleanup_resources.assert_called_once_with("test-label")

    @pytest.mark.asyncio
    async def test_cleanup_default_label(self):
        mock_manager = MagicMock()
        mock_manager.cleanup_resources = AsyncMock(return_value={})
        with patch(
            "distill_gym.sandbox.manager.SandboxManager",
            return_value=mock_manager,
        ):
            await cleanup()
            mock_manager.cleanup_resources.assert_called_once_with(
                "distill-gym=true"
            )
