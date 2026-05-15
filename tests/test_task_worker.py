import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distill_gym.config.schema import TaskItem
from distill_gym.harness.base import HarnessResult
from distill_gym.orchestrator.task_worker import TaskWorker, TaskResult


class _MockHarness:
    def __init__(self, exit_code=0, success=True, stdout="OK", stderr=""):
        self.exit_code = exit_code
        self.success = success
        self.stdout = stdout
        self.stderr = stderr

    async def run_task(self, sandbox, task):
        return HarnessResult(self.exit_code, self.stdout, self.stderr, self.success)


class _MockSandbox:
    def __init__(self):
        self.container_id = "mock-container"

    async def exec(self, command, timeout=300, workdir=None):
        if "git diff" in command:
            return 0, "diff --git a/file.py b/file.py", ""
        if "git status" in command:
            return 0, " M file.py", ""
        return 0, "", ""


class _MockStore(AsyncMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.artifacts = []

    async def update_task(self, task_id, **kwargs):
        pass

    async def create_artifact(self, art):
        self.artifacts.append(art)

    async def close(self):
        pass


class _MockTracePipeline:
    def __init__(self):
        self.registered = []
        self.events = []

    def register_task(self, task_id):
        self.registered.append(task_id)

    async def record_event(self, task_id, event, metadata=None):
        self.events.append((task_id, event, metadata))


class _MockSession:
    def __init__(self, sandbox=None):
        self.sandbox = sandbox or _MockSandbox()

    async def reset(self):
        pass

    async def cleanup(self):
        pass


class TestTaskWorker:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("distill_gym.orchestrator.task_worker.get_artifacts_dir", return_value=Path(tmpdir)):
            harness = _MockHarness(exit_code=0, success=True, stdout="success output")
            store = _MockStore()
            trace_pipeline = _MockTracePipeline()
            worker = TaskWorker(harness, store, trace_pipeline, "run_001")

            task = TaskItem(id="t1", title="Test Task", prompt="do something")
            session = _MockSession()

            result = await worker.execute(task, session)

            assert result.success is True
            assert result.exit_code == 0
            assert result.stdout == "success output"
            assert "t1" in trace_pipeline.registered
            assert len(trace_pipeline.events) == 2
            assert trace_pipeline.events[0][1] == "task_started"
            assert trace_pipeline.events[1][1] == "task_finished"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("distill_gym.orchestrator.task_worker.get_artifacts_dir", return_value=Path(tmpdir)):
            harness = _MockHarness(exit_code=1, success=False, stdout="", stderr="error msg")
            store = _MockStore()
            trace_pipeline = _MockTracePipeline()
            worker = TaskWorker(harness, store, trace_pipeline, "run_001")

            task = TaskItem(id="t2", title="Failing Task", prompt="do something")
            session = _MockSession()

            result = await worker.execute(task, session)

            assert result.success is False
            assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        class FailingHarness:
            async def run_task(self, sandbox, task):
                raise RuntimeError("something went wrong")

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("distill_gym.orchestrator.task_worker.get_artifacts_dir", return_value=Path(tmpdir)):
            harness = FailingHarness()
            store = _MockStore()
            trace_pipeline = _MockTracePipeline()
            worker = TaskWorker(harness, store, trace_pipeline, "run_001")

            task = TaskItem(id="t3", title="Bad Task", prompt="do something")
            session = _MockSession()

            result = await worker.execute(task, session)

            assert result.success is False
            assert result.exit_code == -1
            assert "something went wrong" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_with_test_command(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("distill_gym.orchestrator.task_worker.get_artifacts_dir", return_value=Path(tmpdir)):
            harness = _MockHarness(exit_code=0, success=True)
            store = _MockStore()
            trace_pipeline = _MockTracePipeline()
            worker = TaskWorker(harness, store, trace_pipeline, "run_001")

            task = TaskItem(id="t4", title="Test Task", prompt="do something", test_command="pytest")
            session = _MockSession()

            result = await worker.execute(task, session)

            assert result.success is True
