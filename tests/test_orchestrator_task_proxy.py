import pytest

from distill_gym.config.schema import Config, TaskItem
from distill_gym.orchestrator.orchestrator import _run_tasks_concurrent
from distill_gym.orchestrator.task_worker import TaskResult


class _Plan:
    run_id = "run_001"
    tasks = [
        TaskItem(id="task_a", prompt="a"),
        TaskItem(id="task_b", prompt="b"),
    ]


class _Store:
    async def close(self):
        pass


class _Isolation:
    async def acquire(self, task_id):
        return object()

    async def release(self, session):
        pass


class _EventBus:
    async def emit(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_concurrent_tasks_use_task_specific_proxy_urls(monkeypatch):
    seen_proxy_urls = []

    async def fake_create_worker():
        return _Store()

    class FakeTaskWorker:
        def __init__(self, harness, store, trace_pipeline, run_id):
            seen_proxy_urls.append(harness.proxy_base_url)

        async def execute(self, task, session):
            return TaskResult(task.id, True, 0)

    monkeypatch.setattr(
        "distill_gym.orchestrator.orchestrator.RunStore.create_worker",
        fake_create_worker,
    )
    monkeypatch.setattr(
        "distill_gym.orchestrator.orchestrator.TaskWorker",
        FakeTaskWorker,
    )

    cfg = Config()
    cfg.harness.type = "generic_cli"

    results = await _run_tasks_concurrent(
        cfg,
        _Plan(),
        _Store(),
        _Isolation(),
        trace_pipeline=None,
        concurrency=2,
        event_bus=_EventBus(),
        proxy_base_url="http://proxy.test:5002/v1",
    )

    assert [r.task_id for r in results] == ["task_a", "task_b"]
    assert seen_proxy_urls == [
        "http://proxy.test:5002/tasks/task_a/v1",
        "http://proxy.test:5002/tasks/task_b/v1",
    ]
