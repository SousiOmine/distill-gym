import json
import pytest
import aiosqlite
from pathlib import Path
from datetime import datetime, timezone

from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import RunRecord, TaskRecord, ArtifactRecord
from distill_gym.exporters.openai_messages import export_openai_messages_jsonl
from distill_gym.cache.cache_store import get_artifacts_dir


@pytest.fixture
async def mem_store():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    from distill_gym.storage.db import _ensure_tables
    await _ensure_tables(db)
    store = RunStore(db)
    yield store
    await db.close()


@pytest.mark.asyncio
async def test_export_empty_run(mem_store):
    store = mem_store
    run_id = "test-export-empty"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    task = TaskRecord(id="t1", run_id=run_id, title="task1", prompt="do something", status="completed", success=True)
    await store.create_task(task)

    output = Path("test_export_output.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
            include_reasoning=True, include_tool_results=True, include_failed=False,
        )
        assert count == 1
        with open(output) as f:
            line = json.loads(f.readline())
            assert "messages" in line
            assert "metadata" in line
            assert line["metadata"]["task_id"] == "t1"
            assert line["metadata"]["result"]["success"] is True
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_export_excludes_failed(mem_store):
    store = mem_store
    run_id = "test-export-failed"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)

    task_ok = TaskRecord(id="t1", run_id=run_id, title="ok", prompt="ok", status="completed", success=True)
    task_fail = TaskRecord(id="t2", run_id=run_id, title="fail", prompt="fail", status="completed", success=False)
    await store.create_task(task_ok)
    await store.create_task(task_fail)

    output = Path("test_export_failed.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
            include_failed=False,
        )
        assert count == 1
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_metadata_no_secrets(mem_store):
    store = mem_store
    run_id = "test-no-secrets"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    task = TaskRecord(id="t1", run_id=run_id, title="t1", prompt="p", status="completed", success=True)
    await store.create_task(task)

    output = Path("test_no_secrets.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
        )
        assert count == 1
        with open(output) as f:
            line = json.loads(f.readline())
            meta = json.dumps(line["metadata"])
            assert "sk-" not in meta
            assert "api_key" not in meta.lower()
            assert "authorization" not in meta.lower()
    finally:
        if output.exists():
            output.unlink()
