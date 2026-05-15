import json
import pytest
import aiosqlite
from pathlib import Path
from datetime import datetime, timezone

from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import RunRecord, TaskRecord
from distill_gym.exporters.merger import merge_runs_to_jsonl


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
async def test_merge_single_run(mem_store):
    store = mem_store
    run1 = RunRecord(
        id="run-1", name="run1", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run1)
    await store.create_task(TaskRecord(id="t1", run_id="run-1", title="task1", prompt="do it", status="completed", success=True))

    output = Path("test_merge_single.jsonl")
    try:
        count = await merge_runs_to_jsonl(["run-1"], output, store)
        assert count == 1
        with open(output) as f:
            line = json.loads(f.readline())
            assert "messages" in line
            assert line["source_run_id"] == "run-1"
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_merge_multiple_runs(mem_store):
    store = mem_store
    for rid in ("run-a", "run-b"):
        run = RunRecord(
            id=rid, name=rid, config_yaml="", status="completed",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        await store.create_run(run)
        await store.create_task(TaskRecord(id=f"{rid}_task", run_id=rid, title="t", prompt="p", status="completed", success=True))

    output = Path("test_merge_multi.jsonl")
    try:
        count = await merge_runs_to_jsonl(["run-a", "run-b"], output, store)
        assert count == 2
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_merge_skips_failed(mem_store):
    store = mem_store
    run = RunRecord(
        id="skip-fail", name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    await store.create_task(TaskRecord(id="ok", run_id="skip-fail", title="ok", prompt="ok", status="completed", success=True))
    await store.create_task(TaskRecord(id="fail", run_id="skip-fail", title="fail", prompt="fail", status="completed", success=False))

    output = Path("test_merge_skip.jsonl")
    try:
        count = await merge_runs_to_jsonl(["skip-fail"], output, store, include_failed=False)
        assert count == 1
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_merge_missing_run_skipped(mem_store):
    output = Path("test_merge_missing.jsonl")
    try:
        count = await merge_runs_to_jsonl(["nonexistent"], output, mem_store)
        assert count == 0
    finally:
        if output.exists():
            output.unlink()
