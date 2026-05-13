import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from distill_gym.exporters.chatml import export_chatml_jsonl
from distill_gym.storage.models import RunRecord, TaskRecord
from distill_gym.storage.run_store import RunStore


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
async def test_export_chatml_jsonl_writes_text_records(mem_store):
    run_id = "chatml-run"
    await mem_store.create_run(
        RunRecord(
            id=run_id,
            name="test",
            config_yaml="",
            status="completed",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    await mem_store.create_task(
        TaskRecord(id="t1", run_id=run_id, title="task", prompt="do it", status="completed", success=True)
    )

    output = Path("test_chatml.jsonl")
    try:
        count = await export_chatml_jsonl(run_id, output, mem_store)
        assert count == 1
        row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        assert "text" in row
        assert "metadata" in row
    finally:
        if output.exists():
            output.unlink()
