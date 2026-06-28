import os
import pytest
import aiosqlite
from pathlib import Path
from datetime import datetime, timezone
from distill_gym.storage.run_store import RunStore
from distill_gym.storage.db import get_db, _ensure_tables
from distill_gym.storage.models import RunRecord, TaskRecord, ArtifactRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_store():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    from distill_gym.storage.db import _ensure_tables
    await _ensure_tables(db)
    store = RunStore(db)
    yield store
    await db.close()


def _make_run(run_id: str, **overrides) -> RunRecord:
    now = datetime.now(timezone.utc)
    params = dict(
        id=run_id, name="test", config_yaml="", status="pending",
        created_at=now, updated_at=now,
    )
    params.update(overrides)
    return RunRecord(**params)


# ---------------------------------------------------------------------------
# RunStore tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_and_get_run(mem_store):
    store = mem_store
    run = _make_run("run-1")
    await store.create_run(run)
    got = await store.get_run("run-1")
    assert got is not None
    assert got.id == "run-1"
    assert got.name == "test"
    assert got.status == "pending"


@pytest.mark.asyncio
async def test_get_run_not_found(mem_store):
    store = mem_store
    got = await store.get_run("nonexistent")
    assert got is None


@pytest.mark.asyncio
async def test_update_run(mem_store):
    store = mem_store
    now = datetime.now(timezone.utc)
    run = _make_run("run-upd", created_at=now, updated_at=now)
    await store.create_run(run)
    new_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
    await store.update_run("run-upd",
                           status="completed",
                           success=True,
                           error_message="all good",
                           created_at=new_time)
    got = await store.get_run("run-upd")
    assert got is not None
    assert got.status == "completed"
    assert got.success is True
    assert got.error_message == "all good"
    assert got.created_at == new_time
    assert got.updated_at > now


@pytest.mark.asyncio
async def test_list_runs(mem_store):
    store = mem_store
    for i in range(5):
        run = _make_run(f"run-{i}",
                        created_at=datetime(2024, 1, 1, i, tzinfo=timezone.utc),
                        updated_at=datetime(2024, 1, 1, i, tzinfo=timezone.utc))
        await store.create_run(run)

    all_runs = await store.list_runs()
    assert len(all_runs) == 5
    assert all_runs[0].id == "run-4"
    assert all_runs[-1].id == "run-0"

    limited = await store.list_runs(limit=2, offset=1)
    assert len(limited) == 2
    assert limited[0].id == "run-3"
    assert limited[1].id == "run-2"


@pytest.mark.asyncio
async def test_list_runs_no_limit(mem_store):
    store = mem_store
    for i in range(3):
        await store.create_run(_make_run(f"r-{i}"))
    runs = await store.list_runs()
    assert len(runs) == 3


@pytest.mark.asyncio
async def test_get_task_counts(mem_store):
    store = mem_store
    await store.create_run(_make_run("run-tc"))
    for i in range(3):
        task = TaskRecord(id=f"t-{i}", run_id="run-tc", title=f"t{i}", prompt="p")
        await store.create_task(task)
    counts = await store.get_task_counts()
    assert counts == {"run-tc": 3}


@pytest.mark.asyncio
async def test_get_task_counts_empty(mem_store):
    store = mem_store
    counts = await store.get_task_counts()
    assert counts == {}


@pytest.mark.asyncio
async def test_get_total_run_count(mem_store):
    store = mem_store
    assert await store.get_total_run_count() == 0
    for i in range(3):
        await store.create_run(_make_run(f"run-{i}"))
    assert await store.get_total_run_count() == 3


@pytest.mark.asyncio
async def test_create_and_list_tasks(mem_store):
    store = mem_store
    await store.create_run(_make_run("run-tasks"))
    for i in range(3):
        t = TaskRecord(id=f"task-{i}", run_id="run-tasks", title=f"Task {i}", prompt="p")
        await store.create_task(t)
    tasks = await store.list_tasks("run-tasks")
    assert len(tasks) == 3
    assert [t.id for t in tasks] == ["task-0", "task-1", "task-2"]


@pytest.mark.asyncio
async def test_list_tasks_empty(mem_store):
    store = mem_store
    tasks = await store.list_tasks("no-such-run")
    assert tasks == []


@pytest.mark.asyncio
async def test_update_task(mem_store):
    store = mem_store
    await store.create_run(_make_run("run-ut"))
    task = TaskRecord(id="task-upd", run_id="run-ut", title="orig", prompt="p")
    await store.create_task(task)
    now = datetime.now(timezone.utc)
    await store.update_task("task-upd",
                            status="completed",
                            success=True,
                            finished_at=now)
    tasks = await store.list_tasks("run-ut")
    got = tasks[0]
    assert got.status == "completed"
    assert got.success is True
    assert got.finished_at is not None
    assert got.finished_at == now


@pytest.mark.asyncio
async def test_create_and_list_artifacts(mem_store):
    store = mem_store
    await store.create_run(_make_run("run-art"))
    task = TaskRecord(id="task-art", run_id="run-art", title="a", prompt="p")
    await store.create_task(task)
    art1 = ArtifactRecord(id="art-1", task_id="task-art", run_id="run-art",
                          kind="stdout", path="/out", size=10)
    art2 = ArtifactRecord(id="art-2", task_id="task-art", run_id="run-art",
                          kind="stderr", path="/err", size=5)
    await store.create_artifact(art1)
    await store.create_artifact(art2)

    all_arts = await store.list_artifacts("run-art")
    assert len(all_arts) == 2

    filtered = await store.list_artifacts("run-art", task_id="task-art")
    assert len(filtered) == 2

    empty = await store.list_artifacts("run-art", task_id="nonexistent")
    assert len(empty) == 0


@pytest.mark.asyncio
async def test_create_worker(monkeypatch, tmp_path):
    monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "runs.db")
    store = await RunStore.create_worker()
    assert store._db is not None
    assert store._owned is False
    await store.create_run(_make_run("worker-run"))
    got = await store.get_run("worker-run")
    assert got is not None
    await store.close()


@pytest.mark.asyncio
async def test_close_owned(monkeypatch, tmp_path):
    monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "runs.db")
    store = RunStore(db=None)
    assert store._owned is True
    await store.create_run(_make_run("owned-run"))
    await store.close()
    assert store._db is None


@pytest.mark.asyncio
async def test_close_unowned(mem_store):
    store = mem_store
    assert store._owned is False
    await store.close()
    assert store._db is None


@pytest.mark.asyncio
async def test_close_unowned_twice(mem_store):
    store = mem_store
    await store.close()
    assert store._db is None
    await store.close()
    assert store._db is None


@pytest.mark.asyncio
async def test_lazy_conn(monkeypatch, tmp_path):
    monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "runs.db")
    old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
    os.environ["DISTILL_GYM_CACHE_DIR"] = str(tmp_path)
    try:
        store = RunStore(db=None)
        assert store._db is None
        assert store._owned is True
        await store.create_run(_make_run("lazy-run"))
        assert store._db is not None
        got = await store.get_run("lazy-run")
        assert got is not None
        await store.close()
    finally:
        if old_env is not None:
            os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
        else:
            os.environ.pop("DISTILL_GYM_CACHE_DIR", None)


# ---------------------------------------------------------------------------
# db.py tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_db_custom_path(tmp_path):
    db_path = tmp_path / "custom" / "test.db"
    conn = await get_db(db_path)
    try:
        assert db_path.exists()
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in await cursor.fetchall()}
        assert "runs" in tables
        assert "tasks" in tables
        assert "artifacts" in tables
        cursor2 = await conn.execute("PRAGMA journal_mode")
        row = await cursor2.fetchone()
        assert row[0].lower() == "wal"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_ensure_tables_idempotent(tmp_path):
    db_path = tmp_path / "migrate_test.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        await conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, config_yaml TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, harness_type TEXT DEFAULT '',
            provider_name TEXT DEFAULT '', model TEXT DEFAULT '',
            sandbox_type TEXT DEFAULT '', sandbox_engine TEXT DEFAULT '',
            repo_url TEXT DEFAULT '', commit_hash TEXT DEFAULT '',
            success INTEGER
        )""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, title TEXT DEFAULT '',
            prompt TEXT DEFAULT '', status TEXT DEFAULT 'pending',
            exit_code INTEGER, success INTEGER, started_at TEXT,
            finished_at TEXT, test_command TEXT, tests_passed INTEGER
        )""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL, run_id TEXT NOT NULL,
            kind TEXT NOT NULL, path TEXT NOT NULL, size INTEGER DEFAULT 0
        )""")
        await conn.commit()
        for table in ("runs", "tasks"):
            cursor = await conn.execute(f"PRAGMA table_info({table})")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "error_message" not in cols, (
                f"{table} should NOT have error_message yet"
            )
        await _ensure_tables(conn)
        for table in ("runs", "tasks"):
            cursor = await conn.execute(f"PRAGMA table_info({table})")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "error_message" in cols, (
                f"{table} should have error_message after migration"
            )
        await _ensure_tables(conn)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_get_db_default_path_creates_tables(monkeypatch, tmp_path):
    monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "default" / "runs.db")
    conn = await get_db()
    try:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in await cursor.fetchall()}
        assert "runs" in tables
        assert "tasks" in tables
        assert "artifacts" in tables
    finally:
        await conn.close()
