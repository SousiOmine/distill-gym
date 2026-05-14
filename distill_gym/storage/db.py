import aiosqlite
from pathlib import Path


DB_PATH = Path.home() / ".cache" / "distill-gym" / "runs.db"


async def get_db(path: Path | None = None) -> aiosqlite.Connection:
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path), timeout=10)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=10000")
    await _ensure_tables(conn)
    return conn


CREATE_RUNS_TABLE = """CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    harness_type TEXT DEFAULT '',
    provider_name TEXT DEFAULT '',
    model TEXT DEFAULT '',
    sandbox_type TEXT DEFAULT '',
    sandbox_engine TEXT DEFAULT '',
    repo_url TEXT DEFAULT '',
    commit_hash TEXT DEFAULT '',
    success INTEGER,
    error_message TEXT
)"""

CREATE_TASKS_TABLE = """CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    prompt TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    exit_code INTEGER,
    success INTEGER,
    started_at TEXT,
    finished_at TEXT,
    test_command TEXT,
    tests_passed INTEGER,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
)"""

CREATE_ARTIFACTS_TABLE = """CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER DEFAULT 0,
    FOREIGN KEY (task_id) REFERENCES tasks(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
)"""


async def _ensure_tables(conn: aiosqlite.Connection) -> None:
    for stmt in [CREATE_RUNS_TABLE, CREATE_TASKS_TABLE, CREATE_ARTIFACTS_TABLE]:
        await conn.execute(stmt)
    # Migration: add error_message column for existing databases
    for table in ("runs", "tasks"):
        try:
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN error_message TEXT")
        except aiosqlite.OperationalError:
            pass  # Column already exists
    await conn.commit()
