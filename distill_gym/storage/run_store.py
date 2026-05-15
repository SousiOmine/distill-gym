from datetime import datetime, timezone
from typing import Optional
import aiosqlite
from distill_gym.storage.db import get_db
from distill_gym.storage.models import RunRecord, TaskRecord, ArtifactRecord


class RunStore:
    def __init__(self, db: aiosqlite.Connection | None = None):
        self._db = db
        self._owned = db is None

    @classmethod
    async def create_worker(cls) -> "RunStore":
        db = await get_db()
        return cls(db=db)

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await get_db()
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_run(self, run: RunRecord) -> None:
        conn = await self._conn()
        await conn.execute(
            """INSERT INTO runs (id, name, config_yaml, status, created_at, updated_at,
               harness_type, provider_name, model, sandbox_type, sandbox_engine,
               repo_url, commit_hash, success, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run.id, run.name, run.config_yaml, run.status,
             run.created_at.isoformat(), run.updated_at.isoformat(),
             run.harness_type, run.provider_name, run.model,
             run.sandbox_type, run.sandbox_engine,
             run.repo_url, run.commit_hash, run.success, run.error_message),
        )
        await conn.commit()

    async def update_run(self, run_id: str, **kwargs) -> None:
        conn = await self._conn()
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, val in kwargs.items():
            if isinstance(val, datetime):
                kwargs[key] = val.isoformat()
            if isinstance(val, bool):
                kwargs[key] = int(val)
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [run_id]
        await conn.execute(f"UPDATE runs SET {sets} WHERE id = ?", vals)
        await conn.commit()

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = await self._conn()
        cursor = await conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return RunRecord(**dict(row))

    async def list_runs(self, limit: int | None = None, offset: int = 0) -> list[RunRecord]:
        conn = await self._conn()
        query = "SELECT * FROM runs ORDER BY created_at DESC"
        params: list = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            query += " OFFSET ?"
            params.append(offset)
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [RunRecord(**dict(r)) for r in rows]

    async def get_task_counts(self) -> dict[str, int]:
        conn = await self._conn()
        cursor = await conn.execute(
            "SELECT run_id, COUNT(*) as cnt FROM tasks GROUP BY run_id"
        )
        rows = await cursor.fetchall()
        return {row["run_id"]: row["cnt"] for row in rows}

    async def get_total_run_count(self) -> int:
        conn = await self._conn()
        cursor = await conn.execute("SELECT COUNT(*) as cnt FROM runs")
        row = await cursor.fetchone()
        return row["cnt"]

    async def create_task(self, task: TaskRecord) -> None:
        conn = await self._conn()
        await conn.execute(
            """INSERT INTO tasks (id, run_id, title, prompt, status, exit_code, success,
               started_at, finished_at, test_command, tests_passed, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task.id, task.run_id, task.title, task.prompt, task.status,
             task.exit_code, task.success,
             task.started_at.isoformat() if task.started_at else None,
             task.finished_at.isoformat() if task.finished_at else None,
             task.test_command, task.tests_passed, task.error_message),
        )
        await conn.commit()

    async def update_task(self, task_id: str, **kwargs) -> None:
        conn = await self._conn()
        for key, val in kwargs.items():
            if isinstance(val, datetime):
                kwargs[key] = val.isoformat()
            if isinstance(val, bool):
                kwargs[key] = int(val)
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [task_id]
        await conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", vals)
        await conn.commit()

    async def list_tasks(self, run_id: str) -> list[TaskRecord]:
        conn = await self._conn()
        cursor = await conn.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY id", (run_id,)
        )
        rows = await cursor.fetchall()
        return [TaskRecord(**dict(r)) for r in rows]

    async def create_artifact(self, art: ArtifactRecord) -> None:
        conn = await self._conn()
        await conn.execute(
            """INSERT INTO artifacts (id, task_id, run_id, kind, path, size)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (art.id, art.task_id, art.run_id, art.kind, art.path, art.size),
        )
        await conn.commit()

    async def list_artifacts(self, run_id: str, task_id: str | None = None) -> list[ArtifactRecord]:
        conn = await self._conn()
        if task_id:
            cursor = await conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND task_id = ?",
                (run_id, task_id),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ?", (run_id,)
            )
        rows = await cursor.fetchall()
        return [ArtifactRecord(**dict(r)) for r in rows]
