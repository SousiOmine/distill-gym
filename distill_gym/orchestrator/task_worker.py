import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from distill_gym.config.schema import TaskItem
from distill_gym.harness.base import HarnessAdapter, HarnessResult
from distill_gym.orchestrator.trace_pipeline import TracePipeline
from distill_gym.sandbox.isolation import IsolationSession
from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import ArtifactRecord
from distill_gym.collectors.artifact_collector import ArtifactCollector
from distill_gym.collectors.git_diff import collect_git_diff, collect_changed_files
from distill_gym.collectors.test_result import collect_test_result
from distill_gym.cache.cache_store import get_artifacts_dir

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task_id: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class TaskWorker:
    def __init__(
        self,
        harness: HarnessAdapter,
        store: RunStore,
        trace_pipeline: TracePipeline,
        run_id: str,
    ) -> None:
        self._harness = harness
        self._store = store
        self._trace_pipeline = trace_pipeline
        self._run_id = run_id

    async def execute(
        self, task: TaskItem, session: IsolationSession,
    ) -> TaskResult:
        db_task_id = f"{self._run_id}_{task.id}"
        short_id = task.id

        try:
            await self._store.update_task(
                db_task_id, status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            self._trace_pipeline.register_task(short_id)
            await self._trace_pipeline.record_event(
                short_id, "task_started",
                {"task_id": short_id, "title": task.title},
            )

            result = await self._harness.run_task(session.sandbox, task)

            await self._trace_pipeline.record_event(
                short_id, "task_finished",
                {"exit_code": result.exit_code, "success": result.success},
            )

            artifacts_base = get_artifacts_dir()
            collector = ArtifactCollector(self._run_id, short_id)

            def _rel_path(p: Path) -> str:
                try:
                    return str(p.relative_to(artifacts_base).as_posix())
                except ValueError:
                    return p.name

            stdout_path = collector.save_text("stdout", result.stdout)
            stderr_path = collector.save_text("stderr", result.stderr)

            raw_trace_path = artifacts_base / self._run_id / short_id / "raw_trace.jsonl"
            if raw_trace_path.exists():
                await self._store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                    kind="raw_trace", path=_rel_path(raw_trace_path), size=raw_trace_path.stat().st_size,
                ))

            await self._store.create_artifact(ArtifactRecord(
                id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                kind="stdout", path=_rel_path(stdout_path), size=stdout_path.stat().st_size,
            ))
            await self._store.create_artifact(ArtifactRecord(
                id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                kind="stderr", path=_rel_path(stderr_path), size=stderr_path.stat().st_size,
            ))

            changed_files = await collect_changed_files(session.sandbox)
            if changed_files:
                changed_files_path = collector.save_json("changed_files", changed_files)
                await self._store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                    kind="changed_files", path=_rel_path(changed_files_path), size=changed_files_path.stat().st_size,
                ))

            diff = await collect_git_diff(session.sandbox)
            if diff:
                diff_path = collector.save_text("final", diff)
                await self._store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                    kind="diff", path=_rel_path(diff_path), size=diff_path.stat().st_size,
                ))

            if task.test_command:
                test_res = await collect_test_result(session.sandbox, task.test_command)
                test_path = collector.save_json("test_result", test_res)
                await self._store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                    kind="test_result", path=_rel_path(test_path), size=test_path.stat().st_size,
                ))
                await self._store.update_task(db_task_id, tests_passed=test_res["passed"])

            metadata = {
                "run_id": self._run_id,
                "task_id": short_id,
                "exit_code": result.exit_code,
                "success": result.success,
                "harness_type": type(self._harness).__name__,
            }
            metadata_path = collector.save_json("metadata", metadata)
            await self._store.create_artifact(ArtifactRecord(
                id=str(uuid.uuid4()), task_id=db_task_id, run_id=self._run_id,
                kind="metadata", path=_rel_path(metadata_path), size=metadata_path.stat().st_size,
            ))

            await self._store.update_task(
                db_task_id, status="completed",
                exit_code=result.exit_code, success=result.success,
            )

            return TaskResult(short_id, result.success, result.exit_code, result.stdout, result.stderr)

        except Exception as e:
            error_msg = f"Task '{task.id}' ({task.title}): {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            try:
                await self._store.update_task(
                    db_task_id, status="failed", error_message=error_msg,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception:
                logger.warning("Failed to update task status after error", exc_info=True)
            return TaskResult(short_id, False, -1, error_message=error_msg)
