import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from distill_gym.config.schema import Config, CleanupPolicy
from distill_gym.orchestrator.run_plan import RunPlan
from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import TaskRecord, ArtifactRecord
from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.proxy.app import create_proxy_app
from distill_gym.harness.base import MockHarnessAdapter, HarnessAdapter
from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from distill_gym.harness.codex import CodexHarnessAdapter
from distill_gym.harness.opencode import OpencodeHarnessAdapter
from distill_gym.harness.qwen_code import QwenCodeHarnessAdapter
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.builders.git_repository import GitRepositorySandboxBuilder
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.collectors.artifact_collector import ArtifactCollector
from distill_gym.collectors.git_diff import collect_git_diff
from distill_gym.collectors.test_result import collect_test_result
from distill_gym.cache.cache_store import get_artifacts_dir, get_cache_dir
from distill_gym.storage.db import get_db

import uvicorn
import uuid

logger = logging.getLogger(__name__)


_handlers: dict[str, type[HarnessAdapter]] = {
    "mock": MockHarnessAdapter,
    "generic_cli": GenericCliHarnessAdapter,
    "codex": CodexHarnessAdapter,
    "opencode": OpencodeHarnessAdapter,
    "qwen-code": QwenCodeHarnessAdapter,
}


def _make_harness(config: Config) -> HarnessAdapter:
    cls = _handlers.get(config.harness.type)
    if cls is None:
        raise ValueError(f"Unknown harness type: {config.harness.type}")
    if cls is MockHarnessAdapter:
        return cls()
    return cls(config.harness)


async def run(config: Config, dry_run: bool = False) -> str:
    plan = await RunPlan.from_config(config)
    store = RunStore()
    await store.create_run(plan.to_run_record())

    for tr in plan.to_task_records():
        await store.create_task(tr)

    if dry_run:
        await store.update_run(plan.run_id, status="completed", success=True)
        return plan.run_id

    harness = _make_harness(config)
    sandbox_manager: Optional[SandboxManager] = None
    proxy_server = None

    try:
        if config.harness.type != "mock":
            builder = GitRepositorySandboxBuilder()
            spec = builder.build(config.sandbox)
            sandbox_manager = SandboxManager()
            await sandbox_manager.start(spec)
            await sandbox_manager.prepare_git_repository(config.sandbox)
            await harness.install(sandbox_manager)

        await store.update_run(plan.run_id, status="running")

        task_failed = False
        for task in plan.tasks:
            db_task_id = f"{plan.run_id}_{task.id}"
            await store.update_task(
                db_task_id, status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            try:
                recorder = TraceRecorder(
                    get_artifacts_dir() / plan.run_id / task.id / "raw_trace.jsonl"
                )
                recorder.record_event("task_started", {"task_id": task.id, "title": task.title}, db_task_id)

                if config.harness.type == "mock":
                    result = await harness.run_task(sandbox_manager, task, {})
                else:
                    result = await harness.run_task(sandbox_manager, task, {})

                recorder.record_event("task_finished", {
                    "exit_code": result.exit_code,
                    "success": result.success,
                }, db_task_id)

                collector = ArtifactCollector(plan.run_id, task.id)
                artifacts_base = get_artifacts_dir()

                def _rel_path(p: Path) -> str:
                    try:
                        return str(p.relative_to(artifacts_base).as_posix())
                    except ValueError:
                        return p.name

                stdout_path = collector.save_text("stdout", result.stdout)
                stderr_path = collector.save_text("stderr", result.stderr)

                await store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                    kind="stdout", path=_rel_path(stdout_path), size=stdout_path.stat().st_size,
                ))
                await store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                    kind="stderr", path=_rel_path(stderr_path), size=stderr_path.stat().st_size,
                ))

                if sandbox_manager:
                    diff = await collect_git_diff(sandbox_manager)
                    if diff:
                        diff_path = collector.save_text("final", diff)
                        await store.create_artifact(ArtifactRecord(
                            id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                            kind="diff", path=_rel_path(diff_path), size=diff_path.stat().st_size,
                        ))

                    if task.test_command:
                        test_res = await collect_test_result(sandbox_manager, task.test_command)
                        test_path = collector.save_json("test_result", test_res)
                        await store.create_artifact(ArtifactRecord(
                            id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                            kind="test_result", path=_rel_path(test_path), size=test_path.stat().st_size,
                        ))
                        await store.update_task(db_task_id, tests_passed=test_res["passed"])

                metadata = {
                    "run_id": plan.run_id,
                    "task_id": task.id,
                    "exit_code": result.exit_code,
                    "success": result.success,
                    "harness_type": config.harness.type,
                }
                metadata_path = collector.save_json("metadata", metadata)
                await store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                    kind="metadata", path=_rel_path(metadata_path), size=metadata_path.stat().st_size,
                ))

                await store.update_task(db_task_id, status="completed", exit_code=result.exit_code, success=result.success)

            except Exception as task_e:
                error_msg = f"Task '{task.id}' ({task.title}): {type(task_e).__name__}: {task_e}"
                logger.error(error_msg, exc_info=True)
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                await store.update_task(
                    db_task_id, status="failed", error_message=error_msg,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                task_failed = True
                break  # Stop processing remaining tasks

        if task_failed:
            await store.update_run(plan.run_id, status="failed", success=False, error_message="One or more tasks failed")
        else:
            all_success = True
            tasks_db = await store.list_tasks(plan.run_id)
            for t in tasks_db:
                if t.success is not True:
                    all_success = False
                    break

            await store.update_run(plan.run_id, status="completed", success=all_success)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Run {plan.run_id} failed: {error_msg}", exc_info=True)
        print(f"[ERROR] Run failed: {error_msg}", file=sys.stderr)
        await store.update_run(plan.run_id, status="failed", success=False, error_message=error_msg)
        raise RuntimeError(f"Run {plan.run_id}: {error_msg}") from e

    finally:
        if sandbox_manager and config.run.cleanup == CleanupPolicy.always:
            await sandbox_manager.destroy()
        elif sandbox_manager and config.run.cleanup == CleanupPolicy.on_failure:
            run_rec = await store.get_run(plan.run_id)
            if run_rec and run_rec.success is not True:
                await sandbox_manager.destroy()
        await store.close()

    return plan.run_id


async def cleanup(label: str = "distill-gym=true") -> dict:
    manager = SandboxManager()
    return await manager.cleanup_resources(label)
