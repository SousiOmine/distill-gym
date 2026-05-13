import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from distill_gym.config.schema import Config, CleanupPolicy, HarnessConfig, ProviderConfig
from distill_gym.orchestrator.run_plan import RunPlan
from distill_gym.orchestrator.events import EventBus, get_event_bus
from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import TaskRecord, ArtifactRecord
from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.proxy.app import create_proxy_app
from distill_gym.registry.harness_registry import HarnessRegistry
from distill_gym.registry.builder_registry import BuilderRegistry
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.taskgen.harness_task_generator import HarnessTaskGenerator
from distill_gym.collectors.artifact_collector import ArtifactCollector
from distill_gym.collectors.git_diff import collect_git_diff, collect_changed_files
from distill_gym.collectors.test_result import collect_test_result
from distill_gym.cache.cache_store import get_artifacts_dir

import uvicorn
import uuid

from distill_gym.sandbox.runtimes import create_runtime

logger = logging.getLogger(__name__)


async def _start_proxy(config: Config, recorder: TraceRecorder) -> tuple[uvicorn.Server, asyncio.Task]:
    app = create_proxy_app(config.provider, config.logging_proxy, recorder)
    uvicorn_config = uvicorn.Config(
        app,
        host=config.logging_proxy.listen_host,
        port=config.logging_proxy.listen_port,
        log_level="warning",
    )
    server = uvicorn.Server(uvicorn_config)
    task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            return server, task
        if task.done():
            await task
        await asyncio.sleep(0.1)
    server.should_exit = True
    raise RuntimeError(
        f"logging proxy did not start on "
        f"{config.logging_proxy.listen_host}:{config.logging_proxy.listen_port}"
    )


async def _stop_proxy(server: Optional[uvicorn.Server], task: Optional[asyncio.Task]) -> None:
    if not server or not task:
        return
    server.should_exit = True
    await asyncio.wait([task], timeout=10)


def _make_harness_from_config(
    harness_config: HarnessConfig,
    provider: Optional[ProviderConfig] = None,
    proxy_base_url: Optional[str] = None,
):
    return HarnessRegistry.create(
        harness_config.type,
        harness_config,
        provider=provider,
        proxy_base_url=proxy_base_url,
    )


def _make_harness(config: Config, **extra):
    return _make_harness_from_config(config.harness, **extra)


def _needs_harness_taskgen(config: Config) -> bool:
    return config.taskgen.type == "harness" and not config.taskgen.tasks


async def run(config: Config, dry_run: bool = False) -> str:
    plan = await RunPlan.from_config(config)
    store = RunStore()
    event_bus = get_event_bus()
    await store.create_run(plan.to_run_record())

    tasks_registered = False
    if plan.tasks:
        for tr in plan.to_task_records():
            await store.create_task(tr)
        tasks_registered = True

    if dry_run:
        await store.update_run(plan.run_id, status="completed", success=True)
        return plan.run_id

    proxy_base_url: Optional[str] = None
    sandbox_manager: Optional[SandboxManager] = None
    proxy_server: Optional[uvicorn.Server] = None
    proxy_task: Optional[asyncio.Task] = None
    proxy_recorder: Optional[TraceRecorder] = None
    created_network: str = ""

    try:
        needs_taskgen = _needs_harness_taskgen(config)
        needs_sandbox = config.harness.type != "mock" or needs_taskgen
        needs_proxy = config.harness.type != "mock" or (
            needs_taskgen and config.taskgen.harness.type != "mock"
        )

        if needs_proxy:
            proxy_base_url = f"http://host.containers.internal:{config.logging_proxy.listen_port}/v1"
            proxy_recorder = TraceRecorder(get_artifacts_dir() / plan.run_id / "proxy" / "raw_trace.jsonl")
            proxy_server, proxy_task = await _start_proxy(config, proxy_recorder)
            config.sandbox.env.setdefault("OPENAI_BASE_URL", proxy_base_url)
            config.sandbox.env.setdefault("OPENAI_API_KEY", "distill-gym-proxy")
            config.sandbox.env.setdefault("OPENAI_MODEL", config.provider.model)

        harness = _make_harness(
            config,
            provider=config.provider,
            proxy_base_url=proxy_base_url,
        )

        if needs_sandbox:
            builder = BuilderRegistry.create(config.sandbox.type)
            errors = builder.validate(config.sandbox)
            if errors:
                raise ValueError(f"Sandbox config validation failed: {'; '.join(errors)}")
            spec = builder.build(config.sandbox)

            if config.sandbox.network.mode.value == "proxy_only":
                network_name = f"distill-gym-{plan.run_id}"
                runtime = create_runtime(config.sandbox.engine.value)
                runtime.client.network_create(network_name)
                created_network = network_name
                spec.network_name = network_name

            sandbox_manager = SandboxManager()
            await sandbox_manager.start(spec)
            if spec.steps:
                await sandbox_manager.execute_steps(spec)
            else:
                await sandbox_manager.prepare_git_repository(config.sandbox)
            commit_code, commit_stdout, _ = await sandbox_manager.exec("git rev-parse HEAD", timeout=30)
            if commit_code == 0:
                await store.update_run(plan.run_id, commit_hash=commit_stdout.strip())

        if needs_taskgen:
            if sandbox_manager is None:
                raise RuntimeError("task generation requires a sandbox")
            taskgen_harness = _make_harness_from_config(
                config.taskgen.harness,
                provider=config.provider,
                proxy_base_url=proxy_base_url,
            )
            await taskgen_harness.install(sandbox_manager)
            taskgen = HarnessTaskGenerator(config.taskgen, taskgen_harness, sandbox_manager)
            plan.tasks = await taskgen.generate(config.run.task_count, plan.run_id)
            await taskgen.cleanup_output_file()

        if not tasks_registered:
            for tr in plan.to_task_records():
                await store.create_task(tr)
            tasks_registered = True

        if config.harness.type != "mock":
            await harness.install(sandbox_manager)

        await store.update_run(plan.run_id, status="running")
        await event_bus.emit("run_started", run_id=plan.run_id)

        task_failed = False
        for task in plan.tasks:
            db_task_id = f"{plan.run_id}_{task.id}"
            await store.update_task(
                db_task_id, status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            await event_bus.emit("task_started", run_id=plan.run_id, task_id=task.id, title=task.title)

            try:
                raw_trace_path = get_artifacts_dir() / plan.run_id / task.id / "raw_trace.jsonl"
                recorder = proxy_recorder or TraceRecorder(raw_trace_path)
                recorder.path = raw_trace_path
                recorder.record_event("task_started", {"task_id": task.id, "title": task.title}, db_task_id)

                result = await harness.run_task(sandbox_manager, task)

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
                if raw_trace_path.exists():
                    await store.create_artifact(ArtifactRecord(
                        id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                        kind="raw_trace", path=_rel_path(raw_trace_path), size=raw_trace_path.stat().st_size,
                    ))

                await store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                    kind="stdout", path=_rel_path(stdout_path), size=stdout_path.stat().st_size,
                ))
                await store.create_artifact(ArtifactRecord(
                    id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                    kind="stderr", path=_rel_path(stderr_path), size=stderr_path.stat().st_size,
                ))

                if sandbox_manager:
                    changed_files = await collect_changed_files(sandbox_manager)
                    if changed_files:
                        changed_files_path = collector.save_json("changed_files", changed_files)
                        await store.create_artifact(ArtifactRecord(
                            id=str(uuid.uuid4()), task_id=db_task_id, run_id=plan.run_id,
                            kind="changed_files", path=_rel_path(changed_files_path), size=changed_files_path.stat().st_size,
                        ))

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
                await event_bus.emit("task_finished", run_id=plan.run_id, task_id=task.id, success=result.success)

            except Exception as task_e:
                error_msg = f"Task '{task.id}' ({task.title}): {type(task_e).__name__}: {task_e}"
                logger.error(error_msg, exc_info=True)
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                await store.update_task(
                    db_task_id, status="failed", error_message=error_msg,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                await event_bus.emit("task_failed", run_id=plan.run_id, task_id=task.id, error=error_msg)
                task_failed = True
                break

        if task_failed:
            await store.update_run(plan.run_id, status="failed", success=False, error_message="One or more tasks failed")
            await event_bus.emit("run_failed", run_id=plan.run_id)
        else:
            all_success = True
            tasks_db = await store.list_tasks(plan.run_id)
            for t in tasks_db:
                if t.success is not True:
                    all_success = False
                    break
            await store.update_run(plan.run_id, status="completed", success=all_success)
            await event_bus.emit("run_completed", run_id=plan.run_id, success=all_success)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Run {plan.run_id} failed: {error_msg}", exc_info=True)
        print(f"[ERROR] Run failed: {error_msg}", file=sys.stderr)
        await store.update_run(plan.run_id, status="failed", success=False, error_message=error_msg)
        await event_bus.emit("run_failed", run_id=plan.run_id, error=error_msg)
        raise RuntimeError(f"Run {plan.run_id}: {error_msg}") from e

    finally:
        try:
            if created_network and sandbox_manager:
                runtime = create_runtime(config.sandbox.engine.value)
                runtime.client.network_rm(created_network)
        except Exception:
            logger.warning(f"Failed to remove network {created_network}", exc_info=True)

        await _stop_proxy(proxy_server, proxy_task)
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
