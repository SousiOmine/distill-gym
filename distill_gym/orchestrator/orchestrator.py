import asyncio
import json
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from distill_gym.config.schema import Config, CleanupPolicy, HarnessConfig, ProviderConfig
from distill_gym.orchestrator.run_plan import RunPlan
from distill_gym.orchestrator.events import EventBus, get_event_bus
from distill_gym.harness.base import HarnessResult
from distill_gym.orchestrator.task_worker import TaskWorker, TaskResult
from distill_gym.orchestrator.trace_pipeline import TracePipeline
from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import TaskRecord, ArtifactRecord
from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.proxy.app import create_proxy_app
from distill_gym.proxy.addressing import (
    proxy_base_url_for_sandbox,
    proxy_base_url_for_task,
    proxy_listen_host_for_sandbox,
)
from distill_gym.registry.harness_registry import HarnessRegistry
from distill_gym.registry.builder_registry import BuilderRegistry
import distill_gym.sandbox.builders  # noqa: F401
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.sandbox.pool import SandboxPool
from distill_gym.sandbox.isolation import ContainerIsolation
from distill_gym.taskgen.harness_task_generator import HarnessTaskGenerator
from distill_gym.collectors.artifact_collector import ArtifactCollector
from distill_gym.cache.cache_store import get_artifacts_dir
from distill_gym.platform.compatibility import ensure_podman_ready
from distill_gym.platform.detection import detect

import uvicorn
import uuid

from distill_gym.sandbox.runtime import SandboxRuntime
from distill_gym.sandbox.runtimes import create_runtime

logger = logging.getLogger(__name__)


async def _start_proxy(
    config: Config,
    recorder: TraceRecorder,
    listen_host: Optional[str] = None,
) -> tuple[uvicorn.Server, asyncio.Task]:
    app = create_proxy_app(config.provider, config.logging_proxy, recorder)
    host = listen_host or config.logging_proxy.listen_host
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
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
        f"{host}:{config.logging_proxy.listen_port}"
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


def _proxy_health_url(proxy_base_url: str) -> str:
    parsed = urlparse(proxy_base_url)
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def _effective_concurrency(config: Config) -> int:
    return max(1, config.run.concurrency)


async def _ensure_proxy_reachable_from_sandbox(
    sandbox_manager: SandboxManager,
    proxy_base_url: str,
) -> None:
    health_url = _proxy_health_url(proxy_base_url)
    py = (
        "import sys, urllib.request\n"
        f"url = {health_url!r}\n"
        "try:\n"
        "    response = urllib.request.urlopen(url, timeout=5)\n"
        "    sys.exit(0 if response.status == 200 else 1)\n"
        "except Exception as exc:\n"
        "    print(exc, file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    command = (
        "if command -v python3 >/dev/null 2>&1; then "
        f"python3 -c {shlex.quote(py)}; "
        "elif command -v python >/dev/null 2>&1; then "
        f"python -c {shlex.quote(py)}; "
        "elif command -v curl >/dev/null 2>&1; then "
        f"curl -fsS --max-time 5 {shlex.quote(health_url)} >/dev/null; "
        "else exit 77; fi"
    )
    code, stdout, stderr = await sandbox_manager.exec(command, timeout=10, workdir="/")
    if code == 0:
        return
    if code == 77:
        logger.info(
            "Skipping logging proxy reachability check: sandbox has neither python nor curl"
        )
        return

    detail = (stderr.strip() or stdout.strip() or f"exit code {code}").strip()
    raise RuntimeError(
        "sandbox cannot reach the logging proxy at "
        f"{health_url}: {detail}. On Windows with WSL/Podman, check Windows "
        "firewall rules, the WSL default gateway selected by "
        "logging_proxy.sandbox_host=auto, or set logging_proxy.sandbox_host "
        "explicitly."
    )


async def _setup_run(config: Config) -> tuple[RunPlan, RunStore]:
    plan = await RunPlan.from_config(config)
    store = RunStore()
    await store.create_run(plan.to_run_record())
    return plan, store


async def _execute_run(config: Config, plan: RunPlan, store: RunStore, dry_run: bool = False) -> str:
    event_bus = get_event_bus()
    concurrency = _effective_concurrency(config)

    tasks_registered = False
    if plan.tasks:
        for tr in plan.to_task_records():
            await store.create_task(tr)
        tasks_registered = True

    if dry_run:
        await store.update_run(plan.run_id, status="completed", success=True)
        return plan.run_id

    proxy_base_url: Optional[str] = None
    sandbox_runtime: Optional[SandboxRuntime] = None
    proxy_server: Optional[uvicorn.Server] = None
    proxy_task: Optional[asyncio.Task] = None
    proxy_recorder: Optional[TraceRecorder] = None
    created_network: str = ""
    pool: Optional[SandboxPool] = None

    try:
        needs_taskgen = _needs_harness_taskgen(config)
        needs_sandbox = config.harness.type != "mock" or needs_taskgen
        needs_proxy = config.harness.type != "mock" or (
            needs_taskgen and config.taskgen.harness.type != "mock"
        )

        platform = detect()
        if needs_sandbox and config.sandbox.engine.value == "podman":
            ensure_podman_ready(platform)

        if needs_proxy:
            proxy_base_url = proxy_base_url_for_sandbox(config, platform)
            proxy_recorder = TraceRecorder(get_artifacts_dir() / plan.run_id / "proxy" / "raw_trace.jsonl")
            proxy_listen_host = proxy_listen_host_for_sandbox(config, platform)
            proxy_server, proxy_task = await _start_proxy(
                config,
                proxy_recorder,
                listen_host=proxy_listen_host,
            )
            logger.info(
                "Logging proxy listening on %s:%s; sandbox OPENAI_BASE_URL=%s",
                proxy_listen_host,
                config.logging_proxy.listen_port,
                proxy_base_url,
            )
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

            sandbox_runtime = create_runtime(config.sandbox.engine.value)
            if config.sandbox.network.mode.value == "proxy_only":
                network_name = f"distill-gym-{plan.run_id}"
                await sandbox_runtime.client.network_create(network_name)
                created_network = network_name
                spec.network_name = network_name

            pool = SandboxPool(
                runtime=sandbox_runtime,
                spec_template=spec,
                max_size=concurrency,
                init_hook=harness.install if config.harness.type != "mock" else None,
            )

            setup_sandbox = await pool.acquire()
            try:
                commit_code, commit_stdout, _ = await setup_sandbox.exec("git rev-parse HEAD", timeout=30)
                if commit_code == 0:
                    await store.update_run(plan.run_id, commit_hash=commit_stdout.strip())

                if proxy_base_url:
                    await _ensure_proxy_reachable_from_sandbox(setup_sandbox, proxy_base_url)
            finally:
                await pool.release(setup_sandbox)

        if needs_taskgen:
            if pool is None:
                raise RuntimeError("task generation requires a sandbox")
            taskgen_sandbox = await pool.acquire()
            try:
                taskgen_harness = _make_harness_from_config(
                    config.taskgen.harness,
                    provider=config.provider,
                    proxy_base_url=proxy_base_url,
                )
                await taskgen_harness.install(taskgen_sandbox)
                taskgen = HarnessTaskGenerator(config.taskgen, taskgen_harness, taskgen_sandbox)
                plan.tasks = await taskgen.generate(config.run.task_count, plan.run_id)
                await taskgen.cleanup_output_file()
            finally:
                await pool.release(taskgen_sandbox)

        if not tasks_registered:
            for tr in plan.to_task_records():
                await store.create_task(tr)
            tasks_registered = True

        if config.harness.type == "mock":
            await store.update_run(plan.run_id, status="running")
            await event_bus.emit("run_started", run_id=plan.run_id)
            results = await _run_tasks_mock(plan, store, event_bus)
        else:
            if pool is None:
                raise RuntimeError("sandbox pool not initialized")
            isolation = ContainerIsolation(pool)
            trace_pipeline = TracePipeline(proxy_recorder, plan.run_id)

            await store.update_run(plan.run_id, status="running")
            await event_bus.emit("run_started", run_id=plan.run_id)
            results = await _run_tasks_concurrent(
                config, plan, store, isolation, trace_pipeline, concurrency, event_bus,
                proxy_base_url,
            )

            await trace_pipeline.distribute()

        task_failed = any(not r.success for r in results)

        if task_failed:
            await store.update_run(plan.run_id, status="failed", success=False, error_message="One or more tasks failed")
            await event_bus.emit("run_failed", run_id=plan.run_id)
        else:
            await store.update_run(plan.run_id, status="completed", success=True)
            await event_bus.emit("run_completed", run_id=plan.run_id, success=True)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Run {plan.run_id} failed: {error_msg}", exc_info=True)
        print(f"[ERROR] Run failed: {error_msg}", file=sys.stderr)
        await store.update_run(plan.run_id, status="failed", success=False, error_message=error_msg)
        await event_bus.emit("run_failed", run_id=plan.run_id, error=error_msg)
        raise RuntimeError(f"Run {plan.run_id}: {error_msg}") from e

    finally:
        try:
            if created_network and sandbox_runtime:
                await sandbox_runtime.client.network_rm(created_network)
        except Exception:
            logger.warning(f"Failed to remove network {created_network}", exc_info=True)

        if pool:
            await pool.destroy_all()

        await _stop_proxy(proxy_server, proxy_task)

    return plan.run_id


async def _run_tasks_concurrent(
    config: Config,
    plan: RunPlan,
    store: RunStore,
    isolation: ContainerIsolation,
    trace_pipeline: TracePipeline,
    concurrency: int,
    event_bus: EventBus,
    proxy_base_url: Optional[str] = None,
) -> list[TaskResult]:
    async def _worker(task) -> TaskResult:
        worker_store = await RunStore.create_worker()
        try:
            task_proxy_base_url = (
                proxy_base_url_for_task(proxy_base_url, task.id)
                if proxy_base_url else None
            )
            task_harness = _make_harness(
                config,
                provider=config.provider,
                proxy_base_url=task_proxy_base_url,
            )
            worker = TaskWorker(task_harness, worker_store, trace_pipeline, plan.run_id)
            session = await isolation.acquire(task.id)
            try:
                await event_bus.emit("task_started", run_id=plan.run_id, task_id=task.id, title=task.title)
                result = await worker.execute(task, session)
                await event_bus.emit("task_finished", run_id=plan.run_id, task_id=task.id, success=result.success)
                return result
            finally:
                await isolation.release(session)
        finally:
            await worker_store.close()

    tasks = [_worker(task) for task in plan.tasks]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[TaskResult] = []
    for item in gathered:
        if isinstance(item, BaseException):
            logger.error(f"Unexpected task error: {item}", exc_info=item)
            results.append(TaskResult("", False, -1, error_message=str(item)))
        else:
            results.append(item)
    return results


async def _run_tasks_mock(
    plan: RunPlan,
    store: RunStore,
    event_bus: EventBus,
) -> list[TaskResult]:
    from distill_gym.harness.base import MockHarnessAdapter

    mock = MockHarnessAdapter()
    results: list[TaskResult] = []

    for task in plan.tasks:
        db_task_id = f"{plan.run_id}_{task.id}"
        await store.update_task(
            db_task_id, status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        await event_bus.emit("task_started", run_id=plan.run_id, task_id=task.id, title=task.title)

        harness_result = await mock.run_task(None, task)  # type: ignore[arg-type]

        collector = ArtifactCollector(plan.run_id, task.id)
        collector.save_text("stdout", harness_result.stdout)
        collector.save_text("stderr", harness_result.stderr)

        metadata = {
            "run_id": plan.run_id,
            "task_id": task.id,
            "exit_code": 0,
            "success": True,
            "harness_type": "mock",
        }
        collector.save_json("metadata", metadata)

        await store.update_task(db_task_id, status="completed", exit_code=0, success=True)
        await event_bus.emit("task_finished", run_id=plan.run_id, task_id=task.id, success=True)

        results.append(TaskResult(task.id, True, 0))

    return results


async def run(config: Config, dry_run: bool = False) -> str:
    plan, store = await _setup_run(config)
    try:
        return await _execute_run(config, plan, store, dry_run)
    finally:
        await store.close()


async def cleanup(label: str = "distill-gym=true") -> dict:
    from distill_gym.sandbox.manager import SandboxManager
    manager = SandboxManager()
    return await manager.cleanup_resources(label)
