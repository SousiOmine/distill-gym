import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from distill_gym.config.loader import load_config
from distill_gym.config.schema import Config
from distill_gym.orchestrator.orchestrator import run as run_orch, cleanup as cleanup_orch
from distill_gym.storage.run_store import RunStore
from distill_gym.exporters.openai_messages import export_openai_messages_jsonl
from distill_gym.proxy.app import create_proxy_app
from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.cache.cache_store import get_artifacts_dir
from distill_gym.storage.db import get_db

import uvicorn


app = typer.Typer(help="distill-gym: AI Harness trace collector and SFT dataset generator")


@app.command()
def init():
    """Initialize distill-gym cache directories"""
    from distill_gym.cache.cache_store import ensure_dirs
    ensure_dirs()
    typer.echo("Initialized distill-gym directories.")


@app.command()
def validate(config: str = typer.Argument(..., help="Path to config YAML")):
    """Validate a configuration file"""
    try:
        cfg = load_config(config)
        typer.echo(f"Config valid: {cfg.run.name=} {cfg.harness.type=} {cfg.provider.model=}")
    except Exception as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run without executing"),
):
    """Execute a run from config"""
    cfg = load_config(config)
    if dry_run:
        cfg.harness.type = "mock"

    try:
        run_id = asyncio.run(run_orch(cfg, dry_run=dry_run))
    except Exception as e:
        typer.echo(f"Run failed: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Run completed: {run_id}")


@app.command()
def proxy(
    config: str = typer.Argument(..., help="Path to config YAML"),
    run_id: str = typer.Option("", "--run-id", help="Associate proxy with a run ID"),
):
    """Start the OpenAI-compatible logging proxy"""
    cfg = load_config(config)
    recorder = None
    if run_id:
        trace_path = get_artifacts_dir() / run_id / "proxy" / "raw_trace.jsonl"
        recorder = TraceRecorder(trace_path)

    proxy_app = create_proxy_app(cfg.provider, cfg.logging_proxy, recorder)
    typer.echo(f"Starting proxy on {cfg.logging_proxy.listen_host}:{cfg.logging_proxy.listen_port}")
    uvicorn.run(
        proxy_app,
        host=cfg.logging_proxy.listen_host,
        port=cfg.logging_proxy.listen_port,
        log_level="info",
    )


@app.command()
def export(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to export"),
    format: str = typer.Option("openai-messages", "--format", help="Export format"),
    output: str = typer.Option("out.jsonl", "--output", help="Output file path"),
):
    """Export run data to SFT JSONL"""
    async def _export():
        store = RunStore()
        try:
            output_path = Path(output)
            if format == "openai-messages":
                count = await export_openai_messages_jsonl(
                    run_id=run_id, output=output_path, store=store,
                    include_reasoning=True, include_tool_results=True, include_failed=False,
                )
                typer.echo(f"Exported {count} conversations to {output_path}")
            else:
                typer.echo(f"Unknown format: {format}", err=True)
                raise typer.Exit(code=1)
        finally:
            await store.close()

    asyncio.run(_export())


@app.command()
def cleanup():
    """Remove all distill-gym containers, volumes, and networks"""
    result = asyncio.run(cleanup_orch())
    typer.echo(f"Cleaned up: {result['containers']} containers, {result['volumes']} volumes, {result['networks']} networks")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Web UI listen host"),
    port: int = typer.Option(8000, "--port", help="Web UI listen port"),
):
    """Start the web UI"""
    from distill_gym.web.app import create_web_app
    app = create_web_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
