import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from distill_gym.storage.run_store import RunStore
from distill_gym.cache.cache_store import get_artifacts_dir

logger = logging.getLogger(__name__)


STATIC_DIR = Path(__file__).parent / "static"
HAS_REACT_BUILD = STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists()


def create_web_app() -> FastAPI:
    app = FastAPI(title="distill-gym WebUI")

    # --- REST API Routes ---

    @app.get("/api/runs")
    async def api_list_runs(limit: int = 50, offset: int = 0):
        store = RunStore()
        try:
            runs = await store.list_runs(limit=limit, offset=offset)
            task_counts = await store.get_task_counts()
            total = await store.get_total_run_count()
            return {
                "runs": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "status": r.status,
                        "harness_type": r.harness_type,
                        "model": r.model,
                        "provider_name": r.provider_name,
                        "sandbox_type": r.sandbox_type,
                        "repo_url": r.repo_url,
                        "success": r.success,
                        "error_message": r.error_message,
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                        "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                        "commit_hash": r.commit_hash,
                        "task_count": task_counts.get(r.id, 0),
                    }
                    for r in runs
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        finally:
            await store.close()

    @app.get("/api/runs/{run_id}")
    async def api_get_run(run_id: str):
        store = RunStore()
        try:
            r = await store.get_run(run_id)
            if not r:
                raise HTTPException(404, "Run not found")
            return {
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "harness_type": r.harness_type,
                "model": r.model,
                "provider_name": r.provider_name,
                "sandbox_type": r.sandbox_type,
                "repo_url": r.repo_url,
                "success": r.success,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                "commit_hash": r.commit_hash,
            }
        finally:
            await store.close()

    @app.get("/api/runs/{run_id}/tasks")
    async def api_list_tasks(run_id: str):
        store = RunStore()
        try:
            tasks = await store.list_tasks(run_id)
            return [
                {
                    "id": t.id,
                    "run_id": t.run_id,
                    "title": t.title,
                    "prompt": t.prompt,
                    "status": t.status,
                    "exit_code": t.exit_code,
                    "success": t.success,
                    "test_command": t.test_command,
                    "tests_passed": t.tests_passed,
                    "error_message": t.error_message,
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "finished_at": t.finished_at.isoformat() if t.finished_at else None,
                }
                for t in tasks
            ]
        finally:
            await store.close()

    @app.get("/api/runs/{run_id}/tasks/{task_id}")
    async def api_get_task(run_id: str, task_id: str):
        store = RunStore()
        try:
            tasks = await store.list_tasks(run_id)
            task = next((t for t in tasks if t.id == task_id), None)
            if not task:
                raise HTTPException(404, "Task not found")
            return {
                "id": task.id,
                "run_id": task.run_id,
                "title": task.title,
                "prompt": task.prompt,
                "status": task.status,
                "exit_code": task.exit_code,
                "success": task.success,
                "test_command": task.test_command,
                "tests_passed": task.tests_passed,
                "error_message": task.error_message,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            }
        finally:
            await store.close()

    @app.get("/api/runs/{run_id}/artifacts")
    async def api_list_artifacts(run_id: str, task_id: str | None = None):
        store = RunStore()
        try:
            arts = await store.list_artifacts(run_id, task_id)
            return [
                {
                    "id": a.id,
                    "task_id": a.task_id,
                    "run_id": a.run_id,
                    "kind": a.kind,
                    "path": a.path,
                    "size": a.size,
                }
                for a in arts
            ]
        finally:
            await store.close()

    class CreateRunRequest(BaseModel):
        config_yaml: str

    @app.post("/api/runs")
    async def api_create_run(req: CreateRunRequest):
        import tempfile
        from distill_gym.config.loader import load_config
        from distill_gym.orchestrator.orchestrator import _setup_run, _execute_run

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        )
        try:
            tmp.write(req.config_yaml)
            tmp.flush()
            tmp.close()
            cfg = load_config(tmp.name)
            plan, store = await _setup_run(cfg)
            await store.close()
            asyncio.create_task(_run_background(cfg, plan))
            return {"run_id": plan.run_id}
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    class CreateRunsBatchRequest(BaseModel):
        config_yamls: list[str]

    @app.post("/api/runs/batch")
    async def api_create_runs_batch(req: CreateRunsBatchRequest):
        from distill_gym.config.loader import load_config
        from distill_gym.orchestrator.orchestrator import _setup_run

        results: list[dict] = []
        for config_yaml in req.config_yamls:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8",
            )
            try:
                tmp.write(config_yaml)
                tmp.flush()
                tmp.close()
                cfg = load_config(tmp.name)
                plan, store = await _setup_run(cfg)
                await store.close()
                asyncio.create_task(_run_background(cfg, plan))
                results.append({
                    "run_id": plan.run_id,
                    "name": cfg.run.name,
                    "status": "submitted",
                })
            except Exception as e:
                logger.exception("Batch run submission failed")
                results.append({
                    "run_id": None,
                    "name": "",
                    "status": "error",
                    "error": str(e),
                })
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return {"runs": results}


    async def _run_background(config, plan):
        from distill_gym.orchestrator.orchestrator import _execute_run
        from distill_gym.storage.run_store import RunStore
        store = RunStore()
        try:
            await _execute_run(config, plan, store, dry_run=False)
        except Exception:
            logger.exception("Background run failed")
        finally:
            await store.close()

    class ExportRequest(BaseModel):
        run_id: str
        format: str = "openai-messages"
        include_failed: bool = False

    @app.post("/api/export")
    async def api_export(req: ExportRequest):
        from distill_gym.exporters.openai_messages import export_openai_messages_jsonl

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        store = RunStore()
        try:
            path = Path(tmp.name)
            if req.format == "openai-messages":
                count = await export_openai_messages_jsonl(
                    req.run_id, path, store,
                    include_failed=req.include_failed,
                )
            else:
                raise HTTPException(400, f"Unknown format: {req.format}")
            return {"count": count, "url": f"/api/download/{path.name}"}
        finally:
            await store.close()

    @app.get("/api/download/{filename}")
    async def api_download(filename: str):
        tmp = Path(tempfile.gettempdir()) / filename
        if not tmp.exists():
            raise HTTPException(404, "File not found")
        return FileResponse(str(tmp), media_type="application/jsonl", filename=filename)

    class MergeExportRequest(BaseModel):
        run_ids: list[str]
        format: str = "openai-messages"

    @app.post("/api/export/merge")
    async def api_merge_export(req: MergeExportRequest):
        from distill_gym.exporters.merger import merge_runs_to_jsonl

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        store = RunStore()
        try:
            path = Path(tmp.name)
            if req.format == "openai-messages":
                count = await merge_runs_to_jsonl(req.run_ids, path, store)
            else:
                raise HTTPException(400, f"Unknown format: {req.format}")
            return {"count": count, "url": f"/api/download/{path.name}"}
        finally:
            await store.close()

    @app.get("/api/config/examples")
    async def api_list_examples():
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        files = sorted(examples_dir.glob("*.yaml"))
        names = [f.stem for f in files]
        return names

    @app.get("/api/config/examples/{name}")
    async def api_config_example(name: str):
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        path = examples_dir / name
        if not path.suffix:
            path = path.with_suffix(".yaml")
        if not path.exists():
            raise HTTPException(404, f"Example config not found: {name}")
        return {"yaml": path.read_text(encoding="utf-8")}

    @app.post("/api/config/upload")
    async def api_upload_config(file: UploadFile = File(...)):
        content = await file.read()
        return {"yaml": content.decode("utf-8"), "filename": file.filename}

    # --- Static files (React SPA) ---

    if HAS_REACT_BUILD:
        app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

        @app.get("/artifacts/{path:path}")
        async def serve_artifact(path: str):
            file_path = get_artifacts_dir() / path
            if file_path.exists():
                return FileResponse(str(file_path))
            raise HTTPException(404, "Artifact not found")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(404)
            file_path = STATIC_DIR / "index.html"
            if file_path.exists():
                return HTMLResponse(file_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>distill-gym WebUI</h1><p>Build the frontend with <code>cd distill_gym/web/frontend && npm install && npm run build</code></p>")

    # --- Fallback: legacy Jinja2 templates ---
    else:
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        templates_path = Path(__file__).parent / "templates"
        loader = FileSystemLoader(searchpath=str(templates_path))
        _jinja_env = Environment(
            loader=loader,
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
            cache_size=0,
        )

        async def _render(name: str, context: dict) -> str:
            template = _jinja_env.get_template(name)
            return await template.render_async(context)

        @app.get("/", response_class=HTMLResponse)
        async def list_runs(request: Request):
            store = RunStore()
            runs = await store.list_runs()
            await store.close()
            html = await _render("runs.html", {"request": request, "runs": runs})
            return HTMLResponse(html)

        @app.get("/runs/{run_id}", response_class=HTMLResponse)
        async def run_detail(request: Request, run_id: str):
            store = RunStore()
            run = await store.get_run(run_id)
            tasks = await store.list_tasks(run_id)
            artifacts = await store.list_artifacts(run_id)
            await store.close()
            html = await _render("run_detail.html", {
                "request": request, "run": run, "tasks": tasks, "artifacts": artifacts,
            })
            return HTMLResponse(html)

        @app.get("/runs/{run_id}/tasks/{task_id}", response_class=HTMLResponse)
        async def task_detail(request: Request, run_id: str, task_id: str):
            store = RunStore()
            run = await store.get_run(run_id)
            tasks = await store.list_tasks(run_id)
            task = next((t for t in tasks if t.id == task_id), None)
            artifacts = await store.list_artifacts(run_id, task_id)
            await store.close()
            html = await _render("task_detail.html", {
                "request": request, "run": run, "task": task, "artifacts": artifacts,
            })
            return HTMLResponse(html)

        @app.get("/artifacts/{path:path}")
        async def serve_artifact(path: str):
            file_path = get_artifacts_dir() / path
            if file_path.exists():
                return FileResponse(str(file_path))
            raise HTTPException(404, "Artifact not found")

    return app
