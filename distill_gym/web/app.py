from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from distill_gym.storage.run_store import RunStore


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


def create_web_app() -> FastAPI:
    app = FastAPI(title="distill-gym WebUI")

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
        from distill_gym.cache.cache_store import get_artifacts_dir
        file_path = get_artifacts_dir() / path
        if file_path.exists():
            return FileResponse(str(file_path))
        return HTMLResponse("Not found", status_code=404)

    return app
