import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml

from distill_gym.storage.models import RunRecord, TaskRecord, ArtifactRecord
from distill_gym.storage.run_store import RunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_test_run(store, run_id="test-run-1", name="Test Run"):
    run = RunRecord(
        id=run_id, name=name, config_yaml="version: 1", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    return run


async def _create_test_task(store, run_id, task_id="t1", title="Task 1"):
    task = TaskRecord(
        id=task_id, run_id=run_id, title=title, prompt="do",
        status="completed", success=True,
    )
    await store.create_task(task)
    return task


async def _create_test_artifact(store, run_id, task_id, art_id="a1", kind="stdout"):
    art = ArtifactRecord(
        id=art_id, task_id=task_id, run_id=run_id,
        kind=kind, path=f"/tmp/{art_id}", size=100,
    )
    await store.create_artifact(art)
    return art


async def _prepare_app(monkeypatch, tmp_path, extra_mocks=None):
    monkeypatch.setattr("distill_gym.storage.db.DB_PATH", tmp_path / "test.db")
    if extra_mocks:
        for target, mock in extra_mocks:
            monkeypatch.setattr(target, mock)
    from distill_gym.web.app import create_web_app
    return create_web_app()


# ---------------------------------------------------------------------------
# /api/runs (GET) - list runs
# ---------------------------------------------------------------------------

class TestListRuns:
    @pytest.mark.asyncio
    async def test_api_list_runs_empty(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_api_list_runs_with_data(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1", "Alpha")
        await _create_test_run(store, "r2", "Beta")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 2
        assert data["total"] == 2
        assert data["runs"][0]["name"] == "Beta"
        assert data["runs"][1]["name"] == "Alpha"

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs?limit=1&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 1
        assert data["total"] == 2
        assert data["limit"] == 1
        assert data["offset"] == 1

    @pytest.mark.asyncio
    async def test_api_list_runs_includes_task_counts(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1", "Run One")
        await _create_test_task(store, "r1", "t1")
        await _create_test_task(store, "r1", "t2")
        await _create_test_run(store, "r2", "Run Two")
        await _create_test_task(store, "r2", "t3")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs")
        data = resp.json()
        runs = {r["id"]: r for r in data["runs"]}
        assert runs["r1"]["task_count"] == 2
        assert runs["r2"]["task_count"] == 1
        assert runs["r1"]["task_count"] == 2

    @pytest.mark.asyncio
    async def test_api_list_runs_includes_total(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_run(store, "r2")
        await _create_test_run(store, "r3")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs")
        data = resp.json()
        assert "total" in data
        assert data["total"] == 3


# ---------------------------------------------------------------------------
# /api/runs/{run_id} (GET) - get run
# ---------------------------------------------------------------------------

class TestGetRun:
    @pytest.mark.asyncio
    async def test_api_get_run_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "my-run", "My Run")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/my-run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "my-run"
        assert data["name"] == "My Run"
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_api_get_run_not_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/non-existent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{run_id}/tasks (GET) - list tasks
# ---------------------------------------------------------------------------

class TestListTasks:
    @pytest.mark.asyncio
    async def test_api_list_tasks_empty(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/tasks")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_api_list_tasks_with_data(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_task(store, "r1", "t1", "Task One")
        await _create_test_task(store, "r1", "t2", "Task Two")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["title"] == "Task One"
        assert data[1]["title"] == "Task Two"


# ---------------------------------------------------------------------------
# /api/runs/{run_id}/tasks/{task_id} (GET) - get task
# ---------------------------------------------------------------------------

class TestGetTask:
    @pytest.mark.asyncio
    async def test_api_get_task_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_task(store, "r1", "my-task", "My Task")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/tasks/my-task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "my-task"
        assert data["title"] == "My Task"
        assert data["run_id"] == "r1"

    @pytest.mark.asyncio
    async def test_api_get_task_not_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/tasks/non-existent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{run_id}/artifacts (GET) - list artifacts
# ---------------------------------------------------------------------------

class TestListArtifacts:
    @pytest.mark.asyncio
    async def test_api_list_artifacts_empty(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_task(store, "r1", "t1")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/artifacts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_api_list_artifacts_with_data(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_task(store, "r1", "t1")
        await _create_test_artifact(store, "r1", "t1", "a1")
        await _create_test_artifact(store, "r1", "t1", "a2", kind="stderr")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/artifacts")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_api_list_artifacts_filter_by_task(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        store = RunStore()
        await _create_test_run(store, "r1")
        await _create_test_task(store, "r1", "t1")
        await _create_test_task(store, "r1", "t2")
        await _create_test_artifact(store, "r1", "t1", "a1")
        await _create_test_artifact(store, "r1", "t2", "a2")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/runs/r1/artifacts?task_id=t1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "a1"


# ---------------------------------------------------------------------------
# /api/runs (POST) - create run
# ---------------------------------------------------------------------------

class TestCreateRun:
    @pytest.mark.asyncio
    async def test_api_create_run(self, monkeypatch, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.run.name = "test-run"

        mock_plan = MagicMock()
        mock_plan.run_id = "mock-run-1"

        mock_store = AsyncMock()
        mock_store.close = AsyncMock()

        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.config.loader.load_config", MagicMock(return_value=mock_cfg)),
            ("distill_gym.orchestrator.orchestrator._setup_run",
             AsyncMock(return_value=(mock_plan, mock_store))),
            ("distill_gym.orchestrator.orchestrator._execute_run", AsyncMock()),
        ])

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/runs", json={"config_yaml": "version: 1"})
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "mock-run-1"
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_api_create_run_invalid_yaml(self, monkeypatch, tmp_path):
        from fastapi import HTTPException
        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.config.loader.load_config",
             MagicMock(side_effect=ValueError("Invalid YAML content"))),
        ])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with pytest.raises(Exception):
                await c.post("/api/runs", json={"config_yaml": "invalid"})


# ---------------------------------------------------------------------------
# /api/runs/batch (POST) - batch create runs
# ---------------------------------------------------------------------------

class TestBatchCreate:
    @pytest.mark.asyncio
    async def test_api_create_runs_batch(self, monkeypatch, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.run.name = "batch-run"

        mock_plan1 = MagicMock()
        mock_plan1.run_id = "batch-run-1"
        mock_plan2 = MagicMock()
        mock_plan2.run_id = "batch-run-2"

        mock_store = AsyncMock()
        mock_store.close = AsyncMock()

        mock_setup = AsyncMock()
        mock_setup.side_effect = [
            (mock_plan1, mock_store),
            (mock_plan2, mock_store),
        ]

        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.config.loader.load_config", MagicMock(return_value=mock_cfg)),
            ("distill_gym.orchestrator.orchestrator._setup_run", mock_setup),
            ("distill_gym.orchestrator.orchestrator._execute_run", AsyncMock()),
        ])

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/runs/batch", json={
                "config_yamls": ["version: 1", "version: 1"],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 2
        assert data["runs"][0]["run_id"] == "batch-run-1"
        assert data["runs"][1]["run_id"] == "batch-run-2"
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_api_create_runs_batch_partial_failure(self, monkeypatch, tmp_path):
        mock_cfg_ok = MagicMock()
        mock_cfg_ok.run.name = "ok-run"

        mock_plan = MagicMock()
        mock_plan.run_id = "ok-run-1"

        mock_store = AsyncMock()
        mock_store.close = AsyncMock()

        call_count = 0

        async def _setup_side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_plan, mock_store
            raise ValueError("Simulated failure")

        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.config.loader.load_config", MagicMock(return_value=mock_cfg_ok)),
            ("distill_gym.orchestrator.orchestrator._setup_run", _setup_side_effect),
            ("distill_gym.orchestrator.orchestrator._execute_run", AsyncMock()),
        ])

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/runs/batch", json={
                "config_yamls": ["version: 1", "version: 1"],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 2
        assert data["runs"][0]["status"] == "submitted"
        assert data["runs"][0]["run_id"] == "ok-run-1"
        assert data["runs"][1]["status"] == "error"
        assert data["runs"][1]["run_id"] is None
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# /api/export (POST) - export run
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.mark.asyncio
    async def test_api_export_success(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.exporters.openai_messages.export_openai_messages_jsonl",
             AsyncMock(return_value=5)),
        ])

        store = RunStore()
        await _create_test_run(store, "export-run")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/export", json={
                "run_id": "export-run", "format": "openai-messages",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5
        assert data["url"].startswith("/api/download/")

    @pytest.mark.asyncio
    async def test_api_export_unknown_format(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)

        store = RunStore()
        await _create_test_run(store, "export-run")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/export", json={
                "run_id": "export-run", "format": "unknown-format",
            })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/download/{filename} (GET)
# ---------------------------------------------------------------------------

class TestDownload:
    @pytest.mark.asyncio
    async def test_api_download_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w", encoding="utf-8",
        )
        tmp.write('{"test": true}\n')
        tmp.close()
        fname = Path(tmp.name).name

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get(f"/api/download/{fname}")
            assert resp.status_code == 200
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_api_download_not_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/download/non-existent-file.jsonl")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/export/merge (POST)
# ---------------------------------------------------------------------------

class TestMergeExport:
    @pytest.mark.asyncio
    async def test_api_merge_export_success(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path, extra_mocks=[
            ("distill_gym.exporters.merger.merge_runs_to_jsonl",
             AsyncMock(return_value=10)),
        ])

        store = RunStore()
        await _create_test_run(store, "merge-run-1")
        await _create_test_run(store, "merge-run-2")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/export/merge", json={
                "run_ids": ["merge-run-1", "merge-run-2"],
                "format": "openai-messages",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert data["url"].startswith("/api/download/")

    @pytest.mark.asyncio
    async def test_api_merge_export_unknown_format(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/export/merge", json={
                "run_ids": ["merge-run-1"],
                "format": "unknown-format",
            })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/config/examples (GET)
# ---------------------------------------------------------------------------

class TestConfigExamples:
    @pytest.mark.asyncio
    async def test_api_list_examples(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/config/examples")
        assert resp.status_code == 200
        names = resp.json()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    @pytest.mark.asyncio
    async def test_api_get_example(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/config/examples/mock_run")
        assert resp.status_code == 200
        data = resp.json()
        assert "yaml" in data
        assert isinstance(data["yaml"], str)

    @pytest.mark.asyncio
    async def test_api_get_example_not_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/config/examples/non_existent_example")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/config/upload (POST)
# ---------------------------------------------------------------------------

class TestUploadConfig:
    @pytest.mark.asyncio
    async def test_api_upload_config(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/config/upload",
                files={"file": ("test.yaml", "version: 1\nrun:\n  name: test\n", "text/yaml")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "yaml" in data
        assert data["filename"] == "test.yaml"


# ---------------------------------------------------------------------------
# Legacy Jinja2 templates (HAS_REACT_BUILD=False)
# ---------------------------------------------------------------------------

class TestLegacyTemplates:
    @pytest.mark.asyncio
    async def test_legacy_list_runs_page(self, monkeypatch, tmp_path):
        import distill_gym.web.app as web_app_mod
        monkeypatch.setattr(web_app_mod, "HAS_REACT_BUILD", False)
        app = await _prepare_app(monkeypatch, tmp_path)

        store = RunStore()
        await _create_test_run(store, "r1", "Legacy Run")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_legacy_run_detail_page(self, monkeypatch, tmp_path):
        import distill_gym.web.app as web_app_mod
        monkeypatch.setattr(web_app_mod, "HAS_REACT_BUILD", False)
        app = await _prepare_app(monkeypatch, tmp_path)

        store = RunStore()
        await _create_test_run(store, "r1", "Detail Run")
        await _create_test_task(store, "r1", "t1", "Task One")
        await _create_test_artifact(store, "r1", "t1", "a1")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/runs/r1")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Detail Run" in resp.text

    @pytest.mark.asyncio
    async def test_legacy_task_detail_page(self, monkeypatch, tmp_path):
        import distill_gym.web.app as web_app_mod
        monkeypatch.setattr(web_app_mod, "HAS_REACT_BUILD", False)
        app = await _prepare_app(monkeypatch, tmp_path)

        store = RunStore()
        await _create_test_run(store, "r1", "Task Detail Run")
        await _create_test_task(store, "r1", "my-task", "My Task")
        await _create_test_artifact(store, "r1", "my-task", "a1")
        await store.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/runs/r1/tasks/my-task")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "My Task" in resp.text


# ---------------------------------------------------------------------------
# /artifacts/{path} (GET)
# ---------------------------------------------------------------------------

class TestServeArtifact:
    @pytest.mark.asyncio
    async def test_serve_artifact_found(self, monkeypatch, tmp_path):
        from distill_gym.cache.cache_store import get_artifacts_dir

        app = await _prepare_app(monkeypatch, tmp_path)

        art_dir = get_artifacts_dir()
        art_dir.mkdir(parents=True, exist_ok=True)
        art_file = art_dir / "test_artifact.txt"
        art_file.write_text("artifact content", encoding="utf-8")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/artifacts/test_artifact.txt")
        assert resp.status_code == 200
        assert resp.text == "artifact content"

    @pytest.mark.asyncio
    async def test_serve_artifact_not_found(self, monkeypatch, tmp_path):
        app = await _prepare_app(monkeypatch, tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/artifacts/non_existent_file.txt")
        assert resp.status_code == 404
