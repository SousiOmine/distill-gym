import json

import httpx
import pytest
from fastapi.responses import JSONResponse

from distill_gym.config.schema import LoggingProxyConfig, ProviderConfig
from distill_gym.proxy.app import ProxyApp, create_proxy_app
from distill_gym.proxy.recorder import TraceRecorder


@pytest.mark.asyncio
async def test_task_route_records_path_task_id(tmp_path, monkeypatch):
    async def fake_non_stream(self, body, headers, task_id):
        return JSONResponse({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        })

    monkeypatch.setattr(ProxyApp, "_handle_non_stream", fake_non_stream)

    trace_path = tmp_path / "raw_trace.jsonl"
    app = create_proxy_app(
        ProviderConfig(base_url="https://provider.test/v1"),
        LoggingProxyConfig(capture_raw_request=True, capture_raw_response=False),
        TraceRecorder(trace_path),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/tasks/task_a/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "system", "content": "sys"}],
                "tools": [{"type": "function", "function": {"name": "search"}}],
            },
        )

    assert response.status_code == 200
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["event"] == "llm_request"
    assert event["task_id"] == "task_a"
    assert event["data"]["messages"][0]["role"] == "system"
    assert event["data"]["tools"][0]["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_legacy_route_falls_back_to_x_task_id(tmp_path, monkeypatch):
    async def fake_non_stream(self, body, headers, task_id):
        return JSONResponse({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        })

    monkeypatch.setattr(ProxyApp, "_handle_non_stream", fake_non_stream)

    trace_path = tmp_path / "raw_trace.jsonl"
    app = create_proxy_app(
        ProviderConfig(base_url="https://provider.test/v1"),
        LoggingProxyConfig(capture_raw_request=True, capture_raw_response=False),
        TraceRecorder(trace_path),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-task-id": "task_b"},
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["task_id"] == "task_b"
