import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from distill_gym.config.schema import LoggingProxyConfig, ProviderConfig
from distill_gym.proxy.app import ProxyApp, create_proxy_app
from distill_gym.proxy.recorder import TraceRecorder


class TestProxyAppInternal:
    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_API_KEY", "sk-secret-42")
        provider = ProviderConfig(base_url="https://test.api/v1", api_key_env="CUSTOM_API_KEY")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg)
        assert app._resolve_api_key() == "sk-secret-42"

    def test_resolve_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        provider = ProviderConfig(base_url="https://test.api/v1", api_key_env="MISSING_VAR")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg)
        assert app._resolve_api_key() == "MISSING_VAR"

    def test_make_headers(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "sk-key")
        provider = ProviderConfig(base_url="https://test.api/v1", api_key_env="TEST_KEY")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg)
        mock_request = MagicMock()
        mock_request.headers = {"x-request-id": "req-001"}
        headers = app._make_headers(mock_request)
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer sk-key"
        assert headers["x-request-id"] == "req-001"

    def test_make_headers_no_request_id(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "sk-key")
        provider = ProviderConfig(base_url="https://test.api/v1", api_key_env="TEST_KEY")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg)
        mock_request = MagicMock()
        mock_request.headers = {}
        headers = app._make_headers(mock_request)
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer sk-key"
        assert "x-request-id" not in headers


class TestNonStream:
    @pytest.mark.asyncio
    async def test_non_stream_success(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_raw_response=True)
        app = ProxyApp(provider, proxy_cfg, recorder)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"total_tokens": 5},
            "model": "test-model",
            "id": "chatcmpl-123",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.status_code = 200

        app._client = AsyncMock()
        app._client.post = AsyncMock(return_value=mock_response)

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_non_stream(body, headers, "task_1")

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["choices"][0]["message"]["content"] == "Hello"

        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "llm_response"
        assert event["task_id"] == "task_1"
        assert event["data"]["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_non_stream_no_recorder(self):
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg, recorder=None)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "OK"}}]
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.status_code = 200

        app._client = AsyncMock()
        app._client.post = AsyncMock(return_value=mock_response)

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_non_stream(body, headers, "no_rec")
        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["choices"][0]["message"]["content"] == "OK"

    @pytest.mark.asyncio
    async def test_non_stream_capture_disabled(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_raw_response=False)
        app = ProxyApp(provider, proxy_cfg, recorder)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "OK"}}]
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.status_code = 200

        app._client = AsyncMock()
        app._client.post = AsyncMock(return_value=mock_response)

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_non_stream(body, headers, "task_2")
        assert response.status_code == 200
        assert not trace_path.exists()


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_success(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_stream_chunks=True)
        app = ProxyApp(provider, proxy_cfg, recorder)

        chunk1 = '{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}'
        chunk2 = '{"choices":[{"index":0,"delta":{"content":" world"}}]}'

        async def mock_aiter_lines():
            yield f"data: {chunk1}"
            yield f"data: {chunk2}"
            yield "data: [DONE]"

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.aiter_lines = mock_aiter_lines
            yield mock_resp

        app._client.stream = mock_stream

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_stream(body, headers, "task_stream")

        collected = [chunk async for chunk in response.body_iterator]
        assert len(collected) == 3
        assert collected[0] == f"data: {chunk1}\n\n"
        assert collected[1] == f"data: {chunk2}\n\n"
        assert collected[2] == "data: [DONE]\n\n"

        events = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").strip().split("\n")]
        assert len(events) == 3
        assert events[0]["event"] == "llm_stream_chunk"
        assert events[1]["event"] == "llm_stream_chunk"
        assert events[2]["event"] == "llm_response"
        assert events[2]["data"]["choices"][0]["message"]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_done_records_merged(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_stream_chunks=False)
        app = ProxyApp(provider, proxy_cfg, recorder)

        chunk1 = '{"choices":[{"index":0,"delta":{"role":"assistant","content":"Merged"}}]}'

        async def mock_aiter_lines():
            yield f"data: {chunk1}"
            yield "data: [DONE]"

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.aiter_lines = mock_aiter_lines
            yield mock_resp

        app._client.stream = mock_stream

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_stream(body, headers, "task_merge")
        _ = [chunk async for chunk in response.body_iterator]

        events = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event"] == "llm_response"
        assert events[0]["task_id"] == "task_merge"
        assert events[0]["data"]["choices"][0]["message"]["content"] == "Merged"

    @pytest.mark.asyncio
    async def test_stream_no_recorder(self):
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig()
        app = ProxyApp(provider, proxy_cfg, recorder=None)

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}'
            yield "data: [DONE]"

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.aiter_lines = mock_aiter_lines
            yield mock_resp

        app._client.stream = mock_stream

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_stream(body, headers, "no_rec")

        collected = [chunk async for chunk in response.body_iterator]
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_stream_json_decode_error(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_stream_chunks=True)
        app = ProxyApp(provider, proxy_cfg, recorder)

        async def mock_aiter_lines():
            yield "data: {invalid json}"
            yield "data: [DONE]"

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.aiter_lines = mock_aiter_lines
            yield mock_resp

        app._client.stream = mock_stream

        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        headers = {"Authorization": "Bearer test-key"}
        response = await app._handle_stream(body, headers, "task_bad")

        collected = [chunk async for chunk in response.body_iterator]
        assert len(collected) == 2
        assert collected[0] == "data: {invalid json}\n\n"
        assert collected[1] == "data: [DONE]\n\n"

        events = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event"] == "llm_response"


class TestEndpoints:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig()
        app = create_proxy_app(provider, proxy_cfg)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_chat_completions_non_stream(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello from mock"}}]
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.status_code = 200

        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_raw_request=True, capture_raw_response=True)
        proxy = ProxyApp(provider, proxy_cfg, recorder)
        proxy._client = AsyncMock()
        proxy._client.post = AsyncMock(return_value=mock_response)

        app = FastAPI()
        app.add_api_route("/v1/chat/completions", proxy.chat_completions, methods=["POST"])

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mock"

        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        events = [json.loads(l) for l in lines]
        assert events[0]["event"] == "llm_request"
        assert events[1]["event"] == "llm_response"

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Stream"}}]}'
            yield "data: [DONE]"

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.aiter_lines = mock_aiter_lines
            yield mock_resp

        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_raw_request=True, capture_stream_chunks=True)
        proxy = ProxyApp(provider, proxy_cfg, recorder)
        proxy._client = MagicMock()
        proxy._client.stream = mock_stream

        app = FastAPI()
        app.add_api_route("/v1/chat/completions", proxy.chat_completions, methods=["POST"])

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )

        assert response.status_code == 200

        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 3

    @pytest.mark.asyncio
    async def test_task_specific_route(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        trace_path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(trace_path)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "task response"}}]
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.status_code = 200

        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig(capture_raw_request=True, capture_raw_response=False)
        proxy = ProxyApp(provider, proxy_cfg, recorder)
        proxy._client = AsyncMock()
        proxy._client.post = AsyncMock(return_value=mock_response)

        app = FastAPI()
        app.add_api_route(
            "/tasks/{path_task_id}/v1/chat/completions",
            proxy.chat_completions,
            methods=["POST"],
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/tasks/my_task_42/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        event = json.loads(trace_path.read_text(encoding="utf-8").strip().split("\n")[0])
        assert event["task_id"] == "my_task_42"


class TestCreateProxyApp:
    def test_create_proxy_app_routes(self):
        provider = ProviderConfig(base_url="https://provider.test/v1")
        proxy_cfg = LoggingProxyConfig()
        app = create_proxy_app(provider, proxy_cfg)
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/v1/chat/completions" in routes
        assert "/tasks/{path_task_id}/v1/chat/completions" in routes
