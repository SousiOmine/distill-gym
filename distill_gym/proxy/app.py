import json
import os
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from distill_gym.config.schema import LoggingProxyConfig, ProviderConfig
from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.proxy.normalizer import merge_stream_chunks, strip_sensitive


class ProxyApp:
    def __init__(
        self,
        provider: ProviderConfig,
        proxy_cfg: LoggingProxyConfig,
        recorder: Optional[TraceRecorder] = None,
    ):
        self.provider = provider
        self.proxy_cfg = proxy_cfg
        self.recorder = recorder
        self._client = httpx.AsyncClient(timeout=300)

    def _resolve_api_key(self) -> str:
        env_val = os.environ.get(self.provider.api_key_env)
        if env_val:
            return env_val
        return self.provider.api_key_env or ""

    def _make_headers(self, request: Request) -> dict:
        api_key = self._resolve_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if "x-request-id" in request.headers:
            headers["x-request-id"] = request.headers["x-request-id"]
        return headers

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        task_id = request.headers.get("x-task-id", "")

        if self.recorder and self.proxy_cfg.capture_raw_request:
            self.recorder.record_request(body, task_id)

        stream = body.get("stream", False)
        headers = self._make_headers(request)

        if stream:
            return await self._handle_stream(body, headers, task_id)
        else:
            return await self._handle_non_stream(body, headers, task_id)

    async def _handle_non_stream(self, body: dict, headers: dict, task_id: str) -> Response:
        response = await self._client.post(
            f"{self.provider.base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        resp_body = response.json()

        if self.recorder and self.proxy_cfg.capture_raw_response:
            self.recorder.record_response(strip_sensitive(resp_body), task_id)

        return Response(content=response.text, status_code=response.status_code, media_type="application/json")

    async def _handle_stream(self, body: dict, headers: dict, task_id: str) -> StreamingResponse:
        chunks: list[dict] = []

        async def stream_generator():
            async with self._client.stream(
                "POST",
                f"{self.provider.base_url}/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            if self.recorder:
                                merged, reason = merge_stream_chunks(chunks)
                                self.recorder.record_response(
                                    {"choices": [{"message": merged, "finish_reason": reason}], "usage": {}},
                                    task_id,
                                )
                            yield "data: [DONE]\n\n"
                            continue
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                            if self.recorder and self.proxy_cfg.capture_stream_chunks:
                                self.recorder.record_stream_chunk(strip_sensitive(chunk), task_id)
                            yield f"data: {data_str}\n\n"
                        except json.JSONDecodeError:
                            yield f"data: {data_str}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")


def create_proxy_app(
    provider: ProviderConfig,
    proxy_cfg: LoggingProxyConfig,
    recorder: Optional[TraceRecorder] = None,
) -> FastAPI:
    proxy = ProxyApp(provider, proxy_cfg, recorder)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await proxy._client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.add_api_route("/v1/chat/completions", proxy.chat_completions, methods=["POST"])
    return app
