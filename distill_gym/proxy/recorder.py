import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional


class TraceRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event_type: str, data: dict, task_id: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "task_id": task_id,
            "data": data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def record_request(self, body: dict, task_id: str = "") -> None:
        safe = {}
        keys_to_keep = ["messages", "model", "tools", "tool_choice", "stream", "temperature", "max_tokens", "extra_body"]
        for k in keys_to_keep:
            if k in body:
                safe[k] = body[k]
        self._write("llm_request", safe, task_id)

    def record_response(self, body: dict, task_id: str = "") -> None:
        safe = {}
        if "choices" in body:
            safe["choices"] = body["choices"]
        if "usage" in body:
            safe["usage"] = body["usage"]
        if "model" in body:
            safe["model"] = body["model"]
        if "id" in body:
            safe["id"] = body["id"]
        self._write("llm_response", safe, task_id)

    def record_stream_chunk(self, chunk: dict, task_id: str = "") -> None:
        self._write("llm_stream_chunk", chunk, task_id)

    def record_harness_output(self, stream: str, text: str, task_id: str = "") -> None:
        self._write(f"harness_{stream}", {"text": text}, task_id)

    def record_artifact(self, kind: str, path: str, task_id: str = "") -> None:
        self._write("artifact", {"kind": kind, "path": path}, task_id)

    def record_event(self, event: str, metadata: Optional[dict] = None, task_id: str = "") -> None:
        self._write(event, metadata or {}, task_id)
