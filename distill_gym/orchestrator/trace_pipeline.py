import json
import logging
from pathlib import Path
from typing import Optional

from distill_gym.proxy.recorder import TraceRecorder
from distill_gym.cache.cache_store import get_artifacts_dir

logger = logging.getLogger(__name__)


class TracePipeline:
    def __init__(self, proxy_recorder: Optional[TraceRecorder], run_id: str) -> None:
        self._proxy_recorder = proxy_recorder
        self._run_id = run_id
        self._task_paths: dict[str, Path] = {}

    def register_task(self, task_id: str) -> None:
        path = get_artifacts_dir() / self._run_id / task_id / "raw_trace.jsonl"
        self._task_paths[task_id] = path

    async def record_event(
        self, task_id: str, event: str, metadata: Optional[dict] = None,
    ) -> None:
        path = self._task_paths.get(task_id)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        recorder = TraceRecorder(path)
        recorder.record_event(event, metadata or {}, task_id)

    async def distribute(self) -> None:
        if not self._proxy_recorder or not self._task_paths:
            return
        shared_path = self._proxy_recorder.path
        if not shared_path.exists():
            return

        buffers: dict[str, list[str]] = {}
        with open(shared_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = event.get("task_id", "")
                if task_id in self._task_paths:
                    buffers.setdefault(task_id, []).append(line)

        for task_id, lines in buffers.items():
            path = self._task_paths[task_id]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
