from pathlib import Path
from distill_gym.cache.cache_store import get_artifacts_dir


class ArtifactCollector:
    def __init__(self, run_id: str, task_id: str):
        self.base = get_artifacts_dir() / run_id / task_id
        self.base.mkdir(parents=True, exist_ok=True)

    def save_text(self, kind: str, text: str) -> Path:
        path = self.base / f"{kind}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def save_json(self, kind: str, data: dict | list) -> Path:
        import json
        path = self.base / f"{kind}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def path_for(self, kind: str) -> Path:
        return self.base / kind
