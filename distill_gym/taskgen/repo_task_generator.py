import json
import os
import tempfile
from pathlib import Path

import httpx

from distill_gym.cache.git_cache import clone_from_mirror
from distill_gym.taskgen.base import TaskGenerator
from distill_gym.config.schema import TaskItem, TaskGenConfig, ProviderConfig


class RepoTaskGenerator(TaskGenerator):
    def __init__(
        self,
        config: TaskGenConfig,
        repo_url: str = "",
        ref: str = "main",
        provider_config: ProviderConfig | None = None,
    ):
        self.config = config
        self.repo_url = repo_url
        self.ref = ref
        self.provider_config = provider_config

    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        if self.config.tasks:
            return self.config.tasks[:count]

        generated = await self._generate_with_llm(count)
        if generated:
            return generated

        tasks = []
        for i in range(count):
            tasks.append(TaskItem(
                id=f"task_{i:03d}",
                title=f"Repository maintenance task {i + 1}",
                prompt=(
                    "Inspect this repository, identify one concrete improvement or bug fix, "
                    "implement it, and run the most relevant tests. Keep the change focused."
                ),
            ))
        return tasks

    async def _generate_with_llm(self, count: int) -> list[TaskItem]:
        if not self.repo_url or not self.provider_config:
            return []
        api_key = os.environ.get(self.provider_config.api_key_env, "")
        if not api_key:
            return []

        try:
            with tempfile.TemporaryDirectory(prefix="distill-gym-taskgen-") as td:
                repo_path = Path(td) / "repo"
                clone_from_mirror(self.repo_url, repo_path, self.ref)
                repo_summary = self._summarize_repo(repo_path)
                tasks = await self._request_tasks(repo_summary, count, api_key)
                return tasks
        except Exception:
            return []

    def _summarize_repo(self, repo_path: Path) -> str:
        files = []
        for path in repo_path.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue
            rel = path.relative_to(repo_path).as_posix()
            if len(files) < 200:
                files.append(rel)

        readme = ""
        for name in ("README.md", "README.rst", "README.txt"):
            p = repo_path / name
            if p.exists():
                readme = p.read_text(encoding="utf-8", errors="ignore")[:6000]
                break

        return "Files:\n" + "\n".join(files) + "\n\nREADME:\n" + readme

    async def _request_tasks(self, repo_summary: str, count: int, api_key: str) -> list[TaskItem]:
        prompt = (
            f"Generate {count} focused coding-agent tasks for this repository. "
            "Return only JSON: an array of objects with id, title, prompt, and optional test_command. "
            "Tasks must be solvable by editing the repository and should ask the agent to run tests when possible.\n\n"
            f"{repo_summary}"
        )
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.provider_config.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.provider_config.model,
                    "messages": [
                        {"role": "system", "content": "You generate concise software maintenance tasks."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.4,
                },
            )
            response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        raw_tasks = json.loads(self._extract_json_array(content))
        tasks = []
        for i, item in enumerate(raw_tasks[:count]):
            tasks.append(TaskItem(
                id=str(item.get("id") or f"task_{i:03d}"),
                title=str(item.get("title") or f"Generated task {i + 1}"),
                prompt=str(item["prompt"]),
                test_command=item.get("test_command"),
            ))
        return tasks

    def _extract_json_array(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return text[start:end + 1]
        return text
