import json
import os
from pathlib import Path

import httpx
import yaml

from distill_gym.config.schema import TaskItem, TaskGenConfig
from distill_gym.taskgen.base import TaskGenerator
from distill_gym.registry.taskgen_registry import TaskGenRegistry


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(template_id: str) -> dict:
    path = _TEMPLATES_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task generation template not found: {template_id} (tried {path})")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@TaskGenRegistry.register("template")
class TemplateTaskGenerator(TaskGenerator):
    def __init__(self, config: TaskGenConfig, **kwargs):
        self.config = config
        self._provider_config = kwargs.get("provider_config")

    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        if self.config.tasks:
            return self.config.tasks[:count]

        template_id = self.config.prompt_template or "repo_bugfix_and_test_tasks"
        template = _load_template(template_id)

        if not self._provider_config or not self._provider_config.base_url:
            return self._generate_fallback(template, count)

        context = await self._gather_context(template)
        tasks = await self._generate_with_llm(template, context, count)
        if tasks:
            return tasks

        return self._generate_fallback(template, count)

    async def _gather_context(self, template: dict) -> str:
        context_parts = []

        context_config = template.get("context", {})
        required = context_config.get("required", [])
        for item in required:
            source_type = item.get("type", "static")
            if source_type == "static":
                text = item.get("value", "")
                if text:
                    context_parts.append(text)
            elif source_type == "file":
                from distill_gym.taskgen.contexts import FileContextProvider
                provider = FileContextProvider()
                text = await provider.get_context(item.get("config"))
                if text:
                    context_parts.append(text)

        return "\n\n".join(context_parts)

    async def _generate_with_llm(
        self, template: dict, context: str, count: int,
    ) -> list[TaskItem]:
        api_key = os.environ.get(self._provider_config.api_key_env, "")
        if not api_key:
            return []

        system_prompt = template.get("system_prompt", "You generate software maintenance tasks.")
        task_prompt = template.get("task_prompt", "Generate {count} tasks.")

        filled_prompt = task_prompt.replace("{count}", str(count))
        if context:
            filled_prompt = filled_prompt.replace("{context}", context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled_prompt},
        ]

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._provider_config.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._provider_config.model,
                    "messages": messages,
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

    def _generate_fallback(self, template: dict, count: int) -> list[TaskItem]:
        tasks = []
        for i in range(count):
            tasks.append(TaskItem(
                id=f"task_{i:03d}",
                title=template.get("default_title", f"Task {i + 1}"),
                prompt=template.get("default_prompt", "Perform repository maintenance."),
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
