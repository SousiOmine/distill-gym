import pytest
from pathlib import Path
import json

from distill_gym.config.schema import TaskGenConfig, ProviderConfig
from distill_gym.taskgen.template_task_generator import _load_template
from distill_gym.taskgen.template_task_generator import TemplateTaskGenerator
from distill_gym.registry.taskgen_registry import TaskGenRegistry


class TestTemplateLoader:
    def test_load_bugfix_template(self):
        template = _load_template("repo_bugfix_and_test_tasks")
        assert template["name"] == "repo_bugfix_and_test"
        assert "system_prompt" in template
        assert "task_prompt" in template

    def test_load_enterprise_chat_template(self):
        template = _load_template("enterprise_chat_qa")
        assert template["name"] == "enterprise_chat_qa"
        assert template["default_title"] == "Enterprise chat QA"

    def test_template_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_template("nonexistent_template")


@pytest.mark.asyncio
class TestTemplateTaskGenerator:
    async def test_static_tasks_from_config(self):
        config = TaskGenConfig(
            type="template",
            tasks=[
                {"id": "t1", "title": "Task1", "prompt": "Do thing 1"},
                {"id": "t2", "title": "Task2", "prompt": "Do thing 2"},
            ],
        )
        gen = TemplateTaskGenerator(config)
        tasks = await gen.generate(3)
        assert len(tasks) == 2
        assert tasks[0].id == "t1"

    async def test_fallback_without_provider(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        tasks = await gen.generate(2)
        assert len(tasks) == 2
        assert "improvement" in tasks[0].prompt or "Repository maintenance" in tasks[0].prompt

    def test_extract_json_array(self):
        gen = TemplateTaskGenerator(TaskGenConfig(type="template"))
        content = '[{"id": "t1", "prompt": "hello"}]'
        assert gen._extract_json_array(content) == content

    def test_extract_json_array_with_markdown(self):
        gen = TemplateTaskGenerator(TaskGenConfig(type="template"))
        content = '```json\n[{"id": "t1", "prompt": "hello"}]\n```'
        result = gen._extract_json_array(content)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "t1"

    def test_registered(self):
        cls = TaskGenRegistry.get("template")
        assert cls is not None
        assert cls is TemplateTaskGenerator
