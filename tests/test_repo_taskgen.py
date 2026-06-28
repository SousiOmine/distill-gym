import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from distill_gym.config.schema import TaskGenConfig, ProviderConfig, TaskItem
from distill_gym.taskgen.repo_task_generator import RepoTaskGenerator
from distill_gym.taskgen.template_task_generator import (
    TemplateTaskGenerator,
    _load_template,
)


class TestRepoTaskGenerator:

    @pytest.mark.asyncio
    async def test_generate_returns_config_tasks(self):
        config = TaskGenConfig(
            type="repo_auto",
            tasks=[
                TaskItem(id="t1", title="Task1", prompt="Do 1"),
                TaskItem(id="t2", title="Task2", prompt="Do 2"),
                TaskItem(id="t3", title="Task3", prompt="Do 3"),
            ],
        )
        gen = RepoTaskGenerator(config)
        tasks = await gen.generate(2)
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[1].id == "t2"

    @pytest.mark.asyncio
    async def test_generate_returns_llm_results(self):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config)
        expected = [TaskItem(id="llm1", title="LLM Task", prompt="Do LLM")]
        gen._generate_with_llm = AsyncMock(return_value=expected)
        tasks = await gen.generate(1)
        assert tasks == expected

    @pytest.mark.asyncio
    async def test_generate_fallback_when_llm_empty(self):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config)
        gen._generate_with_llm = AsyncMock(return_value=[])
        tasks = await gen.generate(2)
        assert len(tasks) == 2
        assert tasks[0].id == "task_000"
        assert "improvement" in tasks[0].prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_with_llm_empty_repo_url(self):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config, repo_url="")
        tasks = await gen._generate_with_llm(1)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_generate_with_llm_no_provider_config(self):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(
            config, repo_url="https://example.com/repo.git", provider_config=None
        )
        tasks = await gen._generate_with_llm(1)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_generate_with_llm_no_api_key(self):
        config = TaskGenConfig(type="repo_auto")
        provider_config = ProviderConfig(api_key_env="MISSING_ENV_VAR")
        gen = RepoTaskGenerator(
            config, repo_url="https://example.com/repo.git",
            provider_config=provider_config,
        )
        tasks = await gen._generate_with_llm(1)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_generate_with_llm_success(self):
        config = TaskGenConfig(type="repo_auto")
        provider_config = ProviderConfig(api_key_env="TEST_REPO_API_KEY")
        gen = RepoTaskGenerator(
            config, repo_url="https://example.com/repo.git",
            provider_config=provider_config,
        )

        os.environ["TEST_REPO_API_KEY"] = "test-key"
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": (
                            '[{"id":"t1","title":"Task1",'
                            '"prompt":"Do X","test_command":"pytest"}]'
                        )
                    }
                }]
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "distill_gym.taskgen.repo_task_generator.httpx.AsyncClient",
                return_value=mock_client,
            ):
                with patch(
                    "distill_gym.taskgen.repo_task_generator.clone_from_mirror",
                ) as mock_clone:
                    def _fake_clone(url, path, ref):
                        path.mkdir(parents=True, exist_ok=True)
                    mock_clone.side_effect = _fake_clone

                    tasks = await gen._generate_with_llm(1)
                    assert len(tasks) == 1
                    assert tasks[0].id == "t1"
                    assert tasks[0].title == "Task1"
                    assert tasks[0].prompt == "Do X"
                    assert tasks[0].test_command == "pytest"
        finally:
            del os.environ["TEST_REPO_API_KEY"]

    @pytest.mark.asyncio
    async def test_generate_with_llm_exception_returns_empty(self):
        config = TaskGenConfig(type="repo_auto")
        provider_config = ProviderConfig(api_key_env="TEST_REPO_API_KEY")
        gen = RepoTaskGenerator(
            config, repo_url="https://example.com/repo.git",
            provider_config=provider_config,
        )

        os.environ["TEST_REPO_API_KEY"] = "test-key"
        try:
            with patch(
                "distill_gym.taskgen.repo_task_generator.clone_from_mirror",
                side_effect=Exception("clone failed"),
            ):
                tasks = await gen._generate_with_llm(1)
                assert tasks == []
        finally:
            del os.environ["TEST_REPO_API_KEY"]

    def test_summarize_repo_with_files_and_readme(self, tmp_path):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config)

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test Repo")

        summary = gen._summarize_repo(tmp_path)
        assert "Files:" in summary
        assert "README:" in summary
        assert "src/main.py" in summary
        assert "# Test Repo" in summary

    def test_summarize_repo_excludes_git(self, tmp_path):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config)

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")

        summary = gen._summarize_repo(tmp_path)
        assert ".git/config" not in summary
        assert "src/main.py" in summary

    def test_summarize_repo_file_limit(self, tmp_path):
        config = TaskGenConfig(type="repo_auto")
        gen = RepoTaskGenerator(config)

        for i in range(250):
            (tmp_path / f"file{i}.py").write_text("x")

        summary = gen._summarize_repo(tmp_path)
        readme_idx = summary.index("README:")
        files_section = summary[len("Files:\n"):readme_idx]
        file_count = sum(1 for line in files_section.split("\n") if line.strip())
        assert file_count == 200

    @pytest.mark.asyncio
    async def test_request_tasks_success(self):
        config = TaskGenConfig(type="repo_auto")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_KEY",
            model="test-model",
        )
        gen = RepoTaskGenerator(config, provider_config=provider_config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '[{"id":"t1","title":"T1",'
                        '"prompt":"P1","test_command":"pytest"}]'
                    )
                }
            }]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "distill_gym.taskgen.repo_task_generator.httpx.AsyncClient",
            return_value=mock_client,
        ):
            tasks = await gen._request_tasks("some summary", 1, "test-key")
            assert len(tasks) == 1
            assert tasks[0].id == "t1"
            assert tasks[0].title == "T1"
            assert tasks[0].prompt == "P1"
            assert tasks[0].test_command == "pytest"

    @pytest.mark.asyncio
    async def test_request_tasks_default_id_title(self):
        config = TaskGenConfig(type="repo_auto")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_KEY",
            model="test-model",
        )
        gen = RepoTaskGenerator(config, provider_config=provider_config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '[{"prompt":"P1"}, {"prompt":"P2"}]'
                }
            }]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "distill_gym.taskgen.repo_task_generator.httpx.AsyncClient",
            return_value=mock_client,
        ):
            tasks = await gen._request_tasks("summary", 2, "test-key")
            assert len(tasks) == 2
            assert tasks[0].id == "task_000"
            assert tasks[0].title == "Generated task 1"
            assert tasks[1].id == "task_001"
            assert tasks[1].title == "Generated task 2"

    def test_extract_json_array_plain(self):
        gen = RepoTaskGenerator(TaskGenConfig())
        content = '[{"id":"t1","title":"T1","prompt":"P1"}]'
        assert gen._extract_json_array(content) == content

    def test_extract_json_array_with_markdown(self):
        gen = RepoTaskGenerator(TaskGenConfig())
        content = '```json\n[{"id":"t1","title":"T1","prompt":"P1"}]\n```'
        result = gen._extract_json_array(content)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "t1"

    def test_extract_json_array_with_surrounding_text(self):
        gen = RepoTaskGenerator(TaskGenConfig())
        content = (
            'Here is the result:\n```json\n'
            '[{"id":"t1","title":"T1","prompt":"P1"}]\n```\nEnd'
        )
        result = gen._extract_json_array(content)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "t1"


class TestTemplateTaskGenerator:

    @pytest.mark.asyncio
    async def test_generate_returns_config_tasks(self):
        config = TaskGenConfig(
            type="template",
            tasks=[
                TaskItem(id="t1", title="Task1", prompt="Do 1"),
                TaskItem(id="t2", title="Task2", prompt="Do 2"),
            ],
        )
        gen = TemplateTaskGenerator(config)
        tasks = await gen.generate(5)
        assert len(tasks) == 2
        assert tasks[0].id == "t1"

    @pytest.mark.asyncio
    async def test_generate_fallback_without_provider(self):
        config = TaskGenConfig(
            type="template", prompt_template="repo_bugfix_and_test_tasks",
        )
        gen = TemplateTaskGenerator(config)
        tasks = await gen.generate(2)
        assert len(tasks) == 2
        assert "improvement" in tasks[0].prompt

    @pytest.mark.asyncio
    async def test_generate_fallback_without_base_url(self):
        config = TaskGenConfig(
            type="template", prompt_template="repo_bugfix_and_test_tasks",
        )
        provider_config = ProviderConfig(
            base_url="", api_key_env="TEST_KEY", model="m",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)
        tasks = await gen.generate(2)
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_generate_fallback_when_llm_empty(self):
        config = TaskGenConfig(
            type="template", prompt_template="enterprise_chat_qa",
        )
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_TEMPLATE_KEY",
            model="test-model",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)
        tasks = await gen.generate(3)
        assert len(tasks) == 3
        assert tasks[0].title == "Enterprise chat QA"

    @pytest.mark.asyncio
    async def test_generate_calls_llm_and_returns_tasks(self):
        config = TaskGenConfig(
            type="template", prompt_template="repo_bugfix_and_test_tasks",
        )
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_TEMPLATE_KEY",
            model="test-model",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)

        os.environ["TEST_TEMPLATE_KEY"] = "test-key"
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": (
                            '[{"id":"t1","title":"T1","prompt":"P1"}]'
                        )
                    }
                }]
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "distill_gym.taskgen.template_task_generator.httpx.AsyncClient",
                return_value=mock_client,
            ):
                tasks = await gen.generate(1)
                assert len(tasks) == 1
                assert tasks[0].id == "t1"
                assert tasks[0].title == "T1"
                assert tasks[0].prompt == "P1"
        finally:
            del os.environ["TEST_TEMPLATE_KEY"]

    @pytest.mark.asyncio
    async def test_generate_fallback_when_llm_fails(self):
        config = TaskGenConfig(
            type="template", prompt_template="repo_bugfix_and_test_tasks",
        )
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_TEMPLATE_KEY",
            model="test-model",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)

        os.environ["TEST_TEMPLATE_KEY"] = "test-key"
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "[]"}}]
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "distill_gym.taskgen.template_task_generator.httpx.AsyncClient",
                return_value=mock_client,
            ):
                tasks = await gen.generate(2)
                assert len(tasks) == 2
                assert "improvement" in tasks[0].prompt
        finally:
            del os.environ["TEST_TEMPLATE_KEY"]

    @pytest.mark.asyncio
    async def test_gather_context_static(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        template = {
            "context": {
                "required": [
                    {"type": "static", "value": "static context"}
                ]
            }
        }
        context = await gen._gather_context(template)
        assert context == "static context"

    @pytest.mark.asyncio
    async def test_gather_context_file(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        template = {
            "context": {
                "required": [
                    {"type": "file", "config": {"paths": ["/tmp"]}}
                ]
            }
        }

        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = "file context"

        with patch(
            "distill_gym.taskgen.contexts.FileContextProvider",
            return_value=mock_provider,
        ):
            context = await gen._gather_context(template)
            assert context == "file context"

    @pytest.mark.asyncio
    async def test_gather_context_multiple(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        template = {
            "context": {
                "required": [
                    {"type": "static", "value": "part1"},
                    {"type": "static", "value": "part2"},
                ]
            }
        }
        context = await gen._gather_context(template)
        assert context == "part1\n\npart2"

    @pytest.mark.asyncio
    async def test_generate_with_llm_success(self):
        config = TaskGenConfig(type="template")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="TEST_TEMPLATE_KEY",
            model="test-model",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)

        os.environ["TEST_TEMPLATE_KEY"] = "test-key"
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": (
                            '[{"id":"t1","title":"T1","prompt":"P1"}]'
                        )
                    }
                }]
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "distill_gym.taskgen.template_task_generator.httpx.AsyncClient",
                return_value=mock_client,
            ):
                template = {
                    "system_prompt": "test",
                    "task_prompt": "Generate {count} tasks.",
                }
                tasks = await gen._generate_with_llm(template, "", 1)
                assert len(tasks) == 1
                assert tasks[0].id == "t1"
        finally:
            del os.environ["TEST_TEMPLATE_KEY"]

    @pytest.mark.asyncio
    async def test_generate_with_llm_no_api_key(self):
        config = TaskGenConfig(type="template")
        provider_config = ProviderConfig(
            base_url="http://test.local/v1", api_key_env="MISSING_KEY",
            model="test-model",
        )
        gen = TemplateTaskGenerator(config, provider_config=provider_config)

        template = {
            "system_prompt": "test",
            "task_prompt": "Generate {count} tasks.",
        }
        tasks = await gen._generate_with_llm(template, "", 1)
        assert tasks == []

    def test_generate_fallback_defaults(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        template = {}
        tasks = gen._generate_fallback(template, 2)
        assert len(tasks) == 2
        assert tasks[0].title == "Task 1"
        assert tasks[0].prompt == "Perform repository maintenance."

    def test_generate_fallback_with_template_defaults(self):
        config = TaskGenConfig(type="template")
        gen = TemplateTaskGenerator(config)
        template = {
            "default_title": "Custom Title",
            "default_prompt": "Custom prompt",
        }
        tasks = gen._generate_fallback(template, 1)
        assert tasks[0].title == "Custom Title"
        assert tasks[0].prompt == "Custom prompt"

    def test_load_template_not_found(self):
        with pytest.raises(FileNotFoundError, match="nonexistent_template"):
            _load_template("nonexistent_template")
