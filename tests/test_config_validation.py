import yaml
import pytest
from distill_gym.config.schema import Config
from distill_gym.config.loader import load_config
from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from pydantic import ValidationError


class TestConfigValidation:
    def test_valid_config(self, sample_config_yaml):
        data = yaml.safe_load(sample_config_yaml)
        config = Config.model_validate(data)
        assert config.run.name == "test-run"
        assert config.harness.type == "mock"
        assert config.provider.model == "test-model"

    def test_default_values(self):
        config = Config()
        assert config.run.task_count == 3
        assert config.run.cleanup.value == "always"
        assert config.provider.type == "openai_compatible"

    def test_invalid_cleanup(self):
        data = {"run": {"cleanup": "invalid"}}
        with pytest.raises(ValidationError):
            Config.model_validate(data)

    def test_load_from_file(self, tmp_path, sample_config_yaml):
        path = tmp_path / "config.yaml"
        path.write_text(sample_config_yaml)
        config = load_config(str(path))
        assert config.run.name == "test-run"


class TestConfigHarness:
    def test_harness_install(self):
        data = {
            "harness": {
                "type": "generic_cli",
                "install": {"strategy": "command", "commands": ["echo installed"]},
            }
        }
        config = Config.model_validate(data)
        assert config.harness.install.commands == ["echo installed"]

    def test_task_items(self):
        data = {
            "taskgen": {
                "tasks": [
                    {"id": "t1", "title": "Test", "prompt": "run test"},
                ]
            }
        }
        config = Config.model_validate(data)
        assert len(config.taskgen.tasks) == 1
        assert config.taskgen.tasks[0].id == "t1"

    @pytest.mark.asyncio
    async def test_harness_install_failure_is_reported(self):
        class FailingSandbox:
            async def exec(self, command, **kwargs):
                return 127, "", "/bin/sh: 1: opencode: not found"

        config = Config.model_validate(
            {
                "harness": {
                    "type": "generic_cli",
                    "install": {"commands": ["npm install -g opencode-ai"]},
                }
            }
        )
        adapter = GenericCliHarnessAdapter(config.harness)

        with pytest.raises(RuntimeError, match="npm install -g opencode-ai"):
            await adapter.install(FailingSandbox())
