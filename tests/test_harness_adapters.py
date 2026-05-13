import pytest

from distill_gym.config.schema import HarnessConfig, ProviderConfig, TaskItem
from distill_gym.harness.codex import CodexHarnessAdapter
from distill_gym.harness.opencode import OpencodeHarnessAdapter
from distill_gym.harness.qwen_code import QwenCodeHarnessAdapter
from distill_gym.harness.generic_cli import GenericCliHarnessAdapter


SAMPLE_PROVIDER = ProviderConfig(
    type="openai_compatible",
    name="test-provider",
    base_url="https://api.test.com/v1",
    api_key_env="TEST_API_KEY",
    model="test-model",
)
PROXY_BASE_URL = "http://proxy.test:5002/v1"


class RecordingSandbox:
    def __init__(self):
        self.commands = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return 0, "ok", ""


@pytest.mark.asyncio
async def test_default_harness_commands_are_non_interactive():
    task = TaskItem(id="t1", prompt="fix the tests")

    cases = [
        (CodexHarnessAdapter, "codex exec --json --model test-model 'fix the tests'"),
        (OpencodeHarnessAdapter, "OPENCODE_CONFIG_CONTENT='{\"model\": \"openai/test-model\", \"provider\": {\"openai\": {\"options\": {\"baseURL\": \"http://proxy.test:5002/v1\", \"apiKey\": \"distill-gym-proxy\"}}}}' opencode run --format json 'fix the tests' --model openai/test-model"),
        (QwenCodeHarnessAdapter, "QWEN_MODEL=test-model qwen --model test-model --prompt 'fix the tests' --openai-base-url http://proxy.test:5002/v1 --openai-api-key distill-gym-proxy"),
    ]

    for adapter_cls, expected in cases:
        sandbox = RecordingSandbox()
        adapter = adapter_cls(HarnessConfig(), provider=SAMPLE_PROVIDER, proxy_base_url=PROXY_BASE_URL)
        await adapter.run_task(sandbox, task)
        assert sandbox.commands[-1] == expected, f"{adapter_cls.__name__}"


@pytest.mark.asyncio
async def test_generic_harness_supports_shell_quoted_prompt_placeholder():
    sandbox = RecordingSandbox()
    cfg = HarnessConfig()
    cfg.run.command = "tool run {task.prompt.shell}"
    adapter = GenericCliHarnessAdapter(cfg)

    await adapter.run_task(sandbox, TaskItem(id="t1", prompt="quote ' this"))

    assert sandbox.commands == ['tool run \'quote \'"\'"\' this\'']


@pytest.mark.asyncio
async def test_harness_without_provider_falls_back_gracefully():
    task = TaskItem(id="t1", prompt="hello")
    cases = [
        (CodexHarnessAdapter, "codex exec --json hello"),
        (OpencodeHarnessAdapter, "opencode run --format json hello"),
        (QwenCodeHarnessAdapter, "qwen --prompt hello"),
    ]

    for adapter_cls, expected in cases:
        sandbox = RecordingSandbox()
        adapter = adapter_cls(HarnessConfig())
        await adapter.run_task(sandbox, task)
        assert sandbox.commands[-1] == expected, f"{adapter_cls.__name__}"


@pytest.mark.asyncio
async def test_generic_harness_with_provider_template_vars():
    sandbox = RecordingSandbox()
    cfg = HarnessConfig()
    cfg.run.command = "mycli --model {model} run {task.prompt.shell}"
    adapter = GenericCliHarnessAdapter(cfg, provider=SAMPLE_PROVIDER, proxy_base_url=PROXY_BASE_URL)

    await adapter.run_task(sandbox, TaskItem(id="t1", prompt="do something"))

    assert sandbox.commands[-1] == "mycli --model test-model run 'do something'"


@pytest.mark.asyncio
async def test_opencode_config_content_is_injected():
    task = TaskItem(id="t1", prompt="hello")
    adapter = OpencodeHarnessAdapter(HarnessConfig(), provider=SAMPLE_PROVIDER, proxy_base_url=PROXY_BASE_URL)
    sandbox = RecordingSandbox()
    await adapter.run_task(sandbox, task)

    cmd = sandbox.commands[-1]
    assert cmd.startswith("OPENCODE_CONFIG_CONTENT=")
    assert "--model openai/test-model" in cmd
    assert "--format json" in cmd
    assert "hello" in cmd and "--format json" in cmd
