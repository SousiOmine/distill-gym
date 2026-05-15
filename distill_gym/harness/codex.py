import shlex

from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from distill_gym.harness.base import HarnessResult
from distill_gym.config.schema import HarnessConfig
from distill_gym.registry.harness_registry import HarnessRegistry


@HarnessRegistry.register("codex")
class CodexHarnessAdapter(GenericCliHarnessAdapter):
    name = "codex"
    _default_command = "codex exec --json {task.prompt.shell}"

    def __init__(self, config: HarnessConfig, **kwargs):
        user_cmd = config.run.command
        default_cmd = "{harness} run {task.prompt}"
        self._has_custom_command = (
            user_cmd not in (None, "", default_cmd, self._default_command)
        )
        if not self._has_custom_command:
            config.run.command = self._default_command
        super().__init__(config, **kwargs)

    async def run_task(self, sandbox, task) -> HarnessResult:
        model = self.provider.model if self.provider else ""
        base_url = self.proxy_base_url or ""

        if self._has_custom_command:
            command = self.config.run.command.format(**self._build_format_kwargs(task))
        else:
            prompt = shlex.quote(task.prompt)
            parts = ["codex exec --json"]
            if model:
                parts.append(f"--model {shlex.quote(model)}")
            parts.append(prompt)
            command = " ".join(parts)
            if base_url:
                command = (
                    f"OPENAI_BASE_URL={shlex.quote(base_url)} "
                    f"OPENAI_API_KEY=distill-gym-proxy {command}"
                )

        exit_code, stdout, stderr = await self._exec(
            sandbox, command,
            timeout=self.config.completion.max_idle_seconds,
        )
        return HarnessResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            success=exit_code in self.config.completion.success_exit_codes,
        )
