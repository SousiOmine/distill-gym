import json
import shlex
from typing import Optional
from distill_gym.harness.base import HarnessAdapter, HarnessResult
from distill_gym.config.schema import TaskItem, HarnessConfig


class _TaskFormat:
    def __init__(self, task: TaskItem):
        self._task = task

    def __getattr__(self, name: str):
        value = getattr(self._task, name)
        if isinstance(value, str):
            return _ShellString(value)
        return value


class _ShellString(str):
    @property
    def shell(self) -> str:
        return shlex.quote(str(self))


class GenericCliHarnessAdapter(HarnessAdapter):
    name = "generic_cli"

    async def _exec(self, sandbox: "SandboxManager", command: str, **kwargs) -> tuple[int, str, str]:
        return await sandbox.exec(command, **kwargs)

    def _build_format_kwargs(self, task: TaskItem) -> dict:
        kwargs = {
            "task": _TaskFormat(task),
            "harness": self.name,
        }
        if self.provider:
            kwargs.update(
                model=shlex.quote(self.provider.model),
                provider_name=shlex.quote(self.provider.name),
                base_url=shlex.quote(self.proxy_base_url or ""),
                api_key=shlex.quote("distill-gym-proxy"),
            )
        return kwargs

    async def install(self, sandbox: "SandboxManager") -> None:
        install_cfg = self.config.install
        for cmd in install_cfg.commands:
            exit_code, stdout, stderr = await self._exec(sandbox, cmd)
            if exit_code != 0:
                detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
                raise RuntimeError(f"harness install command '{cmd}' failed: {detail}")

    async def run_task(self, sandbox: "SandboxManager", task: TaskItem) -> HarnessResult:
        command = self.config.run.command.format(**self._build_format_kwargs(task))
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

    def parse_result(self, process_result: dict) -> HarnessResult:
        return HarnessResult(
            exit_code=process_result.get("exit_code", -1),
            stdout=process_result.get("stdout", ""),
            stderr=process_result.get("stderr", ""),
            success=process_result.get("exit_code", -1) in self.config.completion.success_exit_codes,
        )
