from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from distill_gym.config.schema import HarnessConfig


class CodexHarnessAdapter(GenericCliHarnessAdapter):
    name = "codex"

    def __init__(self, config: HarnessConfig):
        if not config.run.command or config.run.command == "{harness} run {task.prompt}":
            config.run.command = "codex exec --task {task.prompt}"
        super().__init__(config)
