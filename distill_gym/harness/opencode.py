from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from distill_gym.config.schema import HarnessConfig


class OpencodeHarnessAdapter(GenericCliHarnessAdapter):
    name = "opencode"

    def __init__(self, config: HarnessConfig):
        if not config.run.command or config.run.command == "{harness} run {task.prompt}":
            config.run.command = "opencode run {task.prompt}"
        super().__init__(config)
