import pytest

from distill_gym.config.schema import TaskGenConfig
from distill_gym.harness.base import HarnessAdapter, HarnessResult
from distill_gym.taskgen.harness_task_generator import HarnessTaskGenerator


class FileSandbox:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("mkdir -p"):
            self.files.pop(self.output_file, None)
            return 0, "", ""
        if command.startswith("test -f"):
            if self.output_file not in self.files:
                return 1, "", ""
            return 0, self.files[self.output_file], ""
        return 0, "", ""


class WritingHarness(HarnessAdapter):
    name = "writing"

    def __init__(self, sandbox: FileSandbox, outputs: list[str]):
        self.sandbox = sandbox
        self.outputs = outputs
        self.tasks = []

    async def install(self, sandbox):
        pass

    async def run_task(self, sandbox, task):
        self.tasks.append(task)
        output = self.outputs[min(len(self.tasks) - 1, len(self.outputs) - 1)]
        self.sandbox.files[self.sandbox.output_file] = output
        return HarnessResult(exit_code=0, stdout="ignored", stderr="", success=True)

    def parse_result(self, process_result):
        return HarnessResult(exit_code=0, stdout="", stderr="", success=True)


def harness_taskgen_config(**kwargs) -> TaskGenConfig:
    data = {
        "type": "harness",
        "output_file": ".distill-gym/taskgen/tasks.json",
        "batch_size": 1,
        "max_rounds": 5,
        "prompts": [
            {"id": "bugfix", "prompt": "Need {target_count}; generate {batch_size} bugfix tasks into {output_file}."},
            {"id": "tests", "prompt": "Avoid {existing_tasks_json} and write {output_file.shell}."},
        ],
    }
    data.update(kwargs)
    return TaskGenConfig.model_validate(data)


@pytest.mark.asyncio
async def test_harness_taskgen_generates_until_requested_count():
    config = harness_taskgen_config()
    sandbox = FileSandbox(config.output_file)
    harness = WritingHarness(
        sandbox,
        [
            '[{"id": "t1", "prompt": "first"}]',
            '[{"id": "t2", "prompt": "second"}]',
            '[{"id": "t3", "prompt": "third"}]',
        ],
    )

    tasks = await HarnessTaskGenerator(config, harness, sandbox).generate(3)

    assert [task.id for task in tasks] == ["t1", "t2", "t3"]
    assert [task.title for task in harness.tasks] == ["bugfix", "tests", "bugfix"]
    assert "Need 3; generate 1" in harness.tasks[0].prompt
    assert len(harness.tasks) == 3


@pytest.mark.asyncio
async def test_harness_taskgen_discards_invalid_and_duplicate_tasks():
    config = harness_taskgen_config(batch_size=2)
    sandbox = FileSandbox(config.output_file)
    harness = WritingHarness(
        sandbox,
        [
            '[{"id": "dup", "prompt": "first"}, {"id": "bad"}]',
            '[{"id": "dup", "prompt": "duplicate"}, {"id": "ok", "prompt": "second"}]',
        ],
    )

    tasks = await HarnessTaskGenerator(config, harness, sandbox).generate(2)

    assert [task.id for task in tasks] == ["dup", "ok"]


@pytest.mark.asyncio
async def test_harness_taskgen_fails_when_count_cannot_be_met():
    config = harness_taskgen_config(max_rounds=2)
    sandbox = FileSandbox(config.output_file)
    harness = WritingHarness(sandbox, ["[]"])

    with pytest.raises(RuntimeError, match="produced 0 valid tasks"):
        await HarnessTaskGenerator(config, harness, sandbox).generate(1)
