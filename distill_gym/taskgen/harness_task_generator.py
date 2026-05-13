import json
import shlex
from typing import Iterable

from pydantic import ValidationError

from distill_gym.config.schema import TaskGenConfig, TaskItem
from distill_gym.harness.base import HarnessAdapter
from distill_gym.taskgen.base import TaskGenerator


class HarnessTaskGenerator(TaskGenerator):
    def __init__(self, config: TaskGenConfig, harness: HarnessAdapter, sandbox: "SandboxManager"):
        self.config = config
        self.harness = harness
        self.sandbox = sandbox

    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        if self.config.tasks:
            return self.config.tasks[:count]
        if not self.config.prompts:
            raise RuntimeError("taskgen.type=harness requires at least one taskgen.prompts item")

        tasks: list[TaskItem] = []
        seen_ids: set[str] = set()

        for round_index in range(self.config.max_rounds):
            if len(tasks) >= count:
                break

            prompt_config = self.config.prompts[round_index % len(self.config.prompts)]
            remaining_count = count - len(tasks)
            batch_limit = min(self.config.batch_size, remaining_count)
            await self._prepare_output_file()

            task = TaskItem(
                id=f"taskgen_{round_index:03d}",
                title=prompt_config.title or prompt_config.id,
                prompt=self._build_prompt(prompt_config.prompt, tasks, remaining_count, batch_limit),
            )
            result = await self.harness.run_task(self.sandbox, task)
            if not result.success:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
                raise RuntimeError(f"task generation harness failed: {detail}")

            for generated in await self._read_generated_tasks():
                if len(tasks) >= count:
                    break
                if not generated.id or generated.id in seen_ids:
                    continue
                if not generated.prompt:
                    continue
                seen_ids.add(generated.id)
                tasks.append(generated)

        if len(tasks) < count:
            raise RuntimeError(
                f"task generation produced {len(tasks)} valid tasks, but {count} were requested"
            )
        return tasks

    def _build_prompt(
        self,
        template: str,
        existing_tasks: list[TaskItem],
        remaining_count: int,
        batch_limit: int,
    ) -> str:
        existing_json = json.dumps(
            [task.model_dump() for task in existing_tasks],
            ensure_ascii=False,
            indent=2,
        )
        output_file = self.config.output_file
        body = template
        replacements = {
            "{target_count}": str(remaining_count),
            "{batch_size}": str(batch_limit),
            "{existing_tasks_json}": existing_json,
            "{output_file}": output_file,
            "{output_file.shell}": shlex.quote(output_file),
        }
        for token, value in replacements.items():
            body = body.replace(token, value)

        return (
            "Generate coding-agent tasks for this repository.\n"
            "Do not rely on stdout for the result. Create or overwrite the output JSON file exactly as requested.\n"
            "The output file must contain a JSON array. Each item must have id and prompt, and may have title and test_command.\n"
            f"Output file: {output_file}\n\n"
            f"{body}"
        )

    async def _prepare_output_file(self) -> None:
        path = shlex.quote(self.config.output_file)
        code, stdout, stderr = await self.sandbox.exec(f"mkdir -p $(dirname {path}) && rm -f {path}", timeout=30)
        if code != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {code}"
            raise RuntimeError(f"failed to prepare task generation output file: {detail}")

    async def cleanup_output_file(self) -> None:
        path = shlex.quote(self.config.output_file)
        await self.sandbox.exec(f"rm -f {path}", timeout=30)

    async def _read_generated_tasks(self) -> list[TaskItem]:
        path = shlex.quote(self.config.output_file)
        code, stdout, stderr = await self.sandbox.exec(f"test -f {path} && cat {path}", timeout=30)
        if code != 0:
            return []

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return list(self._valid_tasks(data))

    def _valid_tasks(self, items: Iterable[object]) -> Iterable[TaskItem]:
        for item in items:
            try:
                yield TaskItem.model_validate(item)
            except ValidationError:
                continue
