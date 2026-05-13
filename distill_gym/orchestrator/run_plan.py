from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
import yaml

from distill_gym.config.schema import Config, TaskItem
from distill_gym.taskgen.repo_task_generator import RepoTaskGenerator
from distill_gym.storage.models import RunRecord, TaskRecord


@dataclass
class RunPlan:
    run_id: str
    config: Config
    tasks: list[TaskItem] = field(default_factory=list)

    @classmethod
    async def from_config(cls, config: Config) -> "RunPlan":
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if config.taskgen.tasks:
            return cls(run_id=run_id, config=config, tasks=config.taskgen.tasks[:config.run.task_count])
        if config.taskgen.type == "harness":
            return cls(run_id=run_id, config=config, tasks=[])

        gen = RepoTaskGenerator(config.taskgen, config.sandbox, config.provider)
        tasks = await gen.generate(config.run.task_count)
        return cls(run_id=run_id, config=config, tasks=tasks)

    def to_run_record(self) -> RunRecord:
        return RunRecord(
            id=self.run_id,
            name=self.config.run.name,
            config_yaml=yaml.dump(self.config.model_dump()),
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            harness_type=self.config.harness.type,
            provider_name=self.config.provider.name,
            model=self.config.provider.model,
            sandbox_type=self.config.sandbox.type,
            sandbox_engine=self.config.sandbox.engine.value,
            repo_url=self.config.sandbox.repo_url,
        )

    def to_task_records(self) -> list[TaskRecord]:
        return [
            TaskRecord(
                id=f"{self.run_id}_{t.id}",
                run_id=self.run_id,
                title=t.title,
                prompt=t.prompt,
                status="pending",
                test_command=t.test_command,
            )
            for t in self.tasks
        ]
