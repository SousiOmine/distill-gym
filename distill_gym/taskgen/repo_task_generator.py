from distill_gym.taskgen.base import TaskGenerator
from distill_gym.config.schema import TaskItem, TaskGenConfig


class RepoTaskGenerator(TaskGenerator):
    def __init__(self, config: TaskGenConfig):
        self.config = config

    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        if self.config.tasks:
            return self.config.tasks[:count]

        tasks = []
        for i in range(count):
            tasks.append(TaskItem(
                id=f"task_{i:03d}",
                title=f"Automated task {i}",
                prompt=f"Please complete task {i} in this repository.",
            ))
        return tasks
