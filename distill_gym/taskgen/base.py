from abc import ABC, abstractmethod
from distill_gym.config.schema import TaskItem


class TaskGenerator(ABC):
    @abstractmethod
    async def generate(self, count: int, run_id: str = "") -> list[TaskItem]:
        ...
