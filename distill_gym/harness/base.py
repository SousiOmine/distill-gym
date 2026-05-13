from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from distill_gym.config.schema import TaskItem


@dataclass
class HarnessResult:
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    sandbox_output_dir: Optional[str] = None


class HarnessAdapter(ABC):
    name: str = "base"

    @abstractmethod
    async def install(self, sandbox: "SandboxManager") -> None:
        ...

    @abstractmethod
    async def run_task(self, sandbox: "SandboxManager", task: TaskItem, proxy_info: dict) -> HarnessResult:
        ...

    @abstractmethod
    def parse_result(self, process_result: dict) -> HarnessResult:
        ...


class MockHarnessAdapter(HarnessAdapter):
    name = "mock"

    async def install(self, sandbox: "SandboxManager") -> None:
        pass

    async def run_task(self, sandbox: "SandboxManager", task: TaskItem, proxy_info: dict) -> HarnessResult:
        return HarnessResult(exit_code=0, stdout=f"mock output for {task.id}", stderr="", success=True)

    def parse_result(self, process_result: dict) -> HarnessResult:
        return HarnessResult(exit_code=0, stdout="", stderr="", success=True)
