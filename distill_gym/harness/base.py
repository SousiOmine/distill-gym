from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from distill_gym.config.schema import TaskItem, ProviderConfig, HarnessConfig
from distill_gym.registry.harness_registry import HarnessRegistry


@dataclass
class HarnessResult:
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    sandbox_output_dir: Optional[str] = None


class HarnessAdapter(ABC):
    name: str = "base"

    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        provider: Optional[ProviderConfig] = None,
        proxy_base_url: Optional[str] = None,
    ):
        self.config = config or HarnessConfig()
        self.provider = provider
        self.proxy_base_url = proxy_base_url

    @abstractmethod
    async def install(self, sandbox: "SandboxManager") -> None:
        ...

    @abstractmethod
    async def run_task(self, sandbox: "SandboxManager", task: TaskItem) -> HarnessResult:
        ...

    @abstractmethod
    def parse_result(self, process_result: dict) -> HarnessResult:
        ...


@HarnessRegistry.register("mock")
class MockHarnessAdapter(HarnessAdapter):
    name = "mock"

    async def install(self, sandbox: "SandboxManager") -> None:
        pass

    async def run_task(self, sandbox: "SandboxManager", task: TaskItem) -> HarnessResult:
        return HarnessResult(exit_code=0, stdout=f"mock output for {task.id}", stderr="", success=True)

    def parse_result(self, process_result: dict) -> HarnessResult:
        return HarnessResult(exit_code=0, stdout="", stderr="", success=True)
