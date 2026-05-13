from abc import ABC, abstractmethod

from distill_gym.sandbox.base import SandboxSpec


class SandboxRuntime(ABC):
    @abstractmethod
    def start(self, spec: SandboxSpec) -> str:
        ...

    @abstractmethod
    def exec(self, container_id: str, command: str, timeout: int = 300, workdir: str | None = None) -> tuple[int, str, str]:
        ...

    @abstractmethod
    def copy_to(self, container_id: str, source: str, target: str) -> None:
        ...

    @abstractmethod
    def copy_from(self, container_id: str, source: str, target: str) -> None:
        ...

    @abstractmethod
    def stop(self, container_id: str) -> None:
        ...

    @abstractmethod
    def remove(self, container_id: str) -> None:
        ...

    @abstractmethod
    def cleanup_resources(self, label: str = "distill-gym=true") -> dict:
        ...
