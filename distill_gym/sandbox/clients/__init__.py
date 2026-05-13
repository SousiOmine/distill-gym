from abc import ABC, abstractmethod

from distill_gym.sandbox.base import SandboxSpec


class ContainerClient(ABC):
    @abstractmethod
    def container_run(self, spec: SandboxSpec) -> str:
        ...

    @abstractmethod
    def container_exec(
        self,
        container_id: str,
        command: str,
        timeout: int = 300,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        ...

    @abstractmethod
    def container_cp_to(self, container_id: str, source: str, target: str) -> None:
        ...

    @abstractmethod
    def container_cp_from(self, container_id: str, source: str, target: str) -> None:
        ...

    @abstractmethod
    def container_stop(self, container_id: str) -> None:
        ...

    @abstractmethod
    def container_rm(self, container_id: str) -> None:
        ...

    @abstractmethod
    def list_containers(self, label: str = "distill-gym=true") -> list[dict]:
        ...

    @abstractmethod
    def list_volumes(self, label: str = "distill-gym=true") -> list[str]:
        ...

    @abstractmethod
    def list_networks(self, label: str = "distill-gym=true") -> list[str]:
        ...

    @abstractmethod
    def network_create(self, name: str, driver: str = "bridge") -> None:
        ...

    @abstractmethod
    def network_rm(self, name: str) -> None:
        ...

    @abstractmethod
    def volume_rm(self, name: str) -> None:
        ...
