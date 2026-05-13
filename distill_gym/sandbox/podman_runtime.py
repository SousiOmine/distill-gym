from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.clients.podman import PodmanClient
from distill_gym.sandbox.clients import ContainerClient
from distill_gym.sandbox.runtime import SandboxRuntime


class PodmanSandboxRuntime(SandboxRuntime):
    def __init__(self, client: ContainerClient | None = None):
        self.client = client or PodmanClient()

    def start(self, spec: SandboxSpec) -> str:
        return self.client.container_run(spec)

    def exec(self, container_id: str, command: str, timeout: int = 300, workdir: str | None = None) -> tuple[int, str, str]:
        return self.client.container_exec(container_id, command, timeout=timeout, workdir=workdir)

    def copy_to(self, container_id: str, source: str, target: str) -> None:
        self.client.container_cp_to(container_id, source, target)

    def copy_from(self, container_id: str, source: str, target: str) -> None:
        self.client.container_cp_from(container_id, source, target)

    def stop(self, container_id: str) -> None:
        self.client.container_stop(container_id)

    def remove(self, container_id: str) -> None:
        self.client.container_rm(container_id)

    def cleanup_resources(self, label: str = "distill-gym=true") -> dict:
        result = {"containers": 0, "volumes": 0, "networks": 0}

        for c in self.client.list_containers(label):
            self.client.container_stop(c.get("Id", ""))
            self.client.container_rm(c.get("Id", ""))
            result["containers"] += 1

        for v in self.client.list_volumes(label):
            self.client.volume_rm(v)
            result["volumes"] += 1

        for n in self.client.list_networks(label):
            self.client.network_rm(n)
            result["networks"] += 1

        return result
