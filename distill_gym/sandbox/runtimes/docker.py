from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.clients import ContainerClient
from distill_gym.sandbox.clients.docker import DockerClient
from distill_gym.sandbox.runtime import SandboxRuntime


class DockerSandboxRuntime(SandboxRuntime):
    def __init__(self, client: ContainerClient | None = None):
        self.client = client or DockerClient()

    async def start(self, spec: SandboxSpec) -> str:
        return await self.client.container_run(spec)

    async def exec(self, container_id: str, command: str, timeout: int = 300, workdir: str | None = None) -> tuple[int, str, str]:
        return await self.client.container_exec(container_id, command, timeout=timeout, workdir=workdir)

    async def copy_to(self, container_id: str, source: str, target: str) -> None:
        await self.client.container_cp_to(container_id, source, target)

    async def copy_from(self, container_id: str, source: str, target: str) -> None:
        await self.client.container_cp_from(container_id, source, target)

    async def stop(self, container_id: str) -> None:
        await self.client.container_stop(container_id)

    async def remove(self, container_id: str) -> None:
        await self.client.container_rm(container_id)

    async def cleanup_resources(self, label: str = "distill-gym=true") -> dict:
        result = {"containers": 0, "volumes": 0, "networks": 0}

        for c in await self.client.list_containers(label):
            await self.client.container_stop(c.get("Id", ""))
            await self.client.container_rm(c.get("Id", ""))
            result["containers"] += 1

        for v in await self.client.list_volumes(label):
            await self.client.volume_rm(v)
            result["volumes"] += 1

        for n in await self.client.list_networks(label):
            await self.client.network_rm(n)
            result["networks"] += 1

        return result
