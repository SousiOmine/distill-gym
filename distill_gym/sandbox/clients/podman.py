import asyncio
import json
import subprocess
from typing import Optional

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.clients import ContainerClient


class PodmanClient(ContainerClient):
    def __init__(self, binary: str = "podman"):
        self.binary = binary

    async def _run(self, args: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
        cmd = [self.binary] + args
        try:
            return await asyncio.to_thread(
                lambda: subprocess.run(cmd, capture_output=True, encoding="utf-8", check=check, **kwargs),
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Podman binary not found: {self.binary}") from e

    async def container_run(self, spec: SandboxSpec) -> str:
        args = ["run", "-d", "--name", spec.container_name]
        for k, v in spec.labels.items():
            args.extend(["--label", f"{k}={v}"])
        for k, v in spec.env.items():
            args.extend(["-e", f"{k}={v}"])
        for vol in spec.volumes:
            if vol.get("type") == "bind" and vol.get("source"):
                args.extend(["-v", f"{vol['source']}:{vol['target']}"])
            elif vol.get("source"):
                args.extend(["-v", f"{vol['source']}:{vol['target']}"])
        if spec.network_name:
            args.extend(["--network", spec.network_name])
        elif spec.network_mode:
            args.extend(["--network", spec.network_mode])
        args.extend(["--hostname", spec.hostname])
        args.append(spec.image)
        args.extend(spec.cmd)
        result = await self._run(args, timeout=60)
        return result.stdout.strip()

    async def container_stop(self, container_id: str) -> None:
        await self._run(["stop", container_id], check=False, timeout=30)

    async def container_rm(self, container_id: str) -> None:
        await self._run(["rm", "-f", container_id], check=False, timeout=30)

    async def container_exec(
        self,
        container_id: str,
        command: str,
        timeout: int = 300,
        workdir: Optional[str] = None,
    ) -> tuple[int, str, str]:
        args = [self.binary, "exec"]
        if workdir:
            args.extend(["--workdir", workdir])
        args.extend([container_id, "/bin/sh", "-c", command])
        try:
            result = await asyncio.to_thread(
                lambda: subprocess.run(
                    args,
                    capture_output=True, encoding="utf-8", timeout=timeout,
                ),
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"

    async def container_cp_to(self, container_id: str, source: str, target: str) -> None:
        await self._run(["cp", source, f"{container_id}:{target}"], timeout=60)

    async def container_cp_from(self, container_id: str, source: str, target: str) -> None:
        await self._run(["cp", f"{container_id}:{source}", target], timeout=60)

    async def list_containers(self, label: str = "distill-gym=true") -> list[dict]:
        result = await self._run(
            ["ps", "-a", "--filter", f"label={label}", "--format", "json"],
            check=False, timeout=30,
        )
        if not result.stdout.strip():
            return []
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    async def list_volumes(self, label: str = "distill-gym=true") -> list[str]:
        result = await self._run(
            ["volume", "ls", "--filter", f"label={label}", "--format", "{{.Name}}"],
            check=False, timeout=30,
        )
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

    async def volume_rm(self, name: str) -> None:
        await self._run(["volume", "rm", "-f", name], check=False, timeout=30)

    async def list_networks(self, label: str = "distill-gym=true") -> list[str]:
        result = await self._run(
            ["network", "ls", "--filter", f"label={label}", "--format", "{{.Name}}"],
            check=False, timeout=30,
        )
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

    async def network_create(self, name: str, driver: str = "bridge") -> None:
        await self._run(["network", "create", "--label", "distill-gym=true", "--driver", driver, name], timeout=30)

    async def network_rm(self, name: str) -> None:
        await self._run(["network", "rm", name], check=False, timeout=30)
