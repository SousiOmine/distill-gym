import json
import subprocess
from typing import Optional

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.clients import ContainerClient


class DockerClient(ContainerClient):
    def __init__(self, binary: str = "docker"):
        self.binary = binary

    def _run(self, args: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
        cmd = [self.binary] + args
        try:
            return subprocess.run(cmd, capture_output=True, encoding="utf-8", check=check, **kwargs)
        except FileNotFoundError as e:
            raise RuntimeError(f"Docker binary not found: {self.binary}") from e

    def container_run(self, spec: SandboxSpec) -> str:
        args = ["run", "-d", "--name", spec.container_name]
        for k, v in spec.labels.items():
            args.extend(["--label", f"{k}={v}"])
        for k, v in spec.env.items():
            args.extend(["-e", f"{k}={v}"])
        for vol in spec.volumes:
            type_ = vol.get("type", "bind")
            source = vol.get("source")
            target = vol.get("target")
            if source and target:
                if type_ == "volume":
                    args.extend(["-v", f"{source}:{target}"])
                else:
                    args.extend(["-v", f"{source}:{target}"])
        if spec.network_mode:
            args.extend(["--network", spec.network_mode])
        args.extend(["--hostname", spec.hostname])
        args.append(spec.image)
        args.extend(spec.cmd)
        result = self._run(args, timeout=60)
        return result.stdout.strip()

    def container_stop(self, container_id: str) -> None:
        self._run(["stop", container_id], check=False, timeout=30)

    def container_rm(self, container_id: str) -> None:
        self._run(["rm", "-f", container_id], check=False, timeout=30)

    def container_exec(
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
            result = subprocess.run(
                args,
                capture_output=True, encoding="utf-8", timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"

    def container_cp_to(self, container_id: str, source: str, target: str) -> None:
        self._run(["cp", source, f"{container_id}:{target}"], timeout=60)

    def container_cp_from(self, container_id: str, source: str, target: str) -> None:
        self._run(["cp", f"{container_id}:{source}", target], timeout=60)

    def list_containers(self, label: str = "distill-gym=true") -> list[dict]:
        result = self._run(
            ["ps", "-a", "--filter", f"label={label}", "--format", "json"],
            check=False, timeout=30,
        )
        if not result.stdout.strip():
            return []
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    def list_volumes(self, label: str = "distill-gym=true") -> list[str]:
        result = self._run(
            ["volume", "ls", "--filter", f"label={label}", "--format", "{{.Name}}"],
            check=False, timeout=30,
        )
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

    def volume_rm(self, name: str) -> None:
        self._run(["volume", "rm", "-f", name], check=False, timeout=30)

    def list_networks(self, label: str = "distill-gym=true") -> list[str]:
        result = self._run(
            ["network", "ls", "--filter", f"label={label}", "--format", "{{.Name}}"],
            check=False, timeout=30,
        )
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

    def network_create(self, name: str, driver: str = "bridge") -> None:
        self._run(["network", "create", "--driver", driver, name], timeout=30)

    def network_rm(self, name: str) -> None:
        self._run(["network", "rm", name], check=False, timeout=30)
