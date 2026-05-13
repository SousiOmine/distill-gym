import shlex
import tempfile
import posixpath
from pathlib import Path
from typing import Optional

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.config.schema import SandboxConfig
from distill_gym.sandbox.podman import PodmanClient
from distill_gym.sandbox.podman_runtime import PodmanSandboxRuntime
from distill_gym.sandbox.runtime import SandboxRuntime
from distill_gym.cache.git_cache import clone_from_mirror


class SandboxManager:
    def __init__(self, podman: Optional[PodmanClient] = None, runtime: Optional[SandboxRuntime] = None):
        self.runtime = runtime or PodmanSandboxRuntime(podman)
        self.container_id: Optional[str] = None
        self.temp_dir: Optional[Path] = None
        self.workdir: str = "/workspace"

    async def start(self, spec: SandboxSpec) -> str:
        self.container_id = self.runtime.start(spec)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="distill-gym-"))
        self.workdir = spec.workdir
        return self.container_id

    async def exec(
        self,
        command: str,
        timeout: int = 300,
        workdir: Optional[str] = None,
    ) -> tuple[int, str, str]:
        if not self.container_id:
            raise RuntimeError("Container not started")
        return self.runtime.exec(
            self.container_id,
            command,
            timeout=timeout,
            workdir=workdir or self.workdir,
        )

    async def checked_exec(
        self,
        command: str,
        timeout: int = 300,
        workdir: Optional[str] = None,
        context: str = "command",
    ) -> tuple[str, str]:
        code, stdout, stderr = await self.exec(command, timeout=timeout, workdir=workdir)
        if code != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {code}"
            raise RuntimeError(f"{context} failed: {detail}")
        return stdout, stderr

    async def prepare_git_repository(self, config: SandboxConfig) -> None:
        quoted_workdir = shlex.quote(config.workdir)
        quoted_repo = shlex.quote(config.repo_url)
        quoted_ref = shlex.quote(config.ref)

        if config.use_git_cache and self.temp_dir:
            repo_dir_name = posixpath.basename(config.workdir.rstrip("/")) or "repo"
            repo_parent = posixpath.dirname(config.workdir.rstrip("/")) or "/"
            host_repo = self.temp_dir / repo_dir_name
            clone_from_mirror(config.repo_url, host_repo, config.ref)
            await self.checked_exec(f"mkdir -p {shlex.quote(repo_parent)}", workdir="/", context="create workdir parent")
            await self.copy_to(str(host_repo), repo_parent)
            for cmd in config.setup:
                await self.checked_exec(cmd, timeout=900, context=f"sandbox setup command '{cmd}'")
            return

        await self.checked_exec(f"mkdir -p {quoted_workdir}", workdir="/", context="create workdir")
        await self.checked_exec(
            "command -v git >/dev/null 2>&1 || "
            "(command -v apt-get >/dev/null 2>&1 && "
            "apt-get update && apt-get install -y git ca-certificates)",
            workdir="/",
            timeout=600,
            context="install git",
        )
        await self.checked_exec(
            f"git clone --branch {quoted_ref} --depth 1 {quoted_repo} {quoted_workdir}",
            workdir="/",
            timeout=900,
            context="clone repository",
        )

        for cmd in config.setup:
            await self.checked_exec(cmd, timeout=900, context=f"sandbox setup command '{cmd}'")

    async def copy_to(self, source: str, target: str) -> None:
        if not self.container_id:
            raise RuntimeError("Container not started")
        self.runtime.copy_to(self.container_id, source, target)

    async def copy_from(self, source: str, target: str) -> None:
        if not self.container_id:
            raise RuntimeError("Container not started")
        self.runtime.copy_from(self.container_id, source, target)

    async def stop(self) -> None:
        if self.container_id:
            self.runtime.stop(self.container_id)

    async def destroy(self) -> None:
        if self.container_id:
            self.runtime.remove(self.container_id)
            self.container_id = None
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    async def cleanup_resources(self, label: str = "distill-gym=true") -> dict:
        return self.runtime.cleanup_resources(label)

    def get_temp_dir(self) -> Optional[Path]:
        return self.temp_dir
