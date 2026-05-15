import asyncio
import copy
import logging
from collections.abc import Callable, Coroutine
from typing import Optional

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.sandbox.runtime import SandboxRuntime

logger = logging.getLogger(__name__)

InitHook = Callable[[SandboxManager], Coroutine]


class SandboxPool:
    def __init__(
        self,
        runtime: SandboxRuntime,
        spec_template: SandboxSpec,
        max_size: int,
        init_hook: Optional[InitHook] = None,
    ) -> None:
        self._runtime = runtime
        self._spec_template = spec_template
        self._max_size = max_size
        self._init_hook = init_hook
        self._semaphore = asyncio.Semaphore(max_size)
        self._idle: asyncio.Queue[SandboxManager] = asyncio.Queue()
        self._all_sandboxes: list[SandboxManager] = []

    async def acquire(self) -> SandboxManager:
        await self._semaphore.acquire()
        if not self._idle.empty():
            sb = await self._idle.get()
            await self._reset(sb)
            return sb
        try:
            sb = await self._create_sandbox()
            return sb
        except BaseException:
            self._semaphore.release()
            raise

    async def release(self, sandbox: SandboxManager) -> None:
        await self._idle.put(sandbox)
        self._semaphore.release()

    async def exec_on_all(self, command: str, timeout: int = 300) -> None:
        for sb in self._all_sandboxes:
            await sb.exec(command, timeout=timeout)

    async def _reset(self, sandbox: SandboxManager) -> None:
        await sandbox.exec("git checkout -- . && git clean -fd 2>/dev/null || true")

    async def _create_sandbox(self) -> SandboxManager:
        spec = copy.deepcopy(self._spec_template)
        idx = len(self._all_sandboxes)
        spec.container_name = f"{spec.container_name}-pool-{idx}"
        sb = SandboxManager(runtime=self._runtime)
        self._all_sandboxes.append(sb)
        await sb.start(spec)
        if self._init_hook:
            await self._init_hook(sb)
        return sb

    async def destroy_all(self) -> None:
        for sb in list(self._all_sandboxes):
            try:
                await sb.destroy()
            except Exception:
                logger.warning("Failed to destroy sandbox", exc_info=True)
        self._all_sandboxes.clear()
