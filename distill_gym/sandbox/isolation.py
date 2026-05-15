from abc import ABC, abstractmethod
from typing import Optional

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.sandbox.pool import SandboxPool
from distill_gym.sandbox.runtime import SandboxRuntime
from distill_gym.config.schema import SandboxConfig


class IsolationSession(ABC):
    sandbox: SandboxManager

    @abstractmethod
    async def reset(self) -> None:
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        ...


class IsolationStrategy(ABC):
    @abstractmethod
    async def acquire(self, task_id: str) -> IsolationSession:
        ...

    @abstractmethod
    async def release(self, session: IsolationSession) -> None:
        ...

    @abstractmethod
    async def destroy_all(self) -> None:
        ...


class ContainerSession(IsolationSession):
    def __init__(self, sandbox: SandboxManager) -> None:
        self.sandbox = sandbox

    async def reset(self) -> None:
        await self.sandbox.exec("git checkout -- . && git clean -fd 2>/dev/null || true")

    async def cleanup(self) -> None:
        await self.sandbox.exec("git checkout -- . && git clean -fd 2>/dev/null || true")


class ContainerIsolation(IsolationStrategy):
    def __init__(self, pool: SandboxPool) -> None:
        self._pool = pool

    async def acquire(self, task_id: str) -> ContainerSession:
        sb = await self._pool.acquire()
        return ContainerSession(sb)

    async def release(self, session: ContainerSession) -> None:
        await session.cleanup()
        await self._pool.release(session.sandbox)

    async def destroy_all(self) -> None:
        await self._pool.destroy_all()
