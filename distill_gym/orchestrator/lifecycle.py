import asyncio
import logging
import signal
import sys
from typing import Any

from distill_gym.orchestrator.events import EventBus, get_event_bus

logger = logging.getLogger(__name__)


_shutting_down = False


def is_shutting_down() -> bool:
    return _shutting_down


class GracefulShutdown:
    def __init__(self, event_bus: EventBus | None = None, timeout: int = 30):
        self.event_bus = event_bus or get_event_bus()
        self.timeout = timeout
        self._orig_handlers: dict[int, Any] = {}

    def __enter__(self):
        self._orig_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        self._orig_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)

        if sys.platform != "win32":
            signal.signal(signal.SIGINT, self._handler)
            signal.signal(signal.SIGTERM, self._handler)
        else:
            signal.signal(signal.SIGINT, self._handler)
            try:
                signal.signal(signal.SIGTERM, self._handler)
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for sig, handler in self._orig_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _handler(self, signum: int, frame):
        global _shutting_down
        if _shutting_down:
            logger.warning("Forced exit")
            sys.exit(1)
        _shutting_down = True
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._shutdown())
        except RuntimeError:
            pass

    async def _shutdown(self):
        logger.info("Emitting shutdown event...")
        await self.event_bus.emit("shutdown")
        await asyncio.sleep(0.5)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self.timeout)
        logger.info("Shutdown complete")
