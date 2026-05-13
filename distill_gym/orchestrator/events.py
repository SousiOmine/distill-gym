import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable[..., Coroutine]]] = {}

    def on(self, event: str, handler: Callable[..., Coroutine]):
        self._handlers.setdefault(event, []).append(handler)
        return handler

    def off(self, event: str, handler: Callable[..., Coroutine]):
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: str, **data):
        for handler in self._handlers.get(event, []):
            try:
                await handler(**data)
            except Exception:
                logger.exception(f"EventBus handler failed for event '{event}'")

    def clear(self):
        self._handlers.clear()


_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus():
    global _global_bus
    _global_bus = None
