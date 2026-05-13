from collections.abc import Callable, Coroutine
from pathlib import Path


class ExporterRegistry:
    _exporters: dict[str, Callable[..., Coroutine]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable[..., Coroutine]):
            cls._exporters[name] = func
            return func
        return decorator

    @classmethod
    def run(cls, name: str, **kwargs):
        if name not in cls._exporters:
            raise ValueError(f"Unknown exporter: {name}. Available: {list(cls._exporters)}")
        return cls._exporters[name](**kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._exporters)

    @classmethod
    def get(cls, name: str) -> Callable | None:
        return cls._exporters.get(name)
