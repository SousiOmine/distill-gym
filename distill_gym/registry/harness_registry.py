from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill_gym.harness.base import HarnessAdapter


class HarnessRegistry:
    _adapters: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_cls: type):
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def create(cls, name: str, config, **kwargs):
        if name not in cls._adapters:
            raise ValueError(f"Unknown harness type: {name}. Available: {list(cls._adapters)}")
        return cls._adapters[name](config, **kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._adapters)

    @classmethod
    def get(cls, name: str) -> type | None:
        return cls._adapters.get(name)
