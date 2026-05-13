from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill_gym.sandbox.builders.base import SandboxBuilder


class BuilderRegistry:
    _builders: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(builder_cls: type):
            cls._builders[name] = builder_cls
            return builder_cls
        return decorator

    @classmethod
    def create(cls, name: str):
        if name not in cls._builders:
            raise ValueError(f"Unknown sandbox builder type: {name}. Available: {list(cls._builders)}")
        return cls._builders[name]()

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._builders)

    @classmethod
    def get(cls, name: str) -> type | None:
        return cls._builders.get(name)
