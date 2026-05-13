from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill_gym.taskgen.base import TaskGenerator


class TaskGenRegistry:
    _generators: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(gen_cls: type):
            cls._generators[name] = gen_cls
            return gen_cls
        return decorator

    @classmethod
    def create(cls, name: str, config, **kwargs):
        if name not in cls._generators:
            raise ValueError(f"Unknown task generator type: {name}. Available: {list(cls._generators)}")
        return cls._generators[name](config, **kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._generators)

    @classmethod
    def get(cls, name: str) -> type | None:
        return cls._generators.get(name)
