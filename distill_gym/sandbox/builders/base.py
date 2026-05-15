from abc import ABC, abstractmethod
from typing import Any

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.config.schema import SandboxConfig


class SandboxBuilder(ABC):
    @abstractmethod
    def build(self, config: SandboxConfig) -> SandboxSpec:
        ...

    def validate(self, config: SandboxConfig) -> list[str]:
        errors: list[str] = []
        if not config.image:
            errors.append("sandbox.image is required")
        return errors
