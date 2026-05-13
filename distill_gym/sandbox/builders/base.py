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
        if getattr(config, 'repo_url', None) and not config.repo_url.startswith(("http://", "https://", "git@", "ssh://")):
            errors.append(f"sandbox.repo_url has unexpected format: {config.repo_url}")
        return errors
