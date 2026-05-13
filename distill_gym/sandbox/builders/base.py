from abc import ABC, abstractmethod
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.config.schema import SandboxConfig


class SandboxBuilder(ABC):
    @abstractmethod
    def build(self, config: SandboxConfig) -> SandboxSpec:
        ...
