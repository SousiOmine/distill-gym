import pytest
from distill_gym.registry.builder_registry import BuilderRegistry
from distill_gym.sandbox.builders.base import SandboxBuilder
from distill_gym.sandbox.builders.git_repository import GitRepositorySandboxBuilder


class TestBuilderRegistry:
    def test_register_and_create(self):
        cls = type("TestBuilder", (SandboxBuilder,), {"build": lambda self, c: None})
        BuilderRegistry.register("test_builder")(cls)
        instance = BuilderRegistry.create("test_builder")
        assert isinstance(instance, cls)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sandbox builder type"):
            BuilderRegistry.create("nonexistent")

    def test_list_types(self):
        types = BuilderRegistry.list_types()
        assert "git_repository" in types

    def test_git_repository_is_registered(self):
        cls = BuilderRegistry.get("git_repository")
        assert cls is not None
        assert cls is GitRepositorySandboxBuilder

    def test_decorator_preserves_class(self):
        cls = BuilderRegistry.get("git_repository")
        assert cls is GitRepositorySandboxBuilder
