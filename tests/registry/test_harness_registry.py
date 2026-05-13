import pytest
from distill_gym.registry.harness_registry import HarnessRegistry
from distill_gym.harness.base import HarnessAdapter, MockHarnessAdapter, HarnessResult
from distill_gym.config.schema import HarnessConfig, TaskItem


class TestHarnessRegistry:
    def test_register_and_create(self):
        class ConcreteHarness(HarnessAdapter):
            name = "test_concrete"

            async def install(self, sandbox):
                pass

            async def run_task(self, sandbox, task):
                return HarnessResult(exit_code=0, stdout="", stderr="", success=True)

            def parse_result(self, process_result):
                return HarnessResult(exit_code=0, stdout="", stderr="", success=True)

        HarnessRegistry.register("test_concrete")(ConcreteHarness)
        instance = HarnessRegistry.create("test_concrete", HarnessConfig())
        assert isinstance(instance, ConcreteHarness)
        assert instance.name == "test_concrete"

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown harness type"):
            HarnessRegistry.create("nonexistent", HarnessConfig())

    def test_list_types(self):
        types = HarnessRegistry.list_types()
        assert "mock" in types
        assert "opencode" in types
        assert "generic_cli" in types

    def test_get_returns_none_for_unknown(self):
        assert HarnessRegistry.get("nonexistent") is None

    def test_get_returns_class(self):
        cls = HarnessRegistry.get("mock")
        assert cls is not None
        assert cls is MockHarnessAdapter

    def test_mock_is_registered(self):
        instance = HarnessRegistry.create("mock", HarnessConfig())
        assert isinstance(instance, MockHarnessAdapter)

    def test_decorator_preserves_class(self):
        cls = HarnessRegistry.get("generic_cli")
        assert cls is not None
        from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
        assert cls is GenericCliHarnessAdapter
