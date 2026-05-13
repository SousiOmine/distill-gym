import pytest
from distill_gym.registry.taskgen_registry import TaskGenRegistry
from distill_gym.taskgen.base import TaskGenerator


class TestTaskGenRegistry:
    def test_register_and_create(self):
        class ConcreteGen(TaskGenerator):
            def __init__(self, config, **kwargs):
                self.config = config

            async def generate(self, count, run_id=""):
                return []

        TaskGenRegistry.register("test_gen")(ConcreteGen)
        instance = TaskGenRegistry.create("test_gen", "myconfig")
        assert isinstance(instance, ConcreteGen)
        assert instance.config == "myconfig"

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task generator type"):
            TaskGenRegistry.create("nonexistent", None)

    def test_list_types(self):
        types = TaskGenRegistry.list_types()
        assert isinstance(types, list)
