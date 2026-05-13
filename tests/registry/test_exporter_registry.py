import pytest
from distill_gym.registry.exporter_registry import ExporterRegistry


class TestExporterRegistry:
    def test_register_and_get(self):
        async def dummy_exporter(**kwargs):
            return 0

        ExporterRegistry.register("test_export")(dummy_exporter)
        assert ExporterRegistry.get("test_export") is dummy_exporter

    def test_list_types(self):
        types = ExporterRegistry.list_types()
        assert isinstance(types, list)

    def test_get_unknown_returns_none(self):
        assert ExporterRegistry.get("nonexistent") is None
