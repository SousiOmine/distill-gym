import pytest
from distill_gym.orchestrator.lifecycle import is_shutting_down, GracefulShutdown
from distill_gym.orchestrator.events import EventBus, reset_event_bus


class TestLifecycle:
    def test_shutting_down_default_false(self):
        assert is_shutting_down() is False

    def test_graceful_shutdown_context_manager(self):
        bus = EventBus()
        results = []

        async def on_shutdown(**data):
            results.append("shutdown")

        bus.on("shutdown", on_shutdown)

        with GracefulShutdown(event_bus=bus):
            assert is_shutting_down() is False

        assert is_shutting_down() is False
