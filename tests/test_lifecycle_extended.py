import asyncio
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from distill_gym.orchestrator.events import EventBus
from distill_gym.orchestrator.lifecycle import GracefulShutdown, is_shutting_down


@pytest.fixture(autouse=True)
def reset_shutting_down():
    import distill_gym.orchestrator.lifecycle
    distill_gym.orchestrator.lifecycle._shutting_down = False
    yield
    distill_gym.orchestrator.lifecycle._shutting_down = False


class TestGracefulShutdownExtended:
    def test_is_shutting_down_default_false(self):
        assert is_shutting_down() is False

    def test_graceful_shutdown_context_manager(self, monkeypatch):
        monkeypatch.setattr(signal, "signal", MagicMock())
        monkeypatch.setattr(signal, "getsignal", MagicMock(return_value=MagicMock()))

        bus = EventBus()
        results = []

        async def on_shutdown(**data):
            results.append("shutdown")

        bus.on("shutdown", on_shutdown)

        with GracefulShutdown(event_bus=bus):
            assert is_shutting_down() is False

        assert is_shutting_down() is False

    def test_graceful_shutdown_handler_first_signal(self, monkeypatch):
        monkeypatch.setattr(signal, "signal", MagicMock())
        monkeypatch.setattr(signal, "getsignal", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(sys, "exit", MagicMock())

        import distill_gym.orchestrator.lifecycle as lifecycle
        lifecycle._shutting_down = False

        shutdown = GracefulShutdown()

        shutdown._handler(signal.SIGINT, None)
        assert lifecycle._shutting_down is True

        shutdown._handler(signal.SIGTERM, None)
        sys.exit.assert_called_once_with(1)

    def test_graceful_shutdown_handler_no_running_loop(self, monkeypatch):
        monkeypatch.setattr(signal, "signal", MagicMock())
        monkeypatch.setattr(signal, "getsignal", MagicMock(return_value=MagicMock()))

        import distill_gym.orchestrator.lifecycle as lifecycle
        shutdown = GracefulShutdown()

        lifecycle._shutting_down = False
        mock_get_event_loop = MagicMock(side_effect=RuntimeError("no event loop"))
        monkeypatch.setattr(asyncio, "get_event_loop", mock_get_event_loop)

        shutdown._handler(signal.SIGINT, None)
        assert lifecycle._shutting_down is True

        lifecycle._shutting_down = False
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        monkeypatch.setattr(asyncio, "get_event_loop", MagicMock(return_value=mock_loop))

        shutdown._handler(signal.SIGINT, None)
        assert lifecycle._shutting_down is True

    def test_graceful_shutdown_exit_restores_signals(self, monkeypatch):
        orig_sigint = MagicMock()
        orig_sigterm = MagicMock()
        getsignal_values = {signal.SIGINT: orig_sigint, signal.SIGTERM: orig_sigterm}
        monkeypatch.setattr(signal, "getsignal", lambda sig: getsignal_values[sig])

        called = []

        def mock_signal(sig, handler):
            called.append((sig, handler))

        monkeypatch.setattr(signal, "signal", mock_signal)

        with GracefulShutdown():
            pass

        assert (signal.SIGINT, orig_sigint) in called
        assert (signal.SIGTERM, orig_sigterm) in called

    def test_graceful_shutdown_custom_event_bus(self):
        bus = EventBus()
        shutdown = GracefulShutdown(event_bus=bus)
        assert shutdown.event_bus is bus

    def test_graceful_shutdown_custom_timeout(self):
        shutdown = GracefulShutdown(timeout=99)
        assert shutdown.timeout == 99
