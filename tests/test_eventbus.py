import pytest
from distill_gym.orchestrator.events import EventBus, get_event_bus, reset_event_bus


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_and_handle(self):
        bus = EventBus()
        results = []

        async def handler(**data):
            results.append(data)

        bus.on("test_event", handler)
        await bus.emit("test_event", foo="bar")
        assert results == [{"foo": "bar"}]

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = EventBus()
        results = []

        async def handler1(**data):
            results.append(f"h1:{data['val']}")

        async def handler2(**data):
            results.append(f"h2:{data['val']}")

        bus.on("evt", handler1)
        bus.on("evt", handler2)
        await bus.emit("evt", val=42)
        assert results == ["h1:42", "h2:42"]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_block_others(self):
        bus = EventBus()
        results = []

        async def failing(**data):
            raise RuntimeError("oops")

        async def ok(**data):
            results.append("ok")

        bus.on("evt", failing)
        bus.on("evt", ok)
        await bus.emit("evt")
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        bus = EventBus()
        results = []

        async def handler(**data):
            results.append("called")

        bus.on("evt", handler)
        bus.off("evt", handler)
        await bus.emit("evt")
        assert results == []

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        bus = EventBus()
        results = []

        async def handler(**data):
            results.append("called")

        bus.on("evt", handler)
        bus.clear()
        await bus.emit("evt")
        assert results == []

    def test_global_bus_singleton(self):
        reset_event_bus()
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
        reset_event_bus()
        bus3 = get_event_bus()
        assert bus3 is not bus1
