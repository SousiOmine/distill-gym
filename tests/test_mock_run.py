import pytest
from distill_gym.config.schema import Config, HarnessConfig, TaskItem
from distill_gym.harness.base import MockHarnessAdapter, HarnessResult
from distill_gym.orchestrator.run_plan import RunPlan


class TestMockRun:
    @pytest.mark.asyncio
    async def test_mock_harness_run(self):
        config = Config()
        config.harness.type = "mock"
        plan = await RunPlan.from_config(config)
        assert len(plan.tasks) > 0

    def test_mock_harness_adapter(self):
        adapter = MockHarnessAdapter()
        task = TaskItem(id="test_001", title="Test", prompt="run tests")
        result = adapter.parse_result({"exit_code": 0})
        assert isinstance(result, HarnessResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_mock_run_creates_tasks(self):
        config = Config()
        config.run.task_count = 3
        config.harness.type = "mock"
        plan = await RunPlan.from_config(config)
        assert len(plan.tasks) == 3
        for t in plan.tasks:
            assert t.id
            assert t.prompt
