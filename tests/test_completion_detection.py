import pytest
from distill_gym.config.schema import HarnessConfig, TaskItem
from distill_gym.harness.generic_cli import GenericCliHarnessAdapter
from distill_gym.harness.base import HarnessResult


class TestCompletionDetection:
    def test_success_by_exit_code(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0, 42]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({"exit_code": 42, "stdout": "", "stderr": ""})
        assert result.success is True

    def test_failure_by_exit_code(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({"exit_code": 1, "stdout": "", "stderr": ""})
        assert result.success is False

    def test_success_by_pattern(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0]
        cfg.completion.success_patterns = ["All tests passed"]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({
            "exit_code": 1,
            "stdout": "All tests passed! 100% coverage",
            "stderr": "",
        })
        assert result.success is True

    def test_failure_by_pattern(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0]
        cfg.completion.failure_patterns = ["FAILED", "Error:"]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({
            "exit_code": 0,
            "stdout": "Tests FAILED: 2 errors",
            "stderr": "",
        })
        assert result.success is False

    def test_failure_pattern_takes_priority(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0]
        cfg.completion.success_patterns = ["All tests passed"]
        cfg.completion.failure_patterns = ["FAILED"]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({
            "exit_code": 1,
            "stdout": "All tests passed\nBut also FAILED",
            "stderr": "",
        })
        assert result.success is False

    def test_success_in_stderr(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [1]
        cfg.completion.success_patterns = ["completed successfully"]
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({
            "exit_code": 1,
            "stdout": "",
            "stderr": "Task completed successfully",
        })
        assert result.success is True

    def test_no_patterns_uses_exit_code_only(self):
        cfg = HarnessConfig()
        cfg.completion.success_exit_codes = [0]
        cfg.completion.success_patterns = []
        cfg.completion.failure_patterns = []
        adapter = GenericCliHarnessAdapter(cfg)

        result = adapter.parse_result({"exit_code": 0, "stdout": "anything", "stderr": ""})
        assert result.success is True

        result = adapter.parse_result({"exit_code": 1, "stdout": "anything", "stderr": ""})
        assert result.success is False
