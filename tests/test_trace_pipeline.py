import json
import tempfile
from pathlib import Path

import pytest

from distill_gym.orchestrator.trace_pipeline import TracePipeline
from distill_gym.proxy.recorder import TraceRecorder


class TestTracePipeline:
    @pytest.mark.asyncio
    async def test_record_event_creates_task_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TracePipeline(proxy_recorder=None, run_id="test-run")
            task_id = "task_001"
            pipeline.register_task(task_id)
            pipeline._task_paths[task_id] = Path(tmpdir) / task_id / "raw_trace.jsonl"

            await pipeline.record_event(task_id, "task_started", {"task_id": task_id})

            trace_file = pipeline._task_paths[task_id]
            assert trace_file.exists()
            lines = trace_file.read_text().strip().split("\n")
            assert len(lines) == 1
            event = json.loads(lines[0])
            assert event["event"] == "task_started"
            assert event["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_distribute_splits_traces_by_task_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir) / "proxy" / "raw_trace.jsonl"
            shared_path.parent.mkdir(parents=True)

            proxy_recorder = TraceRecorder(shared_path)
            proxy_recorder.record_request(
                {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"},
                task_id="task_a",
            )
            proxy_recorder.record_request(
                {"messages": [{"role": "user", "content": "world"}], "model": "gpt-4"},
                task_id="task_b",
            )
            proxy_recorder.record_request(
                {"messages": [{"role": "user", "content": "another"}], "model": "gpt-4"},
                task_id="task_a",
            )

            pipeline = TracePipeline(proxy_recorder=proxy_recorder, run_id="test-run")
            pipeline.register_task("task_a")
            pipeline.register_task("task_b")
            pipeline._task_paths["task_a"] = Path(tmpdir) / "task_a" / "raw_trace.jsonl"
            pipeline._task_paths["task_b"] = Path(tmpdir) / "task_b" / "raw_trace.jsonl"

            await pipeline.distribute()

            task_a_lines = pipeline._task_paths["task_a"].read_text().strip().split("\n")
            task_b_lines = pipeline._task_paths["task_b"].read_text().strip().split("\n")

            assert len(task_a_lines) == 2
            assert len(task_b_lines) == 1
            assert json.loads(task_a_lines[0])["task_id"] == "task_a"
            assert json.loads(task_a_lines[1])["task_id"] == "task_a"
            assert json.loads(task_b_lines[0])["task_id"] == "task_b"

    @pytest.mark.asyncio
    async def test_distribute_keeps_interleaved_task_conversations_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir) / "proxy" / "raw_trace.jsonl"
            shared_path.parent.mkdir(parents=True)

            proxy_recorder = TraceRecorder(shared_path)
            proxy_recorder.record_request(
                {"messages": [{"role": "user", "content": "a"}], "model": "gpt-4"},
                task_id="task_a",
            )
            proxy_recorder.record_request(
                {"messages": [{"role": "user", "content": "b"}], "model": "gpt-4"},
                task_id="task_b",
            )
            proxy_recorder.record_response(
                {"choices": [{"message": {"role": "assistant", "content": "A"}}]},
                task_id="task_a",
            )
            proxy_recorder.record_response(
                {"choices": [{"message": {"role": "assistant", "content": "B"}}]},
                task_id="task_b",
            )

            pipeline = TracePipeline(proxy_recorder=proxy_recorder, run_id="test-run")
            pipeline.register_task("task_a")
            pipeline.register_task("task_b")
            pipeline._task_paths["task_a"] = Path(tmpdir) / "task_a" / "raw_trace.jsonl"
            pipeline._task_paths["task_b"] = Path(tmpdir) / "task_b" / "raw_trace.jsonl"

            await pipeline.distribute()

            task_a_events = [
                json.loads(line) for line in pipeline._task_paths["task_a"].read_text().strip().split("\n")
            ]
            task_b_events = [
                json.loads(line) for line in pipeline._task_paths["task_b"].read_text().strip().split("\n")
            ]

            assert [e["event"] for e in task_a_events] == ["llm_request", "llm_response"]
            assert [e["event"] for e in task_b_events] == ["llm_request", "llm_response"]
            assert {e["task_id"] for e in task_a_events} == {"task_a"}
            assert {e["task_id"] for e in task_b_events} == {"task_b"}

    @pytest.mark.asyncio
    async def test_distribute_no_proxy_recorder(self):
        pipeline = TracePipeline(proxy_recorder=None, run_id="test-run")
        pipeline.register_task("task_001")
        await pipeline.distribute()  # Should not raise

    @pytest.mark.asyncio
    async def test_register_task_sets_path(self):
        pipeline = TracePipeline(proxy_recorder=None, run_id="test-run-id")
        pipeline.register_task("task_x")
        assert "task_x" in pipeline._task_paths
        path = pipeline._task_paths["task_x"]
        assert "test-run-id" in str(path)
        assert "task_x" in str(path)
        assert path.name == "raw_trace.jsonl"
