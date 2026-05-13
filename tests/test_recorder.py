import json
import pytest
from pathlib import Path

from distill_gym.proxy.recorder import TraceRecorder


class TestTraceRecorder:
    def test_record_request(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_request({"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "llm_request"
        assert entry["data"]["model"] == "gpt-4"

    def test_record_request_filters_sensitive(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_request({"api_key": "sk-1234", "messages": []})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert "api_key" not in entry["data"]

    def test_record_response(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_response({"choices": [{"message": {"content": "hello"}}], "usage": {}})

        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["event"] == "llm_response"
        assert entry["data"]["choices"][0]["message"]["content"] == "hello"

    def test_record_stream_chunk(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_stream_chunk({"choices": [{"delta": {"content": "chunk"}}]})

        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["event"] == "llm_stream_chunk"
        assert entry["data"]["choices"][0]["delta"]["content"] == "chunk"

    def test_record_artifact(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_artifact("diff", "/path/to/diff.txt")

        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["event"] == "artifact"
        assert entry["data"]["kind"] == "diff"

    def test_record_event_with_task_id(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_event("task_started", {"task_id": "t1"}, "run1_task1")

        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["event"] == "task_started"
        assert entry["task_id"] == "run1_task1"

    def test_multiple_events(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_request({"messages": []}, "task1")
        recorder.record_response({"choices": []}, "task1")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_timestamp_included(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        recorder = TraceRecorder(path)
        recorder.record_event("test", {})

        entry = json.loads(path.read_text(encoding="utf-8"))
        assert "timestamp" in entry
        assert entry["timestamp"]

    def test_directory_created(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "trace.jsonl"
        recorder = TraceRecorder(nested)
        recorder.record_event("test", {})
        assert nested.exists()
