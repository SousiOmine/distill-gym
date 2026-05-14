import json
import os
import tempfile
import shutil
import pytest
import aiosqlite
from pathlib import Path
from datetime import datetime, timezone

from distill_gym.storage.run_store import RunStore
from distill_gym.storage.models import RunRecord, TaskRecord, ArtifactRecord
from distill_gym.exporters.openai_messages import export_openai_messages_jsonl
from distill_gym.cache.cache_store import get_artifacts_dir


@pytest.fixture
async def mem_store():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    from distill_gym.storage.db import _ensure_tables
    await _ensure_tables(db)
    store = RunStore(db)
    yield store
    await db.close()


def _with_temp_cache(fn):
    """Decorator/helper to run a test with a temporary cache directory."""
    async def wrapper(*args, **kwargs):
        tmpdir = tempfile.mkdtemp()
        old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
        os.environ["DISTILL_GYM_CACHE_DIR"] = tmpdir
        try:
            return await fn(*args, **kwargs)
        finally:
            if old_env:
                os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
            else:
                os.environ.pop("DISTILL_GYM_CACHE_DIR", None)
            shutil.rmtree(tmpdir, ignore_errors=True)
    return wrapper


@pytest.mark.asyncio
async def test_export_empty_run(mem_store):
    store = mem_store
    run_id = "test-export-empty"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    task = TaskRecord(id="t1", run_id=run_id, title="task1", prompt="do something", status="completed", success=True)
    await store.create_task(task)

    output = Path("test_export_output.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
            include_reasoning=True, include_tool_results=True, include_failed=False,
        )
        assert count == 1
        with open(output) as f:
            line = json.loads(f.readline())
            assert "messages" in line
            assert "metadata" in line
            assert line["metadata"]["task_id"] == "t1"
            assert line["metadata"]["result"]["success"] is True
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_export_excludes_failed(mem_store):
    store = mem_store
    run_id = "test-export-failed"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)

    task_ok = TaskRecord(id="t1", run_id=run_id, title="ok", prompt="ok", status="completed", success=True)
    task_fail = TaskRecord(id="t2", run_id=run_id, title="fail", prompt="fail", status="completed", success=False)
    await store.create_task(task_ok)
    await store.create_task(task_fail)

    output = Path("test_export_failed.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
            include_failed=False,
        )
        assert count == 1
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_metadata_no_secrets(mem_store):
    store = mem_store
    run_id = "test-no-secrets"
    run = RunRecord(
        id=run_id, name="test", config_yaml="", status="completed",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    await store.create_run(run)
    task = TaskRecord(id="t1", run_id=run_id, title="t1", prompt="p", status="completed", success=True)
    await store.create_task(task)

    output = Path("test_no_secrets.jsonl")
    try:
        count = await export_openai_messages_jsonl(
            run_id=run_id, output=output, store=store,
        )
        assert count == 1
        with open(output) as f:
            line = json.loads(f.readline())
            meta = json.dumps(line["metadata"])
            assert "sk-" not in meta
            assert "api_key" not in meta.lower()
            assert "authorization" not in meta.lower()
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.asyncio
async def test_export_multi_turn_conversation(mem_store):
    """Multi-turn conversation (tool call → result → answer) is properly ordered."""
    store = mem_store
    run_id = "multi-turn-test"
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
    os.environ["DISTILL_GYM_CACHE_DIR"] = tmpdir
    try:
        run = RunRecord(
            id=run_id, name="multi-turn", config_yaml="", status="completed",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        await store.create_run(run)
        task = TaskRecord(
            id="mt1", run_id=run_id, title="Multi-turn Task",
            prompt="What is the weather?", status="completed", success=True,
        )
        await store.create_task(task)

        artifact_dir = get_artifacts_dir() / run_id / "mt1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / "raw_trace.jsonl"

        trace_lines = [
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "",
                "reasoning_content": "I need to check the weather for Tokyo.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}],
            }}]}}),
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25°C"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "The weather in Tokyo is sunny at 25°C.",
                "reasoning_content": "Based on the weather data, Tokyo is sunny.",
            }}]}}),
        ]
        trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

        output = Path("test_multi_turn.jsonl")
        try:
            count = await export_openai_messages_jsonl(
                run_id=run_id, output=output, store=store,
                include_reasoning=True, include_tool_results=True,
            )
            assert count == 1

            with open(output) as f:
                record = json.loads(f.readline())
                msgs = record["messages"]

            assert len(msgs) == 5
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"
            assert "reasoning_content" in msgs[2]
            assert msgs[2]["reasoning_content"] == "I need to check the weather for Tokyo."
            assert msgs[3]["role"] == "tool"
            assert msgs[3]["content"] == "Sunny, 25°C"
            assert msgs[4]["role"] == "assistant"
            assert msgs[4]["content"] == "The weather in Tokyo is sunny at 25°C."
            assert msgs[4]["reasoning_content"] == "Based on the weather data, Tokyo is sunny."
        finally:
            if output.exists():
                output.unlink()
    finally:
        if old_env:
            os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
        else:
            os.environ.pop("DISTILL_GYM_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_export_multiple_conversations(mem_store):
    """Multiple separate conversations in one task produce separate records."""
    store = mem_store
    run_id = "multi-conv-test"
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
    os.environ["DISTILL_GYM_CACHE_DIR"] = tmpdir
    try:
        run = RunRecord(
            id=run_id, name="multi-conv", config_yaml="", status="completed",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        await store.create_run(run)
        task = TaskRecord(
            id="mc1", run_id=run_id, title="Multi-conv Task",
            prompt="test", status="completed", success=True,
        )
        await store.create_task(task)

        artifact_dir = get_artifacts_dir() / run_id / "mc1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / "raw_trace.jsonl"

        trace_lines = [
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "The capital of France is Paris.",
            }}]}}),
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "The capital of Japan is Tokyo.",
            }}]}}),
        ]
        trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

        output = Path("test_multi_conv.jsonl")
        try:
            count = await export_openai_messages_jsonl(
                run_id=run_id, output=output, store=store,
                include_reasoning=True, include_tool_results=True,
            )
            assert count == 2

            records = []
            with open(output) as f:
                for line in f:
                    records.append(json.loads(line))
            assert len(records) == 2
            assert records[0]["messages"][1]["content"] == "What is the capital of France?"
            assert records[1]["messages"][1]["content"] == "What is the capital of Japan?"
        finally:
            if output.exists():
                output.unlink()
    finally:
        if old_env:
            os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
        else:
            os.environ.pop("DISTILL_GYM_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_export_without_tool_results(mem_store):
    """include_tool_results=False removes tool-role messages."""
    store = mem_store
    run_id = "no-tool-test"
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
    os.environ["DISTILL_GYM_CACHE_DIR"] = tmpdir
    try:
        run = RunRecord(
            id=run_id, name="no-tool", config_yaml="", status="completed",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        await store.create_run(run)
        task = TaskRecord(
            id="nt1", run_id=run_id, title="No Tool Task",
            prompt="test", status="completed", success=True,
        )
        await store.create_task(task)

        artifact_dir = get_artifacts_dir() / run_id / "nt1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / "raw_trace.jsonl"

        trace_lines = [
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}],
            }}]}}),
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25°C"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "The weather in Tokyo is sunny at 25°C.",
            }}]}}),
        ]
        trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

        output = Path("test_no_tool.jsonl")
        try:
            count = await export_openai_messages_jsonl(
                run_id=run_id, output=output, store=store,
                include_reasoning=True, include_tool_results=False,
            )
            assert count == 1

            with open(output) as f:
                record = json.loads(f.readline())
                msgs = record["messages"]

            assert len(msgs) == 4
            for msg in msgs:
                assert msg["role"] != "tool"
            assert msgs[2]["role"] == "assistant"
            assert msgs[2]["tool_calls"]
            assert msgs[3]["role"] == "assistant"
            assert msgs[3]["content"] == "The weather in Tokyo is sunny at 25°C."
        finally:
            if output.exists():
                output.unlink()
    finally:
        if old_env:
            os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
        else:
            os.environ.pop("DISTILL_GYM_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_export_triple_turn_conversation(mem_store):
    """Three turns: tool_call → tool result → tool_call → tool result → final answer."""
    store = mem_store
    run_id = "triple-turn-test"
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("DISTILL_GYM_CACHE_DIR")
    os.environ["DISTILL_GYM_CACHE_DIR"] = tmpdir
    try:
        run = RunRecord(
            id=run_id, name="triple-turn", config_yaml="", status="completed",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        await store.create_run(run)
        task = TaskRecord(
            id="tt1", run_id=run_id, title="Triple Turn",
            prompt="test", status="completed", success=True,
        )
        await store.create_task(task)

        artifact_dir = get_artifacts_dir() / run_id / "tt1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / "raw_trace.jsonl"

        tc1 = {"id": "call_1", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\": \"Tokyo\"}"}}
        tc2 = {"id": "call_2", "type": "function", "function": {"name": "get_area", "arguments": "{\"city\": \"Tokyo\"}"}}

        trace_lines = [
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the population density of Tokyo?"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "", "reasoning_content": "Need population and area.",
                "tool_calls": [tc1],
            }}]}}),
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the population density of Tokyo?"},
                {"role": "assistant", "content": "", "tool_calls": [tc1]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Population: 14M"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "", "reasoning_content": "Got population, now need area.",
                "tool_calls": [tc2],
            }}]}}),
            json.dumps({"event": "llm_request", "data": {"messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the population density of Tokyo?"},
                {"role": "assistant", "content": "", "tool_calls": [tc1]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Population: 14M"},
                {"role": "assistant", "content": "", "tool_calls": [tc2]},
                {"role": "tool", "tool_call_id": "call_2", "content": "Area: 2194 km²"},
            ]}}),
            json.dumps({"event": "llm_response", "data": {"choices": [{"message": {
                "role": "assistant", "content": "The population density of Tokyo is about 6,382 people per km².",
                "reasoning_content": "Population 14M / Area 2194 km² = 6382.",
            }}]}}),
        ]
        trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

        output = Path("test_triple_turn.jsonl")
        try:
            count = await export_openai_messages_jsonl(
                run_id=run_id, output=output, store=store,
                include_reasoning=True, include_tool_results=True,
            )
            assert count == 1

            with open(output) as f:
                record = json.loads(f.readline())
                msgs = record["messages"]

            # system, user, assistant(tool1), tool, assistant(tool2), tool, assistant(answer)
            assert len(msgs) == 7
            roles = [m["role"] for m in msgs]
            assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]

            assert msgs[2]["reasoning_content"] == "Need population and area."
            assert msgs[4]["reasoning_content"] == "Got population, now need area."
            assert msgs[6]["content"] == "The population density of Tokyo is about 6,382 people per km²."
            assert msgs[6]["reasoning_content"] == "Population 14M / Area 2194 km² = 6382."
        finally:
            if output.exists():
                output.unlink()
    finally:
        if old_env:
            os.environ["DISTILL_GYM_CACHE_DIR"] = old_env
        else:
            os.environ.pop("DISTILL_GYM_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)
