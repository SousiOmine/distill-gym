import json
from pathlib import Path
from typing import Optional

from distill_gym.proxy.normalizer import normalize_assistant_message, merge_stream_chunks
from distill_gym.storage.run_store import RunStore
from distill_gym.cache.cache_store import get_artifacts_dir


async def export_openai_messages_jsonl(
    run_id: str,
    output: Path,
    store: RunStore,
    include_reasoning: bool = True,
    include_tool_results: bool = True,
    include_failed: bool = False,
) -> int:
    run = await store.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    tasks = await store.list_tasks(run_id)
    count = 0

    with open(output, "w", encoding="utf-8") as f:
        for task in tasks:
            if not include_failed and task.success is not True:
                continue

            messages, metadata = await _build_conversation(
                run, task, store, include_reasoning, include_tool_results,
            )

            record = {
                "messages": messages,
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1

    return count


async def _build_conversation(
    run, task, store, include_reasoning: bool, include_tool_results: bool,
) -> tuple[list, dict]:
    messages: list[dict] = []
    short_task_id = task.id.removeprefix(f"{run.id}_") if task.id.startswith(f"{run.id}_") else task.id
    raw_trace_path = get_artifacts_dir() / run.id / short_task_id / "raw_trace.jsonl"

    if raw_trace_path.exists():
        system_msg = None
        user_msgs: list[dict] = []
        assistant_msgs: list[dict] = []
        stream_chunks: list[dict] = []
        in_stream = False

        with open(raw_trace_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                data = event.get("data", {})
                if event["event"] == "llm_request":
                    req_messages = data.get("messages", [])
                    for msg in req_messages:
                        if msg.get("role") == "system":
                            system_msg = msg
                        elif msg.get("role") == "user":
                            user_msgs.append(msg)
                        elif msg.get("role") == "assistant":
                            assistant_msgs.append(msg)
                        elif msg.get("role") == "tool":
                            if include_tool_results:
                                assistant_msgs.append(msg)

                elif event["event"] == "llm_stream_chunk":
                    stream_chunks.append(data)

                elif event["event"] == "llm_response":
                    if stream_chunks:
                        merged, _ = merge_stream_chunks(stream_chunks)
                        if not include_reasoning:
                            merged.pop("reasoning_content", None)
                        if merged.get("tool_calls") or merged.get("content"):
                            assistant_msgs.append(merged)
                        stream_chunks = []
                    else:
                        for choice in data.get("choices", []):
                            msg = normalize_assistant_message(choice, normalize_reasoning=include_reasoning)
                            if msg.get("tool_calls") or msg.get("content"):
                                assistant_msgs.append(msg)

        if system_msg:
            messages.append(system_msg)
        messages.extend(user_msgs)
        messages.extend(assistant_msgs)

    artifacts_map = {}
    arts = await store.list_artifacts(run.id, task.id)
    for a in arts:
        artifacts_map[a.kind] = a.path

    metadata = {
        "run_id": run.id,
        "task_id": short_task_id,
        "task_title": task.title or "",
        "harness": {
            "name": run.harness_type or "",
            "command": "",
        },
        "provider": {
            "name": run.provider_name or "",
            "base_url": "",
            "model": run.model or "",
        },
        "sandbox": {
            "type": run.sandbox_type or "",
            "engine": run.sandbox_engine or "",
            "image": "",
            "repo_url": run.repo_url or "",
            "commit": run.commit_hash or "",
        },
        "result": {
            "success": task.success or False,
            "exit_code": task.exit_code,
            "tests_passed": task.tests_passed,
            "changed_files": [],
        },
        "artifacts": artifacts_map,
    }

    return messages, metadata
