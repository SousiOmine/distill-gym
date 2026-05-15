import json
from pathlib import Path

from distill_gym.exporters.openai_messages import _build_conversation
from distill_gym.storage.run_store import RunStore


def _ensure_message(messages: list, task) -> list:
    if not messages and task and task.prompt:
        messages.append({"role": "user", "content": task.prompt})
    if not messages:
        messages.append({"role": "user", "content": "No trace recorded."})
    return messages


async def merge_runs_to_jsonl(
    run_ids: list[str],
    output: Path,
    store: RunStore,
    include_reasoning: bool = True,
    include_tool_results: bool = True,
    include_tools: bool = True,
    include_failed: bool = False,
) -> int:
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for run_id in run_ids:
            run = await store.get_run(run_id)
            if not run:
                continue

            tasks = await store.list_tasks(run_id)
            for task in tasks:
                if not include_failed and task.success is not True:
                    continue

                conversations, metadata = await _build_conversation(
                    run, task, store, include_reasoning, include_tool_results, include_tools,
                )
                for conversation, tools in conversations:
                    conversation = _ensure_message(conversation, task)
                    record = {
                        "messages": conversation,
                        "metadata": metadata,
                        "source_run_id": run_id,
                    }
                    if include_tools and tools:
                        record["tools"] = tools
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    count += 1

    return count

