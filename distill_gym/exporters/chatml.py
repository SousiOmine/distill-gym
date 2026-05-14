import json
from pathlib import Path

from distill_gym.exporters.openai_messages import _build_conversation
from distill_gym.storage.run_store import RunStore


def _message_to_chatml(message: dict) -> str:
    role = message.get("role", "user")
    content = message.get("content") or ""
    if message.get("reasoning_content"):
        content = f"<reasoning>\n{message['reasoning_content']}\n</reasoning>\n{content}"
    if message.get("tool_calls"):
        content = f"{content}\n<tool_calls>\n{json.dumps(message['tool_calls'], ensure_ascii=False)}\n</tool_calls>"
    return f"<|im_start|>{role}\n{content}<|im_end|>"


async def export_chatml_jsonl(
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
            conversations, metadata = await _build_conversation(
                run, task, store, include_reasoning, include_tool_results,
            )
            for conversation in conversations:
                text = "\n".join(_message_to_chatml(m) for m in conversation)
                f.write(json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count
