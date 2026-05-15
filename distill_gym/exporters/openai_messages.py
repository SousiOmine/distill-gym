import json
from pathlib import Path
from typing import Optional
import yaml

from distill_gym.proxy.normalizer import normalize_assistant_message, merge_stream_chunks
from distill_gym.storage.run_store import RunStore
from distill_gym.cache.cache_store import get_artifacts_dir

ConversationWithTools = tuple[list[dict], Optional[list[dict]]]


async def export_openai_messages_jsonl(
    run_id: str,
    output: Path,
    store: RunStore,
    include_reasoning: bool = True,
    include_tool_results: bool = True,
    include_tools: bool = True,
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
                run, task, store, include_reasoning, include_tool_results, include_tools,
            )

            for conversation, tools in conversations:
                record = {
                    "messages": conversation,
                    "metadata": metadata,
                }
                if include_tools and tools:
                    record["tools"] = tools
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                count += 1

    return count


async def _build_conversation(
    run, task, store, include_reasoning: bool, include_tool_results: bool,
    include_tools: bool = True,
) -> tuple[list[ConversationWithTools], dict]:
    short_task_id = task.id.removeprefix(f"{run.id}_") if task.id.startswith(f"{run.id}_") else task.id
    raw_trace_path = get_artifacts_dir() / run.id / short_task_id / "raw_trace.jsonl"

    conversations: list[ConversationWithTools] = []

    if raw_trace_path.exists():
        pairs = _parse_trace_events(raw_trace_path, include_reasoning)
        conv_groups = _group_into_conversations(pairs)

        for group in conv_groups:
            conversation, conv_tools = _reconstruct_conversation(group, include_tool_results, include_reasoning)
            if conversation:
                conversations.append((conversation, conv_tools if include_tools else None))

    # Fallback: if no conversations from trace but task has a prompt, create a default
    if not conversations and task and task.prompt:
        conversations.append(([{"role": "user", "content": task.prompt}], None))

    artifacts_map = {}
    changed_files = []
    arts = await store.list_artifacts(run.id, task.id)
    for a in arts:
        artifacts_map[a.kind] = a.path
        if a.kind == "changed_files":
            changed_path = get_artifacts_dir() / a.path
            if changed_path.exists():
                try:
                    changed_files = json.loads(changed_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    changed_files = []

    config_data = {}
    if getattr(run, "config_yaml", ""):
        try:
            config_data = yaml.safe_load(run.config_yaml) or {}
        except yaml.YAMLError:
            config_data = {}
    provider_cfg = config_data.get("provider", {})
    sandbox_cfg = config_data.get("sandbox", {})
    harness_cfg = config_data.get("harness", {})

    metadata = {
        "run_id": run.id,
        "task_id": short_task_id,
        "task_title": task.title or "",
        "harness": {
            "name": run.harness_type or "",
            "command": (harness_cfg.get("run") or {}).get("command", ""),
        },
        "provider": {
            "name": run.provider_name or "",
            "base_url": provider_cfg.get("base_url", ""),
            "model": run.model or "",
        },
        "sandbox": {
            "type": run.sandbox_type or "",
            "engine": run.sandbox_engine or "",
            "image": sandbox_cfg.get("image", ""),
            "repo_url": run.repo_url or "",
            "commit": run.commit_hash or "",
        },
        "result": {
            "success": task.success or False,
            "exit_code": task.exit_code,
            "tests_passed": task.tests_passed,
            "changed_files": changed_files,
        },
        "artifacts": artifacts_map,
    }

    return conversations, metadata


def _parse_trace_events(
    trace_path: Path, include_reasoning: bool,
) -> list[tuple[list[dict], Optional[list[dict]], dict]]:
    pairs: list[tuple[list[dict], Optional[list[dict]], dict]] = []
    curr_request = None
    curr_tools = None

    with open(trace_path, encoding="utf-8") as f:
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
                curr_request = data.get("messages", [])
                curr_tools = data.get("tools")
            elif event["event"] == "llm_response":
                if curr_request is not None:
                    choices = data.get("choices", [])
                    if choices:
                        response_msg = normalize_assistant_message(
                            choices[0], normalize_reasoning=include_reasoning,
                        )
                        if response_msg.get("tool_calls") or response_msg.get("content") or response_msg.get("reasoning_content"):
                            pairs.append((list(curr_request), curr_tools, response_msg))
                    curr_request = None
                    curr_tools = None

    return pairs


def _group_into_conversations(
    pairs: list[tuple[list[dict], Optional[list[dict]], dict]],
) -> list[list[tuple[list[dict], Optional[list[dict]], dict]]]:
    groups: list[list[tuple[list[dict], Optional[list[dict]], dict]]] = []
    current: list[tuple[list[dict], Optional[list[dict]], dict]] = []

    for request_msgs, tools, response in pairs:
        if current:
            prev_request = current[-1][0]
            if _is_new_conversation(prev_request, request_msgs):
                groups.append(current)
                current = []
        current.append((request_msgs, tools, response))

    if current:
        groups.append(current)

    return groups


def _is_new_conversation(prev_request: list[dict], curr_request: list[dict]) -> bool:
    if len(curr_request) < len(prev_request):
        return True
    if len(curr_request) == len(prev_request):
        for i in range(len(curr_request)):
            p = prev_request[i]
            c = curr_request[i]
            if p.get("role") != c.get("role") or p.get("content") != c.get("content"):
                return True
    return False


def _reconstruct_conversation(
    group: list[tuple[list[dict], Optional[list[dict]], dict]],
    include_tool_results: bool,
    include_reasoning: bool,
) -> tuple[list[dict], Optional[list[dict]]]:
    if not group:
        return [], None

    last_request_messages = group[-1][0]
    last_response = group[-1][2]

    response_msgs = [pair[2] for pair in group if pair[2] is not None]

    tools = group[0][1]

    messages: list[dict] = []
    response_idx = 0

    for msg in last_request_messages:
        role = msg.get("role", "")
        if role == "tool" and not include_tool_results:
            continue

        msg_copy = dict(msg)

        if role == "assistant" and response_idx < len(response_msgs):
            resp = response_msgs[response_idx]
            if include_reasoning and resp.get("reasoning_content") and not msg_copy.get("reasoning_content"):
                msg_copy["reasoning_content"] = resp["reasoning_content"]
            response_idx += 1

        messages.append(msg_copy)

    if last_response:
        final = dict(last_response)
        if not include_reasoning:
            final.pop("reasoning_content", None)
        messages.append(final)

    return messages, tools
