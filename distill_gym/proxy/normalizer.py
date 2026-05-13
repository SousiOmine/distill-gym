from typing import Any


REASONING_FIELDS = ["reasoning_content", "reasoning", "reasoning_text"]


def normalize_assistant_message(choice: dict, normalize_reasoning: bool = True) -> dict:
    delta = choice.get("delta") or choice.get("message") or {}
    message = {"role": "assistant", "content": ""}

    content = delta.get("content", "")
    if content:
        message["content"] = content

    if normalize_reasoning:
        for field in REASONING_FIELDS:
            val = delta.get(field) or choice.get(field)
            if val:
                message["reasoning_content"] = val
                break

    tool_calls = delta.get("tool_calls") or choice.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def merge_stream_chunks(chunks: list[dict]) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])

            for field in REASONING_FIELDS:
                val = delta.get(field)
                if val:
                    reasoning_parts.append(val)
                    break

            tc = delta.get("tool_calls") or []
            for t in tc:
                idx = t.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "type": "", "function": {"name": "", "arguments": ""}}
                if t.get("id"):
                    tool_calls[idx]["id"] = t["id"]
                if t.get("type"):
                    tool_calls[idx]["type"] = t["type"]
                fn = t.get("function", {})
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    tc_list = [v for k, v in sorted(tool_calls.items())]
    if tc_list:
        message["tool_calls"] = tc_list

    return message, finish_reason


def strip_sensitive(data: dict) -> dict:
    safe = {}
    for k, v in data.items():
        if k.lower() in ("authorization", "x-api-key", "api-key"):
            continue
        if isinstance(v, dict):
            safe[k] = strip_sensitive(v)
        elif isinstance(v, list):
            safe[k] = [strip_sensitive(i) if isinstance(i, dict) else i for i in v]
        else:
            safe[k] = v
    return safe
