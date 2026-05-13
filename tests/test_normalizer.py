from distill_gym.proxy.normalizer import (
    normalize_assistant_message,
    merge_stream_chunks,
    strip_sensitive,
)


class TestNormalizer:
    def test_normalize_assistant_message_content(self):
        choice = {"message": {"role": "assistant", "content": "Hello"}}
        result = normalize_assistant_message(choice)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello"

    def test_normalize_assistant_message_reasoning(self):
        choice = {"message": {"role": "assistant", "content": "Hello", "reasoning_content": "thinking..."}}
        result = normalize_assistant_message(choice, normalize_reasoning=True)
        assert result["reasoning_content"] == "thinking..."

    def test_normalize_assistant_message_reasoning_disabled(self):
        choice = {"message": {"role": "assistant", "content": "Hello", "reasoning_content": "thinking..."}}
        result = normalize_assistant_message(choice, normalize_reasoning=False)
        assert "reasoning_content" not in result

    def test_normalize_vendor_reasoning_fields(self):
        choice = {"message": {"role": "assistant", "content": "Hi", "reasoning": "deep think"}}
        result = normalize_assistant_message(choice)
        assert result["reasoning_content"] == "deep think"

    def test_normalize_delta(self):
        choice = {"delta": {"role": "assistant", "content": "stream"}}
        result = normalize_assistant_message(choice)
        assert result["content"] == "stream"


class TestMergeStreamChunks:
    def test_merge_content(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "World"}}]},
            {"choices": [{"delta": {}}, {"finish_reason": "stop"}]},
        ]
        merged, reason = merge_stream_chunks(chunks)
        assert merged["content"] == "Hello World"

    def test_merge_reasoning(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"reasoning_content": "ing"}}]},
            {"choices": [{"delta": {"content": "Answer"}}]},
        ]
        merged, _ = merge_stream_chunks(chunks)
        assert merged["reasoning_content"] == "thinking"
        assert merged["content"] == "Answer"

    def test_merge_tool_calls(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": '{"loc'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ation\": \"NYC\"}"}}]}}]},
        ]
        merged, _ = merge_stream_chunks(chunks)
        assert len(merged["tool_calls"]) == 1
        assert merged["tool_calls"][0]["function"]["name"] == "get_weather"
        assert merged["tool_calls"][0]["function"]["arguments"] == '{"location": "NYC"}'


class TestStripSensitive:
    def test_strip_authorization(self):
        data = {"Authorization": "Bearer sk-1234", "messages": [{"role": "user"}]}
        result = strip_sensitive(data)
        assert "Authorization" not in result
        assert "messages" in result

    def test_strip_nested(self):
        data = {"headers": {"x-api-key": "secret123"}}
        result = strip_sensitive(data)
        assert "x-api-key" not in result["headers"]

    def test_keep_innocent(self):
        data = {"model": "gpt-4", "messages": [{"role": "user"}]}
        result = strip_sensitive(data)
        assert result["model"] == "gpt-4"
