import json
import httpx


class LLMClient:
    """Async LLM client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        timeout: int = 120,
        extra_body: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body or {}

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant content string."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature,
                    **self.extra_body,
                },
            )
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def complete_json(self, system_prompt: str, user_prompt: str) -> list | dict:
        """Send a chat completion request and parse the response as JSON.
        Extracts JSON arrays/objects from markdown code fences or surrounding text.
        """
        content = await self.complete(system_prompt, user_prompt)
        return _parse_json_content(content)


def _parse_json_content(content: str) -> list | dict:
    text = content.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:].strip()
    # find JSON array or object
    for opener, closer in [("[", "]"), ("{", "}")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    return json.loads(text)
