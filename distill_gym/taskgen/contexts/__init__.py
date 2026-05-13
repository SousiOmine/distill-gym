from abc import ABC, abstractmethod
from pathlib import Path


class ContextProvider(ABC):
    @abstractmethod
    async def get_context(self, config: dict | None = None) -> str:
        ...


class FileContextProvider(ContextProvider):
    async def get_context(self, config: dict | None = None) -> str:
        paths = (config or {}).get("paths", [])
        max_files = (config or {}).get("max_files", 50)
        content_parts = []
        count = 0
        for path_str in paths:
            base = Path(path_str)
            if not base.exists():
                continue
            if base.is_file():
                parts = [(base.parent, base)]
            else:
                parts = []
                for f in base.rglob("*"):
                    if f.is_file() and ".git" not in f.parts:
                        parts.append((base, f))
            for base_dir, file_path in parts:
                if count >= max_files:
                    break
                try:
                    rel = file_path.relative_to(base_dir).as_posix()
                    text = file_path.read_text(encoding="utf-8", errors="replace")[:5000]
                    content_parts.append(f"### {rel}\n{text}")
                    count += 1
                except Exception:
                    pass
        return "\n\n".join(content_parts) if content_parts else ""


class StaticContextProvider(ContextProvider):
    async def get_context(self, config: dict | None = None) -> str:
        return (config or {}).get("text", "")
