import shutil
import subprocess
import sys
from pathlib import Path

from distill_gym.cache.cache_store import get_cache_dir


def get_harness_cache_dir() -> Path:
    return get_cache_dir() / "harness-cache"


def get_harness_bin_dir(name: str) -> Path:
    return get_harness_cache_dir() / name


def ensure_harness(name: str) -> Path:
    bin_dir = get_harness_bin_dir(name)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if name == "opencode":
        _ensure_npm_harness(bin_dir, "opencode-ai")
    elif name == "codex":
        _ensure_pip_harness(bin_dir, "codex")
    elif name == "qwen-code":
        _ensure_pip_harness(bin_dir, "qwen-code")
    else:
        raise ValueError(f"Unknown harness: {name}")

    return bin_dir


def _ensure_npm_harness(bin_dir: Path, package: str):
    node_bin = bin_dir / "node_modules" / ".bin"
    if node_bin.exists():
        return
    subprocess.run(
        [sys.executable, "-m", "npm", "install", "--prefix", str(bin_dir), package],
        capture_output=True, timeout=300,
    )


def _ensure_pip_harness(bin_dir: Path, package: str):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", str(bin_dir), package],
        capture_output=True, timeout=300,
    )


def to_volume(name: str) -> dict:
    bin_dir = get_harness_bin_dir(name)
    return {
        "type": "bind",
        "source": str(bin_dir),
        "target": f"/opt/harness/{name}",
    }


def clear_cache(name: str | None = None) -> None:
    if name:
        cache_dir = get_harness_bin_dir(name)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    else:
        cache_dir = get_harness_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
