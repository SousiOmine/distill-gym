import subprocess
import hashlib
from pathlib import Path
from distill_gym.cache.cache_store import get_git_mirror_dir


def _mirror_name(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode()).hexdigest()[:16]


def get_mirror_path(repo_url: str) -> Path:
    return get_git_mirror_dir() / _mirror_name(repo_url)


def ensure_mirror(repo_url: str) -> Path:
    mirror_path = get_mirror_path(repo_url)
    if mirror_path.exists():
        subprocess.run(
            ["git", "-C", str(mirror_path), "remote", "update", "--prune"],
            capture_output=True, timeout=120,
        )
    else:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--mirror", repo_url, str(mirror_path)],
            capture_output=True, check=True, timeout=300,
        )
    return mirror_path


def clone_from_mirror(repo_url: str, target: Path, ref: str = "main") -> None:
    mirror_path = ensure_mirror(repo_url)
    subprocess.run(
        ["git", "clone", str(mirror_path), str(target)],
        capture_output=True, check=True, timeout=120,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", ref],
        capture_output=True, check=True, timeout=30,
    )
