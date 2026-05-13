from pathlib import Path
import os


def get_cache_dir() -> Path:
    override = os.environ.get("DISTILL_GYM_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "distill-gym"


def get_git_mirror_dir() -> Path:
    return get_cache_dir() / "git-mirrors"


def get_artifacts_dir() -> Path:
    return get_cache_dir() / "artifacts"


def ensure_dirs() -> None:
    get_git_mirror_dir().mkdir(parents=True, exist_ok=True)
    get_artifacts_dir().mkdir(parents=True, exist_ok=True)
