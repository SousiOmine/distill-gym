import pytest
from pathlib import Path
from distill_gym.cache.git_cache import _mirror_name, get_mirror_path, ensure_mirror, clone_from_mirror


class TestGitCache:
    def test_mirror_name_consistent(self):
        url = "https://github.com/example/test.git"
        h1 = _mirror_name(url)
        h2 = _mirror_name(url)
        assert h1 == h2
        assert len(h1) == 16

    def test_mirror_name_different_urls(self):
        h1 = _mirror_name("https://github.com/a/b.git")
        h2 = _mirror_name("https://github.com/c/d.git")
        assert h1 != h2

    def test_get_mirror_path(self):
        url = "https://github.com/example/test.git"
        path = get_mirror_path(url)
        assert isinstance(path, Path)
        assert "git-mirrors" in path.parts
