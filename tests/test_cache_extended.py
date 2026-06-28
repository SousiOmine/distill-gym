import hashlib
import os
import sys

import pytest
from unittest import mock

from distill_gym.cache.cache_store import get_cache_dir, get_artifacts_dir, get_git_mirror_dir, ensure_dirs
from distill_gym.cache.git_cache import _mirror_name, get_mirror_path, ensure_mirror, clone_from_mirror
from distill_gym.cache.harness_cache import (
    ensure_harness,
    _ensure_npm_harness,
    _ensure_pip_harness,
    to_volume,
    clear_cache,
    get_harness_bin_dir,
    get_harness_cache_dir,
)


class TestCacheStore:
    def test_get_cache_dir_with_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        assert get_cache_dir() == tmp_path

    def test_get_cache_dir_default(self, monkeypatch):
        monkeypatch.delenv("DISTILL_GYM_CACHE_DIR", raising=False)
        expected = os.path.expanduser("~")  # Path.home() equivalent check
        assert str(get_cache_dir()) == os.path.join(expected, ".cache", "distill-gym")

    def test_get_artifacts_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        assert get_artifacts_dir() == tmp_path / "artifacts"

    def test_get_git_mirror_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        assert get_git_mirror_dir() == tmp_path / "git-mirrors"

    def test_ensure_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        ensure_dirs()
        assert (tmp_path / "git-mirrors").is_dir()
        assert (tmp_path / "artifacts").is_dir()


class TestGitCache:
    def test_ensure_mirror_existing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        url = "https://github.com/example/test.git"
        mirror_path = get_mirror_path(url)
        mirror_path.mkdir(parents=True)

        with mock.patch("distill_gym.cache.git_cache.subprocess.run") as mock_run:
            result = ensure_mirror(url)

        assert result == mirror_path
        mock_run.assert_called_once_with(
            ["git", "-C", str(mirror_path), "remote", "update", "--prune"],
            capture_output=True, timeout=120,
            encoding="utf-8", errors="replace",
        )

    def test_ensure_mirror_new(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        url = "https://github.com/example/test.git"
        mirror_path = get_mirror_path(url)

        with mock.patch("distill_gym.cache.git_cache.subprocess.run") as mock_run:
            result = ensure_mirror(url)

        assert result == mirror_path
        assert mirror_path.parent.is_dir()
        mock_run.assert_called_once_with(
            ["git", "clone", "--mirror", url, str(mirror_path)],
            capture_output=True, check=True, timeout=300,
            encoding="utf-8", errors="replace",
        )

    def test_clone_from_mirror(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        url = "https://github.com/example/test.git"
        target = tmp_path / "clone-target"

        with mock.patch("distill_gym.cache.git_cache.subprocess.run") as mock_run:
            clone_from_mirror(url, target, ref="develop")

        mirror_path = get_mirror_path(url)
        assert mock_run.call_args_list == [
            mock.call(
                ["git", "clone", "--mirror", url, str(mirror_path)],
                capture_output=True, check=True, timeout=300,
                encoding="utf-8", errors="replace",
            ),
            mock.call(
                ["git", "clone", str(mirror_path), str(target)],
                capture_output=True, check=True, timeout=120,
                encoding="utf-8", errors="replace",
            ),
            mock.call(
                ["git", "-C", str(target), "checkout", "develop"],
                capture_output=True, check=True, timeout=30,
                encoding="utf-8", errors="replace",
            ),
        ]

    def test_mirror_name(self):
        url = "https://github.com/example/test.git"
        name = _mirror_name(url)
        assert len(name) == 16
        expected = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert name == expected

    def test_get_mirror_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        url = "https://github.com/example/test.git"
        path = get_mirror_path(url)
        assert path == tmp_path / "git-mirrors" / _mirror_name(url)


class TestHarnessCache:
    @mock.patch("distill_gym.cache.harness_cache._ensure_npm_harness")
    def test_ensure_harness_opencode(self, mock_npm, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        result = ensure_harness("opencode")
        bin_dir = get_harness_bin_dir("opencode")
        assert result == bin_dir
        mock_npm.assert_called_once_with(bin_dir, "opencode-ai")

    @mock.patch("distill_gym.cache.harness_cache._ensure_pip_harness")
    def test_ensure_harness_codex(self, mock_pip, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        result = ensure_harness("codex")
        bin_dir = get_harness_bin_dir("codex")
        assert result == bin_dir
        mock_pip.assert_called_once_with(bin_dir, "codex")

    @mock.patch("distill_gym.cache.harness_cache._ensure_pip_harness")
    def test_ensure_harness_qwen_code(self, mock_pip, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        result = ensure_harness("qwen-code")
        bin_dir = get_harness_bin_dir("qwen-code")
        assert result == bin_dir
        mock_pip.assert_called_once_with(bin_dir, "qwen-code")

    def test_ensure_harness_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Unknown harness: unknown-harness"):
            ensure_harness("unknown-harness")

    @mock.patch("distill_gym.cache.harness_cache.subprocess.run")
    def test_ensure_npm_harness_skipped(self, mock_run, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        bin_dir = get_harness_bin_dir("test-npm")
        node_bin = bin_dir / "node_modules" / ".bin"
        node_bin.mkdir(parents=True)
        _ensure_npm_harness(bin_dir, "test-package")
        mock_run.assert_not_called()

    @mock.patch("distill_gym.cache.harness_cache.subprocess.run")
    def test_ensure_npm_harness_installs(self, mock_run, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        bin_dir = get_harness_bin_dir("test-npm")
        bin_dir.mkdir(parents=True)
        _ensure_npm_harness(bin_dir, "test-package")
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "npm", "install", "--prefix", str(bin_dir), "test-package"],
            capture_output=True, timeout=300,
        )

    @mock.patch("distill_gym.cache.harness_cache.subprocess.run")
    def test_ensure_pip_harness(self, mock_run, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        bin_dir = get_harness_bin_dir("test-pip")
        bin_dir.mkdir(parents=True)
        _ensure_pip_harness(bin_dir, "test-package")
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "--target", str(bin_dir), "test-package"],
            capture_output=True, timeout=300,
        )

    def test_to_volume(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        vol = to_volume("opencode")
        bin_dir = get_harness_bin_dir("opencode")
        assert vol == {
            "type": "bind",
            "source": str(bin_dir),
            "target": "/opt/harness/opencode",
        }

    def test_clear_cache_with_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        bin_dir = get_harness_bin_dir("test-clear")
        bin_dir.mkdir(parents=True)
        (bin_dir / "some-file").touch()
        clear_cache("test-clear")
        assert not bin_dir.exists()

    def test_clear_cache_all(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))
        bin_dir1 = get_harness_bin_dir("test-a")
        bin_dir1.mkdir(parents=True)
        (bin_dir1 / "file-a").touch()
        bin_dir2 = get_harness_bin_dir("test-b")
        bin_dir2.mkdir(parents=True)
        (bin_dir2 / "file-b").touch()
        clear_cache()
        assert not get_harness_cache_dir().exists()
