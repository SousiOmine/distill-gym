import pytest
from pathlib import Path
from distill_gym.cache.harness_cache import (
    get_harness_cache_dir,
    get_harness_bin_dir,
    to_volume,
)


class TestHarnessCache:
    def test_cache_dir(self):
        d = get_harness_cache_dir()
        assert isinstance(d, Path)
        assert "harness-cache" in d.parts

    def test_bin_dir(self):
        d = get_harness_bin_dir("opencode")
        assert isinstance(d, Path)
        assert d.name == "opencode"
        assert d.parent.name == "harness-cache"

    def test_to_volume_opencode(self):
        vol = to_volume("opencode")
        assert vol["type"] == "bind"
        assert "opencode" in vol["source"]
        assert vol["target"] == "/opt/harness/opencode"

    def test_to_volume_codex(self):
        vol = to_volume("codex")
        assert vol["target"] == "/opt/harness/codex"

    def test_to_volume_qwen(self):
        vol = to_volume("qwen-code")
        assert vol["target"] == "/opt/harness/qwen-code"
