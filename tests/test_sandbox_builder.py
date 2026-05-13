from distill_gym.sandbox.builders.git_repository import GitRepositorySandboxBuilder
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.config.schema import (
    SandboxConfig, SandboxVolume, SandboxNetwork, SandboxNetworkMode, SandboxEngine,
)


class TestGitRepositorySandboxBuilder:
    def test_build_basic(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            ref="main",
            image="python:3.12",
            workdir="/workspace/repo",
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert isinstance(spec, SandboxSpec)
        assert spec.image == "python:3.12"
        assert spec.workdir == "/workspace/repo"
        assert "distill-gym" in spec.labels

    def test_build_with_cache_volume(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            image="python:3.12",
            volumes=[
                SandboxVolume(type="cache", name="pip", target="/root/.cache/pip"),
            ],
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert len(spec.volumes) == 1
        assert spec.volumes[0]["target"] == "/root/.cache/pip"

    def test_build_env(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            image="python:3.12",
            env={"MY_VAR": "value"},
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert spec.env.get("MY_VAR") == "value"

    def test_build_network_mode(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            image="python:3.12",
            network=SandboxNetwork(mode=SandboxNetworkMode.bridge),
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert spec.network_mode == "bridge"

    def test_proxy_only_uses_bridge_for_current_podman_backend(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            image="python:3.12",
            network=SandboxNetwork(mode=SandboxNetworkMode.proxy_only),
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert spec.network_mode == "bridge"
