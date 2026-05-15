import pytest
from distill_gym.sandbox.builders.custom import CustomSandboxBuilder
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.config.schema import (
    SandboxConfig, BuildStep, SandboxVolume, SandboxNetwork, SandboxEngine,
)
from distill_gym.registry.builder_registry import BuilderRegistry


class TestCustomSandboxBuilder:
    def test_build_basic(self):
        config = SandboxConfig(
            type="custom",
            image="python:3.12",
            workdir="/workspace",
            steps=[
                BuildStep(type="mkdir", args={"path": "/workspace/data"}),
                BuildStep(type="run", args={"command": "echo hello", "timeout": 30}),
            ],
        )
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert isinstance(spec, SandboxSpec)
        assert spec.image == "python:3.12"
        assert spec.workdir == "/workspace"
        assert len(spec.steps) == 2
        assert spec.steps[0]["type"] == "mkdir"
        assert spec.steps[1]["type"] == "run"

    def test_build_with_git_clone_step(self):
        config = SandboxConfig(
            type="custom",
            image="python:3.12",
            steps=[
                BuildStep(
                    type="git_clone",
                    args={"repo": "https://github.com/example/test.git", "ref": "main", "dest": "/workspace/repo"},
                ),
            ],
        )
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert spec.steps[0]["type"] == "git_clone"
        assert spec.steps[0]["args"]["repo"] == "https://github.com/example/test.git"

    def test_build_with_env(self):
        config = SandboxConfig(
            type="custom",
            image="python:3.12",
            env={"MY_VAR": "value"},
        )
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert spec.env.get("MY_VAR") == "value"

    def test_build_with_volume(self):
        config = SandboxConfig(
            type="custom",
            image="python:3.12",
            volumes=[SandboxVolume(type="cache", name="pip", target="/root/.cache/pip")],
        )
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert len(spec.volumes) == 1
        assert spec.volumes[0]["target"] == "/root/.cache/pip"

    def test_registered(self):
        cls = BuilderRegistry.get("custom")
        assert cls is not None
        assert cls is CustomSandboxBuilder

    def test_build_network_mode(self):
        config = SandboxConfig(
            type="custom",
            image="python:3.12",
            network=SandboxNetwork(mode="bridge"),
        )
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert spec.network_mode == "bridge"

    def test_build_labels(self):
        config = SandboxConfig(type="custom", image="python:3.12")
        builder = CustomSandboxBuilder()
        spec = builder.build(config)
        assert spec.labels.get("distill-gym") == "true"
