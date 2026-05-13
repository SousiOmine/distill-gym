from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.builders.git_repository import GitRepositorySandboxBuilder
from distill_gym.sandbox.manager import SandboxManager
from distill_gym.config.schema import SandboxConfig


class TestCleanupLabels:
    def test_container_label(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.labels.get("distill-gym") == "true"

    def test_builder_labels(self):
        config = SandboxConfig(
            repo_url="https://github.com/example/test.git",
            image="python:3.12",
        )
        builder = GitRepositorySandboxBuilder()
        spec = builder.build(config)
        assert spec.labels["distill-gym"] == "true"
        assert "distill-gym-repo" in spec.labels

    def test_cleanup_label_generation(self):
        manager = SandboxManager()
        label = "distill-gym=true"
        assert "distill-gym" in label
        assert "true" in label
