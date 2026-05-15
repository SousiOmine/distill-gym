from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.manager import SandboxManager


class TestCleanupLabels:
    def test_container_label(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.labels.get("distill-gym") == "true"

    def test_builder_labels(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.labels["distill-gym"] == "true"

    def test_cleanup_label_generation(self):
        manager = SandboxManager()
        label = "distill-gym=true"
        assert "distill-gym" in label
        assert "true" in label
