import pytest
from distill_gym.sandbox.clients import ContainerClient
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.builders.base import SandboxBuilder
from distill_gym.config.schema import SandboxConfig


class _TestableBuilder(SandboxBuilder):
    def build(self, config: SandboxConfig) -> SandboxSpec:
        return SandboxSpec(image=config.image)


class TestContainerClientInterface:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ContainerClient()

    def test_sandbox_spec_defaults(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.image == "python:3.12"
        assert spec.cmd == ["sleep", "infinity"]
        assert spec.labels == {"distill-gym": "true"}
        assert spec.hostname == "distill-gym"

    def test_sandbox_spec_custom_labels(self):
        spec = SandboxSpec(
            image="python:3.12",
            labels={"distill-gym": "true", "custom": "value"},
        )
        assert spec.labels["custom"] == "value"


class TestSandboxBuilderValidate:
    def test_builder_validate_empty_image(self):
        builder = _TestableBuilder()
        config = SandboxConfig(image="")
        errors = builder.validate(config)
        assert any("image" in e for e in errors)

    def test_builder_validate_valid_passes(self):
        builder = _TestableBuilder()
        config = SandboxConfig(image="python:3.12")
        errors = builder.validate(config)
        assert len(errors) == 0
