import pytest
from distill_gym.sandbox.runtime import SandboxRuntime
from distill_gym.sandbox.runtimes import create_runtime, PodmanSandboxRuntime
from distill_gym.sandbox.clients.podman import PodmanClient
from distill_gym.sandbox.clients.docker import DockerClient


class TestRuntimeFactory:
    def test_create_podman_runtime(self):
        runtime = create_runtime("podman")
        assert isinstance(runtime, PodmanSandboxRuntime)

    def test_create_podman_with_custom_client(self):
        client = PodmanClient()
        runtime = create_runtime("podman", client=client)
        assert runtime.client is client

    def test_create_docker_runtime(self):
        runtime = create_runtime("docker")
        assert isinstance(runtime, PodmanSandboxRuntime)
        from distill_gym.sandbox.clients.docker import DockerClient
        assert isinstance(runtime.client, DockerClient)

    def test_create_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown container engine"):
            create_runtime("nonexistent")

    def test_runtime_is_sandbox_runtime(self):
        runtime = create_runtime("podman")
        assert isinstance(runtime, SandboxRuntime)
