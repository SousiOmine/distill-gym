from distill_gym.sandbox.runtime import SandboxRuntime
from distill_gym.sandbox.clients import ContainerClient
from distill_gym.sandbox.clients.podman import PodmanClient
from distill_gym.sandbox.runtimes.podman import PodmanSandboxRuntime
from distill_gym.platform.detection import detect


def create_runtime(
    engine: str = "podman",
    client: ContainerClient | None = None,
) -> SandboxRuntime:
    if engine == "podman":
        return PodmanSandboxRuntime(client or PodmanClient())
    elif engine == "docker":
        from distill_gym.sandbox.clients.docker import DockerClient
        return PodmanSandboxRuntime(client or DockerClient())
    else:
        raise ValueError(f"Unknown container engine: {engine}")


def auto_detect_runtime() -> SandboxRuntime:
    info = detect()
    if info.has_podman:
        return PodmanSandboxRuntime(PodmanClient())
    if info.has_docker:
        from distill_gym.sandbox.clients.docker import DockerClient
        from distill_gym.sandbox.runtimes.podman import PodmanSandboxRuntime
        return PodmanSandboxRuntime(DockerClient())
    raise RuntimeError(
        "No container runtime found. Install Podman (recommended) or Docker."
    )


__all__ = [
    "SandboxRuntime",
    "ContainerClient",
    "PodmanSandboxRuntime",
    "create_runtime",
    "auto_detect_runtime",
]
