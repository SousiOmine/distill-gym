import pytest
from unittest.mock import patch, MagicMock
from distill_gym.sandbox.clients.podman import PodmanClient
from distill_gym.sandbox.base import SandboxSpec


class TestPodmanClient:
    def test_init_default_binary(self):
        client = PodmanClient()
        assert client.binary == "podman"

    def test_init_custom_binary(self):
        client = PodmanClient(binary="docker")
        assert client.binary == "docker"

    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        client = PodmanClient(binary="nonexistent-podman")
        with pytest.raises(RuntimeError, match="Podman binary not found"):
            await client.container_stop("test")

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_creates_correct_args(self, mock_run):
        mock_run.return_value = MagicMock(stdout="container-id\n")
        client = PodmanClient()
        spec = SandboxSpec(
            image="python:3.12",
            container_name="test-container",
            env={"KEY": "val"},
            volumes=[{"type": "bind", "source": "/host", "target": "/container"}],
            network_mode="bridge",
            hostname="test-host",
            labels={"distill-gym": "true"},
        )
        cid = await client.container_run(spec)
        assert cid == "container-id"
        args = mock_run.call_args[0][0]
        assert "run" in args
        assert "-d" in args
        assert "--name" in args
        assert "test-container" in args

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_with_network_name(self, mock_run):
        mock_run.return_value = MagicMock(stdout="cid\n")
        client = PodmanClient()
        spec = SandboxSpec(image="alpine", network_name="custom-net")
        await client.container_run(spec)
        args = mock_run.call_args[0][0]
        assert "--network" in args
        net_idx = args.index("--network")
        assert args[net_idx + 1] == "custom-net"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_stop(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        client = PodmanClient()
        await client.container_stop("abc123")
        args = mock_run.call_args[0][0]
        assert args == ["podman", "stop", "abc123"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_rm(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        client = PodmanClient()
        await client.container_rm("abc123")
        args = mock_run.call_args[0][0]
        assert args == ["podman", "rm", "-f", "abc123"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_exec(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        client = PodmanClient()
        code, out, err = await client.container_exec("cid", "echo hello")
        assert code == 0
        assert out == "ok"
        args = mock_run.call_args[0][0]
        assert args[:3] == ["podman", "exec", "cid"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_network_create(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        client = PodmanClient()
        await client.network_create("my-net")
        args = mock_run.call_args[0][0]
        assert "network" in args
        assert "create" in args
        assert "my-net" in args

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_containers(self, mock_run):
        mock_run.return_value = MagicMock(stdout='[{"Id": "abc", "Names": ["test"]}]\n', returncode=0)
        client = PodmanClient()
        containers = await client.list_containers("my-label")
        assert len(containers) == 1
        assert containers[0]["Id"] == "abc"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_containers_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        client = PodmanClient()
        containers = await client.list_containers("nonexistent")
        assert containers == []

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_volumes(self, mock_run):
        mock_run.return_value = MagicMock(stdout="vol1\nvol2\n", returncode=0)
        client = PodmanClient()
        volumes = await client.list_volumes("my-label")
        assert volumes == ["vol1", "vol2"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_cp_to(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        client = PodmanClient()
        await client.container_cp_to("cid", "/src", "/dst")
        args = mock_run.call_args[0][0]
        assert "cp" in args
        assert args[-2] == "/src"
        assert args[-1] == "cid:/dst"
