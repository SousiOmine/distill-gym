import json
import subprocess

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.clients import ContainerClient
from distill_gym.sandbox.clients.docker import DockerClient
from distill_gym.sandbox.runtimes.docker import DockerSandboxRuntime
from distill_gym.sandbox.runtimes.podman import PodmanSandboxRuntime


def _make_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], stdout=stdout, stderr=stderr, returncode=returncode)


class TestDockerClient:
    def test_init_default_binary(self):
        client = DockerClient()
        assert client.binary == "docker"

    def test_init_custom_binary(self):
        client = DockerClient(binary="docker-custom")
        assert client.binary == "docker-custom"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_creates_correct_args(self, mock_run):
        mock_run.return_value = _make_completed(stdout="container-id\n")
        client = DockerClient()
        spec = SandboxSpec(
            image="python:3.12",
            container_name="test-container",
            env={"KEY": "val", "FOO": "bar"},
            volumes=[
                {"type": "bind", "source": "/host/path", "target": "/container/path"},
                {"type": "volume", "source": "myvol", "target": "/data"},
            ],
            network_mode="custom-net",
            hostname="test-host",
            labels={"distill-gym": "true", "tier": "test"},
            cmd=["python", "-c", "print(1)"],
        )
        cid = await client.container_run(spec)
        assert cid == "container-id"
        assert len(mock_run.call_args_list) == 2

        args0 = mock_run.call_args_list[0].args[0]
        assert args0 == ["docker", "rm", "-f", "test-container"]

        args1 = mock_run.call_args_list[1].args[0]
        assert args1[:3] == ["docker", "run", "-d"]
        assert "--name" in args1
        assert args1[args1.index("--name") + 1] == "test-container"
        assert args1.count("--label") == 2
        label_indices = [i for i, a in enumerate(args1) if a == "--label"]
        assert args1[label_indices[0] + 1] == "distill-gym=true"
        assert args1[label_indices[1] + 1] == "tier=test"
        assert args1.count("-e") == 2
        assert "-v" in args1
        assert args1.count("-v") == 2
        assert "--network" in args1
        assert args1[args1.index("--network") + 1] == "custom-net"
        assert "--hostname" in args1
        assert args1[args1.index("--hostname") + 1] == "test-host"
        assert "python:3.12" in args1
        assert args1[-3:] == ["python", "-c", "print(1)"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_default_args(self, mock_run):
        mock_run.return_value = _make_completed(stdout="cid\n")
        client = DockerClient()
        spec = SandboxSpec(image="alpine")
        await client.container_run(spec)
        args1 = mock_run.call_args_list[1].args[0]
        assert "--label" in args1
        assert args1[args1.index("--label") + 1] == "distill-gym=true"
        assert "-e" not in args1
        assert "-v" not in args1
        assert "--network" in args1
        assert args1[args1.index("--network") + 1] == "bridge"
        assert "--hostname" in args1
        assert args1[args1.index("--hostname") + 1] == "distill-gym"
        assert args1[-2:] == ["sleep", "infinity"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_called_process_error(self, mock_run):
        mock_run.side_effect = [
            _make_completed(),
            subprocess.CalledProcessError(1, [], stderr="Container error"),
        ]
        client = DockerClient()
        spec = SandboxSpec(image="python:3.12", container_name="test")
        with pytest.raises(RuntimeError, match="Failed to start container 'test': Container error"):
            await client.container_run(spec)

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_called_process_error_empty_stderr(self, mock_run):
        mock_run.side_effect = [
            _make_completed(),
            subprocess.CalledProcessError(2, [], stderr=""),
        ]
        client = DockerClient()
        spec = SandboxSpec(image="python:3.12", container_name="test")
        with pytest.raises(RuntimeError, match="Failed to start container 'test': exit code 2"):
            await client.container_run(spec)

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_run_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        client = DockerClient()
        spec = SandboxSpec(image="python:3.12")
        with pytest.raises(RuntimeError, match="Docker binary not found"):
            await client.container_run(spec)

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_stop(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.container_stop("abc123")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "stop", "abc123"]
        assert mock_run.call_args[1]["check"] is False

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_rm(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.container_rm("abc123")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "rm", "-f", "abc123"]
        assert mock_run.call_args[1]["check"] is False

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_exec(self, mock_run):
        mock_run.return_value = _make_completed(stdout="hello", returncode=0)
        client = DockerClient()
        code, out, err = await client.container_exec("cid", "echo hello")
        assert code == 0
        assert out == "hello"
        args = mock_run.call_args[0][0]
        assert args == ["docker", "exec", "cid", "/bin/sh", "-c", "echo hello"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_exec_with_workdir(self, mock_run):
        mock_run.return_value = _make_completed(stdout="ok", returncode=0)
        client = DockerClient()
        code, out, err = await client.container_exec("cid", "pwd", workdir="/workspace")
        assert code == 0
        assert out == "ok"
        args = mock_run.call_args[0][0]
        assert "--workdir" in args
        assert args[args.index("--workdir") + 1] == "/workspace"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_exec_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=5)
        client = DockerClient()
        code, out, err = await client.container_exec("cid", "sleep 100", timeout=5)
        assert code == -1
        assert out == ""
        assert err == "timeout"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_cp_to(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.container_cp_to("cid", "/src", "/dst")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "/src", "cid:/dst"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_container_cp_from(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.container_cp_from("cid", "/src", "/dst")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "cid:/src", "/dst"]

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_containers(self, mock_run):
        mock_run.return_value = _make_completed(stdout='[{"Id": "abc", "Names": ["test"]}]\n')
        client = DockerClient()
        containers = await client.list_containers("my-label")
        assert len(containers) == 1
        assert containers[0]["Id"] == "abc"
        args = mock_run.call_args[0][0]
        assert "docker" in args
        assert "ps" in args
        assert "--filter" in args

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_containers_empty(self, mock_run):
        mock_run.return_value = _make_completed(stdout="")
        client = DockerClient()
        containers = await client.list_containers("nonexistent")
        assert containers == []

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_containers_json_decode_error(self, mock_run):
        mock_run.return_value = _make_completed(stdout="invalid json")
        client = DockerClient()
        containers = await client.list_containers("my-label")
        assert containers == []

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_volumes(self, mock_run):
        mock_run.return_value = _make_completed(stdout="vol1\nvol2\n")
        client = DockerClient()
        volumes = await client.list_volumes("my-label")
        assert volumes == ["vol1", "vol2"]
        args = mock_run.call_args[0][0]
        assert "volume" in args and "ls" in args

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_volumes_empty(self, mock_run):
        mock_run.return_value = _make_completed(stdout="")
        client = DockerClient()
        volumes = await client.list_volumes("my-label")
        assert volumes == []

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_volume_rm(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.volume_rm("myvol")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "volume", "rm", "-f", "myvol"]
        assert mock_run.call_args[1]["check"] is False

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_networks(self, mock_run):
        mock_run.return_value = _make_completed(stdout="net1\nnet2\n")
        client = DockerClient()
        networks = await client.list_networks("my-label")
        assert networks == ["net1", "net2"]
        args = mock_run.call_args[0][0]
        assert "network" in args and "ls" in args

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_list_networks_empty(self, mock_run):
        mock_run.return_value = _make_completed(stdout="")
        client = DockerClient()
        networks = await client.list_networks("my-label")
        assert networks == []

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_network_create(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.network_create("my-net", driver="overlay")
        args = mock_run.call_args[0][0]
        assert args[:-1] == ["docker", "network", "create", "--driver", "overlay"]
        assert args[-1] == "my-net"

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_network_rm(self, mock_run):
        mock_run.return_value = _make_completed()
        client = DockerClient()
        await client.network_rm("my-net")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "network", "rm", "my-net"]
        assert mock_run.call_args[1]["check"] is False


class TestDockerSandboxRuntime:
    async def test_init_default_client(self):
        runtime = DockerSandboxRuntime()
        assert isinstance(runtime.client, DockerClient)

    async def test_init_custom_client(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        assert runtime.client is mock_client

    @pytest.mark.asyncio
    async def test_start(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        spec = SandboxSpec(image="python:3.12")
        mock_client.container_run.return_value = "container-id"
        result = await runtime.start(spec)
        assert result == "container-id"
        mock_client.container_run.assert_awaited_once_with(spec)

    @pytest.mark.asyncio
    async def test_exec(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        mock_client.container_exec.return_value = (0, "out", "")
        rc, out, err = await runtime.exec("cid", "echo hi", timeout=10, workdir="/tmp")
        assert rc == 0
        assert out == "out"
        mock_client.container_exec.assert_awaited_once_with("cid", "echo hi", timeout=10, workdir="/tmp")

    @pytest.mark.asyncio
    async def test_exec_default_timeout(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        mock_client.container_exec.return_value = (0, "", "")
        await runtime.exec("cid", "ls")
        mock_client.container_exec.assert_awaited_once_with("cid", "ls", timeout=300, workdir=None)

    @pytest.mark.asyncio
    async def test_copy_to(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        await runtime.copy_to("cid", "/src", "/dst")
        mock_client.container_cp_to.assert_awaited_once_with("cid", "/src", "/dst")

    @pytest.mark.asyncio
    async def test_copy_from(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        await runtime.copy_from("cid", "/src", "/dst")
        mock_client.container_cp_from.assert_awaited_once_with("cid", "/src", "/dst")

    @pytest.mark.asyncio
    async def test_stop(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        await runtime.stop("cid")
        mock_client.container_stop.assert_awaited_once_with("cid")

    @pytest.mark.asyncio
    async def test_remove(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = DockerSandboxRuntime(client=mock_client)
        await runtime.remove("cid")
        mock_client.container_rm.assert_awaited_once_with("cid")

    @pytest.mark.asyncio
    async def test_cleanup_resources(self):
        mock_client = AsyncMock(spec=ContainerClient)
        mock_client.list_containers.return_value = [{"Id": "c1"}, {"Id": "c2"}]
        mock_client.list_volumes.return_value = ["v1"]
        mock_client.list_networks.return_value = ["n1", "n2"]
        runtime = DockerSandboxRuntime(client=mock_client)
        result = await runtime.cleanup_resources()
        assert result == {"containers": 2, "volumes": 1, "networks": 2}
        mock_client.list_containers.assert_awaited_once()
        mock_client.list_volumes.assert_awaited_once()
        mock_client.list_networks.assert_awaited_once()
        mock_client.container_stop.assert_has_calls([call("c1"), call("c2")])
        mock_client.container_rm.assert_has_calls([call("c1"), call("c2")])
        mock_client.volume_rm.assert_awaited_once_with("v1")
        mock_client.network_rm.assert_has_calls([call("n1"), call("n2")])

    @pytest.mark.asyncio
    async def test_cleanup_resources_stop_before_rm(self):
        mock_client = AsyncMock(spec=ContainerClient)
        mock_client.list_containers.return_value = [{"Id": "c1"}]
        mock_client.list_volumes.return_value = []
        mock_client.list_networks.return_value = []
        runtime = DockerSandboxRuntime(client=mock_client)
        await runtime.cleanup_resources()
        expected_calls = [call("c1"), call("c1")]
        mock_client.container_stop.assert_called_once_with("c1")
        mock_client.container_rm.assert_called_once_with("c1")
        assert mock_client.mock_calls.index(call.container_stop("c1")) < mock_client.mock_calls.index(call.container_rm("c1"))

    @pytest.mark.asyncio
    async def test_cleanup_resources_empty(self):
        mock_client = AsyncMock(spec=ContainerClient)
        mock_client.list_containers.return_value = []
        mock_client.list_volumes.return_value = []
        mock_client.list_networks.return_value = []
        runtime = DockerSandboxRuntime(client=mock_client)
        result = await runtime.cleanup_resources()
        assert result == {"containers": 0, "volumes": 0, "networks": 0}
        mock_client.container_stop.assert_not_called()
        mock_client.container_rm.assert_not_called()
        mock_client.volume_rm.assert_not_called()
        mock_client.network_rm.assert_not_called()


class TestPodmanSandboxRuntime:
    async def test_init_default_client(self):
        runtime = PodmanSandboxRuntime()
        from distill_gym.sandbox.clients.podman import PodmanClient
        assert isinstance(runtime.client, PodmanClient)

    async def test_init_custom_client(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        assert runtime.client is mock_client

    @pytest.mark.asyncio
    async def test_start(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        spec = SandboxSpec(image="python:3.12")
        mock_client.container_run.return_value = "cid"
        result = await runtime.start(spec)
        assert result == "cid"
        mock_client.container_run.assert_awaited_once_with(spec)

    @pytest.mark.asyncio
    async def test_exec(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        mock_client.container_exec.return_value = (42, "stdout", "stderr")
        rc, out, err = await runtime.exec("cid", "cmd", timeout=99, workdir="/x")
        assert rc == 42
        assert out == "stdout"
        assert err == "stderr"
        mock_client.container_exec.assert_awaited_once_with("cid", "cmd", timeout=99, workdir="/x")

    @pytest.mark.asyncio
    async def test_exec_default_timeout(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        await runtime.exec("cid", "ls")
        mock_client.container_exec.assert_awaited_once_with("cid", "ls", timeout=300, workdir=None)

    @pytest.mark.asyncio
    async def test_copy_to(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        await runtime.copy_to("cid", "/src", "/dst")
        mock_client.container_cp_to.assert_awaited_once_with("cid", "/src", "/dst")

    @pytest.mark.asyncio
    async def test_copy_from(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        await runtime.copy_from("cid", "/src", "/dst")
        mock_client.container_cp_from.assert_awaited_once_with("cid", "/src", "/dst")

    @pytest.mark.asyncio
    async def test_stop(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        await runtime.stop("cid")
        mock_client.container_stop.assert_awaited_once_with("cid")

    @pytest.mark.asyncio
    async def test_remove(self):
        mock_client = AsyncMock(spec=ContainerClient)
        runtime = PodmanSandboxRuntime(client=mock_client)
        await runtime.remove("cid")
        mock_client.container_rm.assert_awaited_once_with("cid")

    @pytest.mark.asyncio
    async def test_cleanup_resources(self):
        mock_client = AsyncMock(spec=ContainerClient)
        mock_client.list_containers.return_value = [{"Id": "c1"}]
        mock_client.list_volumes.return_value = ["v1", "v2"]
        mock_client.list_networks.return_value = ["n1"]
        runtime = PodmanSandboxRuntime(client=mock_client)
        result = await runtime.cleanup_resources()
        assert result == {"containers": 1, "volumes": 2, "networks": 1}
        mock_client.container_stop.assert_awaited_once_with("c1")
        mock_client.container_rm.assert_awaited_once_with("c1")
        mock_client.volume_rm.assert_has_calls([call("v1"), call("v2")])
        mock_client.network_rm.assert_awaited_once_with("n1")

    @pytest.mark.asyncio
    async def test_cleanup_resources_empty(self):
        mock_client = AsyncMock(spec=ContainerClient)
        mock_client.list_containers.return_value = []
        mock_client.list_volumes.return_value = []
        mock_client.list_networks.return_value = []
        runtime = PodmanSandboxRuntime(client=mock_client)
        result = await runtime.cleanup_resources()
        assert result == {"containers": 0, "volumes": 0, "networks": 0}
        mock_client.container_stop.assert_not_called()
        mock_client.container_rm.assert_not_called()
        mock_client.volume_rm.assert_not_called()
        mock_client.network_rm.assert_not_called()
