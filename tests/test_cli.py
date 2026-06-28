from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from distill_gym.cli.main import app

runner = CliRunner()

SAMPLE_YAML = """\
version: 1
run:
  name: test-run
  task_count: 1
  concurrency: 1
  cleanup: always
  repo_url: https://github.com/example/test.git
provider:
  type: openai_compatible
  name: test
  base_url: http://localhost:5002/v1
  api_key_env: TEST_API_KEY
  model: test-model
logging_proxy:
  listen_host: 127.0.0.1
  listen_port: 5002
sandbox:
  type: custom
  engine: podman
  image: python:3.12
  workdir: /workspace
  steps: []
harness:
  type: mock
taskgen:
  type: repo_auto
"""


@pytest.fixture(autouse=True)
def env_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTILL_GYM_CACHE_DIR", str(tmp_path))


# ---------- init ----------

@patch("distill_gym.cache.cache_store.ensure_dirs")
def test_init_creates_dirs(mock_ensure_dirs):
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Initialized" in result.stdout
    mock_ensure_dirs.assert_called_once()


# ---------- validate ----------

def test_validate_valid_config(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(SAMPLE_YAML)
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0
    assert "Config valid" in result.stdout


def test_validate_invalid_config(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("version: 1\nrun:\n  task_count: not_a_number\n")
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 1
    assert "Validation error" in result.stderr


def test_validate_nonexistent_file(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.yaml")])
    assert result.exit_code == 1


# ---------- run ----------

@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_success(mock_run_orch, tmp_path):
    mock_run_orch.return_value = "run-id-001"
    config_path = tmp_path / "test.yaml"
    config_path.write_text(SAMPLE_YAML)
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0
    assert "Run completed: run-id-001" in result.stdout
    mock_run_orch.assert_called_once()


@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_dry_run(mock_run_orch, tmp_path):
    yaml = SAMPLE_YAML.replace("harness:\n  type: mock", "harness:\n  type: opencode")
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml)
    result = runner.invoke(app, ["run", str(config_path), "--dry-run"])
    assert result.exit_code == 0
    mock_run_orch.assert_called_once()
    cfg = mock_run_orch.call_args[0][0]
    assert cfg.harness.type == "mock"


@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_exception(mock_run_orch, tmp_path):
    mock_run_orch.side_effect = RuntimeError("execution error")
    config_path = tmp_path / "test.yaml"
    config_path.write_text(SAMPLE_YAML)
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    assert "Run failed" in result.stderr


# ---------- run-repo ----------

@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_repo_success(mock_run_orch):
    mock_run_orch.return_value = "repo-run-id"
    result = runner.invoke(app, [
        "run-repo",
        "https://github.com/example/test.git",
        "--model", "test-model",
        "--provider-base-url", "http://localhost:5002/v1",
    ])
    assert result.exit_code == 0
    assert "Run completed: repo-run-id" in result.stdout
    mock_run_orch.assert_called_once()


@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_repo_exception(mock_run_orch):
    mock_run_orch.side_effect = RuntimeError("repo error")
    result = runner.invoke(app, [
        "run-repo",
        "https://github.com/example/test.git",
        "--model", "test-model",
        "--provider-base-url", "http://localhost:5002/v1",
    ])
    assert result.exit_code == 1
    assert "Run failed" in result.stderr


@patch("distill_gym.cli.main.run_orch", new_callable=AsyncMock)
def test_run_repo_builds_config(mock_run_orch):
    mock_run_orch.return_value = "cfg-test"
    result = runner.invoke(app, [
        "run-repo",
        "https://github.com/example/my-repo.git",
        "--model", "gpt-4",
        "--provider-base-url", "http://localhost:5002/v1/",
        "--api-key-env", "MY_KEY",
        "--ref", "develop",
        "--task-count", "5",
        "--harness", "codex",
    ])
    assert result.exit_code == 0
    mock_run_orch.assert_called_once()
    cfg = mock_run_orch.call_args[0][0]
    assert cfg.run.name == "codex-my-repo"
    assert cfg.run.task_count == 5
    assert cfg.run.repo_url == "https://github.com/example/my-repo.git"
    assert cfg.run.ref == "develop"
    assert cfg.provider.base_url == "http://localhost:5002/v1"
    assert cfg.provider.api_key_env == "MY_KEY"
    assert cfg.provider.model == "gpt-4"
    assert cfg.harness.type == "codex"
    assert cfg.sandbox.steps[0]["type"] == "git_clone"


# ---------- proxy ----------

@patch("distill_gym.cli.main.detect")
@patch("distill_gym.cli.main.proxy_listen_host_for_sandbox")
@patch("distill_gym.cli.main.uvicorn.run")
@patch("distill_gym.cli.main.create_proxy_app")
@patch("distill_gym.cli.main.load_config")
def test_proxy_starts(
    mock_load_config,
    mock_create_proxy_app,
    mock_uvicorn_run,
    mock_proxy_listen_host,
    mock_detect,
    tmp_path,
):
    from distill_gym.config.schema import Config
    cfg = Config()
    cfg.provider.base_url = "http://localhost:5002/v1"
    cfg.logging_proxy.listen_port = 5002
    mock_load_config.return_value = cfg
    mock_proxy_listen_host.return_value = "0.0.0.0"

    config_path = tmp_path / "proxy.yaml"
    config_path.write_text(SAMPLE_YAML)
    result = runner.invoke(app, ["proxy", str(config_path)])
    assert result.exit_code == 0
    assert "Starting proxy" in result.stdout
    mock_load_config.assert_called_once()
    mock_create_proxy_app.assert_called_once_with(
        cfg.provider,
        cfg.logging_proxy,
        None,
    )
    mock_uvicorn_run.assert_called_once()


@patch("distill_gym.cli.main.TraceRecorder")
@patch("distill_gym.cli.main.detect")
@patch("distill_gym.cli.main.proxy_listen_host_for_sandbox")
@patch("distill_gym.cli.main.uvicorn.run")
@patch("distill_gym.cli.main.create_proxy_app")
@patch("distill_gym.cli.main.load_config")
def test_proxy_with_run_id(
    mock_load_config,
    mock_create_proxy_app,
    mock_uvicorn_run,
    mock_proxy_listen_host,
    mock_detect,
    mock_trace_recorder,
    tmp_path,
):
    from distill_gym.config.schema import Config
    cfg = Config()
    cfg.provider.base_url = "http://localhost:5002/v1"
    cfg.logging_proxy.listen_port = 5002
    mock_load_config.return_value = cfg
    mock_proxy_listen_host.return_value = "0.0.0.0"

    config_path = tmp_path / "proxy.yaml"
    config_path.write_text(SAMPLE_YAML)
    result = runner.invoke(app, [
        "proxy", str(config_path),
        "--run-id", "run-with-recorder",
    ])
    assert result.exit_code == 0
    assert "Starting proxy" in result.stdout
    mock_trace_recorder.assert_called_once()


# ---------- merge ----------

@patch("distill_gym.exporters.merger.merge_runs_to_jsonl", new_callable=AsyncMock)
def test_merge_success(mock_merge, tmp_path):
    mock_merge.return_value = 5
    output = tmp_path / "merged.jsonl"
    result = runner.invoke(app, [
        "merge",
        "--run-ids", "run1,run2,run3",
        "--format", "openai-messages",
        "--output", str(output),
    ])
    assert result.exit_code == 0
    assert "Merged" in result.stdout
    mock_merge.assert_called_once()


def test_merge_unknown_format(tmp_path):
    output = tmp_path / "out.jsonl"
    result = runner.invoke(app, [
        "merge",
        "--run-ids", "run1",
        "--format", "unknown-format",
        "--output", str(output),
    ])
    assert result.exit_code == 1


# ---------- export ----------

@patch("distill_gym.cli.main.export_openai_messages_jsonl", new_callable=AsyncMock)
def test_export_success(mock_export, tmp_path):
    mock_export.return_value = 3
    output = tmp_path / "out.jsonl"
    result = runner.invoke(app, [
        "export",
        "--run-id", "test-run",
        "--format", "openai-messages",
        "--output", str(output),
    ])
    assert result.exit_code == 0
    assert "Exported" in result.stdout
    mock_export.assert_called_once()


def test_export_unknown_format(tmp_path):
    output = tmp_path / "out.jsonl"
    result = runner.invoke(app, [
        "export",
        "--run-id", "test-run",
        "--format", "unknown-format",
        "--output", str(output),
    ])
    assert result.exit_code == 1


# ---------- cleanup ----------

@patch("distill_gym.cli.main.cleanup_orch", new_callable=AsyncMock)
def test_cleanup_success(mock_cleanup):
    mock_cleanup.return_value = {"containers": 2, "volumes": 1, "networks": 0}
    result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "Cleaned up" in result.stdout
    assert "2 containers" in result.stdout
    assert "1 volumes" in result.stdout
    assert "0 networks" in result.stdout
    mock_cleanup.assert_called_once()


# ---------- web ----------

@patch("distill_gym.web.app.create_web_app")
@patch("distill_gym.cli.main.uvicorn.run")
def test_web_starts(mock_uvicorn_run, mock_create_web_app):
    mock_app = MagicMock()
    mock_create_web_app.return_value = mock_app
    result = runner.invoke(app, ["web", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0
    mock_create_web_app.assert_called_once()
    mock_uvicorn_run.assert_called_once_with(
        mock_app,
        host="0.0.0.0",
        port=9000,
        log_level="info",
    )
