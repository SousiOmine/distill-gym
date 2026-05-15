import pytest
import tempfile
from pathlib import Path
from distill_gym.config.schema import Config


@pytest.fixture
def sample_config_yaml():
    return """
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


@pytest.fixture
def sample_config(sample_config_yaml) -> Config:
    import yaml
    data = yaml.safe_load(sample_config_yaml)
    return Config.model_validate(data)
