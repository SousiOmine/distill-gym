from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from enum import Enum


class CleanupPolicy(str, Enum):
    always = "always"
    never = "never"
    on_failure = "on_failure"


class SandboxNetworkMode(str, Enum):
    proxy_only = "proxy_only"
    isolated = "isolated"
    bridge = "bridge"
    host = "host"


class SandboxEngine(str, Enum):
    podman = "podman"
    docker = "docker"


class SandboxVolume(BaseModel):
    type: Literal["bind", "volume", "cache"] = "bind"
    source: Optional[str] = None
    name: Optional[str] = None
    target: str


class SandboxNetwork(BaseModel):
    mode: SandboxNetworkMode = SandboxNetworkMode.proxy_only


class BuildStep(BaseModel):
    type: Literal[
        "git_clone", "run", "pip_install", "apt_get",
        "npm_install", "copy_file", "env_set", "mkdir",
    ]
    args: dict[str, Any] = Field(default_factory=dict)


class SandboxConfig(BaseModel):
    type: str = "git_repository"
    engine: SandboxEngine = SandboxEngine.podman
    repo_url: str = ""
    ref: str = "main"
    use_git_cache: bool = True
    image: str = "docker.io/library/python:3.12-bookworm"
    workdir: str = "/workspace/repo"
    setup: list[str] = Field(default_factory=list)
    steps: list[BuildStep] = Field(default_factory=list)
    volumes: list[SandboxVolume] = Field(default_factory=list)
    network: SandboxNetwork = Field(default_factory=SandboxNetwork)
    env: dict[str, str] = Field(default_factory=dict)


class LoggingProxyConfig(BaseModel):
    listen_host: str = "127.0.0.1"
    listen_port: int = 5002
    capture_stream_chunks: bool = True
    capture_raw_request: bool = True
    capture_raw_response: bool = True
    normalize_reasoning: bool = True


class ProviderConfig(BaseModel):
    type: str = "openai_compatible"
    name: str = "default"
    base_url: str = "https://api.deepinfra.com/v1/openai"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "example-model"
    extra_body: dict = Field(default_factory=dict)


class HarnessInstallConfig(BaseModel):
    strategy: Literal["command", "pip", "npm"] = "command"
    commands: list[str] = Field(default_factory=list)


class HarnessRunConfig(BaseModel):
    command: str = "{harness} run {task.prompt}"


class HarnessCompletionConfig(BaseModel):
    success_exit_codes: list[int] = Field(default_factory=lambda: [0])
    max_idle_seconds: int = 120
    success_patterns: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)


class HarnessConfig(BaseModel):
    type: str = "generic_cli"
    install: HarnessInstallConfig = Field(default_factory=HarnessInstallConfig)
    run: HarnessRunConfig = Field(default_factory=HarnessRunConfig)
    completion: HarnessCompletionConfig = Field(default_factory=HarnessCompletionConfig)


class TaskItem(BaseModel):
    id: str
    title: str = ""
    prompt: str
    test_command: Optional[str] = None


class TaskGenPrompt(BaseModel):
    id: str
    title: str = ""
    prompt: str


class TaskGenConfig(BaseModel):
    type: str = "repo_auto"
    prompt_template: str = "repo_bugfix_and_test_tasks"
    tasks: list[TaskItem] = Field(default_factory=list)
    prompts: list[TaskGenPrompt] = Field(default_factory=list)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    output_file: str = ".distill-gym/taskgen/tasks.json"
    max_rounds: int = Field(default=5, ge=1)
    batch_size: int = Field(default=3, ge=1)


class ArtifactConfig(BaseModel):
    collect: list[str] = Field(default_factory=lambda: ["git_diff", "stdout", "stderr", "test_results", "files_changed"])


class ExportFormats(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["openai_messages_jsonl"])
    include_reasoning_content: bool = True
    include_tool_results: bool = True
    include_failed_runs: bool = False


class RunConfig(BaseModel):
    name: str = "default-run"
    task_count: int = 3
    concurrency: int = 1
    timeout_minutes_per_task: int = 30
    cleanup: CleanupPolicy = CleanupPolicy.always


class Config(BaseModel):
    version: int = 1
    run: RunConfig = Field(default_factory=RunConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    logging_proxy: LoggingProxyConfig = Field(default_factory=LoggingProxyConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    taskgen: TaskGenConfig = Field(default_factory=TaskGenConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    export: ExportFormats = Field(default_factory=ExportFormats)
