from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SandboxSpec:
    image: str
    container_name: str = ""
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    volumes: list[dict] = field(default_factory=list)
    network_mode: str = "bridge"
    cmd: list[str] = field(default_factory=lambda: ["sleep", "infinity"])
    labels: dict[str, str] = field(default_factory=lambda: {"distill-gym": "true"})
    hostname: str = "distill-gym"
    source_type: str = ""
    source_url: str = ""
    source_ref: str = ""
