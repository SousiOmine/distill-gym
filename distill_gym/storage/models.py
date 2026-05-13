from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class RunRecord(BaseModel):
    id: str
    name: str
    config_yaml: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    harness_type: str = ""
    provider_name: str = ""
    model: str = ""
    sandbox_type: str = ""
    sandbox_engine: str = ""
    repo_url: str = ""
    commit_hash: str = ""
    success: Optional[bool] = None
    error_message: Optional[str] = None


class TaskRecord(BaseModel):
    id: str
    run_id: str
    title: str = ""
    prompt: str = ""
    status: str = "pending"
    exit_code: Optional[int] = None
    success: Optional[bool] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    test_command: Optional[str] = None
    tests_passed: Optional[bool] = None
    error_message: Optional[str] = None


class ArtifactRecord(BaseModel):
    id: str
    task_id: str
    run_id: str
    kind: str  # stdout, stderr, diff, test_result, raw_trace, metadata, changed_files
    path: str
    size: int = 0
