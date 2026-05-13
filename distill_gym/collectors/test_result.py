import json


async def collect_test_result(sandbox, test_command: str) -> dict:
    if not test_command or not sandbox or not sandbox.container_id:
        return {"exit_code": None, "stdout": "", "stderr": "", "passed": None}

    code, stdout, stderr = await sandbox.exec(test_command)
    return {
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "passed": code == 0,
    }
