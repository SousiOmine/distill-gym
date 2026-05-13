async def collect_git_diff(sandbox) -> str:
    if not sandbox or not sandbox.container_id:
        return ""
    code, stdout, stderr = await sandbox.exec("git diff HEAD 2>/dev/null || true")
    return stdout if code == 0 else ""
