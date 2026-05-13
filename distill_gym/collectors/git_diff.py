async def collect_git_diff(sandbox) -> str:
    if not sandbox or not sandbox.container_id:
        return ""
    code, stdout, stderr = await sandbox.exec("git diff HEAD 2>/dev/null || true")
    return stdout if code == 0 else ""


async def collect_changed_files(sandbox) -> list[str]:
    if not sandbox or not sandbox.container_id:
        return []
    code, stdout, stderr = await sandbox.exec("git status --porcelain 2>/dev/null || true")
    if code != 0:
        return []
    files = []
    for line in stdout.splitlines():
        if len(line) > 3:
            files.append(line[3:].strip())
    return files
