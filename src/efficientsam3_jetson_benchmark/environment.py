"""Describe when and on what software a run happened, without host names or paths."""

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git(repository: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_commit(repository: Path) -> str | None:
    """Return the checked-out commit of a git repository, or None outside one."""
    return _git(repository, "rev-parse", "HEAD")


def git_is_dirty(repository: Path) -> bool | None:
    """Return whether tracked files have uncommitted changes, or None outside git."""
    status = _git(repository, "status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def describe(repository: Path, upstream: Path) -> dict[str, Any]:
    """Return the run time, platform, Python version, and both git commits."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "benchmark_commit": git_commit(repository),
        "benchmark_dirty": git_is_dirty(repository),
        "upstream_commit": git_commit(upstream),
    }
