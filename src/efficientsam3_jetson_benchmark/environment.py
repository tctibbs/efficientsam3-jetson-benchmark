"""Describe when and on what software a run happened, without host names or paths."""

import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


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


def parse_l4t_release(text: str) -> str | None:
    """Return the L4T version from /etc/nv_tegra_release text, such as 36.4.4."""
    match = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", text)
    return f"{match[1]}.{match[2]}" if match else None


def nvpmodel_mode() -> str | None:
    """Return the Jetson power mode name, or None where nvpmodel is unavailable."""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"NV Power Mode: (\S+)", result.stdout)
    return match[1] if match else None


def jetson_details(root: Path = Path("/")) -> dict[str, str | None] | None:
    """Return the Jetson board, L4T release, and power mode, or None off a Jetson."""
    release_file = root / "etc" / "nv_tegra_release"
    if not release_file.is_file():
        return None
    model_file = root / "proc" / "device-tree" / "model"
    return {
        "model": model_file.read_text().rstrip("\0\n")
        if model_file.is_file()
        else None,
        "l4t": parse_l4t_release(release_file.read_text()),
        "power_mode": nvpmodel_mode(),
    }


def memory_mb() -> dict[str, int]:
    """Return total and currently available system memory in MiB."""
    memory = psutil.virtual_memory()
    return {"total": memory.total // 2**20, "available": memory.available // 2**20}


def describe(repository: Path, upstream: Path) -> dict[str, Any]:
    """Return the run time, platform, memory, Jetson details, and both git commits."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "memory_mb": memory_mb(),
        "jetson": jetson_details(),
        "benchmark_commit": git_commit(repository),
        "benchmark_dirty": git_is_dirty(repository),
        "upstream_commit": git_commit(upstream),
    }
