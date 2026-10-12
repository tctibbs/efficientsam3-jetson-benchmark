import platform
import subprocess
from pathlib import Path

import pytest

from efficientsam3_jetson_benchmark import environment


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    """A git repository with one committed file."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("one")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-q",
        "-m",
        "init",
    )
    return tmp_path


def test_git_commit_returns_head(git_repository: Path) -> None:
    assert environment.git_commit(git_repository) == _git(
        git_repository, "rev-parse", "HEAD"
    )


def test_git_is_dirty_ignores_untracked_files(git_repository: Path) -> None:
    (git_repository / "untracked.txt").write_text("new")

    assert environment.git_is_dirty(git_repository) is False


def test_git_is_dirty_sees_modified_tracked_files(git_repository: Path) -> None:
    (git_repository / "tracked.txt").write_text("two")

    assert environment.git_is_dirty(git_repository) is True


def test_git_helpers_return_none_outside_git(tmp_path: Path) -> None:
    assert environment.git_commit(tmp_path) is None
    assert environment.git_is_dirty(tmp_path) is None


def test_describe_leaves_out_the_host_name(
    git_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "node", lambda: "private-host-name")

    description = environment.describe(git_repository, git_repository)

    assert "private-host-name" not in str(description)
    assert description["benchmark_commit"] == description["upstream_commit"]
    assert description["benchmark_dirty"] is False
    assert description["timestamp"].endswith("+00:00")
