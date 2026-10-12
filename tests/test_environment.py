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
    assert description["memory_mb"]["total"] > 0
    assert "jetson" in description


L4T_RELEASE = (
    "# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64\n"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [(L4T_RELEASE, "36.4.4"), ("not a release file", None)],
)
def test_parse_l4t_release(text: str, expected: str | None) -> None:
    assert environment.parse_l4t_release(text) == expected


def test_nvpmodel_mode_parses_the_power_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NV Power Mode: MAXN_SUPER\n2\n"
        ),
    )

    assert environment.nvpmodel_mode() == "MAXN_SUPER"


def test_nvpmodel_mode_is_none_without_nvpmodel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvpmodel")

    monkeypatch.setattr(environment.subprocess, "run", missing)

    assert environment.nvpmodel_mode() is None


def test_jetson_details_reads_board_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "nv_tegra_release").write_text(L4T_RELEASE)
    (tmp_path / "proc" / "device-tree").mkdir(parents=True)
    (tmp_path / "proc" / "device-tree" / "model").write_text("NVIDIA Jetson Board\0")
    monkeypatch.setattr(environment, "nvpmodel_mode", lambda: "MAXN_SUPER")

    assert environment.jetson_details(tmp_path) == {
        "model": "NVIDIA Jetson Board",
        "l4t": "36.4.4",
        "power_mode": "MAXN_SUPER",
    }


def test_jetson_details_is_none_off_a_jetson(tmp_path: Path) -> None:
    assert environment.jetson_details(tmp_path) is None


def test_memory_mb_reports_total_and_available() -> None:
    memory = environment.memory_mb()

    assert 0 < memory["available"] <= memory["total"]
