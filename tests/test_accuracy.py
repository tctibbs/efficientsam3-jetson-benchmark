import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from efficientsam3_jetson_benchmark import accuracy, download, environment
from efficientsam3_jetson_benchmark.accuracy import CocoAccuracyRun, IncompleteRunError
from efficientsam3_jetson_benchmark.download import HubFile, PinnedFile

CHECKPOINT = "efficient_sam3_efficientvit_m"


def _pin(content: bytes) -> PinnedFile:
    return PinnedFile(len(content), hashlib.sha256(content).hexdigest())


def _summary(ious: list[float], skips: int = 0) -> dict[str, Any]:
    return {
        "torch": "2.8.0",
        "numpy": "1.26.4",
        "device_type": "mps",
        "device_name": "Apple Chip",
        "autocast": {"enabled": True, "dtype": "float16"},
        "ious": ious,
        "postprocess_skips": skips,
        "wall_clock_s": 1.5,
    }


@pytest.fixture
def accuracy_run(
    tmp_path: Path, annotations_file: Path, monkeypatch: pytest.MonkeyPatch
) -> CocoAccuracyRun:
    """A run whose checkpoint and annotations exist and match patched pins."""
    weights = b"weights"
    hub_file = HubFile("owner/repo", "a" * 40, f"stage1/{CHECKPOINT}.pt", _pin(weights))
    monkeypatch.setitem(download.CHECKPOINTS, CHECKPOINT, hub_file)
    checkpoint = download.checkpoint_path(CHECKPOINT, tmp_path / "checkpoints")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(weights)

    coco_root = tmp_path / "coco"
    annotations = coco_root / download.COCO_ANNOTATIONS_MEMBER
    annotations.parent.mkdir(parents=True)
    shutil.copy(annotations_file, annotations)
    monkeypatch.setattr(
        download, "COCO_VAL_ANNOTATIONS", _pin(annotations.read_bytes())
    )
    monkeypatch.setattr(
        environment,
        "describe",
        lambda repository, upstream: {"timestamp": "2026-10-11T19:30:00+00:00"},
    )
    return CocoAccuracyRun(
        checkpoint=CHECKPOINT,
        first=2,
        checkpoints_root=tmp_path / "checkpoints",
        coco_root=coco_root,
        upstream=tmp_path / "upstream",
    )


def test_expected_instances_skips_crowd_annotations(annotations_file: Path) -> None:
    assert accuracy.expected_instances(annotations_file, [30, 10]) == 2


def test_verified_input_names_the_fixing_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run `fix it`"):
        accuracy.verified_input(tmp_path / "missing", PinnedFile(1, "0" * 64), "fix it")


@pytest.mark.parametrize(
    ("ious", "skips", "message"),
    [
        ([0.5], 0, "scored 1 of 2 instances"),
        ([0.5, 0.5], 3, "skipped mask post-processing 3 times"),
    ],
)
def test_check_complete_rejects_skipped_work(
    ious: list[float], skips: int, message: str
) -> None:
    with pytest.raises(IncompleteRunError, match=message):
        accuracy.check_complete(_summary(ious, skips), expected=2)


def test_check_complete_accepts_full_run() -> None:
    accuracy.check_complete(_summary([0.5, 0.5]), expected=2)


def test_run_eval_coco_links_only_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "efficient_sam3_efficientvit_m.pt"
    checkpoint.write_bytes(b"weights")
    run = CocoAccuracyRun(CHECKPOINT, 2, tmp_path, tmp_path / "coco", tmp_path / "up")
    seen = {}

    def fake_run(
        command: list[str], cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        models_dir = Path(command[command.index("--output_dir") + 1])
        seen["models"] = sorted(path.name for path in models_dir.iterdir())
        seen["target"] = (models_dir / checkpoint.name).resolve()
        seen["cwd"], seen["user_site"] = cwd, env["PYTHONNOUSERSITE"]
        Path(command[3]).write_text(json.dumps(_summary([0.5])))

    monkeypatch.setattr(accuracy.subprocess, "run", fake_run)

    summary = accuracy.run_eval_coco(checkpoint, run, tmp_path / "summary.json")

    assert summary["ious"] == [0.5]
    assert seen == {
        "models": [checkpoint.name],
        "target": checkpoint.resolve(),
        "cwd": run.upstream,
        "user_site": "1",
    }


def test_run_coco_accuracy_writes_record(
    tmp_path: Path, accuracy_run: CocoAccuracyRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        accuracy, "run_eval_coco", lambda *args: _summary([0.25, 0.3333334])
    )

    path = accuracy.run_coco_accuracy(accuracy_run, tmp_path / "results")

    record = json.loads(path.read_text())
    assert path.name == f"20261011T193000Z-coco-{CHECKPOINT}-first2-mps.json"
    assert record["metrics"] == {"miou": pytest.approx(0.29166670), "instances": 2}
    assert record["dataset"]["image_ids"] == [30, 10]
    assert record["checkpoint"]["repo_id"] == "owner/repo"
    assert record["runtime"]["autocast"] == {"enabled": True, "dtype": "float16"}
    assert record["ious"] == [0.25, 0.333333]


def test_run_coco_accuracy_requires_downloaded_checkpoint(
    tmp_path: Path, accuracy_run: CocoAccuracyRun
) -> None:
    download.checkpoint_path(CHECKPOINT, accuracy_run.checkpoints_root).unlink()

    with pytest.raises(FileNotFoundError, match="download checkpoint"):
        accuracy.run_coco_accuracy(accuracy_run, tmp_path / "results")


def test_run_coco_accuracy_writes_nothing_for_incomplete_run(
    tmp_path: Path, accuracy_run: CocoAccuracyRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accuracy, "run_eval_coco", lambda *args: _summary([0.5]))

    with pytest.raises(IncompleteRunError):
        accuracy.run_coco_accuracy(accuracy_run, tmp_path / "results")
    assert not (tmp_path / "results").exists()
