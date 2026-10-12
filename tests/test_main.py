import json
from pathlib import Path

import pytest

from efficientsam3_jetson_benchmark import __main__ as cli
from efficientsam3_jetson_benchmark import accuracy, download


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_unknown_checkpoint_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["download", "checkpoint", "not-a-model"])


def test_download_checkpoint_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_download(name: str, root: Path) -> Path:
        calls.append((name, root))
        return root / f"{name}.pt"

    monkeypatch.setattr(download, "download_checkpoint", fake_download)

    cli.main(["download", "checkpoint", "sam3", "--root", str(tmp_path)])

    assert calls == [("sam3", tmp_path)]
    assert capsys.readouterr().out.strip() == str(tmp_path / "sam3.pt")


def test_download_coco_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    annotations = tmp_path / "instances_val2017.json"
    monkeypatch.setattr(download, "download_coco_annotations", lambda root: annotations)
    monkeypatch.setattr(download, "first_image_ids", lambda path, count: [1, 2, 3])
    monkeypatch.setattr(
        download,
        "download_coco_images",
        lambda path, ids, root: [root / f"{image_id}.jpg" for image_id in ids],
    )

    cli.main(["download", "coco", "--first", "3", "--root", str(tmp_path)])

    assert capsys.readouterr().out.strip() == (
        f"3 images in {tmp_path / 'images' / 'val2017'}"
    )


@pytest.mark.parametrize("first", ["0", "-3"])
def test_first_must_be_positive(first: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["download", "coco", "--first", first])


def test_accuracy_rejects_the_teacher_checkpoint() -> None:
    # eval_coco.py only evaluates efficient_sam3 student checkpoints.
    with pytest.raises(SystemExit):
        cli.main(["accuracy", "coco", "--checkpoint", "sam3", "--first", "1"])


def test_accuracy_coco_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = []
    results = tmp_path / "result.json"
    results.write_text(json.dumps({"metrics": {"miou": 0.62831, "instances": 96}}))

    def fake_run(run: accuracy.CocoAccuracyRun, results_dir: Path) -> Path:
        runs.append((run, results_dir))
        return results

    monkeypatch.setattr(accuracy, "run_coco_accuracy", fake_run)

    cli.main(
        [
            "accuracy",
            "coco",
            "--checkpoint",
            "efficient_sam3_efficientvit_m",
            "--first",
            "10",
            "--results",
            str(tmp_path),
        ]
    )

    run, results_dir = runs[0]
    assert (run.checkpoint, run.first, results_dir) == (
        "efficient_sam3_efficientvit_m",
        10,
        tmp_path,
    )
    assert run.upstream == Path("third_party/efficientsam3")
    assert capsys.readouterr().out.strip() == (
        f"mIoU 0.6283 over 96 instances: {results}"
    )
