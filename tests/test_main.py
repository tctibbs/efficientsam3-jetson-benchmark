import json
from pathlib import Path
from typing import Any

import pytest

from efficientsam3_jetson_benchmark import __main__ as cli
from efficientsam3_jetson_benchmark import (
    accuracy,
    download,
    examples,
    figures,
    latency,
    saco_gold,
)


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


def test_accuracy_rejects_full_models() -> None:
    # Full models are not named efficient_sam3_*, so eval_coco.py would skip them.
    with pytest.raises(SystemExit):
        cli.main(
            [
                "accuracy",
                "coco",
                "--checkpoint",
                "efficientsam3_tinyvit",
                "--first",
                "1",
            ]
        )


def test_latency_pytorch_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = []
    results = tmp_path / "result.json"
    results.write_text(json.dumps({"end_to_end_ms": {"p50_ms": 61.25, "fps": 15.5}}))

    def fake_run(run: latency.LatencyRun, results_dir: Path) -> Path:
        runs.append(run)
        return results

    monkeypatch.setattr(latency, "run_latency", fake_run)

    cli.main(
        [
            "latency",
            "pytorch",
            "--model",
            "efficientsam3_tinyvit",
            "--precision",
            "fp16",
        ]
    )

    run = runs[0]
    assert (run.model, run.precision, run.prompt) == (
        "efficientsam3_tinyvit",
        "fp16",
        "person",
    )
    assert (run.warmup, run.iterations) == (50, 300)
    assert run.image.name == "000000397133.jpg"
    assert capsys.readouterr().out.strip() == (
        f"end to end p50 61.2 ms, 15.5 FPS from the mean: {results}"
    )


def test_download_saco_gold_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(download, "download_saco_gold", lambda root: root / "all")

    cli.main(["download", "saco-gold", "--root", str(tmp_path)])

    assert capsys.readouterr().out.strip() == str(tmp_path / "all")


def test_accuracy_saco_gold_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = []
    results = tmp_path / "result.json"
    average = {"cgF1": 0.0512, "IL_MCC": 0.214, "positive_micro_F1": 0.2391}
    results.write_text(json.dumps({"average": average}))

    def fake_run(run: saco_gold.SacoGoldRun, results_dir: Path) -> Path:
        runs.append(run)
        return results

    monkeypatch.setattr(saco_gold, "run_saco_gold", fake_run)

    cli.main(["accuracy", "saco-gold", "--model", "efficientsam3_repvit"])

    run = runs[0]
    assert (run.model, run.precision, run.images_per_subset) == (
        "efficientsam3_repvit",
        "bf16",
        100,
    )
    assert run.subsets == tuple(saco_gold.SUBSETS)
    assert capsys.readouterr().out.strip() == (
        f"cgF1 5.12, IL_MCC 0.21, pmF1 23.91 averaged over 7 domains: {results}"
    )


def test_accuracy_saco_gold_takes_chosen_subsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = []
    results = tmp_path / "result.json"
    average = {"cgF1": 0.1, "IL_MCC": 0.3, "positive_micro_F1": 0.3}
    results.write_text(json.dumps({"average": average}))

    def fake_run(run: saco_gold.SacoGoldRun, results_dir: Path) -> Path:
        runs.append(run)
        return results

    monkeypatch.setattr(saco_gold, "run_saco_gold", fake_run)

    cli.main(
        [
            "accuracy",
            "saco-gold",
            "--model",
            "efficientsam3_tinyvit",
            "--subsets",
            "crowded",
            "sa1b",
            "--images-per-subset",
            "3",
        ]
    )

    assert runs[0].subsets == ("crowded", "sa1b")
    assert runs[0].images_per_subset == 3


@pytest.mark.parametrize(
    ("argv", "results"),
    [
        (
            [
                "accuracy",
                "coco",
                "--checkpoint",
                "efficient_sam3_tinyvit_s",
                "--first",
                "1",
            ],
            "results/accuracy/coco",
        ),
        (
            ["accuracy", "saco-gold", "--model", "efficientsam3_tinyvit"],
            "results/accuracy/saco-gold",
        ),
        (
            ["latency", "pytorch", "--model", "efficientsam3_tinyvit"],
            "results/latency/pytorch",
        ),
    ],
)
def test_each_benchmark_writes_to_its_own_folder(argv: list[str], results: str) -> None:
    assert cli.build_parser().parse_args(argv).results == Path(results)


def test_figures_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        figures, "write_figures", lambda results: [results / "speed.svg"]
    )

    cli.main(["figures", "--results", str(tmp_path)])

    assert capsys.readouterr().out.strip() == str(tmp_path / "speed.svg")


def test_examples_coco_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = []
    results = tmp_path / "result.json"
    outcomes = {"found": 3, "partial": 1, "missed": 2}
    results.write_text(json.dumps({"summary": {"outcomes": outcomes}}))

    def fake_run(run: examples.ExamplesRun, results_dir: Path) -> Path:
        runs.append((run, results_dir))
        return results

    monkeypatch.setattr(examples, "run_examples", fake_run)

    cli.main(["examples", "coco"])

    run, results_dir = runs[0]
    assert (run.model, run.precision, run.images) == (
        "efficientsam3_tinyvit",
        "bf16",
        300,
    )
    assert results_dir == Path("results/examples/coco")
    assert run.pins == (("large", 397133, "person"),)
    assert capsys.readouterr().out.strip() == (
        f"found 3, partial 1, missed 2: {results}"
    )


def test_examples_gallery_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    photo = {
        "image_id": 7,
        "flickr_url": "http://x/7.jpg",
        "license": "Attribution License",
    }
    pinned = []

    def fake_redraw(
        results: Path, coco_root: Path, pins: tuple[tuple[str, int, str], ...]
    ) -> list[dict[str, Any]]:
        pinned.append(pins)
        return [photo]

    monkeypatch.setattr(examples, "redraw_gallery", fake_redraw)

    cli.main(
        ["examples", "gallery", "--results", str(tmp_path), "--pins", "mixup:5:hot dog"]
    )

    assert pinned == [(("mixup", 5, "hot dog"),)]
    assert capsys.readouterr().out.strip() == "7 http://x/7.jpg (Attribution License)"


def test_gallery_pins_must_name_a_row(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["examples", "gallery", "--pins", "best:5:cat"])

    assert "row must be one of" in capsys.readouterr().err
