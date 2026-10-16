import json
from pathlib import Path
from typing import Any

import pytest

from efficientsam3_jetson_benchmark import figures


def write(folder: Path, name: str, record: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(record))


def timing(milliseconds: float) -> dict[str, float]:
    return {"p50_ms": milliseconds, "fps": 1000 / milliseconds}


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """A results folder with a record of each kind for every model."""
    for model in figures.MODELS:
        for precision in figures.PRECISIONS:
            record = {
                "model": {"name": model},
                "settings": {"precision": precision, "iterations": 300},
                "end_to_end_ms": timing(400.0),
                "stages_ms": {step: timing(80.0) for step in figures.STEPS},
            }
            write(tmp_path / "latency" / "pytorch", f"{model}-{precision}", record)
        record = {
            "model": {"name": model},
            "settings": {"images_per_subset": 100},
            "average": {"cgF1": 0.1},
            "subsets": {
                domain: {"metrics": {"cgF1": 0.1}} for domain in figures.DOMAINS
            },
        }
        write(tmp_path / "accuracy" / "saco-gold", model, record)
    for encoder in figures.ENCODERS:
        for size in figures.SIZES:
            record = {
                "checkpoint": {
                    "name": f"efficient_sam3_{encoder}_{size}",
                    "revision": figures.PARAMETERS_REVISION,
                },
                "dataset": {"first": 5000},
                "metrics": {"miou": 0.65},
            }
            write(tmp_path / "accuracy" / "coco", f"{encoder}-{size}", record)
    return tmp_path


def test_write_figures_draws_one_svg_per_benchmark(results: Path) -> None:
    paths = figures.write_figures(results)

    assert paths == [
        results / "latency" / "pytorch" / "speed.svg",
        results / "accuracy" / "coco" / "miou.svg",
        results / "accuracy" / "saco-gold" / "cgf1.svg",
    ]
    assert all(path.read_text().startswith("<?xml") for path in paths)


def test_redrawing_changes_nothing(results: Path) -> None:
    first = [path.read_bytes() for path in figures.write_figures(results)]
    second = [path.read_bytes() for path in figures.write_figures(results)]

    assert first == second
    assert all(b"<dc:date>" not in svg for svg in first)


def test_benchmarks_without_results_are_skipped(tmp_path: Path) -> None:
    assert figures.write_figures(tmp_path) == []


def test_largest_keeps_the_biggest_sample_then_the_newest() -> None:
    records = [
        {"name": "a", "size": 300, "order": 1},
        {"name": "a", "size": 10, "order": 2},
        {"name": "b", "size": 10, "order": 3},
        {"name": "b", "size": 10, "order": 4},
    ]

    chosen = figures.largest(
        records, lambda record: record["name"], lambda record: record["size"]
    )

    assert {name: record["order"] for name, record in chosen.items()} == {
        "a": 1,
        "b": 4,
    }


def test_incomplete_records_are_left_out(results: Path) -> None:
    (results / "latency" / "pytorch" / "efficientsam3_tinyvit-bf16.json").unlink()
    partial = {
        "model": {"name": "efficientsam3_tinyvit"},
        "settings": {"images_per_subset": 100},
        "average": {"cgF1": 0.1},
        "subsets": {"crowded": {"metrics": {"cgF1": 0.1}}},
    }
    write(results / "accuracy" / "saco-gold", "zz-partial", partial)
    uncounted = {
        "checkpoint": {
            "name": "efficient_sam3_tinyvit_xl",
            "revision": figures.PARAMETERS_REVISION,
        },
        "dataset": {"first": 5000},
        "metrics": {"miou": 0.7},
    }
    write(results / "accuracy" / "coco", "zz-uncounted", uncounted)

    assert len(figures.write_figures(results)) == 3


def test_benchmarks_with_nothing_to_draw_are_skipped(tmp_path: Path) -> None:
    speed = {
        "model": {"name": "efficientsam3_tinyvit"},
        "settings": {"precision": "fp16", "iterations": 300},
    }
    write(tmp_path / "latency" / "pytorch", "fp16-only", speed)
    coco = {
        "checkpoint": {"name": "efficient_sam3_tinyvit_m", "revision": "b" * 40},
        "dataset": {"first": 5000},
        "metrics": {"miou": 0.7},
    }
    write(tmp_path / "accuracy" / "coco", "other-revision", coco)

    assert figures.write_figures(tmp_path) == []
