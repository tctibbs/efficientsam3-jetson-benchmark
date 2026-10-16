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
                "settings": {"precision": precision},
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
                "checkpoint": {"name": f"efficient_sam3_{encoder}_{size}"},
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
