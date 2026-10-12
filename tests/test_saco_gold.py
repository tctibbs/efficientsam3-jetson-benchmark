import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy
import pytest
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from efficientsam3_jetson_benchmark import (
    coco_runner,
    download,
    environment,
    latency,
    saco_gold,
)
from efficientsam3_jetson_benchmark.download import HubFile, PinnedFile
from efficientsam3_jetson_benchmark.saco_gold import SacoGoldRun

MODEL = "efficientsam3_tinyvit"
EVALUATOR = (
    Path(__file__).parents[1]
    / "third_party"
    / "efficientsam3"
    / "sam3"
    / "sam3"
    / "eval"
    / "cgf1_eval.py"
)
needs_upstream = pytest.mark.skipif(
    not EVALUATOR.is_file(), reason="upstream submodule is not checked out"
)


def square(columns: slice = slice(2, 6)) -> numpy.ndarray:
    """An 8x8 mask with rows 2 to 5 set in the given columns."""
    mask = numpy.zeros((8, 8), dtype=bool)
    mask[2:6, columns] = True
    return mask


def as_masks(mask: numpy.ndarray) -> torch.Tensor:
    """Shape one mask the way Sam3Processor returns them, N x 1 x H x W."""
    return torch.from_numpy(mask)[None, None]


def labels() -> dict[str, Any]:
    """Three pairs: a cat on a.jpg, no dog on a.jpg, and an unscored pair on b.jpg."""
    encoded = mask_utils.encode(numpy.asfortranarray(square().astype(numpy.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    pair = {"width": 8, "height": 8, "is_instance_exhaustive": 1}
    return {
        "images": [
            {**pair, "id": 1, "file_name": "a.jpg", "text_input": "cat"},
            {**pair, "id": 2, "file_name": "a.jpg", "text_input": "dog"},
            {
                **pair,
                "id": 3,
                "file_name": "b.jpg",
                "text_input": "cat",
                "is_instance_exhaustive": 0,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "area": 16,
                "bbox": [2, 2, 4, 4],
                "segmentation": encoded,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }


def load_evaluator(upstream: Path, name: str) -> ModuleType:
    """Load upstream's cgf1_eval.py alone, without importing the sam3 package."""
    spec = importlib.util.spec_from_file_location("cgf1_eval", EVALUATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    """Finds a square for "cat" and nothing for any other phrase."""

    device = torch.device("cpu")
    confidence_threshold = 0.5

    def __init__(self) -> None:
        self.encoded = 0
        self.resets = 0

    def set_image(self, image: Image.Image) -> dict[str, Any]:
        self.encoded += 1
        return {}

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        found = prompt == "cat"
        state["masks"] = as_masks(square()) if found else torch.zeros((0, 1, 8, 8))
        state["scores"] = torch.tensor([0.9] if found else [])
        return state

    def reset_all_prompts(self, state: dict[str, Any]) -> None:
        self.resets += 1
        del state["masks"], state["scores"]


def write_dataset(root: Path, subset: str) -> None:
    """Write one domain's labels for all annotators, and its two images."""
    for annotator in saco_gold.ANNOTATORS:
        path = saco_gold.annotation_path(root, subset, annotator)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels()))
    images = saco_gold.image_root(root, saco_gold.SUBSETS[subset])
    images.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(images / name)


@pytest.fixture
def gold_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SacoGoldRun:
    """A run whose checkpoint matches a patched pin."""
    weights = b"weights"
    pin = PinnedFile(len(weights), hashlib.sha256(weights).hexdigest())
    hub_file = HubFile("owner/repo", "a" * 40, f"full/{MODEL}.pt", pin)
    monkeypatch.setitem(download.CHECKPOINTS, MODEL, hub_file)
    checkpoint = download.checkpoint_path(MODEL, tmp_path / "checkpoints")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(weights)
    return SacoGoldRun(
        model=MODEL,
        precision="fp32",
        images_per_subset=5,
        subsets=("attributes",),
        saco_root=tmp_path / "saco-gold",
        checkpoints_root=tmp_path / "checkpoints",
        upstream=tmp_path / "upstream",
    )


def test_sample_pairs_takes_every_phrase_of_the_first_images() -> None:
    pairs = labels()["images"]

    assert [pair["id"] for pair in saco_gold.sample_pairs(pairs, 1)] == [1, 2]
    assert saco_gold.sample_pairs(pairs, 5) == pairs


def test_restrict_keeps_only_the_given_pairs() -> None:
    restricted = saco_gold.restrict(labels(), {2})

    assert [pair["id"] for pair in restricted["images"]] == [2]
    assert restricted["annotations"] == []
    assert restricted["categories"] == labels()["categories"]


def test_load_labels_keeps_the_sampled_pairs_for_every_annotator(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, "crowded")

    pairs, labels_by_annotator = saco_gold.load_labels(tmp_path, "crowded", 1)

    assert [pair["id"] for pair in pairs] == [1, 2]
    assert len(labels_by_annotator) == 3
    for dataset in labels_by_annotator:
        assert [pair["id"] for pair in dataset["images"]] == [1, 2]
        assert [label["image_id"] for label in dataset["annotations"]] == [1]


def test_to_results_round_trips_masks_as_rle() -> None:
    results = saco_gold.to_results(7, as_masks(square()), torch.tensor([0.8]))

    decoded = mask_utils.decode(results[0]["segmentation"])
    assert (decoded.astype(bool) == square()).all()
    assert (results[0]["image_id"], results[0]["category_id"]) == (7, 1)
    assert results[0]["score"] == pytest.approx(0.8)
    assert json.loads(json.dumps(results)) == results


def test_predict_encodes_each_image_once_and_resets_each_phrase(
    tmp_path: Path,
) -> None:
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    processor = FakeProcessor()

    results = saco_gold.predict(processor, labels()["images"], tmp_path, "fp32")

    assert (processor.encoded, processor.resets) == (2, 3)
    assert [result["image_id"] for result in results] == [1, 3]


@needs_upstream
@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (slice(2, 6), 1.0),
        # IoU 16/24 matches at IoU thresholds 0.50 to 0.65, 4 of the 10.
        (slice(2, 8), 0.4),
    ],
)
def test_score_with_upstreams_evaluator(
    monkeypatch: pytest.MonkeyPatch, columns: slice, expected: float
) -> None:
    monkeypatch.setattr(latency, "import_upstream", load_evaluator)
    results = saco_gold.to_results(1, as_masks(square(columns)), torch.tensor([0.9]))

    metrics, scored_pairs = saco_gold.score(Path("upstream"), [labels()] * 3, results)

    # The evaluator smooths its ratios, so a perfect score is 0.9998.
    assert scored_pairs == 2
    assert metrics["IL_MCC"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["positive_micro_F1"] == pytest.approx(expected, abs=1e-3)
    assert metrics["cgF1"] == pytest.approx(expected, abs=1e-3)


@needs_upstream
def test_run_saco_gold_writes_a_record(
    tmp_path: Path, gold_run: SacoGoldRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_dataset(gold_run.saco_root, "attributes")
    monkeypatch.setattr(
        environment,
        "describe",
        lambda repository, upstream: {"timestamp": "2026-10-12T09:00:00+00:00"},
    )
    monkeypatch.setattr(
        latency,
        "load_pipeline",
        lambda model_name, checkpoint, upstream: (None, FakeProcessor()),
    )
    monkeypatch.setattr(latency, "import_upstream", load_evaluator)
    monkeypatch.setattr(coco_runner, "device_name", lambda device_type: "CPU")

    path = saco_gold.run_saco_gold(gold_run, tmp_path / "results")

    record = json.loads(path.read_text())
    assert path.name == f"20261012T090000Z-saco-gold-{MODEL}-fp32-5images-cpu.json"
    subset = record["subsets"]["attributes"]
    assert (subset["images"], subset["pairs"], subset["scored_pairs"]) == (2, 3, 2)
    assert subset["predicted_masks"] == 2
    assert record["average"]["cgF1"] == pytest.approx(1.0, abs=1e-3)
    assert record["settings"] == {
        "precision": "fp32",
        "images_per_subset": 5,
        "confidence_threshold": 0.5,
    }
    assert record["dataset"]["sha256"] == download.SACO_GOLD_ARCHIVE.sha256
    assert record["model"]["text_context_length"] == 16


def test_run_saco_gold_requires_labels(gold_run: SacoGoldRun) -> None:
    with pytest.raises(FileNotFoundError, match="download saco-gold"):
        saco_gold.run_saco_gold(gold_run, gold_run.saco_root / "results")


def test_run_saco_gold_requires_a_verified_checkpoint(gold_run: SacoGoldRun) -> None:
    download.checkpoint_path(MODEL, gold_run.checkpoints_root).unlink()

    with pytest.raises(FileNotFoundError, match="download checkpoint"):
        saco_gold.run_saco_gold(gold_run, gold_run.saco_root / "results")
