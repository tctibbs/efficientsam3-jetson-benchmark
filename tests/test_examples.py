import hashlib
import json
from pathlib import Path
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
    examples,
    latency,
)
from efficientsam3_jetson_benchmark.download import HubFile, PinnedFile
from efficientsam3_jetson_benchmark.examples import ExamplesRun

MODEL = "efficientsam3_tinyvit"
CATEGORIES = [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"},
    {"id": 3, "name": "bus"},
]


def square(left: int, height: int = 8, width: int = 12) -> numpy.ndarray:
    """A mask with a 4-pixel square whose top-left corner is at (left, left)."""
    mask = numpy.zeros((height, width), dtype=bool)
    mask[left : left + 4, left : left + 4] = True
    return mask


def rle(mask: numpy.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(numpy.asfortranarray(mask.astype(numpy.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def label(label_id: int, image_id: int, category: int, left: int) -> dict[str, Any]:
    """A COCO label whose polygon outlines square(left)."""
    right = left + 4
    return {
        "id": label_id,
        "image_id": image_id,
        "category_id": category,
        "iscrowd": 0,
        "area": 16.0,
        "bbox": [left, left, 4, 4],
        "segmentation": [[left, left, right, left, right, right, left, right]],
    }


def dataset() -> dict[str, Any]:
    """Landscape images: a cat, two cats, a bus, an unshowable cat, and a bus."""
    sizes = {1: (12, 8), 2: (24, 16), 3: (12, 8), 4: (12, 8), 5: (12, 8)}
    licenses = {1: 4, 2: 7, 3: 4, 4: 1, 5: 4}
    images = [
        {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "license": licenses[image_id],
            "width": width,
            "height": height,
            "flickr_url": f"http://example.com/{image_id}.jpg",
        }
        for image_id, (width, height) in sizes.items()
    ]
    return {
        "images": images,
        "annotations": [
            label(1, 1, 1, 2),
            label(2, 2, 1, 2),
            label(3, 2, 1, 10),
            label(4, 3, 3, 2),
            label(5, 4, 1, 2),
            label(6, 5, 3, 2),
        ],
        "categories": CATEGORIES,
        "licenses": [
            {"id": 1, "name": "Attribution-NonCommercial-ShareAlike License"},
            {"id": 4, "name": "Attribution License"},
            {"id": 7, "name": "No known copyright restrictions"},
        ],
    }


class FakeProcessor:
    """Finds a square at (2, 2) for "cat" and "dog", and nothing for "bus"."""

    device = torch.device("cpu")
    confidence_threshold = 0.5

    def set_image(self, image: Image.Image) -> dict[str, Any]:
        return {"width": image.size[0], "height": image.size[1]}

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        height, width = state["height"], state["width"]
        if prompt == "bus":
            state["masks"] = torch.zeros((0, 1, height, width))
            state["scores"] = torch.tensor([])
        else:
            state["masks"] = torch.from_numpy(square(2, height, width))[None, None]
            state["scores"] = torch.tensor([0.9])
        return state

    def reset_all_prompts(self, state: dict[str, Any]) -> None:
        del state["masks"], state["scores"]


@pytest.fixture
def examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExamplesRun:
    """A run with a verified checkpoint and the synthetic COCO data on disk."""
    weights = b"weights"
    pin = PinnedFile(len(weights), hashlib.sha256(weights).hexdigest())
    hub_file = HubFile("owner/repo", "a" * 40, f"full/{MODEL}.pt", pin)
    monkeypatch.setitem(download.CHECKPOINTS, MODEL, hub_file)
    checkpoint = download.checkpoint_path(MODEL, tmp_path / "checkpoints")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(weights)
    coco_root = tmp_path / "coco"
    annotations = coco_root / download.COCO_ANNOTATIONS_MEMBER
    annotations.parent.mkdir(parents=True)
    annotations.write_text(json.dumps(dataset()))
    images = coco_root / "images" / "val2017"
    images.mkdir(parents=True)
    for image in dataset()["images"]:
        size = (image["width"], image["height"])
        Image.new("RGB", size, "gray").save(images / image["file_name"])
    return ExamplesRun(
        model=MODEL,
        precision="fp32",
        images=10,
        coco_root=coco_root,
        checkpoints_root=tmp_path / "checkpoints",
        upstream=tmp_path / "upstream",
    )


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the model, the run description, and the device name."""
    monkeypatch.setattr(
        environment,
        "describe",
        lambda repository, upstream: {"timestamp": "2026-10-16T10:00:00+00:00"},
    )
    monkeypatch.setattr(
        latency,
        "load_pipeline",
        lambda model_name, checkpoint, upstream: (None, FakeProcessor()),
    )
    monkeypatch.setattr(coco_runner, "device_name", lambda device_type: "CPU")


def prompt(
    image_id: int, outcome: str, name: str, truth: int = 1, found: int = 0, **extra: Any
) -> dict[str, Any]:
    """A prompt as the record stores it, on a landscape image."""
    return {
        "image_id": image_id,
        "outcome": outcome,
        "name": name,
        "truth": truth,
        "found": found,
        "predicted": extra.get("predicted", found),
        "width": extra.get("width", 640),
        "height": 480,
    }


def test_choose_images_keeps_showable_images_with_objects() -> None:
    chosen = examples.choose_images(dataset(), 2)

    assert [image["id"] for image in chosen] == [1, 2]


def test_prompts_for_lists_present_categories_then_absent_ones() -> None:
    labels = [label(1, 1, 1, 2), {**label(2, 1, 2, 2), "iscrowd": 1}]

    prompts = examples.prompts_for(dataset(), labels, image_id=1)

    # Dog appears only as a crowd region, so it is neither present nor absent.
    assert [(item["name"], item["present"]) for item in prompts] == [
        ("cat", True),
        ("bus", False),
    ]


def test_absent_categories_are_the_same_on_every_run() -> None:
    labels = [label(1, 1, 1, 2)]

    first = examples.prompts_for(dataset(), labels, image_id=7)

    assert len([item for item in first if not item["present"]]) == 2
    assert examples.prompts_for(dataset(), labels, image_id=7) == first


def test_match_pairs_each_object_with_at_most_one_prediction() -> None:
    truth = [rle(square(2, 16, 16)), rle(square(10, 16, 16))]
    shifted = square(2, 16, 16)
    shifted[:, 2] = False
    shifted[2:6, 6] = True  # IoU 12 / 20 = 0.6 with square(2)

    assert examples.match([rle(square(2, 16, 16))] * 2, truth) == {(0, 0)}
    assert examples.match([rle(shifted)], truth) == {(0, 0)}
    assert examples.match([rle(square(3, 16, 16))], truth) == set()  # IoU 9 / 23
    assert examples.match([], truth) == set()


@pytest.mark.parametrize(
    ("truth", "found", "predicted", "expected"),
    [
        (2, 2, 2, "found"),
        (2, 2, 3, "partial"),
        (2, 1, 1, "partial"),
        (2, 0, 1, "missed"),
        (0, 0, 1, "false_alarm"),
        (0, 0, 0, "correct_rejection"),
    ],
)
def test_outcome(truth: int, found: int, predicted: int, expected: str) -> None:
    assert examples.outcome(truth, found, predicted) == expected


def test_summarize_counts_outcomes_and_objects_by_size() -> None:
    prompts = [
        {"outcome": "found", "object_areas": [500.0, 20000.0], "objects_found": [1, 1]},
        {"outcome": "missed", "object_areas": [500.0], "objects_found": [0]},
        {"outcome": "false_alarm", "object_areas": [], "objects_found": []},
    ]

    summary = examples.summarize(prompts)

    assert summary["outcomes"]["found"] == 1
    assert summary["outcomes"]["false_alarm"] == 1
    assert summary["objects_by_size"]["small"] == {"objects": 2, "found": 1}
    assert summary["objects_by_size"]["large"] == {"objects": 1, "found": 1}


def test_select_ranks_hits_and_failures_by_object_count() -> None:
    prompts = [
        prompt(1, "found", "cat", truth=2, found=2),
        prompt(2, "found", "cat", truth=5, found=5),
        prompt(3, "found", "dog", truth=3, found=3),
        prompt(4, "found", "bus", truth=9, found=9, width=400),  # not landscape
        prompt(5, "found", "car", truth=1, found=1),
        prompt(6, "missed", "cup", truth=2),
        prompt(7, "missed", "fork", truth=6),
        prompt(8, "partial", "kite", truth=5, found=1, predicted=1),
        prompt(9, "partial", "boat", truth=3, found=2, predicted=2),
        prompt(10, "false_alarm", "zebra", truth=0, predicted=1),
        prompt(11, "false_alarm", "bear", truth=0, predicted=3),
    ]

    chosen = examples.select(prompts)

    assert [(item["image_id"], item["name"]) for item in chosen] == [
        (2, "cat"),
        (3, "dog"),
        (5, "car"),
        (7, "fork"),
        (8, "kite"),
        (11, "bear"),
    ]


def test_select_leaves_out_outcomes_that_never_happened() -> None:
    prompts = [prompt(1, "found", "cat", found=1)]

    assert examples.select(prompts) == prompts


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (prompt(1, "found", "cat", truth=2, found=2), '"cat": found 2 of 2'),
        (
            prompt(1, "partial", "cat", truth=2, found=1, predicted=3),
            '"cat": found 1 of 2, 2 wrong',
        ),
        (
            prompt(1, "false_alarm", "dog", truth=0, predicted=1),
            '"dog": none in photo, 1 wrong',
        ),
    ],
)
def test_caption(item: dict[str, Any], expected: str) -> None:
    assert examples.caption(item) == expected


@pytest.mark.usefixtures("fake_model")
def test_run_examples_writes_a_record_and_gallery(
    tmp_path: Path, examples_run: ExamplesRun
) -> None:
    path = examples.run_examples(examples_run, tmp_path / "results")

    record = json.loads(path.read_text())
    assert path.name == f"20261016T100000Z-examples-coco-{MODEL}-fp32-cpu.json"
    assert record["settings"]["images"] == 4
    assert record["summary"]["outcomes"] == {
        "found": 1,
        "partial": 1,
        "missed": 2,
        "false_alarm": 6,
        "correct_rejection": 2,
    }
    assert record["summary"]["objects_by_size"]["small"] == {"objects": 5, "found": 2}
    partial = next(item for item in record["prompts"] if item["outcome"] == "partial")
    assert partial["matched_masks"] == [True]
    assert partial["objects_found"] == [True, False]
    assert (tmp_path / "results" / "gallery.jpg").read_bytes()[:2] == b"\xff\xd8"


@pytest.mark.usefixtures("fake_model")
def test_redraw_gallery_uses_the_newest_record(
    tmp_path: Path, examples_run: ExamplesRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    examples.run_examples(examples_run, results)
    (results / "gallery.jpg").unlink()
    fetched = []
    monkeypatch.setattr(
        download,
        "download_coco_images",
        lambda annotations, ids, root: fetched.extend(ids),
    )

    credits = examples.redraw_gallery(results, examples_run.coco_root)

    assert (results / "gallery.jpg").is_file()
    assert fetched == [item["image_id"] for item in credits]
    assert credits[0] == {
        "image_id": 1,
        "flickr_url": "http://example.com/1.jpg",
        "license": "Attribution License",
    }


def test_redraw_gallery_needs_a_record(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no examples record"):
        examples.redraw_gallery(tmp_path, tmp_path)


def test_empty_gallery_draws_nothing(tmp_path: Path) -> None:
    record = {"prompts": [prompt(1, "correct_rejection", "cat", truth=0)]}

    assert examples.write_gallery(record, None, tmp_path, tmp_path / "g.jpg") == []
    assert not (tmp_path / "g.jpg").exists()


def test_run_examples_requires_a_verified_checkpoint(examples_run: ExamplesRun) -> None:
    download.checkpoint_path(MODEL, examples_run.checkpoints_root).unlink()

    with pytest.raises(FileNotFoundError, match="download checkpoint"):
        examples.run_examples(examples_run, examples_run.coco_root / "results")


def test_run_examples_requires_coco_annotations(examples_run: ExamplesRun) -> None:
    (examples_run.coco_root / download.COCO_ANNOTATIONS_MEMBER).unlink()

    with pytest.raises(FileNotFoundError, match="download coco"):
        examples.run_examples(examples_run, examples_run.coco_root / "results")
