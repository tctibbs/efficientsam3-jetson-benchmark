"""Example text-prompt results on COCO images, sorted into hits and misses.

Each chosen image is prompted with the name of every COCO category it contains,
plus two it doesn't. The predicted masks are matched to COCO's outlines at an
IoU of at least 0.5, and each prompt gets one outcome: found every object,
found some, missed all, or a false alarm on a category that isn't there. The
gallery shows the strongest and weakest results by object count.
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from efficientsam3_jetson_benchmark import download, environment, latency, saco_gold

# COCO licence ids whose images may be shown with credit: CC BY 2.0, and no known
# copyright restrictions.
SHOWABLE_LICENSES = (4, 7)
ABSENT_PROMPTS = 2
MATCH_IOU = 0.5
# COCO's own object size bands, by area in pixels.
SIZE_BANDS = (("small", 32**2), ("medium", 96**2), ("large", float("inf")))
OUTCOMES = ("found", "partial", "missed", "false_alarm", "correct_rejection")
# The gallery: three hits, then one of each kind of failure, ranked by how many
# objects or masks the outcome involves.
GALLERY = ("found", "found", "found", "missed", "partial", "false_alarm")
RANKING = {
    "found": lambda prompt: prompt["truth"],
    "missed": lambda prompt: prompt["truth"],
    "partial": lambda prompt: prompt["truth"] - prompt["found"],
    "false_alarm": lambda prompt: prompt["predicted"],
}
# Gallery photos must be landscape, so the panels line up.
MIN_ASPECT = 1.3
MATCHED_COLOR = (0.18, 0.8, 0.44)
WRONG_COLOR = (0.91, 0.3, 0.24)
OUTLINE_COLOR = "#ffd400"


@dataclass(frozen=True)
class ExamplesRun:
    """Inputs of one examples run on one model."""

    model: str
    precision: str
    images: int
    coco_root: Path
    checkpoints_root: Path
    upstream: Path


def choose_images(dataset: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Return the first count showable images with objects, in file order."""
    has_objects = {
        label["image_id"] for label in dataset["annotations"] if not label["iscrowd"]
    }
    chosen = [
        image
        for image in dataset["images"]
        if image["license"] in SHOWABLE_LICENSES and image["id"] in has_objects
    ]
    return chosen[:count]


def prompts_for(
    dataset: dict[str, Any], labels: list[dict[str, Any]], image_id: int
) -> list[dict[str, Any]]:
    """Return the prompts for one image: categories it has, then two it lacks.

    Categories that only appear as crowd regions in the image are left out,
    because a crowd region has no single outline to match.
    """
    names = {category["id"]: category["name"] for category in dataset["categories"]}
    crowd = {label["category_id"] for label in labels if label["iscrowd"]}
    present = sorted(
        {label["category_id"] for label in labels if not label["iscrowd"]} - crowd
    )
    seen = {label["category_id"] for label in labels}
    unseen = [category for category in sorted(names) if category not in seen]
    absent = random.Random(image_id).sample(unseen, min(ABSENT_PROMPTS, len(unseen)))
    return [
        {"category_id": category, "name": names[category], "present": True}
        for category in present
    ] + [
        {"category_id": category, "name": names[category], "present": False}
        for category in absent
    ]


def match(
    predicted: list[dict[str, Any]], truth: list[dict[str, Any]]
) -> set[tuple[int, int]]:
    """Return (prediction, object) index pairs where a mask covers an object.

    A prediction covers an object when their IoU is at least MATCH_IOU. Pairs
    are matched greedily from the highest IoU down, one prediction per object.
    """
    from pycocotools import mask as mask_utils

    if not predicted or not truth:
        return set()
    ious = mask_utils.iou(predicted, truth, [0] * len(truth))
    # Highest IoU first; ties go to the earlier prediction.
    candidates = sorted(
        (-ious[row][column], row, column)
        for row in range(len(predicted))
        for column in range(len(truth))
    )
    pairs: set[tuple[int, int]] = set()
    for negative_iou, row, column in candidates:
        if -negative_iou < MATCH_IOU:
            break
        if all(
            row != used_row and column != used_column for used_row, used_column in pairs
        ):
            pairs.add((row, column))
    return pairs


def outcome(truth: int, found: int, predicted: int) -> str:
    """Name what happened for one prompt."""
    if truth == 0:
        return "false_alarm" if predicted else "correct_rejection"
    if found == 0:
        return "missed"
    if found == truth and predicted == truth:
        return "found"
    return "partial"


def size_band(area: float) -> str:
    """Return COCO's size band for an object's area."""
    return next(name for name, limit in SIZE_BANDS if area < limit)


def summarize(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the outcomes and the share of objects found in each size band."""
    counts = dict.fromkeys(OUTCOMES, 0)
    bands = {name: {"objects": 0, "found": 0} for name, _ in SIZE_BANDS}
    for prompt in prompts:
        counts[prompt["outcome"]] += 1
        for area, was_found in zip(
            prompt["object_areas"], prompt["objects_found"], strict=True
        ):
            band = bands[size_band(area)]
            band["objects"] += 1
            band["found"] += was_found
    return {"outcomes": counts, "objects_by_size": bands}


def select(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the gallery: the three biggest hits, then the biggest of each failure.

    Hits are ranked by objects found, misses by objects missed, partial
    results by objects missed, and false alarms by wrong masks. Each pick
    comes from a different landscape photo, the hits from different
    categories, and ties go to the earlier prompt.
    """
    chosen: list[dict[str, Any]] = []
    for wanted in GALLERY:
        used_images = {example["image_id"] for example in chosen}
        used_names = {example["name"] for example in chosen}
        candidates = [
            (index, prompt)
            for index, prompt in enumerate(prompts)
            if prompt["outcome"] == wanted
            and prompt["width"] >= MIN_ASPECT * prompt["height"]
            and prompt["image_id"] not in used_images
            and (wanted != "found" or prompt["name"] not in used_names)
        ]
        if candidates:
            rank = RANKING[wanted]
            chosen.append(
                max(candidates, key=lambda item: (rank(item[1]), -item[0]))[1]
            )
    return chosen


def caption(prompt: dict[str, Any]) -> str:
    """Describe one gallery example in a few words."""
    if prompt["outcome"] == "false_alarm":
        return f'"{prompt["name"]}": none in photo, {prompt["predicted"]} wrong'
    extra = prompt["predicted"] - prompt["found"]
    text = f'"{prompt["name"]}": found {prompt["found"]} of {prompt["truth"]}'
    return f"{text}, {extra} wrong" if extra else text


def draw_gallery(
    examples: list[dict[str, Any]],
    images: Path,
    outlines: list[list[dict[str, Any]]],
    path: Path,
) -> Path:
    """Draw each example over its photo, three to a row.

    Masks that match an object are green and masks that don't are red. COCO's
    outlines of the objects are yellow.

    Args:
        examples: Selected prompts, with their predicted masks as RLE.
        images: The folder holding the COCO images.
        outlines: COCO's outlines as RLE, one list per example.
        path: Where to write the JPEG.
    """
    import numpy
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from PIL import Image
    from pycocotools import mask as mask_utils

    columns = 3
    rows = (len(examples) + columns - 1) // columns
    figure = Figure(figsize=(5 * columns, 3.75 * rows + 0.6))
    axes_grid = figure.subplots(
        rows, columns, squeeze=False, gridspec_kw={"wspace": 0.03, "hspace": 0.16}
    )
    for axes, example, truth in zip(axes_grid.flat, examples, outlines, strict=False):
        photo = numpy.asarray(Image.open(images / example["file_name"]).convert("RGB"))
        overlay = numpy.zeros((*photo.shape[:2], 4))
        for mask, matched in zip(
            example["masks"], example["matched_masks"], strict=True
        ):
            color = MATCHED_COLOR if matched else WRONG_COLOR
            overlay[mask_utils.decode(mask).astype(bool)] = (*color, 0.55)
        axes.imshow(photo)
        axes.imshow(overlay)
        for outline in truth:
            axes.contour(
                mask_utils.decode(outline),
                levels=[0.5],
                colors=OUTLINE_COLOR,
                linewidths=1.5,
            )
        axes.set_title(caption(example), fontsize=12)
    for axes in axes_grid.flat:
        axes.set_axis_off()
    for row, label in zip(axes_grid, ("Strongest", "Weakest"), strict=False):
        row[0].text(
            -0.03,
            0.5,
            label,
            rotation=90,
            transform=row[0].transAxes,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
    figure.legend(
        handles=[
            Patch(color=MATCHED_COLOR, alpha=0.7, label="Mask on an object"),
            Patch(color=WRONG_COLOR, alpha=0.7, label="Mask on nothing"),
            Line2D([], [], color=OUTLINE_COLOR, linewidth=2, label="COCO's outline"),
        ],
        loc="upper center",
        ncols=3,
        frameon=False,
        fontsize=12,
    )
    figure.savefig(path, format="jpeg", dpi=110, bbox_inches="tight")
    return path


def write_gallery(
    record: dict[str, Any], coco: Any, images: Path, path: Path
) -> list[dict[str, Any]]:
    """Draw the gallery from a run's record and return the examples it shows."""
    examples = select(record["prompts"])
    outlines = [
        [
            coco.annToRLE(label)
            for label in coco.loadAnns(
                coco.getAnnIds(
                    imgIds=example["image_id"],
                    catIds=[example["category_id"]],
                    iscrowd=False,
                )
            )
        ]
        for example in examples
    ]
    if examples:
        draw_gallery(examples, images, outlines, path)
    return examples


def credits(coco: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return each gallery photo's COCO id, Flickr address, and licence."""
    licenses = {license["id"]: license for license in coco.dataset["licenses"]}
    return [
        {
            "image_id": example["image_id"],
            "flickr_url": coco.imgs[example["image_id"]]["flickr_url"],
            "license": licenses[coco.imgs[example["image_id"]]["license"]]["name"],
        }
        for example in examples
    ]


def load_coco(coco_root: Path) -> Any:
    """Load COCO val2017's annotations.

    Raises:
        FileNotFoundError: The annotations are missing.
    """
    from pycocotools.coco import COCO

    annotations = coco_root / download.COCO_ANNOTATIONS_MEMBER
    if not annotations.is_file():
        raise FileNotFoundError(
            f"{annotations} is missing; run "
            "`efficientsam3-jetson-benchmark download coco --first 5000`"
        )
    return COCO(str(annotations))


def run_examples(run: ExamplesRun, results_dir: Path) -> Path:
    """Prompt the chosen COCO images, sort the results, and write a record and gallery.

    Raises:
        FileNotFoundError: The checkpoint is missing or unverified, or the COCO
            annotations are missing.
    """
    from efficientsam3_jetson_benchmark.coco_runner import device_name

    hub_file = download.CHECKPOINTS[run.model]
    checkpoint = download.checkpoint_path(run.model, run.checkpoints_root)
    if not download.matches(checkpoint, hub_file.pin):
        raise FileNotFoundError(
            f"{checkpoint} is missing or unverified; run "
            f"`efficientsam3-jetson-benchmark download checkpoint {run.model}`"
        )
    coco = load_coco(run.coco_root)
    images = run.coco_root / "images" / "val2017"
    chosen = choose_images(coco.dataset, run.images)
    pairs: list[dict[str, Any]] = []
    for image in chosen:
        labels = coco.loadAnns(coco.getAnnIds(imgIds=image["id"]))
        for prompt in prompts_for(coco.dataset, labels, image["id"]):
            truth = [
                label
                for label in labels
                if label["category_id"] == prompt["category_id"]
            ]
            pairs.append(
                {
                    **prompt,
                    "id": len(pairs),
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                    "text_input": prompt["name"],
                    "labels": truth,
                }
            )

    context = environment.describe(Path.cwd(), run.upstream)
    _, processor = latency.load_pipeline(run.model, checkpoint, run.upstream)
    results = saco_gold.predict(processor, pairs, images, run.precision)
    by_pair: dict[int, list[dict[str, Any]]] = {pair["id"]: [] for pair in pairs}
    for result in results:
        by_pair[result["image_id"]].append(result["segmentation"])

    prompts = []
    for pair in pairs:
        predicted = by_pair[pair["id"]]
        truth = [coco.annToRLE(label) for label in pair["labels"]]
        pairs_matched = match(predicted, truth)
        matched_rows = {row for row, _ in pairs_matched}
        matched_columns = {column for _, column in pairs_matched}
        prompts.append(
            {
                key: pair[key]
                for key in (
                    "image_id",
                    "file_name",
                    "width",
                    "height",
                    "category_id",
                    "name",
                )
            }
            | {
                "truth": len(truth),
                "found": len(pairs_matched),
                "predicted": len(predicted),
                "outcome": outcome(len(truth), len(pairs_matched), len(predicted)),
                "object_areas": [label["area"] for label in pair["labels"]],
                "objects_found": [
                    index in matched_columns for index in range(len(truth))
                ],
                "masks": predicted,
                "matched_masks": [
                    index in matched_rows for index in range(len(predicted))
                ],
            }
        )

    device_type = processor.device.type
    stamp = datetime.fromisoformat(context["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-examples-coco-{run.model}-{run.precision}-{device_type}"
    record = {
        "run_id": run_id,
        "benchmark": "coco-text-prompt-examples",
        "environment": context,
        "runtime": latency.runtime_record(device_type, device_name(device_type)),
        "model": {
            "name": run.model,
            "repo_id": hub_file.repo_id,
            "revision": hub_file.revision,
            "filename": hub_file.filename,
            "sha256": hub_file.pin.sha256,
        },
        "settings": {
            "precision": run.precision,
            "images": len(chosen),
            "absent_prompts_per_image": ABSENT_PROMPTS,
            "match_iou": MATCH_IOU,
            "confidence_threshold": processor.confidence_threshold,
        },
        "dataset": {"annotations_sha256": download.COCO_VAL_ANNOTATIONS.sha256},
        "summary": summarize(prompts),
        "prompts": prompts,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps(record) + "\n")
    write_gallery(record, coco, images, results_dir / "gallery.jpg")
    return path


def redraw_gallery(results_dir: Path, coco_root: Path) -> list[dict[str, Any]]:
    """Redraw the gallery from the newest record, downloading any missing photos.

    Returns:
        The credits for each photo in the gallery.

    Raises:
        FileNotFoundError: There is no record, or the COCO annotations are missing.
    """
    records = sorted(results_dir.glob("*-examples-coco-*.json"))
    if not records:
        raise FileNotFoundError(f"no examples record in {results_dir}")
    record = json.loads(records[-1].read_text())
    coco = load_coco(coco_root)
    examples = select(record["prompts"])
    download.download_coco_images(
        coco_root / download.COCO_ANNOTATIONS_MEMBER,
        [example["image_id"] for example in examples],
        coco_root,
    )
    shown = write_gallery(
        record, coco, coco_root / "images" / "val2017", results_dir / "gallery.jpg"
    )
    return credits(coco, shown)
