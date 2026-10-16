"""Example text-prompt results on COCO images, sorted into hits and misses.

Each chosen image is prompted with the name of every COCO category it contains,
plus two it doesn't. The predicted masks are matched to COCO's outlines at an
IoU of at least 0.5, and each prompt gets one outcome: found every object,
found some, missed all, or a false alarm on a category that isn't there. The
gallery shows examples of what the model does well and badly.
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
# The gallery rows, from strengths to weaknesses.
GALLERY_ROWS = (
    ("large", "Large objects"),
    ("several", "Several at once"),
    ("small_found", "Small, found"),
    ("small_missed", "Small, missed"),
    ("mixup", "Mix-ups"),
)
PER_ROW = 5
# Photos chosen by hand, shown first in their row: (COCO image id, prompt).
PINNED = {"large": ((397133, "person"),)}
# The large-object row shows the categories found most often among those with
# at least this many objects.
MIN_CATEGORY_OBJECTS = 10
LARGE_AREA = SIZE_BANDS[1][1]
SMALL_AREA = SIZE_BANDS[0][1]
# Gallery photos must be landscape, between 4:3 and 3:2, so the panels line up.
MIN_ASPECT = 1.3
MAX_ASPECT = 1.55
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


def small_found(prompt: dict[str, Any]) -> int:
    """Return how many small objects a prompt found."""
    return sum(
        was_found
        for area, was_found in zip(
            prompt["object_areas"], prompt["objects_found"], strict=True
        )
        if area < SMALL_AREA
    )


def fits(row: str, prompt: dict[str, Any]) -> bool:
    """Return whether a prompt shows what a gallery row is about."""
    areas = prompt["object_areas"]
    if row == "large":
        return prompt["outcome"] == "found" and min(areas) >= LARGE_AREA
    if row == "several":
        return prompt["outcome"] == "found" and prompt["truth"] >= 3
    if row == "small_found":
        return small_found(prompt) > 0
    if row == "small_missed":
        return prompt["outcome"] == "missed" and max(areas) < SMALL_AREA
    covered = prompt.get("covered", [])
    return prompt["outcome"] == "false_alarm" and bool(covered) and all(covered)


def find_rates(prompts: list[dict[str, Any]]) -> dict[str, float]:
    """Return the share of objects found per category with enough objects."""
    totals: dict[str, list[int]] = {}
    for prompt in prompts:
        found, objects = totals.get(prompt["name"], [0, 0])
        totals[prompt["name"]] = [found + prompt["found"], objects + prompt["truth"]]
    return {
        name: found / objects
        for name, (found, objects) in totals.items()
        if objects >= MIN_CATEGORY_OBJECTS
    }


def select(
    prompts: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Pick the gallery rows, PER_ROW prompts that fit each.

    Pinned photos come first in their row. The rest are ranked: large objects
    by how often their category is found, among categories with at least
    MIN_CATEGORY_OBJECTS objects; several at once and small misses by most
    objects; small finds by most small objects found; and mix-ups in file
    order. Each prompt comes from a different landscape photo between 4:3
    and 3:2. A row uses different categories where it can, then fills with
    repeats.

    Returns:
        (row label, prompts) for each row with at least one prompt.
    """
    rates = find_rates(prompts)
    ranking = {
        "large": lambda prompt: -rates.get(prompt["name"], -1.0),
        "several": lambda prompt: -prompt["truth"],
        "small_found": lambda prompt: -small_found(prompt),
        "small_missed": lambda prompt: -prompt["truth"],
        "mixup": lambda prompt: 0,
    }
    used_images: set[int] = set()
    rows = []
    for row, label in GALLERY_ROWS:
        chosen = [
            prompt
            for image_id, name in PINNED.get(row, ())
            for prompt in prompts
            if prompt["image_id"] == image_id and prompt["name"] == name
        ]
        used_images.update(prompt["image_id"] for prompt in chosen)
        ranked = sorted(
            (
                (index, prompt)
                for index, prompt in enumerate(prompts)
                if fits(row, prompt)
                and MIN_ASPECT <= prompt["width"] / prompt["height"] <= MAX_ASPECT
                and (row != "large" or prompt["name"] in rates)
            ),
            key=lambda item: (ranking[row](item[1]), item[0]),
        )
        for distinct in (True, False):
            for _, prompt in ranked:
                names = {example["name"] for example in chosen}
                if (
                    len(chosen) < PER_ROW
                    and prompt["image_id"] not in used_images
                    and not (distinct and prompt["name"] in names)
                ):
                    chosen.append(prompt)
                    used_images.add(prompt["image_id"])
        if chosen:
            rows.append((label, chosen))
    return rows


def caption(prompt: dict[str, Any]) -> str:
    """Describe one gallery example: objects found, and any wrong masks."""
    text = f'"{prompt["name"]}": found {prompt["found"]} of {prompt["truth"]}'
    wrong = prompt["predicted"] - prompt["found"]
    if not wrong:
        return text
    return f"{text}, plus {wrong} wrong mask{'' if wrong == 1 else 's'}"


def lookalikes(prompt: dict[str, Any], coco: Any) -> list[str | None]:
    """Name the COCO object each wrong mask covers, or None where it covers none.

    A wrong mask covers an object of another category when their IoU is at
    least MATCH_IOU.
    """
    from pycocotools import mask as mask_utils

    labels = coco.loadAnns(coco.getAnnIds(imgIds=prompt["image_id"], iscrowd=False))
    outlines = [coco.annToRLE(label) for label in labels]
    names = {
        category["id"]: category["name"] for category in coco.dataset["categories"]
    }
    covered: list[str | None] = []
    for mask, matched in zip(prompt["masks"], prompt["matched_masks"], strict=True):
        if matched:
            continue
        if not outlines:
            covered.append(None)
            continue
        ious = mask_utils.iou([mask], outlines, [0] * len(outlines))[0]
        best = max(range(len(outlines)), key=lambda index: ious[index])
        covered.append(
            names[labels[best]["category_id"]] if ious[best] >= MATCH_IOU else None
        )
    return covered


def draw_gallery(
    rows: list[tuple[str, list[dict[str, Any]]]],
    images: Path,
    outlines: dict[int, list[dict[str, Any]]],
    path: Path,
) -> Path:
    """Draw each row of examples over their photos, with the row's label.

    Masks that match an object are green and masks that don't are red. COCO's
    outlines of the objects are yellow.

    Args:
        rows: (label, prompts) per row, with each prompt's masks as RLE.
        images: The folder holding the COCO images.
        outlines: COCO's outlines as RLE, keyed by the prompt's id().
        path: Where to write the JPEG.
    """
    import numpy
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from PIL import Image
    from pycocotools import mask as mask_utils

    figure = Figure(figsize=(4 * PER_ROW, 3.15 * len(rows) + 0.5))
    axes_grid = figure.subplots(
        len(rows),
        PER_ROW,
        squeeze=False,
        gridspec_kw={
            "wspace": 0.03,
            "hspace": 0.18,
            "top": 0.95,
            "bottom": 0.01,
            "left": 0.03,
            "right": 0.99,
        },
    )
    for axes_row, (label, examples) in zip(axes_grid, rows, strict=True):
        for axes, example in zip(axes_row, examples, strict=False):
            photo = numpy.asarray(
                Image.open(images / example["file_name"]).convert("RGB")
            )
            overlay = numpy.zeros((*photo.shape[:2], 4))
            for mask, matched in zip(
                example["masks"], example["matched_masks"], strict=True
            ):
                color = MATCHED_COLOR if matched else WRONG_COLOR
                overlay[mask_utils.decode(mask).astype(bool)] = (*color, 0.55)
            axes.imshow(photo)
            axes.imshow(overlay)
            for outline in outlines[id(example)]:
                axes.contour(
                    mask_utils.decode(outline),
                    levels=[0.5],
                    colors=OUTLINE_COLOR,
                    linewidths=1.5,
                )
            axes.set_title(caption(example), fontsize=10)
        axes_row[0].text(
            -0.04,
            0.5,
            label,
            rotation=90,
            transform=axes_row[0].transAxes,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
    for axes in axes_grid.flat:
        axes.set_axis_off()
    figure.legend(
        handles=[
            Patch(color=MATCHED_COLOR, alpha=0.7, label="Mask on an object"),
            Patch(color=WRONG_COLOR, alpha=0.7, label="Mask on nothing"),
            Line2D([], [], color=OUTLINE_COLOR, linewidth=2, label="COCO's outline"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=3,
        frameon=False,
        fontsize=12,
    )
    figure.savefig(path, format="jpeg", dpi=100, bbox_inches="tight")
    return path


def write_gallery(
    record: dict[str, Any], coco: Any, images: Path, path: Path
) -> list[dict[str, Any]]:
    """Draw the gallery from a run's record and return the examples it shows."""
    rows = select(with_lookalikes(record["prompts"], coco))
    examples = [example for _, row in rows for example in row]
    outlines = {
        id(example): [
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
    }
    if rows:
        draw_gallery(rows, images, outlines, path)
    return examples


def with_lookalikes(prompts: list[dict[str, Any]], coco: Any) -> list[dict[str, Any]]:
    """Add what each false alarm's wrong masks covered, under "covered"."""
    return [
        prompt | {"covered": lookalikes(prompt, coco)}
        if prompt["outcome"] == "false_alarm"
        else prompt
        for prompt in prompts
    ]


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
    download.download_coco_images(
        coco_root / download.COCO_ANNOTATIONS_MEMBER,
        [
            example["image_id"]
            for _, row in select(with_lookalikes(record["prompts"], coco))
            for example in row
        ],
        coco_root,
    )
    shown = write_gallery(
        record, coco, coco_root / "images" / "val2017", results_dir / "gallery.jpg"
    )
    return credits(coco, shown)
