"""Text-prompt accuracy on SA-Co/Gold, scored with SAM 3's own cgF1 evaluator.

SA-Co/Gold pairs images with noun phrases across seven domains, and each pair
is labelled by three annotators. For each domain this takes the first N images
in annotation order with all of their phrases, runs every phrase through
upstream's Sam3Processor the way upstream's SA-Co/Gold script does, and scores
the returned masks with upstream's CGF1Evaluator against all three annotators,
restricted to the same pairs.
"""

import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from efficientsam3_jetson_benchmark import download, environment, latency

# Domain name in the annotation file names, and the image source it uses.
SUBSETS = {
    "metaclip": "metaclip",
    "sa1b": "sa1b",
    "attributes": "metaclip",
    "crowded": "metaclip",
    "wiki_common": "metaclip",
    "fg_food": "metaclip",
    "fg_sports_equipment": "metaclip",
}
ANNOTATORS = ("a", "b", "c")
AVERAGED_METRICS = ("cgF1", "IL_MCC", "positive_micro_F1")


@dataclass(frozen=True)
class SacoGoldRun:
    """Inputs of one SA-Co/Gold accuracy run on one model."""

    model: str
    precision: str
    images_per_subset: int
    subsets: tuple[str, ...]
    saco_root: Path
    checkpoints_root: Path
    upstream: Path


def annotation_path(root: Path, subset: str, annotator: str) -> Path:
    """Return the path of one annotator's labels for a domain."""
    name = f"gold_{subset}_merged_{annotator}_release_test.json"
    return root / "all" / "gt-annotations" / name


def image_root(root: Path, source: str) -> Path:
    """Return the directory that annotation file names are relative to."""
    return root / "all" / f"{source}-images"


def sample_pairs(pairs: list[dict[str, Any]], image_count: int) -> list[dict[str, Any]]:
    """Return every pair of the first image_count images, in annotation order."""
    first_images = list(dict.fromkeys(pair["file_name"] for pair in pairs))
    chosen = set(first_images[:image_count])
    return [pair for pair in pairs if pair["file_name"] in chosen]


def restrict(dataset: dict[str, Any], pair_ids: set[int]) -> dict[str, Any]:
    """Return a copy of an annotation file that keeps only the given pairs."""
    return {
        **dataset,
        "images": [pair for pair in dataset["images"] if pair["id"] in pair_ids],
        "annotations": [
            label for label in dataset["annotations"] if label["image_id"] in pair_ids
        ],
    }


def load_labels(
    root: Path, subset: str, image_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the sampled pairs and each annotator's labels for only those pairs.

    Only one full annotation file is held at a time; the largest are 50 MB.
    """

    def read(annotator: str) -> dict[str, Any]:
        return json.loads(annotation_path(root, subset, annotator).read_text())

    pairs = sample_pairs(read(ANNOTATORS[0])["images"], image_count)
    pair_ids = {pair["id"] for pair in pairs}
    return pairs, [restrict(read(annotator), pair_ids) for annotator in ANNOTATORS]


def to_results(pair_id: int, masks: Any, scores: Any) -> list[dict[str, Any]]:
    """Convert one prompt's masks and scores to COCO result entries."""
    import numpy
    from pycocotools import mask as mask_utils

    results = []
    for mask, score in zip(masks, scores, strict=True):
        binary = numpy.asfortranarray(mask.squeeze(0).cpu().numpy().astype(numpy.uint8))
        encoded = mask_utils.encode(binary)
        encoded["counts"] = encoded["counts"].decode("ascii")
        results.append(
            {
                "image_id": pair_id,
                "category_id": 1,
                "segmentation": encoded,
                "score": float(score),
            }
        )
    return results


def predict(
    processor: Any, pairs: list[dict[str, Any]], images: Path, precision: str
) -> list[dict[str, Any]]:
    """Run every pair's phrase on its image, encoding each image once."""
    from PIL import Image

    by_image: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_image.setdefault(pair["file_name"], []).append(pair)
    results = []
    for file_name, image_pairs in by_image.items():
        image = Image.open(images / file_name).convert("RGB")
        with latency.autocast(processor.device.type, precision):
            state = processor.set_image(image)
            for pair in image_pairs:
                state = processor.set_text_prompt(pair["text_input"], state)
                results.extend(to_results(pair["id"], state["masks"], state["scores"]))
                processor.reset_all_prompts(state)
    return results


def score(
    upstream: Path, labels: list[dict[str, Any]], results: list[dict[str, Any]]
) -> tuple[dict[str, float], int]:
    """Score predictions with upstream's CGF1Evaluator against every annotator.

    Returns:
        The metrics without their cgF1_eval_segm_ prefix, and how many pairs
        were scored; the evaluator skips pairs not every annotator labelled
        exhaustively.
    """
    module = latency.import_upstream(upstream, "sam3.eval.cgf1_eval")
    with tempfile.TemporaryDirectory() as scratch:
        paths = []
        for annotator, dataset in zip(ANNOTATORS, labels, strict=True):
            path = Path(scratch) / f"labels_{annotator}.json"
            path.write_text(json.dumps(dataset))
            paths.append(str(path))
        predictions = Path(scratch) / "predictions.json"
        predictions.write_text(json.dumps(results))
        evaluator = module.CGF1Evaluator(gt_path=paths, iou_type="segm")
        scores = evaluator.evaluate(str(predictions))
    metrics = {
        name.removeprefix("cgF1_eval_segm_"): value for name, value in scores.items()
    }
    return metrics, len(evaluator.eval_img_ids)


def evaluate_subset(run: SacoGoldRun, processor: Any, subset: str) -> dict[str, Any]:
    """Sample, predict, and score one domain."""
    pairs, labels = load_labels(run.saco_root, subset, run.images_per_subset)
    start = time.perf_counter()
    results = predict(
        processor, pairs, image_root(run.saco_root, SUBSETS[subset]), run.precision
    )
    elapsed = time.perf_counter() - start
    metrics, scored_pairs = score(run.upstream, labels, results)
    return {
        "images": len({pair["file_name"] for pair in pairs}),
        "pairs": len(pairs),
        "scored_pairs": scored_pairs,
        "predicted_masks": len(results),
        "inference_s": round(elapsed, 1),
        "metrics": metrics,
    }


def run_saco_gold(run: SacoGoldRun, results_dir: Path) -> Path:
    """Run the sampled SA-Co/Gold evaluation on one model and write a record.

    Raises:
        FileNotFoundError: The checkpoint is missing or unverified, or the
            labels of a requested domain are missing.
    """
    from efficientsam3_jetson_benchmark.coco_runner import device_name

    hub_file = download.CHECKPOINTS[run.model]
    checkpoint = download.checkpoint_path(run.model, run.checkpoints_root)
    if not download.matches(checkpoint, hub_file.pin):
        raise FileNotFoundError(
            f"{checkpoint} is missing or unverified; run "
            f"`efficientsam3-jetson-benchmark download checkpoint {run.model}`"
        )
    for subset in run.subsets:
        for annotator in ANNOTATORS:
            path = annotation_path(run.saco_root, subset, annotator)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing; run "
                    "`efficientsam3-jetson-benchmark download saco-gold`"
                )
    context = environment.describe(Path.cwd(), run.upstream)
    _, processor = latency.load_pipeline(run.model, checkpoint, run.upstream)
    subsets = {
        subset: evaluate_subset(run, processor, subset) for subset in run.subsets
    }
    device_type = processor.device.type
    stamp = datetime.fromisoformat(context["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{stamp}-saco-gold-{run.model}-{run.precision}-"
        f"{run.images_per_subset}images-{device_type}"
    )
    record = {
        "run_id": run_id,
        "benchmark": "saco-gold-text-prompt-cgf1",
        "environment": context,
        "runtime": latency.runtime_record(device_type, device_name(device_type)),
        "model": {
            "name": run.model,
            "text_encoder": latency.TEXT_ENCODER,
            "text_context_length": latency.TEXT_CONTEXT_LENGTH,
            "repo_id": hub_file.repo_id,
            "revision": hub_file.revision,
            "filename": hub_file.filename,
            "sha256": hub_file.pin.sha256,
        },
        "settings": {
            "precision": run.precision,
            "images_per_subset": run.images_per_subset,
            "confidence_threshold": processor.confidence_threshold,
        },
        "dataset": {
            "url": download.SACO_GOLD_URL,
            "sha256": download.SACO_GOLD_ARCHIVE.sha256,
        },
        "average": {
            name: sum(item["metrics"][name] for item in subsets.values()) / len(subsets)
            for name in AVERAGED_METRICS
        },
        "subsets": subsets,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
