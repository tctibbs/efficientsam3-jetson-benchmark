"""Tier A accuracy: upstream's COCO box-prompt mIoU, measured by eval_coco.py.

eval_coco.py prompts the model with each non-crowd ground-truth box and reports
the mean IoU of the predicted masks. Its timing is wall clock including model
load and image I/O, not latency.
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from efficientsam3_jetson_benchmark import download, environment

DOWNLOAD_COMMAND = "efficientsam3-jetson-benchmark download"


class IncompleteRunError(RuntimeError):
    """eval_coco.py skipped work, so its score is not comparable."""


@dataclass(frozen=True)
class CocoAccuracyRun:
    """Inputs of one eval_coco.py run on one checkpoint."""

    checkpoint: str
    first: int
    checkpoints_root: Path
    coco_root: Path
    upstream: Path


def expected_instances(annotations: Path, image_ids: list[int]) -> int:
    """Count the non-crowd annotations eval_coco.py scores for these images."""
    wanted = set(image_ids)
    data = json.loads(annotations.read_text())
    return sum(
        1
        for annotation in data["annotations"]
        if annotation["image_id"] in wanted and not annotation["iscrowd"]
    )


def verified_input(path: Path, pin: download.PinnedFile, fix: str) -> Path:
    """Return path if it matches its pin; otherwise raise with the fixing command.

    Raises:
        FileNotFoundError: The file is missing or differs from its pin.
    """
    if not download.matches(path, pin):
        raise FileNotFoundError(f"{path} is missing or unverified; run `{fix}`")
    return path


def run_eval_coco(
    checkpoint: Path, run: CocoAccuracyRun, summary_path: Path
) -> dict[str, Any]:
    """Run eval_coco.py on one checkpoint in a child process; return its summary."""
    with tempfile.TemporaryDirectory() as models_dir:
        # eval_coco.py scores every efficient_sam3*.pt in --output_dir; link only one.
        (Path(models_dir) / checkpoint.name).symlink_to(checkpoint.resolve())
        command = [
            sys.executable,
            "-m",
            "efficientsam3_jetson_benchmark.coco_runner",
            str(summary_path),
            "--",
            "--coco_root",
            str(run.coco_root.resolve()),
            "--output_dir",
            models_dir,
            "--num_samples",
            str(run.first),
        ]
        child_environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
        subprocess.run(command, cwd=run.upstream, env=child_environment, check=True)
    return json.loads(summary_path.read_text())


def check_complete(summary: dict[str, Any], expected: int) -> None:
    """Raise if eval_coco.py skipped instances or mask post-processing.

    Raises:
        IncompleteRunError: The run is not comparable.
    """
    scored = len(summary["ious"])
    if scored != expected:
        raise IncompleteRunError(
            f"scored {scored} of {expected} instances: upstream skipped images "
            "or failed to load the model"
        )
    skips = summary["postprocess_skips"]
    if skips:
        raise IncompleteRunError(f"upstream skipped mask post-processing {skips} times")


def run_coco_accuracy(run: CocoAccuracyRun, results_dir: Path) -> Path:
    """Run Tier A on one checkpoint and write a results record.

    Returns:
        Path of the results JSON.

    Raises:
        FileNotFoundError: The checkpoint or annotations are missing or unverified.
        IncompleteRunError: eval_coco.py skipped instances or post-processing.
    """
    hub_file = download.CHECKPOINTS[run.checkpoint]
    checkpoint = verified_input(
        download.checkpoint_path(run.checkpoint, run.checkpoints_root),
        hub_file.pin,
        f"{DOWNLOAD_COMMAND} checkpoint {run.checkpoint}",
    )
    annotations = verified_input(
        run.coco_root / download.COCO_ANNOTATIONS_MEMBER,
        download.COCO_VAL_ANNOTATIONS,
        f"{DOWNLOAD_COMMAND} coco --first {run.first}",
    )
    image_ids = download.first_image_ids(annotations, run.first)
    expected = expected_instances(annotations, image_ids)
    context = environment.describe(Path.cwd(), run.upstream)
    with tempfile.TemporaryDirectory() as scratch:
        summary = run_eval_coco(checkpoint, run, Path(scratch) / "summary.json")
    check_complete(summary, expected)

    stamp = datetime.fromisoformat(context["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-coco-{run.checkpoint}-first{run.first}-{summary['device_type']}"
    ious = summary["ious"]
    record = {
        "run_id": run_id,
        "benchmark": "coco-val2017-box-prompt-miou",
        "environment": context,
        "runtime": {
            key: summary[key]
            for key in ("torch", "numpy", "device_type", "device_name", "autocast")
        },
        "checkpoint": {
            "name": run.checkpoint,
            "repo_id": hub_file.repo_id,
            "revision": hub_file.revision,
            "filename": hub_file.filename,
            "sha256": hub_file.pin.sha256,
        },
        "dataset": {
            "split": "val2017",
            "annotations_sha256": download.COCO_VAL_ANNOTATIONS.sha256,
            "first": run.first,
            "image_ids": image_ids,
        },
        "metrics": {
            "miou": statistics.fmean(ious) if ious else None,
            "instances": len(ious),
        },
        "wall_clock_s_including_model_load": summary["wall_clock_s"],
        "ious": [round(iou, 6) for iou in ious],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
