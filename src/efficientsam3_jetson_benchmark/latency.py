"""PyTorch latency of EfficientSAM3's text-prompt pipeline, stage by stage.

Upstream's public flow runs unmodified: Sam3Processor.set_image, then
set_text_prompt. The model calls inside it are wrapped with a stage timer, so
the stages are the ones upstream defines: image encoder
(backbone.forward_image), text encoder (backbone.forward_text), and decoder
(forward_grounding). Preprocessing and postprocessing are the rest of
set_image and set_text_prompt.
"""

import contextlib
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from efficientsam3_jetson_benchmark import download, environment
from efficientsam3_jetson_benchmark.tegrastats import (
    TegrastatsSampler,
    summarize_samples,
)
from efficientsam3_jetson_benchmark.timing import StageTimer, summarize

FULL_MODELS = {
    "efficientsam3_efficientvit": ("efficientvit", "b1"),
    "efficientsam3_repvit": ("repvit", "m1.1"),
    "efficientsam3_tinyvit": ("tinyvit", "11m"),
}
TEXT_ENCODER = "MobileCLIP-S0"
TEXT_CONTEXT_LENGTH = 16
PRECISIONS = ("fp32", "fp16", "bf16")
TEXT_BATCH_SIZES = (1, 4, 16)
TEXT_WARMUP = 10


@dataclass(frozen=True)
class LatencyRun:
    """Inputs of one latency measurement."""

    model: str
    precision: str
    prompt: str
    image: Path
    warmup: int
    iterations: int
    checkpoints_root: Path
    upstream: Path


def autocast(device_type: str, precision: str) -> contextlib.AbstractContextManager:
    """Return the autocast context for a precision; fp32 runs without autocast."""
    import torch

    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def synchronizer(device_type: str) -> Callable[[], None]:
    """Return a function that waits for queued work on the device."""
    import torch

    if device_type == "cuda":
        return torch.cuda.synchronize
    if device_type == "mps":
        return torch.mps.synchronize
    return lambda: None


def load_pipeline(run: LatencyRun, checkpoint: Path) -> tuple[Any, Any]:
    """Build the model and upstream's processor, with both on the same device."""
    # Upstream is importable only once its package directory is on sys.path.
    sys.path.insert(0, str((run.upstream / "sam3").resolve()))
    builder = importlib.import_module("sam3.model_builder")
    processing = importlib.import_module("sam3.model.sam3_image_processor")

    backbone, variant = FULL_MODELS[run.model]
    model = builder.build_efficientsam3_image_model(
        checkpoint_path=str(checkpoint),
        backbone_type=backbone,
        model_name=variant,
        text_encoder_type=TEXT_ENCODER,
        text_encoder_context_length=TEXT_CONTEXT_LENGTH,
        load_from_HF=False,
    )
    processor = processing.Sam3Processor(model)
    # The builder moves the model only for CUDA; the processor picks CUDA or MPS.
    model.to(processor.device)
    return model, processor


def instrument(model: Any, processor: Any, timer: StageTimer) -> None:
    """Wrap the upstream calls whose durations make up the pipeline."""
    backbone = model.backbone
    backbone.forward_image = timer.wrap("image_encoder", backbone.forward_image)
    backbone.forward_text = timer.wrap("text_encoder", backbone.forward_text)
    model.forward_grounding = timer.wrap("decoder", model.forward_grounding)
    processor.set_image = timer.wrap("set_image", processor.set_image)
    processor.set_text_prompt = timer.wrap("set_text_prompt", processor.set_text_prompt)


def stage_samples(samples: dict[str, list[float]]) -> dict[str, list[float]]:
    """Split each recorded iteration into the five pipeline stages."""
    return {
        "preprocess": [
            total - encoder
            for total, encoder in zip(
                samples["set_image"], samples["image_encoder"], strict=True
            )
        ],
        "image_encoder": samples["image_encoder"],
        "text_encoder": samples["text_encoder"],
        "decoder": samples["decoder"],
        "postprocess": [
            total - text - decoder
            for total, text, decoder in zip(
                samples["set_text_prompt"],
                samples["text_encoder"],
                samples["decoder"],
                strict=True,
            )
        ],
    }


def text_encoder_latency(
    model: Any, processor: Any, run: LatencyRun, timer: StageTimer, size: int
) -> dict[str, float]:
    """Time the text encoder alone on a batch of size copies of the prompt."""
    import torch

    prompts = [run.prompt] * size

    def encode() -> None:
        with torch.inference_mode(), autocast(processor.device.type, run.precision):
            model.backbone.forward_text(prompts, device=processor.device)

    timer.time_calls(encode, TEXT_WARMUP)
    return summarize(timer.time_calls(encode, run.iterations)).as_record()


def measure(run: LatencyRun) -> dict[str, Any]:
    """Load the model and measure every stage, end to end, and the text encoder.

    Raises:
        FileNotFoundError: The checkpoint is missing or unverified.
    """
    from PIL import Image

    hub_file = download.CHECKPOINTS[run.model]
    checkpoint = download.checkpoint_path(run.model, run.checkpoints_root)
    if not download.matches(checkpoint, hub_file.pin):
        raise FileNotFoundError(
            f"{checkpoint} is missing or unverified; run "
            f"`efficientsam3-jetson-benchmark download checkpoint {run.model}`"
        )
    model, processor = load_pipeline(run, checkpoint)
    device_type = processor.device.type
    timer = StageTimer(synchronizer(device_type))
    instrument(model, processor, timer)
    image = Image.open(run.image).convert("RGB")
    outputs: dict[str, Any] = {}

    def step() -> None:
        with autocast(device_type, run.precision):
            state = processor.set_image(image)
            outputs["state"] = processor.set_text_prompt(run.prompt, state)

    timer.time_calls(step, run.warmup)
    timer.recording = True
    timer.time_calls(step, run.iterations)
    timer.recording = False
    with TegrastatsSampler() as sampler:
        end_to_end = summarize(timer.time_calls(step, run.iterations)).as_record()
    power = summarize_samples(sampler.samples)

    scores = outputs["state"]["scores"]
    return {
        "device_type": device_type,
        "stages_ms": {
            name: summarize(values).as_record()
            for name, values in stage_samples(timer.samples).items()
        },
        "end_to_end_ms": end_to_end,
        "text_encoder_batch_ms": {
            str(size): text_encoder_latency(model, processor, run, timer, size)
            for size in TEXT_BATCH_SIZES
        },
        "output": {
            "detections": len(scores),
            "top_score": float(scores.max()) if len(scores) else None,
        },
        "power": power,
        "energy_per_frame_mj": power["vdd_in_mean_mw"] * end_to_end["mean_ms"] / 1000
        if power and power["vdd_in_mean_mw"] is not None
        else None,
    }


def run_latency(run: LatencyRun, results_dir: Path) -> Path:
    """Measure one model and precision, and write a results record.

    Returns:
        Path of the results JSON.
    """
    from efficientsam3_jetson_benchmark.coco_runner import device_name

    context = environment.describe(Path.cwd(), run.upstream)
    measured = measure(run)
    hub_file = download.CHECKPOINTS[run.model]
    backbone, variant = FULL_MODELS[run.model]
    stamp = datetime.fromisoformat(context["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    device_type = measured.pop("device_type")
    run_id = f"{stamp}-latency-{run.model}-{run.precision}-{device_type}"
    record = {
        "run_id": run_id,
        "benchmark": "pytorch-text-prompt-latency",
        "environment": context,
        "runtime": runtime_record(device_type, device_name(device_type)),
        "model": {
            "name": run.model,
            "backbone": backbone,
            "variant": variant,
            "text_encoder": TEXT_ENCODER,
            "text_context_length": TEXT_CONTEXT_LENGTH,
            "repo_id": hub_file.repo_id,
            "revision": hub_file.revision,
            "filename": hub_file.filename,
            "sha256": hub_file.pin.sha256,
        },
        "input": {
            "image": run.image.name,
            "image_sha256": download.sha256_of(run.image),
            "prompt": run.prompt,
        },
        "settings": {
            "precision": run.precision,
            "warmup": run.warmup,
            "iterations": run.iterations,
        },
        **measured,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def runtime_record(device_type: str, name: str) -> dict[str, Any]:
    """Return the torch, CUDA, and cuDNN versions and the device that ran the model."""
    import torch
    import torch.version

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "device_type": device_type,
        "device_name": name,
    }
