import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from efficientsam3_jetson_benchmark import coco_runner, download, environment, latency
from efficientsam3_jetson_benchmark.download import HubFile, PinnedFile
from efficientsam3_jetson_benchmark.latency import LatencyRun
from efficientsam3_jetson_benchmark.timing import StageTimer

MODEL = "efficientsam3_tinyvit"


class FakeBackbone:
    def forward_image(self, image: object) -> dict[str, object]:
        return {"image": image}

    def forward_text(self, prompts: list[str], device: object = None) -> dict:
        return {"prompts": prompts}


class FakeModel:
    def __init__(self) -> None:
        self.backbone = FakeBackbone()

    def forward_grounding(self, **kwargs: object) -> dict[str, object]:
        return kwargs


class FakeProcessor:
    """Calls the model the way upstream's Sam3Processor does."""

    device = torch.device("cpu")

    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def set_image(self, image: object) -> dict[str, Any]:
        return {"backbone_out": self.model.backbone.forward_image(image)}

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        text = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text)
        self.model.forward_grounding(backbone_out=state["backbone_out"])
        state["scores"] = torch.tensor([0.9, 0.4])
        return state


class FakeSampler:
    def __init__(self) -> None:
        self.samples = [{"vdd_in_mw": 5000.0}, {"vdd_in_mw": 7000.0}]

    def __enter__(self) -> "FakeSampler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def latency_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LatencyRun:
    """A run whose checkpoint matches a patched pin and whose image exists."""
    weights = b"weights"
    pin = PinnedFile(len(weights), hashlib.sha256(weights).hexdigest())
    hub_file = HubFile("owner/repo", "a" * 40, f"full/{MODEL}.pt", pin)
    monkeypatch.setitem(download.CHECKPOINTS, MODEL, hub_file)
    checkpoint = download.checkpoint_path(MODEL, tmp_path / "checkpoints")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(weights)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 6), "white").save(image)
    return LatencyRun(
        model=MODEL,
        precision="fp32",
        prompt="person",
        image=image,
        warmup=2,
        iterations=3,
        checkpoints_root=tmp_path / "checkpoints",
        upstream=tmp_path / "upstream",
    )


def test_autocast_matches_the_precision() -> None:
    with latency.autocast("cpu", "fp32"):
        assert not torch.is_autocast_enabled("cpu")
    with latency.autocast("cpu", "bf16"):
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        assert torch.is_autocast_enabled("cpu")


def test_event_factory_only_on_cuda() -> None:
    assert latency.event_factory("cpu") is None
    assert latency.event_factory("mps") is None
    assert latency.event_factory("cuda") is not None


def test_synchronizer_per_device() -> None:
    assert latency.synchronizer("cuda") is torch.cuda.synchronize
    assert latency.synchronizer("mps") is torch.mps.synchronize
    assert latency.synchronizer("cpu")() is None


def test_stage_samples_split_each_iteration() -> None:
    samples = {
        "set_image": [10.0, 12.0],
        "image_encoder": [8.0, 9.0],
        "set_text_prompt": [20.0, 22.0],
        "text_encoder": [2.0, 3.0],
        "decoder": [15.0, 16.0],
    }

    stages = latency.stage_samples(samples)

    assert stages["preprocess"] == [2.0, 3.0]
    assert stages["postprocess"] == [3.0, 3.0]
    assert stages["decoder"] == [15.0, 16.0]


def test_instrument_times_each_upstream_call() -> None:
    model = FakeModel()
    processor = FakeProcessor(model)
    timer = StageTimer(synchronize=lambda: None, recording=True)

    latency.instrument(model, processor, timer)
    processor.set_text_prompt("cat", processor.set_image("image"))

    assert sorted(timer.samples) == [
        "decoder",
        "image_encoder",
        "set_image",
        "set_text_prompt",
        "text_encoder",
    ]


def test_measure_reports_stages_end_to_end_text_batches_and_energy(
    latency_run: LatencyRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        latency, "load_pipeline", lambda run, checkpoint: (model, FakeProcessor(model))
    )
    monkeypatch.setattr(latency, "TegrastatsSampler", FakeSampler)

    measured = latency.measure(latency_run)

    assert measured["device_type"] == "cpu"
    assert measured["timing_method"] == "synchronized_wall_clock"
    assert {stats["count"] for stats in measured["stages_ms"].values()} == {3}
    assert measured["end_to_end_ms"]["count"] == 3
    assert sorted(measured["text_encoder_batch_ms"]) == ["1", "16", "4"]
    assert measured["output"] == {"detections": 2, "top_score": pytest.approx(0.9)}
    assert measured["power"]["vdd_in_mean_mw"] == pytest.approx(6000.0)
    assert measured["energy_per_frame_mj"] == pytest.approx(
        6000.0 * measured["end_to_end_ms"]["mean_ms"] / 1000
    )


def test_measure_requires_a_verified_checkpoint(latency_run: LatencyRun) -> None:
    download.checkpoint_path(MODEL, latency_run.checkpoints_root).unlink()

    with pytest.raises(FileNotFoundError, match="download checkpoint"):
        latency.measure(latency_run)


def test_run_latency_writes_a_record(
    tmp_path: Path, latency_run: LatencyRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        environment,
        "describe",
        lambda repository, upstream: {"timestamp": "2026-10-12T08:00:00+00:00"},
    )
    monkeypatch.setattr(
        latency,
        "measure",
        lambda run: {
            "device_type": "cuda",
            "timing_method": "cuda_events",
            "end_to_end_ms": {"mean_ms": 50.0},
        },
    )
    monkeypatch.setattr(coco_runner, "device_name", lambda device_type: "Orin")

    path = latency.run_latency(latency_run, tmp_path / "results")

    record = json.loads(path.read_text())
    assert path.name == f"20261012T080000Z-latency-{MODEL}-fp32-cuda.json"
    assert record["model"]["variant"] == "11m"
    assert record["input"]["prompt"] == "person"
    assert record["settings"] == {
        "precision": "fp32",
        "warmup": 2,
        "iterations": 3,
        "timing_method": "cuda_events",
    }
    assert record["end_to_end_ms"] == {"mean_ms": 50.0}
    assert record["runtime"]["device_name"] == "Orin"
    assert "device_type" not in record


def test_load_pipeline_builds_on_the_processor_device(
    latency_run: LatencyRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = latency_run.upstream / "sam3" / "sam3"
    (package / "model").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "model" / "__init__.py").write_text("")
    (package / "model_builder.py").write_text(
        "class Model:\n"
        "    def to(self, device):\n"
        "        self.device = device\n"
        "def build_efficientsam3_image_model(**kwargs):\n"
        "    model = Model()\n"
        "    model.kwargs = kwargs\n"
        "    return model\n"
    )
    (package / "model" / "sam3_image_processor.py").write_text(
        "class Sam3Processor:\n"
        "    def __init__(self, model):\n"
        "        self.model = model\n"
        "        self.device = 'cpu'\n"
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name.split(".")[0] == "sam3"]:
        monkeypatch.delitem(sys.modules, name)  # restored after the test

    try:
        model, processor = latency.load_pipeline(latency_run, Path("weights.pt"))
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "sam3"]:
            del sys.modules[name]

    assert processor.model is model
    assert model.device == "cpu"
    assert model.kwargs["model_name"] == "11m"
    assert model.kwargs["text_encoder_context_length"] == 16
