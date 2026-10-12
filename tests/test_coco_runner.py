import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.version

from efficientsam3_jetson_benchmark import coco_runner

FAKE_EVAL_COCO = """
import sys
import warnings

import torch


def calculate_iou(pred_mask, gt_mask):
    return pred_mask / gt_mask


def get_device():
    return torch.device("cpu")


def main():
    count = int(sys.argv[sys.argv.index("--num_samples") + 1])
    for index in range(count):
        calculate_iou(index + 1, 4)
    warnings.warn(
        "kernel failed\\n\\nSkipping the post-processing step due to the error above.",
        UserWarning,
    )
"""


def _warn_skip() -> None:
    warnings.warn(
        f"error\n\n{coco_runner.POSTPROCESS_SKIP} due to the error above.",
        UserWarning,
        stacklevel=1,
    )


def test_recorder_records_every_iou() -> None:
    module = SimpleNamespace(
        calculate_iou=lambda pred_mask, gt_mask: pred_mask / gt_mask
    )
    recorder = coco_runner.Recorder()

    recorder.instrument(module)
    results = [module.calculate_iou(1, 4), module.calculate_iou(3, 4)]

    assert results == [0.25, 0.75]
    assert recorder.ious == [0.25, 0.75]


def test_recorder_counts_each_postprocess_skip_and_shows_warnings() -> None:
    recorder = coco_runner.Recorder()

    def entry_point() -> None:
        _warn_skip()
        _warn_skip()
        warnings.warn("unrelated", UserWarning, stacklevel=1)

    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        recorder.run(entry_point)

    assert recorder.postprocess_skips == 2
    assert len(shown) == 3


def test_recorder_shows_warnings_when_the_run_fails() -> None:
    recorder = coco_runner.Recorder()

    def failing_entry_point() -> None:
        _warn_skip()
        raise RuntimeError("model failed")

    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="model failed"):
            recorder.run(failing_entry_point)

    assert recorder.postprocess_skips == 1
    assert len(shown) == 1


def test_autocast_state_reports_enabled_dtype() -> None:
    assert coco_runner.autocast_state("cpu") == {"enabled": False, "dtype": None}

    with torch.autocast("cpu", dtype=torch.bfloat16):
        state = coco_runner.autocast_state("cpu")

    assert state == {"enabled": True, "dtype": "bfloat16"}


def test_device_name_for_each_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Orin")
    monkeypatch.setattr(
        coco_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Apple Chip\n"),
    )

    assert coco_runner.device_name("cuda") == "Orin"
    assert coco_runner.device_name("mps") == "Apple Chip"
    assert coco_runner.device_name("cpu")


@pytest.mark.parametrize("separator", [["--"], []])
def test_main_runs_eval_coco_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, separator: list[str]
) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "eval_coco.py").write_text(FAKE_EVAL_COCO)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["coco_runner"])
    summary_path = tmp_path / "summary.json"

    with pytest.warns(UserWarning, match=coco_runner.POSTPROCESS_SKIP):
        coco_runner.main([str(summary_path), *separator, "--num_samples", "3"])

    summary = json.loads(summary_path.read_text())
    assert summary["ious"] == [0.25, 0.5, 0.75]
    assert summary["postprocess_skips"] == 1
    assert summary["device_type"] == "cpu"
    assert summary["autocast"] == {"enabled": False, "dtype": None}
    assert summary["torch"] == torch.__version__
    assert summary["cuda"] == torch.version.cuda
    assert "cudnn" in summary
