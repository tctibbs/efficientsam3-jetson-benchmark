"""Run upstream eval/eval_coco.py unmodified and record what it measured.

Run with the upstream checkout as the working directory, because eval_coco.py
loads its tokenizer vocabulary from a relative path:

    python -m efficientsam3_jetson_benchmark.coco_runner SUMMARY -- ARGS...

ARGS go to eval_coco.py unchanged. The script is imported, not edited:
calculate_iou is wrapped to record every IoU, and the warnings upstream emits
when it skips mask post-processing are counted.
"""

import importlib.util
import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy
import torch

EVAL_COCO = Path("eval/eval_coco.py")
POSTPROCESS_SKIP = "Skipping the post-processing step"


@dataclass
class Recorder:
    """Per-instance IoUs and post-processing skips seen during one run."""

    ious: list[float] = field(default_factory=list)
    postprocess_skips: int = 0

    def instrument(self, eval_coco: Any) -> None:
        """Wrap the loaded eval_coco module's calculate_iou to record every IoU."""
        calculate_iou = eval_coco.calculate_iou

        def recording_calculate_iou(pred_mask: Any, gt_mask: Any) -> Any:
            iou = calculate_iou(pred_mask, gt_mask)
            self.ious.append(float(iou))
            return iou

        eval_coco.calculate_iou = recording_calculate_iou

    def run(self, entry_point: Callable[[], object]) -> None:
        """Call entry_point and count warnings that it skipped post-processing.

        Warnings are recorded during the call, then shown again, so the console
        output is unchanged apart from timing.
        """
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.filterwarnings("always", message=f"(?s).*{POSTPROCESS_SKIP}")
                entry_point()
        finally:
            for item in caught:
                warnings.showwarning(
                    item.message, item.category, item.filename, item.lineno
                )
            self.postprocess_skips += sum(
                POSTPROCESS_SKIP in str(item.message) for item in caught
            )


def load_eval_coco(path: Path = EVAL_COCO) -> ModuleType:
    """Import upstream's eval_coco.py from its file without running main()."""
    spec = importlib.util.spec_from_file_location("eval_coco", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def autocast_state(device_type: str) -> dict[str, Any]:
    """Return whether autocast is enabled for device_type, and at which dtype."""
    enabled = torch.is_autocast_enabled(device_type)
    dtype = str(torch.get_autocast_dtype(device_type)).removeprefix("torch.")
    return {"enabled": enabled, "dtype": dtype if enabled else None}


def device_name(device_type: str) -> str:
    """Return a readable name for the accelerator that ran the model."""
    if device_type == "cuda":
        return torch.cuda.get_device_name(0)
    if device_type == "mps":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    return platform.processor() or platform.machine()


def main(argv: list[str] | None = None) -> None:
    """Run eval_coco.py with the given arguments and write a JSON summary."""
    args = sys.argv[1:] if argv is None else argv
    summary_path, eval_args = Path(args[0]), args[1:]
    if eval_args[:1] == ["--"]:
        eval_args = eval_args[1:]

    recorder = Recorder()
    eval_coco = load_eval_coco()
    recorder.instrument(eval_coco)
    sys.argv = [str(EVAL_COCO), *eval_args]
    start = time.perf_counter()
    recorder.run(eval_coco.main)
    elapsed = time.perf_counter() - start

    device_type = eval_coco.get_device().type
    summary = {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "device_type": device_type,
        "device_name": device_name(device_type),
        "autocast": autocast_state(device_type),
        "ious": recorder.ious,
        "postprocess_skips": recorder.postprocess_skips,
        "wall_clock_s": round(elapsed, 2),
    }
    summary_path.write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
