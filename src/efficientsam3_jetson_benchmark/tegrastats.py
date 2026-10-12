"""Sample NVIDIA tegrastats while a benchmark runs, and summarize the samples."""

import re
import shutil
import subprocess
import threading
from types import TracebackType
from typing import Any, TextIO

PATTERNS = {
    "ram_used_mb": re.compile(r"RAM (\d+)/\d+MB"),
    "gr3d_percent": re.compile(r"GR3D_FREQ (\d+)%"),
    "vdd_in_mw": re.compile(r"VDD_IN (\d+)mW"),
    "gpu_temperature_c": re.compile(r"\bgpu@([\d.]+)C"),
    "junction_temperature_c": re.compile(r"\btj@([\d.]+)C"),
}


def parse_line(line: str) -> dict[str, float]:
    """Return the readings tegrastats printed on one line; missing ones are left out."""
    readings = {}
    for name, pattern in PATTERNS.items():
        match = pattern.search(line)
        if match:
            readings[name] = float(match[1])
    return readings


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, Any] | None:
    """Return mean power and GPU load, and peak memory and temperatures."""
    if not samples:
        return None

    def values(name: str) -> list[float]:
        return [sample[name] for sample in samples if name in sample]

    def mean(name: str) -> float | None:
        found = values(name)
        return sum(found) / len(found) if found else None

    def peak(name: str) -> float | None:
        found = values(name)
        return max(found) if found else None

    return {
        "samples": len(samples),
        "vdd_in_mean_mw": mean("vdd_in_mw"),
        "vdd_in_peak_mw": peak("vdd_in_mw"),
        "gr3d_mean_percent": mean("gr3d_percent"),
        "ram_used_peak_mb": peak("ram_used_mb"),
        "gpu_temperature_peak_c": peak("gpu_temperature_c"),
        "junction_temperature_peak_c": peak("junction_temperature_c"),
    }


class TegrastatsSampler:
    """Collect tegrastats readings in the background for the duration of a block.

    Off a Jetson, where tegrastats is not installed, no samples are collected.
    """

    def __init__(
        self, interval_ms: int = 100, command: list[str] | None = None
    ) -> None:
        """Prepare a sampler; nothing runs until the block is entered."""
        if command is None and shutil.which("tegrastats"):
            command = ["tegrastats", "--interval", str(interval_ms)]
        self.command = command
        self.samples: list[dict[str, float]] = []
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    def __enter__(self) -> "TegrastatsSampler":
        """Start tegrastats and a thread that parses its output."""
        if self.command:
            self._process = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, text=True
            )
            self._reader = threading.Thread(
                target=self._read, args=(self._process.stdout,), daemon=True
            )
            self._reader.start()
        return self

    def _read(self, stream: TextIO) -> None:
        for line in stream:
            readings = parse_line(line)
            if readings:
                self.samples.append(readings)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop tegrastats and wait for the reader to finish."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
        if self._reader is not None:
            self._reader.join(timeout=5)
