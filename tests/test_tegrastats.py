import sys
import time

import pytest

from efficientsam3_jetson_benchmark import tegrastats
from efficientsam3_jetson_benchmark.tegrastats import TegrastatsSampler

# A line printed by tegrastats on a Jetson Orin Nano Super.
LINE = (
    "10-12-2026 07:48:26 RAM 5028/7620MB (lfb 32x1MB) SWAP 340/3810MB (cached 0MB) "
    "CPU [10%@1728,9%@1728,17%@1728,17%@1728,9%@1728,4%@1728] GR3D_FREQ 37% "
    "cpu@49.875C soc2@48.656C soc0@49.125C gpu@49.437C tj@49.875C soc1@48.562C "
    "VDD_IN 6615mW/6615mW VDD_CPU_GPU_CV 1861mW/1861mW VDD_SOC 2535mW/2535mW"
)


def test_parse_line_reads_power_memory_load_and_temperatures() -> None:
    assert tegrastats.parse_line(LINE) == {
        "ram_used_mb": 5028.0,
        "gr3d_percent": 37.0,
        "vdd_in_mw": 6615.0,
        "gpu_temperature_c": 49.437,
        "junction_temperature_c": 49.875,
    }


def test_parse_line_leaves_out_missing_readings() -> None:
    assert tegrastats.parse_line("RAM 100/200MB") == {"ram_used_mb": 100.0}


def test_summarize_samples_reports_means_and_peaks() -> None:
    samples = [
        {"vdd_in_mw": 6000.0, "gr3d_percent": 90.0, "ram_used_mb": 5000.0},
        {"vdd_in_mw": 8000.0, "gr3d_percent": 70.0, "ram_used_mb": 5200.0},
    ]

    summary = tegrastats.summarize_samples(samples)

    assert summary is not None
    assert summary["samples"] == 2
    assert summary["vdd_in_mean_mw"] == pytest.approx(7000.0)
    assert summary["vdd_in_peak_mw"] == 8000.0
    assert summary["gr3d_mean_percent"] == pytest.approx(80.0)
    assert summary["ram_used_peak_mb"] == 5200.0
    assert summary["gpu_temperature_peak_c"] is None


def test_summarize_samples_is_none_without_samples() -> None:
    assert tegrastats.summarize_samples([]) is None


def test_sampler_collects_lines_from_its_command() -> None:
    command = [
        sys.executable,
        "-c",
        f"import time; print({LINE!r}, flush=True); print('no reading', flush=True); "
        "time.sleep(30)",
    ]

    with TegrastatsSampler(command=command) as sampler:
        deadline = time.monotonic() + 10
        while not sampler.samples and time.monotonic() < deadline:
            time.sleep(0.01)

    assert sampler.samples == [tegrastats.parse_line(LINE)]


def test_sampler_does_nothing_without_tegrastats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tegrastats.shutil, "which", lambda name: None)

    with TegrastatsSampler() as sampler:
        pass

    assert sampler.command is None
    assert sampler.samples == []
