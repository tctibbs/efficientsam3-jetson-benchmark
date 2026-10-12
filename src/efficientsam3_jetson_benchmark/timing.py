"""Latency statistics and per-stage timing around device synchronization."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LatencyStats:
    """Summary of repeated latency measurements in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def as_record(self) -> dict[str, float]:
        """Return the statistics, plus frames per second from the mean."""
        return {**asdict(self), "fps": 1000 / self.mean_ms}


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return a percentile of sorted values, interpolating linearly between ranks."""
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(durations_ms: list[float]) -> LatencyStats:
    """Summarize latency samples.

    Raises:
        ValueError: There are no samples.
    """
    if not durations_ms:
        raise ValueError("no latency samples")
    ordered = sorted(durations_ms)
    return LatencyStats(
        count=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=percentile(ordered, 0.50),
        p90_ms=percentile(ordered, 0.90),
        p99_ms=percentile(ordered, 0.99),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


@dataclass
class StageTimer:
    """Times named stages, synchronizing the device so queued GPU work counts.

    While recording is off, wrapped stages run without synchronizing, so an
    end-to-end measurement is not slowed by per-stage synchronization.
    """

    synchronize: Callable[[], None]
    recording: bool = False
    samples: dict[str, list[float]] = field(default_factory=dict)

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        """Return function wrapped so each call is timed as stage name."""

        def timed(*args: Any, **kwargs: Any) -> Any:
            if not self.recording:
                return function(*args, **kwargs)
            self.synchronize()
            start = time.perf_counter_ns()
            result = function(*args, **kwargs)
            self.synchronize()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            self.samples.setdefault(name, []).append(elapsed_ms)
            return result

        return timed

    def time_calls(self, step: Callable[[], object], iterations: int) -> list[float]:
        """Return the wall-clock duration of each call to step, in milliseconds."""
        durations = []
        for _ in range(iterations):
            self.synchronize()
            start = time.perf_counter_ns()
            step()
            self.synchronize()
            durations.append((time.perf_counter_ns() - start) / 1e6)
        return durations
