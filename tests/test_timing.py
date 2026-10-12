import numpy
import pytest

from efficientsam3_jetson_benchmark import timing
from efficientsam3_jetson_benchmark.timing import StageTimer


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_percentile_matches_numpy(fraction: float) -> None:
    values = sorted([12.0, 3.5, 7.25, 9.0, 1.0, 30.0, 4.0])

    assert timing.percentile(values, fraction) == pytest.approx(
        numpy.percentile(values, fraction * 100)
    )


def test_summarize_reports_spread_and_fps() -> None:
    stats = timing.summarize([10.0, 20.0, 30.0, 40.0])

    record = stats.as_record()
    assert (stats.count, stats.mean_ms, stats.min_ms, stats.max_ms) == (4, 25, 10, 40)
    assert stats.p50_ms == pytest.approx(25.0)
    assert record["fps"] == pytest.approx(40.0)


def test_summarize_rejects_no_samples() -> None:
    with pytest.raises(ValueError, match="no latency samples"):
        timing.summarize([])


def test_stage_timer_records_only_while_recording() -> None:
    syncs = []
    timer = StageTimer(synchronize=lambda: syncs.append(1))
    add = timer.wrap("add", lambda left, right: left + right)

    assert add(1, 2) == 3
    timer.recording = True
    assert add(2, 3) == 5

    assert len(timer.samples["add"]) == 1
    assert len(syncs) == 2


def test_time_calls_times_each_call() -> None:
    calls = []
    timer = StageTimer(synchronize=lambda: None)

    durations = timer.time_calls(lambda: calls.append(1), iterations=5)

    assert len(durations) == len(calls) == 5
    assert all(duration >= 0 for duration in durations)


class FakeEvent:
    """A timing event whose clock advances by one millisecond per record."""

    clock = 0

    def record(self) -> None:
        FakeEvent.clock += 1
        self.at = FakeEvent.clock

    def elapsed_time(self, end: "FakeEvent") -> float:
        return float(end.at - self.at)


def test_events_time_stages_without_synchronizing_each_one() -> None:
    syncs = []
    timer = StageTimer(
        synchronize=lambda: syncs.append(1), make_event=FakeEvent, recording=True
    )
    encode = timer.wrap("encode", lambda: None)
    decode = timer.wrap("decode", lambda: None)

    durations = timer.time_calls(lambda: (encode(), decode()), iterations=3)

    assert durations == [5.0, 5.0, 5.0]
    assert timer.samples == {"encode": [1.0] * 3, "decode": [1.0] * 3}
    assert len(syncs) == 6
