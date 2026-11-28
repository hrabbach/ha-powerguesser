"""Engine tests for the D-12 earlier-runs replay count (CAP-11, D-12).

Plan 09-02 Task 2: ``earlier_runs`` fits the capture's own single-run D-02
tolerance artefact, replays it over a bounded lookback window ending at the
captured rise and reports the count with the span it actually covered.
"""

from custom_components.powerguesser.engine.capture import earlier_runs as earlier_mod
from custom_components.powerguesser.engine.capture.config import CaptureConfig
from tests.fixtures.learning.synth import SynthSpec, build_buffer

END_MS = 1_700_000_000_000
GRID_S = 30


def _context(buffer_path: str) -> dict:
    return {
        "buffer_path": buffer_path,
        "grid_s": GRID_S,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "session_windows": [],
        "tz_name": "UTC",
    }


def _capture(on_ms: int, off_ms: int, *, rise_w: float = 2000.0) -> dict:
    dur_s = (off_ms - on_ms) / 1000.0
    return {
        "capture_id": "c1",
        "result": {
            "verdict": "clean",
            "rise_ms": on_ms,
            "fall_ms": off_ms,
            "step_w": rise_w,
            "fall_w": -rise_w,
            "plateau_w": rise_w,
            "dur_s": dur_s,
            "energy_wh": 100.0,
            "grid_s": float(GRID_S),
            "threshold_w": 120.0,
        },
        "features": {
            "rise_w": rise_w,
            "fall_w": -rise_w,
            "dur_s": dur_s,
            "plateau_w": rise_w,
            "energy_wh": 100.0,
            "hour": 7,
            "dow": 0,
            "l1_share": None,
            "l2_share": None,
            "l3_share": None,
        },
    }


def _covered_days(basis: str) -> float:
    return float(basis.split(" on the last ")[1].split(" ")[0])


def test_earlier_runs_counts_prior_kettle_runs(tmp_path) -> None:
    """n covers the prior kettle runs in the lookback window, never the captured run."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=10, interval_s=(30.0, 30.0)), END_MS)
    on_ms, off_ms = data.kettle_runs[-1]

    out = earlier_mod.earlier_runs(_context(path), _capture(on_ms, off_ms))
    lookback = 28 * 86_400_000
    prior = [
        on
        for on, _off in data.kettle_runs
        if on != on_ms and on_ms - lookback <= on < on_ms
    ]
    assert 0.9 * len(prior) <= out["n"] <= len(prior)
    assert isinstance(out["n"], int)


def test_earlier_runs_lookback_is_bounded(tmp_path, monkeypatch) -> None:
    """The buffer read is bounded to the lookback plus one grid step of alignment."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=10, interval_s=(30.0, 30.0)), END_MS)
    on_ms, off_ms = data.kettle_runs[-1]
    capture = _capture(on_ms, off_ms)
    context = _context(path)
    step = GRID_S * 1000

    calls: list[list[int]] = []
    real = earlier_mod._load_job_residual

    def spy(job):
        calls.append(list(job["day_bounds_ms"]))
        return real(job)

    monkeypatch.setattr(earlier_mod, "_load_job_residual", spy)

    out2 = earlier_mod.earlier_runs(
        context, capture, config=CaptureConfig(earlier_runs_lookback_days=2)
    )
    assert len(calls) == 1
    start, end = calls[0]
    assert end == (int(on_ms) // step) * step
    assert end - start <= 2 * 86_400_000 + step
    assert 1.0 <= _covered_days(out2["basis"]) <= 2.5

    calls.clear()
    out28 = earlier_mod.earlier_runs(context, capture)
    assert len(calls) == 1
    start, end = calls[0]
    assert end - start <= 28 * 86_400_000 + step
    # A buffer shorter than the lookback names the span actually covered.
    assert 8.0 <= _covered_days(out28["basis"]) < 28.0
