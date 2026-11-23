"""Tracer tests: a kettle in a gridded residual is detected end to end (D-09)."""

import math

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.online import DeviceModel, OnlineDetector
from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.intervals import EventType
from custom_components.powerguesser.engine.learning.episodes import (
    detect_edges,
    pair_switched,
    suppress_alignment_artefacts,
)
from tests.fixtures.detect_traces import add_run, grid_trace, switched_artefact

GRID_S = 30
THRESHOLD = 120.0


def _offline_table(r: np.ndarray):
    idx, delta = detect_edges(r, THRESHOLD)
    idx, delta = suppress_alignment_artefacts(
        idx, delta, enabled_event_types=[EventType.STEP]
    )
    return pair_switched(r, idx, delta, float(GRID_S))


def _drive(det: OnlineDetector, r: np.ndarray):
    return [det.step(float(i) * GRID_S, float(r[i])) for i in range(len(r))]


def test_kettle_on_off_and_energy() -> None:
    """2000 W kettle: ON one sample after the rise, OFF at the fall, exact energy."""
    r = grid_trace(200, noise_w=2.0)
    add_run(r, start_i=20, n_samples=6, w=2000.0)
    model = DeviceModel.from_artefact("kettle", switched_artefact("kettle"))
    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    results = _drive(det, r)

    on_steps = [i for i, res in enumerate(results) if "kettle" in res.on]
    assert on_steps[0] == 22  # rise pre-edge sample 20, confirmed one sample later
    assert all("kettle" in results[i].on for i in range(on_steps[0], 28))
    assert "kettle" not in results[28].on  # fall pre-edge sample 26, OFF one sample later

    total_wh = sum(res.energy_wh_delta.get("kettle", 0.0) for res in results)
    expected_wh = 2000.0 * 6 * GRID_S / 3600.0
    assert abs(total_wh - expected_wh) / expected_wh <= 0.02

    table = _offline_table(r)
    offline_wh = float(np.sum(table["energy_wh"]))
    assert offline_wh > 0.0
    assert abs(total_wh - offline_wh) / offline_wh <= 0.02


def test_measured_step_inside_box_else_p50() -> None:
    """D-06: measured on-edge step inside the reference box, else the box p50."""
    # Inside the box: the measured step is reported.
    r = grid_trace(60, noise_w=0.0)
    add_run(r, start_i=10, n_samples=4, w=1950.0)
    model = DeviceModel.from_artefact("k", switched_artefact("k"))
    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    results = _drive(det, r)
    on_res = next(res for res in results if "k" in res.on)
    assert on_res.device_w["k"] == 1950.0

    # Outside the nearest-p50 state's box: the box p50 bounds the power.
    # A 1050 W rise lands inside the wide state's box only; the low state is
    # the nearest-p50 reference, so D-06 falls back to its p50 (975 W).
    arte = switched_artefact(
        "two",
        rise=(1000.0, 2000.0, 3000.0),
        fall=(-3000.0, -2000.0, -1000.0),
        dur=(60.0, 120.0, 300.0),
        plateau=(1000.0, 2000.0, 3000.0),
    )
    arte["states"].append(
        {
            "name": "low",
            "rise_w": {"lo": 900.0, "p50": 975.0, "hi": 1000.0},
            "fall_w": {"lo": -1100.0, "p50": -1000.0, "hi": -900.0},
            "dur_s": {"lo": 60.0, "p50": 120.0, "hi": 300.0},
            "plateau_w": {"lo": 900.0, "p50": 975.0, "hi": 1000.0},
            "phase_share": None,
        }
    )
    r2 = grid_trace(60, noise_w=0.0)
    add_run(r2, start_i=10, n_samples=4, w=1050.0)
    model2 = DeviceModel.from_artefact("two", arte)
    det2 = OnlineDetector([model2], grid_s=GRID_S, threshold_w=THRESHOLD)
    results2 = _drive(det2, r2)
    on_res2 = next(res for res in results2 if "two" in res.on)
    assert on_res2.device_w["two"] == 975.0  # low-state p50, not the 1050 W measured step


def test_replay_online_returns_the_episode() -> None:
    """replay_online reports the same episode indices as pair_switched."""
    r = grid_trace(120, noise_w=5.0)
    add_run(r, start_i=30, n_samples=6, w=2000.0)
    arte = switched_artefact("kettle")
    replay = replay_online([arte], r, float(GRID_S))

    table = _offline_table(r)
    assert len(table["start_i"]) == 1
    assert replay.episodes["kettle"][0].start_i == int(table["start_i"][0])
    assert replay.episodes["kettle"][0].end_i == int(table["end_i"][0])
    assert abs(replay.episodes["kettle"][0].energy_wh - float(table["energy_wh"][0])) / float(
        table["energy_wh"][0]
    ) <= 0.02
    assert replay.unverified == 0


def test_replay_online_collect_power() -> None:
    """collect_power=True records the per-step device power and committed energy.

    power_w[sig] equals the StepResult device_w sequence a direct detector
    drive reports; the per-step energy sums to the detector's committed
    energy_wh. With the default False both dicts stay empty (Plan 05-02).
    """
    r = grid_trace(200, noise_w=2.0)
    add_run(r, start_i=20, n_samples=6, w=2000.0)
    arte = switched_artefact("kettle")

    replay = replay_online([arte], r, float(GRID_S), collect_power=True)
    sig = arte["signature_id"]
    expected_power = np.zeros(len(r))
    expected_energy = np.zeros(len(r))
    model = DeviceModel.from_artefact(sig, arte)
    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    for i in range(len(r)):
        res = det.step(float(i) * GRID_S, float(r[i]))
        expected_power[i] = res.device_w.get(sig, 0.0)
        expected_energy[i] = res.energy_wh_delta.get(sig, 0.0)

    np.testing.assert_array_equal(replay.power_w[sig], expected_power)
    assert float(np.sum(replay.energy_step_wh[sig])) == pytest.approx(
        replay.energy_wh[sig], rel=1e-9
    )
    # ON inside the run, OFF outside it
    assert replay.power_w[sig][25] == pytest.approx(2000.0, rel=0.02)
    assert replay.power_w[sig][0] == 0.0

    default_replay = replay_online([arte], r, float(GRID_S))
    assert default_replay.power_w == {}
    assert default_replay.energy_step_wh == {}


def test_closed_log_bounded_and_drained() -> None:
    """1000 closed episodes leave the closed log bounded and drain_closed carries
    verified and unverified closes with their scores (D-15, RESEARCH Pitfall 7)."""
    from custom_components.powerguesser.engine.detect.online import CLOSED_LOG_MAX

    model = DeviceModel.from_artefact("kettle", switched_artefact("kettle"))
    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    n_cycles = 500
    r = grid_trace(n_cycles * 32 + 10, noise_w=2.0)
    i = 5
    while i + 40 < len(r):
        add_run(r, i, 6, 2000.0)  # verified close (dur 180 s)
        i += 12
        add_run(r, i, 3, 2000.0)  # unverified close (dur 90 s < 120)
        i += 20
    _drive(det, r)
    det.flush()

    closed = det.drain_closed()
    assert 0 < len(closed) <= CLOSED_LOG_MAX
    assert len(det.drain_closed()) == 0  # a drain clears the log
    verified = [ep for ep in closed if ep.verified]
    unverified = [ep for ep in closed if not ep.verified]
    assert verified and unverified  # both kinds carry their scores
    assert all(ep.key == "kettle" for ep in closed)
    assert all(0.0 <= ep.match_score <= 1.0 for ep in closed)
    assert all(ep.match_score == pytest.approx(1.0, abs=0.1) for ep in verified)
    assert all(math.isfinite(ep.end_ts) for ep in closed)
    # The verified-episode log (replay semantics) still holds the verified closes.
    assert det.drain_episodes()["kettle"]


def test_unattributed_activity_never_enters_drift_window() -> None:
    """Refused rises near the box and an ON run that never close record no
    ClosedEpisode; a fall attributed to a runtime with state_index -1 (restored
    mid-episode without one) records none either; the next normal close carries
    the whole refused burst once, in its cumulative count (D-15, review R1)."""
    from custom_components.powerguesser.engine.validation.drift import LiveDrift

    kettle = DeviceModel.from_artefact("kettle", switched_artefact("kettle"))
    heater = DeviceModel.from_artefact(
        "heater",
        switched_artefact(
            "heater",
            rise=(1950.0, 2050.0, 2150.0),
            fall=(-2150.0, -2050.0, -1950.0),
            plateau=(1950.0, 2050.0, 2150.0),
        ),
    )
    det = OnlineDetector([kettle, heater], grid_s=GRID_S, threshold_w=THRESHOLD)
    r = grid_trace(600, noise_w=2.0)
    add_run(r, 20, 3, 2040.0)  # inside BOTH boxes -> refused burst, no episode
    add_run(r, 60, 3, 2040.0)  # second refused burst
    add_run(r, 100, 6, 1920.0)  # kettle-only episode, closes verified
    r[200:] += 1920.0  # ON run that never closes before the trace ends
    _drive(det, r)
    det.flush()

    closed = det.drain_closed()
    assert len(closed) == 1  # only the normal kettle close
    ep = closed[0]
    assert ep.key == "kettle"
    assert ep.verified is True
    assert ep.match_score == pytest.approx(0.2, abs=0.05)  # 1920 vs the 2000 W p50 (half 100)
    assert ep.unattributed == 2  # the whole burst, counted once, cumulatively
    assert det.drain_closed() == []

    # A runtime restored mid-episode WITHOUT a known state closes with nothing:
    # no fallback score can enter a drift window.
    det2 = OnlineDetector([kettle], grid_s=GRID_S, threshold_w=THRESHOLD)
    det2.restore(
        {
            "grid_s": GRID_S,
            "last_grid_ts": 30.0,
            "devices": {
                "kettle": {
                    "on": True,
                    "unknown": False,
                    "state_index": -1,
                    "on_w": 2000.0,
                    "edge_ts": 30.0,
                    "attributed_seq": 1,
                    "unattributed": 0,
                    "unverified": 0,
                }
            },
            "energy_wh": {"kettle": 0.0},
        },
        now_ts=60.0,
    )
    det2.step(60.0, 2250.0)  # seed the fresh edge stream at the ON level
    det2.step(90.0, 250.0)  # the fall attributes to the restored runtime
    det2.step(120.0, 250.0)
    assert det2.drain_closed() == []
    assert det2._runtimes["kettle"].unverified == 1

    # Fed into LiveDrift the burst lands in exactly one window entry with the
    # refused delta equal to the burst and no entry at a 0.0 fallback score
    # (the delta is relative to the previously recorded close).
    tracker = LiveDrift()
    tracker.record(ep.end_ts - 30.0, 1.0, 0)  # the previous close
    tracker.record(ep.end_ts, ep.match_score, ep.unattributed)
    evidence = tracker.to_dict()["evidence"]
    assert len(evidence) == 2
    assert evidence[-1]["refused"] == 2.0
    assert evidence[-1]["score"] == pytest.approx(0.2, abs=0.05)
    assert evidence[-1]["score"] != 0.0
