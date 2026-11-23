"""Attribution rules D-06..D-09, D-28..D-31 pinned by named tests."""

import json
import math

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.online import (
    GAP_HOLD_STEPS,
    LOW_RESIDUAL_STEPS,
    MAX_PAIR_S,
    AMBIGUITY_MARGIN,
    DeviceModel,
    OnlineDetector,
    StepResult,
    state_keys,
)
from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.intervals import EventType
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact
from custom_components.powerguesser.engine.learning.episodes import (
    compute_features,
    detect_edges,
    pair_switched,
    suppress_alignment_artefacts,
)
from custom_components.powerguesser.engine.learning.score import replay as score_replay
from tests.fixtures.detect_traces import add_run, grid_trace, switched_artefact

GRID_S = 30
THRESHOLD = 120.0
BASE = 250.0


def _model(arte: dict, key: str | None = None) -> DeviceModel:
    return DeviceModel.from_artefact(key or str(arte["signature_id"]), arte)


def _det(artes, *, grid_s: int = GRID_S, unknown=(), threshold: float = THRESHOLD):
    models = [a if isinstance(a, DeviceModel) else _model(a) for a in artes]
    return OnlineDetector(
        models, grid_s=grid_s, threshold_w=threshold, suppress_alignment=True, unknown=unknown
    )


def _drive(det: OnlineDetector, r: np.ndarray, grid_s: int = GRID_S) -> list[StepResult]:
    return [det.step(float(i) * grid_s, float(r[i])) for i in range(len(r))]


def _on_run(r: np.ndarray, start_i: int, n: int, w: float) -> None:
    add_run(r, start_i, n, w)


def _assert_unknown(res: StepResult, det: OnlineDetector, key: str) -> None:
    assert key in res.unknown
    assert key in det.unknown
    assert key not in res.on
    assert res.device_w.get(key, 0.0) == 0.0
    assert res.energy_wh_delta.get(key, 0.0) == 0.0


def test_duration_cap_clamped_and_corrupt_boxes_rejected() -> None:
    """max_on_s clamps to MAX_PAIR_S; corrupt artefacts raise ValueError."""
    arte = switched_artefact("big", dur=(60.0, 120.0, 1e9))
    model = _model(arte)
    assert model.max_on_s == float(MAX_PAIR_S)

    r = grid_trace(760, noise_w=0.0)
    _on_run(r, 0, 758, 2000.0)  # still ON at the end: no fall clears the state
    det = _det([model])
    results = _drive(det, r)
    forced = [i for i, res in enumerate(results) if res.forced_off >= 1]
    assert forced[0] == 720  # edge_ts 0 + MAX_PAIR_S at grid_ts 21600
    _assert_unknown(results[forced[0]], det, "big")
    plateau = det._runtimes["big"].plateau
    assert len(plateau) <= int(model.max_on_s / GRID_S) + 2

    bad_nan = switched_artefact("n", rise=(np.nan, 2000.0, 2100.0))
    with pytest.raises(ValueError):
        _model(bad_nan)
    bad_inf = switched_artefact("i", plateau=(math.inf, 2000.0, 2100.0))
    with pytest.raises(ValueError):
        _model(bad_inf)
    bad_order = switched_artefact("o", rise=(2100.0, 2000.0, 1900.0))
    with pytest.raises(ValueError):
        _model(bad_order)
    bad_dur = switched_artefact("d", dur=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        _model(bad_dur)
    bad_kind = switched_artefact("k")
    bad_kind["kind"] = "program"
    with pytest.raises(ValueError):
        _model(bad_kind)


def test_fall_time_check_accepts_any_state() -> None:
    """An episode verified through ANY state is reported (score.replay parity)."""
    arte = switched_artefact(
        "dev",
        rise=(1000.0, 2000.0, 3000.0),
        fall=(-3000.0, -2000.0, -1000.0),
        dur=(240.0, 300.0, 420.0),
        plateau=(1000.0, 2000.0, 3000.0),
    )
    arte["states"].append(
        {
            "name": "low",
            "rise_w": {"lo": 950.0, "p50": 1000.0, "hi": 1040.0},
            "fall_w": {"lo": -1100.0, "p50": -1000.0, "hi": -950.0},
            "dur_s": {"lo": 60.0, "p50": 90.0, "hi": 150.0},
            "plateau_w": {"lo": 950.0, "p50": 1000.0, "hi": 1040.0},
            "phase_share": None,
        }
    )
    r = grid_trace(60, noise_w=0.0)
    _on_run(r, 10, 10, 1050.0)  # dur 300 s fits only the wide state
    rep = replay_online([arte], r, float(GRID_S))
    assert len(rep.episodes["dev"]) == 1
    assert rep.unverified == 0
    assert rep.episodes["dev"][0].step_w == 1000.0  # nearest-p50 (low) state power


def test_ambiguous_edge_inside_two_boxes_is_refused() -> None:
    """An edge inside two OFF devices' rise boxes is refused (D-07)."""
    a = switched_artefact("a", rise=(1800.0, 2000.0, 2200.0), plateau=(1800.0, 2000.0, 2200.0))
    b = switched_artefact("b", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0))
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 10, 4, 2000.0)
    det = _det([a, b])
    results = _drive(det, r)
    assert all("a" not in res.on and "b" not in res.on for res in results)
    assert results[-1].unattributed_edges == {"a": 1, "b": 1}
    assert results[-1].unattributed_total == 2  # refused rise + later unmatched fall


def test_runner_up_within_margin_is_refused() -> None:
    """A runner-up outside its box but within AMBIGUITY_MARGIN is refused."""
    a = switched_artefact("a", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0))
    # b's box excludes 2000 W (hi 1999.9) but its p50 sits just below the
    # edge: normalised distance 0.1, inside AMBIGUITY_MARGIN of a's 0.0.
    b = switched_artefact("b", rise=(1500.0, 1999.9, 1999.9), plateau=(1500.0, 1999.9, 1999.9))
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 10, 4, 2000.0)
    det = _det([a, b])
    results = _drive(det, r)
    assert all("a" not in res.on and "b" not in res.on for res in results)
    assert results[-1].unattributed_edges["a"] == 1  # only box-containing devices count
    assert results[-1].unattributed_edges.get("b", 0) == 0
    assert results[-1].unattributed_total == 2  # refused rise + later unmatched fall

    # Clear winner: a far-away box meets the margin.
    c = switched_artefact("c", rise=(100.0, 400.0, 700.0), plateau=(100.0, 400.0, 700.0))
    r2 = grid_trace(40, noise_w=0.0)
    _on_run(r2, 10, 4, 2000.0)
    det2 = _det([a, c])
    results2 = _drive(det2, r2)
    assert any("a" in res.on for res in results2)


def test_unmatched_edge_leaves_state() -> None:
    """An edge inside no box changes nothing and increments the global count."""
    arte = switched_artefact("k")
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 10, 4, 500.0)  # below threshold? no: 500 >= 120, inside no box
    det = _det([arte])
    results = _drive(det, r)
    assert all("k" not in res.on for res in results)
    assert results[-1].unattributed_total == 2  # refused rise + later unmatched fall
    assert results[-1].unattributed_edges["k"] == 0


def test_box_bound_is_inclusive() -> None:
    """A delta exactly on a box bound (lo or hi) counts as inside."""
    arte = switched_artefact("k", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0))
    for w in (1900.0, 2100.0):
        r = grid_trace(40, noise_w=0.0)
        _on_run(r, 10, 4, w)
        det = _det([arte])
        results = _drive(det, r)
        assert any("k" in res.on for res in results), f"bound {w} must attribute"


def test_model_order_does_not_matter() -> None:
    """Permuting model order yields identical StepResults."""
    a = switched_artefact("a", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0))
    b = switched_artefact("b", rise=(400.0, 500.0, 600.0), plateau=(400.0, 500.0, 600.0))
    r = grid_trace(80, noise_w=0.0)
    _on_run(r, 10, 4, 2000.0)
    _on_run(r, 30, 4, 500.0)
    res1 = _drive(_det([a, b]), r)
    res2 = _drive(_det([b, a]), r)
    assert res1 == res2


def test_clamp_clips_newest_first() -> None:
    """The D-06 clamp clips the most recently attributed device first."""
    a = switched_artefact(
        "a", rise=(400.0, 500.0, 600.0), plateau=(400.0, 500.0, 600.0), dur=(60.0, 120.0, 6000.0)
    )
    b = switched_artefact(
        "b", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0), dur=(60.0, 120.0, 6000.0)
    )
    r = grid_trace(45, noise_w=0.0)
    _on_run(r, 5, 39, 500.0)
    _on_run(r, 20, 24, 2000.0)
    det = _det([a, b])
    results = _drive(det, r)
    on_idx = next(i for i, res in enumerate(results) if "a" in res.on and "b" in res.on)
    # Residual crashes to 100 W with both ON: the oldest (a) keeps the budget,
    # the newest (b) is clipped to zero first.
    res = det.step(float(on_idx + 1) * GRID_S, 100.0)
    assert "a" in res.on and "b" in res.on
    assert res.device_w["a"] == 100.0
    assert res.device_w["b"] == 0.0

    # residual <= 0: every device reports 0.0 while still ON
    res2 = det.step(float(on_idx + 2) * GRID_S, -50.0)
    assert "a" in res2.on and "b" in res2.on
    assert res2.device_w == {"a": 0.0, "b": 0.0}
    assert res2.estimated_w == 0.0


def test_two_falls_ambiguous_refused() -> None:
    """Two ON devices matching one fall within FALL_MARGIN -> refused."""
    a = switched_artefact(
        "a",
        rise=(1900.0, 2000.0, 2100.0),
        plateau=(1900.0, 2000.0, 2100.0),
        dur=(60.0, 120.0, 6000.0),
    )
    b = switched_artefact(
        "b",
        rise=(2020.0, 2050.0, 2150.0),
        plateau=(2020.0, 2050.0, 2150.0),
        dur=(60.0, 120.0, 6000.0),
    )
    r = grid_trace(90, noise_w=0.0)
    _on_run(r, 5, 39, 2000.0)  # a turns off at index 44 (-2000 W edge)
    _on_run(r, 20, 69, 2050.0)  # b stays on
    det = _det([a, b])
    results = _drive(det, r)
    refused = next(res for res in results if res.unattributed_total >= 1)
    assert "a" in refused.on and "b" in refused.on  # both stay ON (D-08)
    assert refused.unattributed_edges == {"a": 1, "b": 1}
    assert refused.unattributed_total == 1


def test_stuck_on_forced_off_by_duration() -> None:
    """The duration trigger forces OFF into the unknown state (D-29a)."""
    arte = switched_artefact("k", dur=(60.0, 120.0, 240.0))  # max_on_s = 360 s
    r = grid_trace(30, noise_w=0.0)
    _on_run(r, 2, 27, 2000.0)  # never falls within the trace
    det = _det([_model(arte)])
    results = _drive(det, r)
    # edge_ts = 60; 60 + 360 = 420 -> index 14
    assert results[13].forced_off == 0
    assert results[14].forced_off == 1
    _assert_unknown(results[14], det, "k")


def test_stuck_on_forced_off_by_low_residual() -> None:
    """The low-residual trigger is evidence: plain OFF at 0 W, not unknown (D-29b)."""
    arte = switched_artefact("k")
    r = grid_trace(60, noise_w=0.0)
    _on_run(r, 2, 10, 2000.0)
    r[13:] -= 1400.0  # partial drop: edge outside the fall box, residual 850 W
    det = _det([arte])
    results = _drive(det, r)
    # LOW_RESIDUAL_STEPS consecutive samples below 0.5 * 2000 = 1000 W
    for i in (13, 14, 15):
        assert "k" in results[i].on
    assert "k" not in results[16].on
    assert "k" not in results[16].unknown
    assert results[16].device_w["k"] == 0.0
    assert results[16].forced_off == 1


def _enter_unknown_via_duration(key: str = "k") -> tuple[OnlineDetector, list[StepResult]]:
    arte = switched_artefact(key, dur=(60.0, 120.0, 180.0))  # max_on_s = 270 s
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 2, 37, 2000.0)  # plateau held to the end: no fall, no low residual
    det = _det([arte])
    results = _drive(det, r)
    assert results[11].forced_off == 1  # edge_ts 60 + 270 = 330 -> index 11
    _assert_unknown(results[11], det, key)
    return det, results


@pytest.mark.parametrize(
    "entry",
    ["duration", "gap", "restore_after_gap", "set_models_drop"],
)
def test_forced_off_without_evidence_enters_unknown(entry: str) -> None:
    """Every evidence-free forced OFF enters the unknown state (D-28/D-29a)."""
    arte = switched_artefact("k", dur=(60.0, 120.0, 180.0))
    model = _model(arte)

    if entry == "duration":
        det, results = _enter_unknown_via_duration()
        idx = 11
    elif entry == "gap":
        r = grid_trace(30, noise_w=0.0)
        _on_run(r, 2, 27, 2000.0)  # still ON past the gap: no fall clears it
        r[10:14] = np.nan  # 4 NaN grid points > GAP_HOLD_STEPS
        det = _det([model])
        results = _drive(det, r)
        idx = 12  # third NaN (index 12) exceeds the hold of 2
        assert "k" in results[11].on  # still held at 2 NaN steps
    elif entry == "restore_after_gap":
        arte_long = switched_artefact("k", dur=(60.0, 120.0, 6000.0))
        model_long = _model(arte_long)
        r = grid_trace(20, noise_w=0.0)
        _on_run(r, 2, 17, 2000.0)  # still ON at the end of the trace
        det1 = _det([model_long])
        _drive(det1, r)
        state = det1.state()
        assert state["devices"]["k"]["on"] is True
        det = _det([model_long])
        det.restore(state, float(len(r)) * GRID_S + (GAP_HOLD_STEPS + 1) * GRID_S)
        assert "k" in det.unknown
        results = [det.step(float(i) * GRID_S, 2250.0) for i in range(20, 30)]
        idx = 0
    else:  # set_models_drop
        det, results = _enter_unknown_via_duration()
        det.set_models([])
        assert "k" in det.unknown
        det.set_models([model])  # parked flag re-applied when the model returns
        assert "k" in det.unknown
        return

    res = results[idx]
    _assert_unknown(res, det, "k")
    if entry != "restore_after_gap":
        later = results[idx + 1]
        _assert_unknown(later, det, "k")
        assert later.energy_wh_delta["k"] == 0.0
        # energy committed before the forced OFF is kept (D-09)
        committed = det.energy_wh["k"]
        assert committed > 0.0


def test_unknown_ends_on_matching_fall_or_attributed_rise() -> None:
    """D-29c: a matching fall or the next attributed rise ends the unknown state."""
    arte = switched_artefact("k")
    det, _ = _enter_unknown_via_duration()
    total_before = det.state()["unattributed_total"]
    unverified_before = det.state()["devices"]["k"]["unverified"]

    # (a) a fall inside the fall box ends the unknown state at 0 W
    res = det.step(40 * GRID_S, 250.0 + 2000.0)
    _assert_unknown(res, det, "k")
    det.step(41 * GRID_S, 250.0)  # the fall manifests (pre-edge sample 40)
    res2 = det.step(42 * GRID_S, 250.0)  # fall finalised: plain OFF, no counts
    assert "k" not in res2.unknown
    assert "k" not in res2.on
    assert res2.device_w["k"] == 0.0
    assert res2.unattributed_total == total_before
    assert res2.unverified_episodes["k"] == unverified_before

    # an edge pair outside every box is unmatched and leaves the device OFF
    # (held one sample between the edges so the cancelling pair is not
    # adjacent and survives alignment-artefact suppression)
    det.step(50 * GRID_S, 250.0 + 1300.0)  # unmatched rise manifests (+1300)
    det.step(51 * GRID_S, 250.0 + 1300.0)  # rise finalised: unmatched
    det.step(52 * GRID_S, 250.0)  # unmatched fall manifests (-1300)
    res3 = det.step(53 * GRID_S, 250.0)  # fall finalised
    assert "k" not in res3.on
    assert "k" not in res3.unknown
    assert res3.unattributed_total == total_before + 2

    # (b) the next attributed rise turns it ON one sample later (D-09)
    det.step(60 * GRID_S, 250.0 + 2000.0)
    res4 = det.step(61 * GRID_S, 250.0 + 2000.0)
    assert "k" in res4.on
    assert "k" not in res4.unknown
    assert res4.device_w["k"] == 2000.0


def test_unknown_fall_ambiguity_refused() -> None:
    """D-07 with an unknown candidate: the fall is refused, states unchanged."""
    a = switched_artefact("a")
    b = switched_artefact("b")
    det = _det([a, b], unknown={"a", "b"})
    assert det.unknown == frozenset({"a", "b"})
    det.step(0.0, BASE)
    det.step(GRID_S, BASE + 2000.0)  # rise manifests (unmatched: both boxes)
    res2 = det.step(2 * GRID_S, BASE + 2000.0)  # rise finalised: refused
    _assert_unknown(res2, det, "a")
    _assert_unknown(res2, det, "b")
    assert res2.unattributed_edges == {"a": 1, "b": 1}
    assert res2.unattributed_total == 1
    det.step(3 * GRID_S, BASE)  # fall manifests
    res2b = det.step(4 * GRID_S, BASE)  # fall inside both fall boxes: refused
    _assert_unknown(res2b, det, "a")
    _assert_unknown(res2b, det, "b")
    assert res2b.unattributed_edges == {"a": 2, "b": 2}
    assert res2b.unattributed_total == 2
    # an ON device plus an unknown device as fall candidates is also refused
    b2 = switched_artefact("b", rise=(2020.0, 2050.0, 2150.0), plateau=(2020.0, 2050.0, 2150.0))
    det2 = _det([a, b2], unknown={"b"})
    det2.step(0.0, BASE)
    det2.step(GRID_S, BASE + 2000.0)
    res3 = det2.step(2 * GRID_S, BASE + 2000.0)
    assert "a" in res3.on
    det2.step(3 * GRID_S, BASE)  # fall manifests, inside a's box and b's fall box
    res4 = det2.step(4 * GRID_S, BASE)
    assert "a" in res4.on  # the ON device keeps waiting (precision-first)
    _assert_unknown(res4, det2, "b")
    assert res4.unattributed_edges["a"] == 1
    assert res4.unattributed_edges["b"] == 1


def test_unknown_ends_on_sustained_low_residual() -> None:
    """D-30a: LOW_RESIDUAL_STEPS low samples end the unknown state at 0 W."""
    arte = switched_artefact("k")
    model = _model(arte)

    def _seeded() -> OnlineDetector:
        det = _det([model], unknown={"k"})
        assert "k" in det.unknown
        return det

    def _exit_low(det: OnlineDetector, start_i: int, current_v: float) -> StepResult:
        """Decline to base without edges (-100 W drops separated by holds)."""
        i = start_i
        v = current_v
        while v - 100.0 > BASE:
            v -= 100.0
            det.step(float(i) * GRID_S, v)
            i += 1
            det.step(float(i) * GRID_S, v)  # hold between drops: no merge
            i += 1
        for _ in range(LOW_RESIDUAL_STEPS):
            res = det.step(float(i) * GRID_S, BASE)
            i += 1
        return res

    # parametrized over duration-triggered and seeded-unknown entries; the
    # duration-triggered detector ends on a 2250 W plateau, so decline gently
    det_dur, _ = _enter_unknown_via_duration()
    res = _exit_low(det_dur, 100, 250.0 + 2000.0)
    assert "k" not in res.unknown
    assert "k" not in res.on
    assert res.device_w["k"] == 0.0
    assert res.energy_wh_delta["k"] == 0.0
    # the exit is not counted anywhere
    assert res.forced_off == 1  # counted once at the duration entry
    assert res.unattributed_total == 0
    assert res.unverified_episodes["k"] == 0

    det_seed = _seeded()
    res = _exit_low(det_seed, 0, BASE)
    assert "k" not in res.unknown
    assert "k" not in res.on
    assert res.device_w["k"] == 0.0
    assert res.energy_wh_delta["k"] == 0.0
    assert res.forced_off == 0  # a seeded entry never counted
    assert res.unattributed_total == 0
    assert res.unverified_episodes["k"] == 0

    # a NaN sample resets the counter
    det = _seeded()
    det.step(0.0, 250.0)
    det.step(GRID_S, 250.0)
    det.step(2 * GRID_S, float("nan"))
    det.step(3 * GRID_S, 250.0)
    assert "k" in det.unknown  # only 2 consecutive finite low samples
    det.step(4 * GRID_S, 250.0)
    det.step(5 * GRID_S, 250.0)
    res = det.step(6 * GRID_S, 250.0)
    assert "k" not in res.unknown  # 4th consecutive low finite sample

    # a sample at or above the threshold resets the counter and keeps it unknown
    det = _seeded()
    det.step(0.0, 250.0)
    det.step(GRID_S, 250.0)
    det.step(2 * GRID_S, 250.0 + 2000.0)
    det.step(3 * GRID_S, 250.0)
    det.step(4 * GRID_S, 250.0)
    assert "k" in det.unknown

    # a residual that stays above the threshold keeps it unknown indefinitely
    det = _seeded()
    for i in range(10):
        res = det.step(float(i) * GRID_S, 250.0 + 2000.0)
        assert "k" in res.unknown

    # a parked key without a model stays unknown whatever the residual and gets
    # the exit once set_models adds its model
    det = _det([], unknown={"k"})
    for i in range(LOW_RESIDUAL_STEPS + 2):
        res = det.step(float(i) * GRID_S, 250.0)
        assert "k" in res.unknown
    # the exit fires once the model arrives
    det.set_models([model])
    for i in range(LOW_RESIDUAL_STEPS - 1):
        det.step(float(20 + i) * GRID_S, 250.0)
    res = det.step(float(20 + LOW_RESIDUAL_STEPS - 1) * GRID_S, 250.0)
    assert "k" not in res.unknown

    # a two-state device uses the smaller rise p50 (min_step_w)
    arte2 = switched_artefact("two")
    arte2["states"].append(
        {
            "name": "low",
            "rise_w": {"lo": 400.0, "p50": 500.0, "hi": 600.0},
            "fall_w": {"lo": -600.0, "p50": -500.0, "hi": -400.0},
            "dur_s": {"lo": 60.0, "p50": 90.0, "hi": 150.0},
            "plateau_w": {"lo": 400.0, "p50": 500.0, "hi": 600.0},
            "phase_share": None,
        }
    )
    det2 = _det([_model(arte2, "two")], unknown={"two"})
    assert _model(arte2, "two").min_step_w == 500.0
    # threshold = 0.5 * 500 = 250: values at 200 W count as low
    for i in range(LOW_RESIDUAL_STEPS - 1):
        res = det2.step(float(i) * GRID_S, 200.0)
        assert "two" in res.unknown
    res = det2.step(float(LOW_RESIDUAL_STEPS - 1) * GRID_S, 200.0)
    assert "two" not in res.unknown


def test_restore_leaves_uncovered_seeded_keys_unknown() -> None:
    """D-30b engine half: restore covers only its keys; seeds survive."""
    arte_a = switched_artefact("a")
    arte_b = switched_artefact("b")
    det1 = _det([arte_a])
    for i in range(5):
        det1.step(float(i) * GRID_S, 250.0)
    state = det1.state()
    assert state_keys(state) == frozenset({"a"})
    assert state_keys(None) == frozenset()
    assert state_keys({"x": 1}) == frozenset()

    det2 = _det([arte_a, arte_b], unknown={"b"})
    assert "b" in det2.unknown
    det2.restore(state, 4 * GRID_S)  # within GAP_HOLD_STEPS of last_grid_ts
    res = det2.step(5 * GRID_S, 250.0 + 2000.0)
    assert "a" not in res.unknown
    assert "a" not in res.on
    assert res.device_w["a"] == 0.0
    assert "b" in res.unknown  # seeded and uncovered: keeps the unknown flag
    assert "b" in det2.unknown


@pytest.mark.parametrize("entry", ["stream_gap", "restore_gap"])
@pytest.mark.parametrize("first_value", ["high", "boundary", "low"])
def test_off_device_after_gap_unknown_only_if_plausible(entry, first_value) -> None:
    """D-31: after a long gap an OFF device is unknown only if the first valid
    residual is at least LOW_RESIDUAL_FRACTION x min_step_w."""
    # min_step 600 -> low-residual threshold 300 W; the exit decline stays
    # under the 120 W edge threshold so D-30a (not a fall edge) ends it.
    arte = switched_artefact(
        "k",
        rise=(500.0, 600.0, 700.0),
        fall=(-700.0, -600.0, -500.0),
        dur=(60.0, 120.0, 240.0),
        plateau=(500.0, 600.0, 700.0),
    )
    model = _model(arte)
    min_step = model.min_step_w  # 600
    values = {
        "high": BASE + min_step,  # 850 >= 300: unknown
        "boundary": BASE + 0.5 * min_step,  # exactly 0.5 x step: unknown
        "low": BASE,  # 250 < 300: stays OFF at 0 W
    }
    first = values[first_value]

    det = _det([model])
    if entry == "stream_gap":
        for i in range(5):
            det.step(float(i) * GRID_S, BASE)
        for i in range(5, 5 + GAP_HOLD_STEPS + 1):
            det.step(float(i) * GRID_S, float("nan"))
        idx = 5 + GAP_HOLD_STEPS + 1
    else:
        det1 = _det([model])
        for i in range(5):
            det1.step(float(i) * GRID_S, BASE)
        state = det1.state()
        now = 4 * GRID_S + (GAP_HOLD_STEPS + 1) * GRID_S
        det = _det([model])
        det.restore(state, float(now))
        idx = 5

    res = det.step(float(idx) * GRID_S, first)
    if first_value == "low":
        assert "k" not in res.unknown
        assert "k" not in res.on
        assert res.device_w["k"] == 0.0
        assert res.forced_off == 0
    else:
        _assert_unknown(res, det, "k")
        assert res.forced_off == 0  # a D-31 entry ends no ON period
        # stays unknown while the residual holds, then a gradual decline with
        # -100 W drops separated by holds (so no pair merges into an edge and
        # nothing is suppressed) exits it via D-30a at LOW_RESIDUAL_STEPS low
        res_hold = det.step(float(idx + 1) * GRID_S, first)
        assert "k" in res_hold.unknown
        v = first
        step_i = idx + 2
        while v - 100.0 > BASE:
            v -= 100.0
            r2 = det.step(float(step_i) * GRID_S, v)  # the -100 W drop
            assert "k" in r2.unknown  # still at or above the threshold
            step_i += 1
            r2 = det.step(float(step_i) * GRID_S, v)  # the hold between drops
            assert "k" in r2.unknown
            step_i += 1
        for _ in range(LOW_RESIDUAL_STEPS - 1):
            r2 = det.step(float(step_i) * GRID_S, BASE)
            assert "k" in r2.unknown
            step_i += 1
        r3 = det.step(float(step_i) * GRID_S, BASE)
        assert "k" not in r3.unknown
        assert r3.device_w["k"] == 0.0
        # the exits changed no counts
        assert r3.unattributed_total == res.unattributed_total
        assert r3.unverified_episodes["k"] == res.unverified_episodes["k"]
        assert r3.forced_off == res.forced_off


def test_off_device_after_gap_check_runs_once_and_short_gap_arms_nothing() -> None:
    """D-31 one-shot semantics, short-gap arming and persisted armed flag."""
    arte = switched_artefact("k")
    model = _model(arte)

    # the check runs exactly once: a later high residual with no edge stays OFF
    det = _det([model])
    for i in range(5):
        det.step(float(i) * GRID_S, BASE)
    for i in range(5, 5 + GAP_HOLD_STEPS + 1):
        det.step(float(i) * GRID_S, float("nan"))
    det.step(float(8) * GRID_S, BASE + 0.25 * model.min_step_w)  # low: stays OFF
    det.step(float(9) * GRID_S, BASE + model.min_step_w)  # check already ran
    res = det.step(float(10) * GRID_S, BASE + model.min_step_w)
    assert "k" not in res.unknown
    assert "k" not in res.on

    # NaN points after the armed gap keep it armed until the first finite point
    det = _det([model])
    for i in range(5):
        det.step(float(i) * GRID_S, BASE)
    for i in range(5, 5 + GAP_HOLD_STEPS + 3):
        det.step(float(i) * GRID_S, float("nan"))
    res = det.step(float(10) * GRID_S, BASE + model.min_step_w)
    assert "k" in res.unknown  # armed through the extra NaN points

    # a NaN run of exactly GAP_HOLD_STEPS arms nothing
    det = _det([model])
    for i in range(5):
        det.step(float(i) * GRID_S, BASE)
    for i in range(5, 5 + GAP_HOLD_STEPS):
        det.step(float(i) * GRID_S, float("nan"))
    res = det.step(float(7) * GRID_S, BASE + model.min_step_w)
    assert "k" not in res.unknown

    # the armed flag survives state() -> restore() before the first finite point
    det = _det([model])
    for i in range(5):
        det.step(float(i) * GRID_S, BASE)
    for i in range(5, 5 + GAP_HOLD_STEPS + 1):
        det.step(float(i) * GRID_S, float("nan"))
    state = det.state()
    assert state["post_gap_armed"] is True
    det2 = _det([model])
    det2.restore(state, float(7) * GRID_S)
    res = det2.step(float(8) * GRID_S, BASE + model.min_step_w)
    assert "k" in res.unknown

    # a stored state that never stepped (last_grid_ts None) arms nothing
    det3 = _det([model])
    state3 = det3.state()
    assert state3["last_grid_ts"] is None
    det3.restore(state3, 99999.0)
    res = det3.step(100000.0, BASE + model.min_step_w)
    assert "k" not in res.unknown

    # a device ON before the gap is unknown by the gap rule, not the check;
    # a seeded unknown device stays unknown
    det = _det([model], unknown={"ghost"})
    for i in range(5):
        det.step(float(i) * GRID_S, BASE)
    det.step(float(5) * GRID_S, BASE + 2000.0)
    det.step(float(6) * GRID_S, BASE + 2000.0)  # ON from step 6
    for i in range(7, 7 + GAP_HOLD_STEPS + 1):
        det.step(float(i) * GRID_S, float("nan"))
    res = det.step(float(10) * GRID_S, BASE + 2000.0)
    assert "k" in res.unknown  # gap rule
    assert "ghost" in res.unknown


def test_gap_hold_then_unknown() -> None:
    """<= GAP_HOLD_STEPS NaN keeps an ON device ON; more ends it unknown."""
    arte = switched_artefact("k")
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 2, 30, 2000.0)
    r[10:12] = np.nan  # exactly 2 NaN samples: held ON, no energy
    det = _det([arte])
    results = _drive(det, r)
    for i in (10, 11, 12):
        assert "k" in results[i].on
    assert results[11].energy_wh_delta["k"] == 0.0
    assert results[12].energy_wh_delta["k"] == 0.0

    r2 = grid_trace(40, noise_w=0.0)
    _on_run(r2, 2, 37, 2000.0)  # plateau to the end: no fall clears the state
    r2[10:13] = np.nan  # 3 NaN samples: unknown from the third
    det2 = _det([arte])
    results2 = _drive(det2, r2)
    assert "k" in results2[11].on
    _assert_unknown(results2[12], det2, "k")


def test_non_increasing_grid_ts_ignored() -> None:
    """A grid ts not later than the previous one adds nothing."""
    arte = switched_artefact("k")
    det = _det([arte])
    det.step(0.0, BASE)
    det.step(GRID_S, BASE + 2000.0)
    before = det.state()["unattributed_total"]
    energy_before = det.energy_wh.get("k", 0.0)
    res = det.step(GRID_S, BASE + 5000.0)  # same ts: ignored
    assert res.unattributed_total == before
    assert det.energy_wh.get("k", 0.0) == energy_before
    res2 = det.step(0.5 * GRID_S, BASE + 9000.0)  # older ts: ignored
    assert res2.unattributed_total == before
    assert det.energy_wh.get("k", 0.0) == energy_before


def test_phase_share_gate() -> None:
    """A rise whose first-step phase shares fall outside the state boxes is refused."""
    arte = switched_artefact("k", phase_share={"l1": (0.8, 1.0)})
    model = _model(arte)
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 10, 4, 2000.0)

    # jump on L2: l1 share ~ 0 -> not a candidate
    phases = {
        "l1": np.full(40, BASE),
        "l2": np.full(40, BASE),
        "l3": np.full(40, BASE),
    }
    phases["l2"][11:] += 2000.0
    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    for i in range(40):
        res = det.step(float(i) * GRID_S, float(r[i]), (phases["l1"][i], phases["l2"][i], phases["l3"][i]))
    assert all("k" not in res.on for res in [res])
    assert det.state()["unattributed_total"] == 2  # rise and fall both unmatched

    # jump on L1: l1 share ~ 1.0 -> attributed
    phases2 = {
        "l1": np.full(40, BASE),
        "l2": np.full(40, BASE),
        "l3": np.full(40, BASE),
    }
    phases2["l1"][11:] += 2000.0
    det2 = OnlineDetector([model], grid_s=GRID_S, threshold_w=THRESHOLD)
    results = [
        det2.step(float(i) * GRID_S, float(r[i]), (phases2["l1"][i], phases2["l2"][i], phases2["l3"][i]))
        for i in range(40)
    ]
    assert any("k" in res.on for res in results)

    # without live phases the phase check is skipped
    det3 = _det([model])
    results3 = _drive(det3, r)
    assert any("k" in res.on for res in results3)


def test_fall_time_check_rejects_box_failures() -> None:
    """Episodes failing the full box rule are unverified; score.replay agrees."""
    arte = switched_artefact(
        "k",
        rise=(1700.0, 2000.0, 2100.0),
        fall=(-2100.0, -2000.0, -1700.0),
        dur=(60.0, 120.0, 300.0),
        plateau=(1900.0, 2000.0, 2100.0),
    )
    r = grid_trace(160, noise_w=0.0)
    _on_run(r, 10, 6, 2000.0)  # clean episode -> verified
    _on_run(r, 30, 12, 2000.0)  # dur 360 s > dur box hi 300 (below max_on 450)
    _on_run(r, 60, 6, 1800.0)  # plateau 1800 < plateau box lo 1900
    _on_run(r, 83, 6, 2000.0)
    r[85] = np.nan  # single NaN inside the fourth episode (held ON by the gap rule)

    rep = replay_online([arte], r, float(GRID_S))
    assert len(rep.episodes["k"]) == 1  # only the clean episode is reported
    assert rep.episodes["k"][0].start_i == 10
    assert rep.unverified == 3
    # the device still turned OFF at each matched fall and kept its energy
    assert rep.energy_wh["k"] > 0.0

    # the offline side rejects the same episodes (the NaN-spanning pair is
    # discarded by pair_switched, which the online side reports as unverified)
    idx, delta = detect_edges(r, THRESHOLD)
    idx, delta = suppress_alignment_artefacts(idx, delta, enabled_event_types=[EventType.STEP])
    table = pair_switched(r, idx, delta, float(GRID_S))
    n_eps = len(table["start_i"])
    assert n_eps == 3, f"expected 3 offline episodes, got {n_eps}"
    block = np.zeros(n_eps, dtype=np.int32)
    result = score_replay(
        [SignatureArtefact.from_dict(arte)], {**table, "block": block}, 0, residual_wh=1.0
    )
    claims = result.claims["k"]
    assert len(claims) == 1
    clean_pos = int(np.where(table["start_i"] == 10)[0][0])
    assert claims == {clean_pos}


def test_match_score_only_while_on() -> None:
    """match_score is set only while ON (D-14)."""
    arte = switched_artefact("k", rise=(1900.0, 2000.0, 2100.0))
    r = grid_trace(40, noise_w=0.0)
    _on_run(r, 10, 6, 1950.0)  # |1950-2000| / (2100-2000) = 0.5 -> score 0.5
    det = _det([arte])
    results = _drive(det, r)
    on_res = next(res for res in results if "k" in res.on)
    assert on_res.match_score["k"] == 0.5
    off_res = next(res for res in results if "k" not in res.on and res.grid_ts > on_res.grid_ts)
    assert "k" not in off_res.match_score
    # dead-centre step scores 1.0
    arte2 = switched_artefact("k2")
    r2 = grid_trace(40, noise_w=0.0)
    _on_run(r2, 10, 6, 2000.0)
    det2 = _det([arte2])
    results2 = _drive(det2, r2)
    on_res2 = next(res for res in results2 if "k2" in res.on)
    assert on_res2.match_score["k2"] == 1.0


def test_state_restore_round_trip() -> None:
    """state() is JSON-safe and restore() round-trips flags, counts and energy."""
    arte = switched_artefact("k", dur=(60.0, 120.0, 6000.0))
    det = _det([arte], unknown={"ghost"})
    r = grid_trace(60, noise_w=0.0)
    _on_run(r, 10, 49, 2000.0)  # still ON at the end
    results = _drive(det, r)
    assert "k" in results[-1].on
    state = det.state()
    payload = json.dumps(state)  # JSON-safe
    state = json.loads(payload)

    det2 = _det([arte], unknown={"ghost"})
    det2.restore(state, float(len(r)) * GRID_S)  # no gap: same grid
    assert det2.unknown == det.unknown
    res = det2.step(float(len(r)) * GRID_S, BASE + 2000.0)
    assert "k" in res.on
    assert res.device_w["k"] == 2000.0
    assert "ghost" in res.unknown
    assert det2.energy_wh["k"] == det.energy_wh["k"]
    st1, st2 = det.state(), det2.state()
    assert st1["devices"]["k"]["unattributed"] == st2["devices"]["k"]["unattributed"]
    assert st1["devices"]["k"]["unverified"] == st2["devices"]["k"]["unverified"]
    assert st1["unattributed_total"] == st2["unattributed_total"]
    assert st1["forced_off"] == st2["forced_off"]
    assert st1["post_gap_armed"] == st2["post_gap_armed"]

    # a restored-ON episode counts as unverified at its fall
    det2.step(float(len(r) + 1) * GRID_S, BASE)
    res2 = det2.step(float(len(r) + 2) * GRID_S, BASE)
    assert "k" not in res2.on
    assert res2.unverified_episodes["k"] == 1


def test_restore_after_long_gap_enters_unknown() -> None:
    """A restore after a long gap puts ON devices in unknown with no gap energy."""
    arte = switched_artefact("k", dur=(60.0, 120.0, 6000.0))
    det1 = _det([arte])
    r = grid_trace(30, noise_w=0.0)
    _on_run(r, 5, 24, 2000.0)  # still ON at the end of the trace
    results = _drive(det1, r)
    assert "k" in results[-1].on
    energy_before = det1.energy_wh["k"]
    state = det1.state()

    det2 = _det([arte])
    now = float(29) * GRID_S + (GAP_HOLD_STEPS + 1) * GRID_S
    det2.restore(state, now)
    assert "k" in det2.unknown
    assert det2.state()["devices"]["k"]["on"] is False
    res = det2.step(now + GRID_S, BASE + 2000.0)
    _assert_unknown(res, det2, "k")
    assert det2.energy_wh["k"] == energy_before  # no energy for the blind period


def test_restore_with_changed_grid_enters_unknown() -> None:
    """D-26/D-29a: a state saved on another grid restores ON devices as unknown."""
    arte30 = switched_artefact("k", grid_s=30, dur=(60.0, 120.0, 6000.0))
    det1 = _det([arte30])
    r = grid_trace(30, noise_w=0.0)
    _on_run(r, 5, 24, 2000.0)
    results = _drive(det1, r)
    assert "k" in results[-1].on
    state = det1.state()
    last_ts = state["last_grid_ts"]

    # rebuild on a 15 s grid, quick restart (no gap in 15 s steps)
    arte15 = switched_artefact("k", grid_s=15, dur=(60.0, 120.0, 6000.0))
    det15 = _det([arte15], grid_s=15)
    det15.restore(state, float(last_ts) + GAP_HOLD_STEPS * 15)
    assert "k" in det15.unknown
    res = det15.step(float(last_ts) + (GAP_HOLD_STEPS + 1) * 15, BASE + 2000.0)
    _assert_unknown(res, det15, "k")
    assert res.forced_off == 1

    # a stored OFF key restores plain OFF; with no gap the D-31 check is not armed
    det1b = _det([arte30])
    for i in range(10):
        det1b.step(float(i) * GRID_S, BASE)
    state_off = det1b.state()
    det15b = _det([arte15], grid_s=15)
    det15b.restore(state_off, float(state_off["last_grid_ts"]) + 15)
    res = det15b.step(float(state_off["last_grid_ts"]) + 30, BASE + 2000.0)
    assert "k" not in res.unknown
    assert "k" not in res.on

    # a stored unknown key stays unknown; a stored ON key without a model on
    # the new grid is parked unknown
    det1c = _det([arte30], unknown={"u"})
    r_c = grid_trace(20, noise_w=0.0)
    _on_run(r_c, 5, 14, 2000.0)  # k ON at the end (no fall)
    results_c = _drive(det1c, r_c)
    assert "k" in results_c[-1].on
    state_c = json.loads(json.dumps(det1c.state()))
    det15c = _det([], grid_s=15, unknown={"u"})
    det15c.restore(state_c, float(state_c["last_grid_ts"]) + 15)
    assert "u" in det15c.unknown
    assert "k" in det15c.unknown  # stored ON key without a model: parked

    # a stored state without a grid_s key counts as changed
    state_nogrid = {k: v for k, v in state.items() if k != "grid_s"}
    det15d = _det([arte15], grid_s=15)
    det15d.restore(state_nogrid, float(last_ts) + 15)
    assert "k" in det15d.unknown

    # control: same grid restores ON with its on_w
    det30 = _det([arte30])
    det30.restore(state, float(last_ts) + 30)
    assert "k" not in det30.unknown
    res = det30.step(float(last_ts) + 60, BASE + 2000.0)
    assert "k" in res.on
    assert res.device_w["k"] == 2000.0

    # a restart with a gap on the new grid arms the D-31 check for OFF keys
    det1e = _det([arte30])
    for i in range(10):
        det1e.step(float(i) * GRID_S, BASE)
    state_e = det1e.state()
    det15e = _det([arte15], grid_s=15)
    det15e.restore(state_e, float(state_e["last_grid_ts"]) + (GAP_HOLD_STEPS + 1) * 15)
    res = det15e.step(float(state_e["last_grid_ts"]) + (GAP_HOLD_STEPS + 2) * 15, BASE + 2000.0)
    assert "k" in res.unknown
