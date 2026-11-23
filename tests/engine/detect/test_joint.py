"""Joint overlap attribution: one candidate list, splits and refusals (DET-03).

Pins D-06/D-07/D-36/D-41: singletons and small subsets are scored in one list
under one runner-up margin, a single device never wins over a tying subset just
because it was checked first, an ambiguous edge is refused (never split) and
counted as ``overlap_refused``, and a uniquely explained edge is split across
its subset in proportion to each member's learned p50.
"""

import json
import math

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.joint import (
    JOINT_MAX_K,
    JOINT_PENALTY,
    JOINT_PREFILTER,
    Explanation,
    best_subsets,
    pick,
    prefilter,
    split_delta,
)
from custom_components.powerguesser.engine.detect.online import (
    AMBIGUITY_MARGIN,
    OnlineDetector,
    compile_model,
)
from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.validation.drift import LiveDrift
from tests.fixtures.detect_traces import add_run, grid_trace, switched_artefact
from tests.fixtures.programs import (
    dishwasher_stages,
    kettle_overlay,
    program_artefact,
    program_run,
    washer_stages,
)

GRID_S = 30
THRESHOLD = 120.0
LONG_DUR = (300.0, 600.0, 3600.0)


def _switched(key: str, box, *, dur=LONG_DUR):
    """A switched model whose rise box is ``box`` and whose fall mirrors it."""
    lo, p50, hi = box
    return compile_model(
        key,
        switched_artefact(
            key,
            rise=(lo, p50, hi),
            fall=(-hi, -p50, -lo),
            dur=dur,
            plateau=(lo, p50, hi),
        ),
    )


def _detector(*models, unknown=()):
    return OnlineDetector(
        [*models], grid_s=GRID_S, threshold_w=THRESHOLD, unknown=unknown
    )


def _drive(det: OnlineDetector, r: np.ndarray, *, start: int = 0, stop: int | None = None):
    end = len(r) if stop is None else stop
    return [det.step(float(i) * GRID_S, float(r[i])) for i in range(start, end)]


def _first_refused(results):
    return next(res for res in results if res.overlap_refused >= 1)


# -- the canonical SC2 cases ----------------------------------------------


def test_kettle_during_wash_program() -> None:
    """A kettle mid wash program: the kettle is attributed, the program keeps its stage."""
    washer = compile_model("washer", program_artefact("washer", washer_stages()))
    kettle = _switched("kettle", (1950.0, 2000.0, 2050.0))
    det = _detector(washer)
    r = grid_trace(400)
    program_run(r, 100, washer_stages())  # samples 101..230: stage 1, 2 and 3
    kettle_overlay(r, 180, 3, w=2000.0)  # a short boil inside stage 2

    # The kettle joins once the program runs: with the washer idle, its 2000 W
    # stage-1 entry and the kettle's box both contain the program's own start
    # edge, which is the P4 D-07 refusal (Plan 06-01's D-41 production path).
    _drive(det, r, stop=180)
    det.set_models([washer, kettle])
    results = _drive(det, r, start=180)

    # the program reached stage 2 (index 1) before the boil and held it through it
    assert "washer" in results[0].on
    assert "washer" not in results[0].unknown
    stage2 = program_artefact("washer", washer_stages())["session"]["stages"][1]["level_w"]
    boil = [res for res in results if 181 * GRID_S <= res.grid_ts <= 184 * GRID_S]
    assert boil
    for res in boil:
        assert "washer" in res.on
        assert "washer" not in res.unknown
        assert stage2["lo"] <= res.device_w["washer"] <= stage2["hi"]

    # the kettle is attributed with its measured step and turned OFF at its fall
    # (the D-06 clamp caps its share on the sample where the load drops again)
    kettle_on = [res for res in results if "kettle" in res.on]
    assert [res.grid_ts for res in kettle_on] == [
        182 * GRID_S,
        183 * GRID_S,
        184 * GRID_S,
    ]
    assert kettle_on[0].device_w["kettle"] == pytest.approx(2000.0, abs=60.0)
    assert kettle_on[1].device_w["kettle"] == pytest.approx(2000.0, abs=60.0)
    assert kettle_on[-1].device_w["kettle"] <= 300.0  # clamped by the falling residual
    assert "kettle" not in results[-1].on
    assert "kettle" not in results[-1].unknown
    assert results[-1].joint_attributed == 0  # a singleton attribution, not joint
    assert results[-1].overlap_refused == 0

    # the program never diverged and finished in its last stage
    assert det._runtimes["washer"].stage_index == 2
    assert "washer" not in det.unknown
    stage3 = program_artefact("washer", washer_stages())["session"]["stages"][2]["level_w"]
    late = [res for res in results if res.grid_ts >= 214 * GRID_S]
    assert any(
        stage3["lo"] <= res.device_w["washer"] <= stage3["hi"] for res in late
    ), "the program never reported its last stage"


def test_simultaneous_rises_attributed_jointly() -> None:
    """Kettle 2000 W + heater 800 W in one sample: both ON, split by p50 (D-36)."""
    det = _detector(
        _switched("kettle", (1950.0, 2000.0, 2050.0)),
        _switched("heater", (750.0, 800.0, 850.0)),
    )
    r = grid_trace(60, noise_w=0.0)
    add_run(r, 10, 6, 2800.0)

    results = _drive(det, r)

    joint = next(res for res in results if res.joint_attributed >= 1)
    assert joint.on == frozenset({"heater", "kettle"})
    assert joint.device_w["kettle"] == pytest.approx(2000.0)
    assert joint.device_w["heater"] == pytest.approx(800.0)
    assert joint.joint_attributed == 1
    assert joint.unattributed_total == 0
    assert joint.overlap_refused == 0
    assert joint.estimated_w == pytest.approx(2800.0)


def test_ambiguous_pair_refused() -> None:
    """Three equally plausible pairs of one summed box: refused, counted (D-07)."""
    det = _detector(
        _switched("a", (950.0, 1000.0, 1050.0)),
        _switched("b", (950.0, 1000.0, 1050.0)),
        _switched("c", (950.0, 1000.0, 1050.0)),
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)

    results = _drive(det, r)

    res = _first_refused(results)
    assert res.unattributed_total == 1
    assert res.unattributed_edges == {"a": 1, "b": 1, "c": 1}
    assert all("a" not in r2.on and "b" not in r2.on and "c" not in r2.on for r2 in results)
    assert all(res.overlap_refused == 1 for res in results[results.index(res) :])
    assert det.energy_wh == {"a": 0.0, "b": 0.0, "c": 0.0}


# -- D-41: one candidate list, one margin ---------------------------------


def test_singleton_vs_pair_tie_refused() -> None:
    """D-41 counter-example: a device's box and a tying pair refuse, in any order."""
    box = (1950.0, 2000.0, 2050.0)
    pair = (950.0, 1000.0, 1050.0)
    forward = _detector(
        _switched("a", box), _switched("b", pair), _switched("c", pair)
    )
    reverse = _detector(
        _switched("c", pair), _switched("b", pair), _switched("a", box)
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)

    results = _drive(forward, r)
    other = _drive(reverse, r)

    res = _first_refused(results)
    assert res.unattributed_total == 1  # before the later unmatched fall
    assert res.unattributed_edges == {"a": 1, "b": 1, "c": 1}
    assert all(
        key not in step.on for step in results for key in ("a", "b", "c")
    )
    # A alone scores 0.0; {b, c} sums to box [1900, 2100] p50 2000 and scores
    # exactly JOINT_PENALTY, inside AMBIGUITY_MARGIN -> refused, not attributed.
    assert _first_refused(results).overlap_refused == 1
    assert forward.state() == reverse.state()
    assert results == other


def test_singleton_beats_pair_by_margin() -> None:
    """A subset that loses by more than the margin cannot stop the singleton."""
    det = _detector(
        _switched("a", (1950.0, 2000.0, 2050.0)),
        _switched("b", (850.0, 900.0, 1050.0)),
        _switched("c", (850.0, 900.0, 1050.0)),
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)

    results = _drive(det, r)

    assert any("a" in res.on for res in results)
    assert all("b" not in res.on and "c" not in res.on for res in results)
    assert results[-1].overlap_refused == 0
    assert results[-1].joint_attributed == 0
    a_on = next(res for res in results if "a" in res.on)
    assert a_on.device_w["a"] == pytest.approx(2000.0)

    # the {b, c} score is computed from the expression, never typed as a literal
    subsets = best_subsets(
        2000.0,
        [
            ("a", 0, (1950.0, 2000.0, 2050.0)),
            ("b", 0, (850.0, 900.0, 1050.0)),
            ("c", 0, (850.0, 900.0, 1050.0)),
        ],
    )
    assert [keys for _score, keys in subsets] == [("b", "c")]
    expected = abs(2000.0 - 1800.0) / (2100.0 - 1800.0) + JOINT_PENALTY
    assert subsets[0][0] == pytest.approx(expected)
    assert subsets[0][0] - 0.0 >= AMBIGUITY_MARGIN


def test_two_containing_singletons_refuse_despite_better_subset() -> None:
    """P4 D-07 precedes D-41: two containing singletons refuse, whatever a subset says."""
    det = _detector(
        _switched("a", (1450.0, 1500.0, 2600.0)),
        _switched("d", (1350.0, 1400.0, 2700.0)),
        _switched("b", (990.0, 1000.0, 1010.0)),
        _switched("c", (990.0, 1000.0, 1010.0)),
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)

    results = _drive(det, r)

    res = _first_refused(results)
    assert res.overlap_refused == 1
    assert all(not step.on for step in results)
    # the pair {b, c} really is better than both singletons: it cannot rescue
    # an edge two single devices both explain.
    subs = best_subsets(
        2000.0,
        [
            ("a", 0, (1450.0, 1500.0, 2600.0)),
            ("d", 0, (1350.0, 1400.0, 2700.0)),
            ("b", 0, (990.0, 1000.0, 1010.0)),
            ("c", 0, (990.0, 1000.0, 1010.0)),
        ],
    )
    pair = next(score for score, keys in subs if keys == ("b", "c"))
    a_score = abs(2000.0 - 1500.0) / (2600.0 - 1500.0)
    d_score = abs(2000.0 - 1400.0) / (2700.0 - 1400.0)
    assert pair < a_score and pair < d_score
    assert pair == pytest.approx(abs(2000.0 - 2000.0) / (2020.0 - 2000.0) + JOINT_PENALTY)


def test_joint_split_proportional_to_p50() -> None:
    """D-36: delta x p50 / sum p50, clamped to each member's own box."""
    kettle: Explanation = ("kettle", 0, (1950.0, 2000.0, 2050.0))
    heater: Explanation = ("heater", 0, (750.0, 850.0, 950.0))

    inside = dict(split_delta(2800.0, [kettle, heater]))
    assert inside["kettle"] == pytest.approx(2800.0 * 2000.0 / 2850.0, abs=1e-6)
    assert inside["heater"] == pytest.approx(2800.0 * 850.0 / 2850.0, abs=1e-6)
    assert 1950.0 <= inside["kettle"] <= 2050.0
    assert 750.0 <= inside["heater"] <= 950.0

    clamped = dict(split_delta(3000.0, [kettle, heater]))
    assert clamped["kettle"] == 2050.0  # the raw share leaves its box
    assert clamped["heater"] == pytest.approx(3000.0 * 850.0 / 2850.0, abs=1e-6)

    # the jointly attributed devices report those shares as their step
    det = _detector(
        _switched("kettle", (1950.0, 2000.0, 2050.0)),
        _switched("heater", (750.0, 850.0, 950.0)),
    )
    r = grid_trace(60, noise_w=0.0)
    add_run(r, 10, 6, 2800.0)
    joint = next(res for res in _drive(det, r) if res.joint_attributed >= 1)
    assert joint.device_w["kettle"] == pytest.approx(inside["kettle"])
    assert joint.device_w["heater"] == pytest.approx(inside["heater"])


# -- must_have edges: bounds, ordering, single candidates, NaN ------------


def test_subset_summed_bounds_are_inclusive() -> None:
    """A delta exactly on a subset's summed lo or hi counts as contained."""
    box = (900.0, 1000.0, 1100.0)
    for delta, share in ((1800.0, 900.0), (2200.0, 1100.0)):
        det = _detector(_switched("a", box), _switched("b", box))
        r = grid_trace(40, noise_w=0.0)
        add_run(r, 10, 4, delta)
        results = _drive(det, r)
        joint = next(res for res in results if res.joint_attributed >= 1)
        assert joint.on == frozenset({"a", "b"})
        assert joint.device_w["a"] == pytest.approx(share)
        assert joint.device_w["b"] == pytest.approx(share)
        assert joint.joint_attributed == 1


def test_margin_boundary_resolves_to_the_best() -> None:
    """A best and runner-up differing by exactly the margin resolve to the best."""
    attribute = pick(
        [(0.0, "a", True), (AMBIGUITY_MARGIN, "b", False)], [], AMBIGUITY_MARGIN
    )
    assert attribute == ("attribute", (("a",),))
    refused = pick(
        [(0.0, "a", True), (AMBIGUITY_MARGIN - 1e-9, "b", False)],
        [],
        AMBIGUITY_MARGIN,
    )
    assert refused[0] == "refuse"
    assert refused[1] == (("a",),)  # only the containing singleton competes
    # two containing singletons refuse outright on the rise path, however far
    # the runner-up scores
    strict = pick([(0.0, "a", True), (99.0, "b", True)], [], AMBIGUITY_MARGIN)
    assert strict == ("refuse", (("a",), ("b",)))


def test_no_containing_entry_counts_only_unattributed() -> None:
    """An edge no entry explains leaves the overlap counters untouched (D-07)."""
    det = _detector(_switched("k", (3000.0, 3100.0, 3200.0)))
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 500.0)  # 500 W is inside no box

    results = _drive(det, r)

    rise = next(res for res in results if res.unattributed_total >= 1)
    assert rise.unattributed_total == 1
    assert rise.overlap_refused == 0
    assert rise.joint_attributed == 0
    assert "k" not in rise.on
    assert results[-1].unattributed_edges["k"] == 0


def test_single_plausible_candidate_decides_as_today() -> None:
    """Exactly one containing singleton: no subset exists, it decides as before."""
    det = _detector(
        _switched("k", (1950.0, 2000.0, 2050.0)),
        _switched("far", (100.0, 400.0, 700.0)),
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)

    results = _drive(det, r)

    assert any("k" in res.on for res in results)
    assert all("far" not in res.on for res in results)
    assert results[-1].joint_attributed == 0
    assert results[-1].overlap_refused == 0
    k_on = next(res for res in results if "k" in res.on)
    assert k_on.device_w["k"] == pytest.approx(2000.0)


def test_nan_sample_attributes_nothing() -> None:
    """A NaN sample never changes an overlap counter (DET-03 bookkeeping)."""
    det = _detector(
        _switched("a", (950.0, 1000.0, 1050.0)),
        _switched("b", (950.0, 1000.0, 1050.0)),
        _switched("c", (950.0, 1000.0, 1050.0)),
    )
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)  # three tying pairs: refused
    results = _drive(det, r)
    refused = results[-1]
    assert refused.overlap_refused == 1

    nan_res = det.step(1000.0 * GRID_S, math.nan)
    assert nan_res.overlap_refused == refused.overlap_refused
    assert nan_res.joint_attributed == 0
    assert nan_res.estimated_w == 0.0


def test_prefilter_bounds_and_drops_non_contributing() -> None:
    """The prefilter drops impossible boxes and keeps JOINT_PREFILTER candidates."""
    rise: list[Explanation] = [("high", 0, (3000.0, 3100.0, 3200.0))] + [
        (f"d{k}", 0, (900.0 + 10.0 * k, 1000.0 + 10.0 * k, 1100.0 + 10.0 * k))
        for k in range(8)
    ]
    kept = prefilter(2000.0, rise, JOINT_PREFILTER)
    keys = [key for key, _a, _b in kept]
    assert "high" not in keys  # its lo (3000) already exceeds the delta
    assert len(kept) == JOINT_PREFILTER
    # the six nearest by (distance, key): d2..d7, the two farthest dropped
    assert set(keys) == {f"d{k}" for k in range(2, 8)}
    # the scorer itself sorts the prefiltered pool by key, so any input order
    # yields the same enumeration (D-41)
    assert best_subsets(2000.0, kept) == best_subsets(2000.0, list(reversed(kept)))

    fall: list[Explanation] = [("low", 0, (-900.0, -800.0, -700.0))]
    assert prefilter(-800.0, fall, JOINT_PREFILTER)[0][0] == "low"
    # a box lying entirely below the delta cannot contribute to a shallower sum
    deep: list[Explanation] = [("deep", 0, (-3100.0, -3000.0, -2900.0))]
    assert prefilter(-2000.0, deep, JOINT_PREFILTER) == []
    # ... but it can contribute to a deeper summed fall
    assert prefilter(-3500.0, deep, JOINT_PREFILTER)[0][0] == "deep"


def test_best_subsets_k_cap_and_penalty() -> None:
    """Subsets of 2..JOINT_MAX_K only, each paying the per-extra-device penalty."""
    box = (900.0, 1000.0, 1100.0)
    boxes: list[Explanation] = [(f"d{k}", 0, box) for k in range(4)]
    pairs = best_subsets(2000.0, boxes)
    assert all(len(keys) == 2 for _s, keys in pairs)
    assert all(score == pytest.approx(JOINT_PENALTY) for score, _k in pairs)

    triple = best_subsets(
        3000.0, [("a", 0, box), ("b", 0, box), ("c", 0, box)], max_k=JOINT_MAX_K
    )
    assert [keys for _s, keys in triple] == [("a", "b", "c")]
    assert triple[0][0] == pytest.approx(
        abs(3000.0 - 3000.0) / (3300.0 - 3000.0) + JOINT_PENALTY * 2
    )


# -- the drift coupling (review L133/L139@8b3be9c) ------------------------


def test_overlap_refusal_counts_in_drift_refused_delta() -> None:
    """Overlap refusals reach LiveDrift as B's refused delta, once per edge."""
    det = _detector(
        _switched("a", (1950.0, 2000.0, 2050.0)),
        _switched("b", (900.0, 950.0, 1000.0)),
        _switched("c", (1000.0, 1050.0, 1100.0)),
        _switched("d", (950.0, 1050.0, 1150.0)),
    )
    r = grid_trace(300, noise_w=0.0)
    # one plain episode for B (alone: nothing else contains 920 W)
    add_run(r, 10, 4, 920.0)
    # three D-41 tie edges: A contains 2000, and B belongs to two competing
    # containing subsets ({b, c} and {b, d}) while {c, d} is not competitive
    add_run(r, 40, 4, 2000.0)
    add_run(r, 60, 4, 2000.0)
    add_run(r, 80, 4, 2000.0)
    # B's second episode, so the refused burst lands in the next close
    add_run(r, 110, 4, 920.0)

    results = _drive(det, r)
    tally = results[-1].unattributed_edges
    assert tally["b"] == 3, tally  # 3 refusals, the two-subset edge counted once
    assert tally["c"] == 3 and tally["d"] == 3  # c and d compete in two edges each
    assert tally["a"] == 3
    assert results[-1].overlap_refused == 3

    closed = det.drain_closed()
    b_closes = [ep for ep in closed if ep.key == "b"]
    assert len(b_closes) == 2
    tracker = LiveDrift(kind="switched")
    for ep in b_closes:
        tracker.record(ep.end_ts, ep.match_score, ep.unattributed)
    evidence = tracker.to_dict()["evidence"]
    assert len(evidence) == 2
    assert evidence[0]["refused"] == 0  # the first close has no delta
    assert evidence[1]["refused"] == 3


# -- joint falls, programs and the counters' surface ----------------------


def test_joint_fall_switches_both_off() -> None:
    """Kettle and heater switching OFF in one sample turn OFF together (D-41)."""
    det = _detector(
        _switched("kettle", (1950.0, 2000.0, 2050.0)),
        _switched("heater", (750.0, 800.0, 850.0)),
    )
    r = grid_trace(60, noise_w=0.0)
    add_run(r, 10, 6, 2800.0)

    results = _drive(det, r)

    # the rise and the fall are each attributed jointly, neither absorbing the
    # other and nothing left ambiguous
    assert results[-1].joint_attributed == 2
    assert results[-1].overlap_refused == 0
    assert results[-1].on == frozenset()
    assert results[-1].device_w == {"heater": 0.0, "kettle": 0.0}
    assert det.energy_wh["kettle"] > 0.0 and det.energy_wh["heater"] > 0.0
    # each member is verified against its own D-36 share by the unchanged box
    # rule: the neighbouring load contaminates both plateaus, so both closes are
    # unverified — exactly the existing verify semantics, no extra inflation.
    assert results[-1].unverified_episodes == {"heater": 1, "kettle": 1}


def _dishwasher_model():
    return compile_model("dishwasher", program_artefact("dishwasher", dishwasher_stages()))


def test_program_stage_and_kettle_rise_attributed_jointly() -> None:
    """A stage-3 entry and a kettle switching in one sample are attributed jointly."""
    dishwasher = _dishwasher_model()
    kettle = _switched("kettle", (1050.0, 1100.0, 1150.0))
    det = _detector(dishwasher, kettle)
    r = grid_trace(260)
    # samples 101..186: stage 1 (2000 W), stage 2 (150 W), stage 3 (2000 W), stage 4
    program_run(r, 100, dishwasher_stages())
    # the kettle joins on the very sample the program enters stage 3: the edge is
    # the stage-3 entry (+1850) plus the kettle (+1100) in one 30 s sample
    kettle_overlay(r, 150, 3, w=1100.0)

    results = _drive(det, r)

    joint = next(res for res in results if res.joint_attributed >= 1)
    # the program advanced into stage 3 (its level box, not stage 2's 150 W) and
    # the kettle turned ON with its D-36 share in the same sample
    assert joint.on == frozenset({"dishwasher", "kettle"})
    assert joint.device_w["dishwasher"] == pytest.approx(2000.0, abs=100.0)
    assert joint.device_w["kettle"] == pytest.approx(1100.0, abs=60.0)
    assert det._runtimes["dishwasher"].stage_index >= 2
    assert "dishwasher" not in det.unknown
    assert results[-1].overlap_refused == 0
    kettle_on = [res for res in results if "kettle" in res.on]
    assert kettle_on, "the kettle was never attributed"


def test_program_stage_and_switched_device_tie_refused() -> None:
    """A stage entry and a switched box containing the same edge refuse (D-07)."""
    dishwasher = _dishwasher_model()
    near = _switched("near", (1800.0, 1850.0, 1900.0))
    det = _detector(dishwasher, near)
    r = grid_trace(200)
    program_run(r, 100, dishwasher_stages())

    results = _drive(det, r)

    res = _first_refused(results)
    assert res.overlap_refused == 1
    assert res.unattributed_edges["dishwasher"] >= 1
    assert res.unattributed_edges["near"] == 1
    assert "near" not in res.on
    assert det._runtimes["dishwasher"].program_phase != "unknown"
    assert res.joint_attributed == 0


def test_replay_reports_overlap_counters() -> None:
    """replay_online reports the totals and, with collect_power, the cum arrays."""
    artefacts = [
        switched_artefact("kettle", rise=(1950.0, 2000.0, 2050.0)),
        switched_artefact("heater", rise=(750.0, 800.0, 850.0)),
        switched_artefact("d", rise=(3900.0, 4000.0, 4100.0)),
        switched_artefact("e", rise=(3900.0, 4000.0, 4100.0)),
    ]
    r = grid_trace(60, noise_w=0.0)
    # d and e both contain 4000 W: two containing singletons, refused (D-07)
    add_run(r, 20, 4, 4000.0)
    # kettle + heater together at 2800 W: no other entry contains it (the 4000 W
    # pair cannot contribute), so it is attributed jointly (D-36/D-41)
    add_run(r, 50, 20, 2800.0)

    rep = replay_online(artefacts, r, float(GRID_S), collect_power=True)
    assert rep.overlap_refused == 1
    assert rep.joint_attributed == 1
    assert rep.overlap_refused_cum is not None
    assert rep.joint_attributed_cum is not None
    assert len(rep.overlap_refused_cum) == len(r)
    assert rep.overlap_refused_cum[-1] == rep.overlap_refused
    assert rep.joint_attributed_cum[-1] == rep.joint_attributed
    assert np.all(np.diff(rep.overlap_refused_cum) >= 0)
    assert np.all(np.diff(rep.joint_attributed_cum) >= 0)

    # the default path stays allocation-free but reports the same totals
    plain = replay_online(artefacts, r, float(GRID_S))
    assert plain.overlap_refused_cum is None and plain.joint_attributed_cum is None
    assert plain.overlap_refused == 1 and plain.joint_attributed == 1


def test_state_restore_keeps_overlap_counters() -> None:
    """state() carries the overlap counters and restore() reads them back."""
    boxes = [
        _switched("a", (950.0, 1000.0, 1050.0)),
        _switched("b", (950.0, 1000.0, 1050.0)),
        _switched("c", (950.0, 1000.0, 1050.0)),
    ]
    det = _detector(*boxes)
    r = grid_trace(40, noise_w=0.0)
    add_run(r, 10, 4, 2000.0)
    _drive(det, r)

    state = json.loads(json.dumps(det.state()))  # JSON-safe
    assert state["overlap_refused_total"] == 1
    assert state["joint_attributed_total"] == 0
    assert state["overlap_refused_by_device"] == {"a": 1, "b": 1, "c": 1}

    det2 = _detector(*boxes)
    det2.restore(state, float(len(r)) * GRID_S)
    assert det2.state()["overlap_refused_total"] == 1
    assert det2.state()["joint_attributed_total"] == 0
    assert det2.state()["overlap_refused_by_device"] == {"a": 1, "b": 1, "c": 1}
    # a restored detector keeps counting from the restored value
    r2 = grid_trace(40, noise_w=0.0)
    add_run(r2, 10, 4, 2000.0)
    later = [
        det2.step(float(len(r) + i) * GRID_S, float(r2[i])) for i in range(len(r2))
    ]
    assert later[-1].overlap_refused == 2
