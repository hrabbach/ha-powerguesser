"""Replay parity: the online detector claims what the learning replay claims.

14 days at 30 s with isolated runs of the four SC2 switching loads after
D-24 (kettle, water heater, oven, resistive dryer element) plus noise;
artefacts are built from the planted episodes with the compile.py box
widening rule.  On isolated loads replay_online must report exactly
score.replay's claims (as (start_i, end_i) of the claimed table rows); with
overlapping oven sessions every online episode must still be an offline
claim and at least 90 % of the offline claims must be found online.
"""

import numpy as np

from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.intervals import EventType
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact
from custom_components.powerguesser.engine.learning.episodes import (
    detect_edges,
    pair_switched,
    suppress_alignment_artefacts,
)
from custom_components.powerguesser.engine.learning.score import replay as score_replay
from tests.fixtures.detect_traces import add_run, cycling_run, grid_trace, switched_artefact

STEP_S = 30.0
GRID_S = 30
THRESHOLD = 120.0
DAYS = 14
N = int(DAYS * 24 * 3600 / GRID_S)  # 40320


def _fit_boxes(eps: list[dict], threshold: float = THRESHOLD) -> dict:
    """compile.py box widening (lines 66-90) over planted episodes."""
    WIDEN_FRAC = 0.05

    def box(key: str, widen: bool) -> tuple[float, float, float]:
        vals = np.array([e[key] for e in eps], dtype=np.float64)
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        if widen:
            lo = p5 - max(threshold / 2.0, WIDEN_FRAC * abs(p5))
            hi = p95 + max(threshold / 2.0, WIDEN_FRAC * abs(p95))
        else:
            lo = max(0.0, p5 - WIDEN_FRAC * p5)
            hi = p95 + WIDEN_FRAC * p95
        return (round(float(lo), 1), round(float(p50), 1), round(float(hi), 1))

    rise = box("rise_w", True)
    fall = box("fall_w", True)
    dur = box("dur_s", False)
    plateau = box("plateau_w", True)
    return switched_artefact(
        "dev",
        rise=rise,
        fall=(-fall[2], -fall[1], -fall[0]),
        dur=dur,
        plateau=plateau,
    )


def _episode(r: np.ndarray, start_i: int, n: int, w: float) -> dict:
    add_run(r, start_i, n, w)
    return {
        "rise_w": float(w),
        "fall_w": -float(w),
        "dur_s": float(n) * STEP_S,
        "plateau_w": float(w),
    }


def _build_isolated_trace() -> tuple[np.ndarray, dict]:
    r = grid_trace(N, noise_w=3.0, seed=11)
    eps: dict[str, list[dict]] = {"kettle": [], "water_heater": [], "oven": [], "dryer": []}
    for start in (200, 3000, 9000, 16000, 24000, 33000):
        eps["kettle"].append(_episode(r, start, 6, 2000.0))
    for start in (1200, 11000, 21000, 31000):
        eps["water_heater"].append(_episode(r, start, 60, 3000.0))
    for start in (5000, 25000):
        cycling_run(r, start, on_samples=12, off_samples=9, cycles=6, w=2200.0)
        for _ in range(6):
            eps["oven"].append(
                {"rise_w": 2200.0, "fall_w": -2200.0, "dur_s": 12 * STEP_S, "plateau_w": 2200.0}
            )
    # dryer: 250 W drum-motor baseline step with 2400 W element cycles on top
    for start in (7000, 28000):
        add_run(r, start, 130, 250.0)  # motor baseline for the whole session
        cycling_run(r, start + 2, on_samples=8, off_samples=5, cycles=8, w=2400.0)
        for _ in range(8):
            eps["dryer"].append(
                {"rise_w": 2400.0, "fall_w": -2400.0, "dur_s": 8 * STEP_S, "plateau_w": 2400.0}
            )
        r[start + 131] -= 250.0  # motor off again: its edges are claimed by neither
    artes = {key: _fit_boxes(val) for key, val in eps.items()}
    for key, arte in artes.items():
        arte["signature_id"] = key
    return r, artes


def _offline_claims(r: np.ndarray, artes: dict[str, dict]):
    idx, delta = detect_edges(r, THRESHOLD)
    idx, delta = suppress_alignment_artefacts(idx, delta, enabled_event_types=[EventType.STEP])
    table = pair_switched(r, idx, delta, STEP_S)
    block = np.zeros(len(table["start_i"]), dtype=np.int32)
    result = score_replay(
        [SignatureArtefact.from_dict(a) for a in artes.values()],
        {**table, "block": block},
        0,
        residual_wh=1.0,
    )
    claims: dict[str, set[tuple[int, int]]] = {}
    for key in artes:
        claims[key] = {
            (int(table["start_i"][i]), int(table["end_i"][i]))
            for i in result.claims[key]
        }
    return claims, table


def test_parity_isolated_loads() -> None:
    """Isolated SC2 runs: online episodes equal score.replay claims exactly."""
    r, artes = _build_isolated_trace()
    claims, table = _offline_claims(r, artes)
    rep = replay_online(list(artes.values()), r, STEP_S)

    for key in ("kettle", "water_heater", "oven", "dryer"):
        online = {(e.start_i, e.end_i) for e in rep.episodes.get(key, [])}
        assert online == claims[key], (
            f"{key}: online {sorted(online)} != offline claims {sorted(claims[key])}"
        )
    # the dryer motor step (250 W) is claimed by neither side
    pos = {(int(s), int(e)): i for i, (s, e) in enumerate(zip(table["start_i"], table["end_i"], strict=True))}
    for key_claims in claims.values():
        for pair in key_claims:
            assert abs(float(table["rise_w"][pos[pair]])) > 1000.0


def _add_overlap_ovens(r: np.ndarray) -> None:
    """Extra oven sessions nested around kettle runs (edges never coincide and
    start/end order is preserved, so online box pairing and offline LIFO
    pairing agree; the crossing case is the flagged real-data risk)."""
    for start in (2995, 8995, 15990, 23995):
        cycling_run(r, start, on_samples=30, off_samples=9, cycles=2, w=2200.0)


def _overlap_oven_artefact() -> dict:
    eps = [
        {"rise_w": 2200.0, "fall_w": -2200.0, "dur_s": d * STEP_S, "plateau_w": 2200.0}
        for d in ([12] * 8 + [30] * 8)
    ]
    arte = _fit_boxes(eps)
    arte["signature_id"] = "oven"
    return arte


def test_parity_overlaps_precision() -> None:
    """Overlapping oven sessions: every online episode is an offline claim."""
    r, artes = _build_isolated_trace()
    _add_overlap_ovens(r)
    artes["oven"] = _overlap_oven_artefact()
    for key, a in artes.items():
        a["signature_id"] = key

    claims, _table = _offline_claims(r, artes)
    rep = replay_online(list(artes.values()), r, STEP_S)

    n_claims = sum(len(c) for c in claims.values())
    n_found = 0
    for key in artes:
        online = {(e.start_i, e.end_i) for e in rep.episodes.get(key, [])}
        assert online <= claims[key], f"{key}: online episodes not offline claims: {online - claims[key]}"
        n_found += len(online & claims[key])
    assert n_found >= 0.9 * n_claims, f"only {n_found}/{n_claims} offline claims found online"


def test_constants_tuned_on_replay() -> None:
    """Default constants: zero wrong-device episodes; refused-edge data printed."""
    r, artes = _build_isolated_trace()
    _add_overlap_ovens(r)
    artes["oven"] = _overlap_oven_artefact()
    for key, a in artes.items():
        a["signature_id"] = key

    claims, _table = _offline_claims(r, artes)
    rep = replay_online(list(artes.values()), r, STEP_S)
    for key in artes:
        for e in rep.episodes.get(key, []):
            assert (e.start_i, e.end_i) in claims[key], f"{key}: wrong-device episode {e}"
    print(
        "refused-edge data on the overlap trace:",
        f"unattributed_total={rep.unattributed_total}",
        f"forced_off={rep.forced_off}",
        f"unverified={rep.unverified}",
    )
