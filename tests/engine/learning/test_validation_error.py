"""D-16: the learning-split validation error of a switched candidate (Plan 04-07).

validation_error_pct measures live-detector vs learning-extraction energy
consistency on the validation days: E_online is the summed energy of the
detector's verified episodes for the artefact alone, E_ref is the summed
offline energy of the validation episodes score.replay claims for it, so both
sides cover the same episode population. Energy committed for episodes that
fail the switched match rule (e.g. a duration outside the learned dur box)
enters neither sum and cannot inflate the figure (D-16).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.replay import (
    replay_online,
    validation_error_pct,
)
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact
from custom_components.powerguesser.engine.learning.score import replay as score_replay
from custom_components.powerguesser.engine.learning.worker import run_stage, stage_outputs
from tests.fixtures.detect_traces import add_run, grid_trace, switched_artefact
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds

STEP_S = 30.0


def _kettle_artefact(sig_id: str = "sig-kettle") -> dict[str, Any]:
    """A switched artefact whose boxes contain the planted 2000 W kettle runs."""
    return switched_artefact(
        sig_id,
        rise=(1800.0, 2000.0, 2200.0),
        fall=(-2200.0, -2000.0, -1800.0),
        dur=(120.0, 180.0, 300.0),
        plateau=(1800.0, 2000.0, 2200.0),
    )


def _table(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build an episodes table the way score.replay consumes it."""
    keys = ("start_i", "end_i", "rise_w", "fall_w", "dur_s", "plateau_w", "energy_wh", "kind", "block")
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            out[k].append(row[k])
    return {
        "start_i": np.array(out["start_i"], dtype=np.int64),
        "end_i": np.array(out["end_i"], dtype=np.int64),
        "rise_w": np.array(out["rise_w"], dtype=np.float64),
        "fall_w": np.array(out["fall_w"], dtype=np.float64),
        "dur_s": np.array(out["dur_s"], dtype=np.float64),
        "plateau_w": np.array(out["plateau_w"], dtype=np.float64),
        "energy_wh": np.array(out["energy_wh"], dtype=np.float64),
        "kind": np.array(out["kind"], dtype=np.int32),
        "block": np.array(out["block"], dtype=np.int32),
    }


def _episode(start_i: int, n_samples: int, w: float = 2000.0) -> dict[str, Any]:
    """One planted switched episode matching add_run's rectangle semantics."""
    return {
        "start_i": start_i,
        "end_i": start_i + n_samples,
        "rise_w": w,
        "fall_w": -w,
        "dur_s": n_samples * STEP_S,
        "plateau_w": w,
        "energy_wh": w * n_samples * STEP_S / 3600.0,
        "kind": 0,
        "block": 1,
    }


def _reference_wh(artefact: dict[str, Any], table: dict[str, np.ndarray]) -> float:
    """E_ref: the summed offline energy of the claimed switched validation episodes."""
    art = SignatureArtefact.from_dict(artefact)
    val_res = score_replay([art], table, block=1, residual_wh=1.0)
    rows = [
        int(i)
        for i in val_res.claims[art.signature_id]
        if int(table["kind"][int(i)]) == 0
    ]
    return float(np.sum(table["energy_wh"][rows])) if rows else 0.0


def test_validation_error_pct_matches_definition() -> None:
    """The value equals 100 x |E_online - E_ref| / E_ref over the same population.

    Trace: two kettle runs the extraction claims (180 s and 240 s) plus one
    360 s run whose duration is above the dur box — score.replay does not
    claim it and the detector reports it unverified, so it enters neither sum.
    """
    artefact = _kettle_artefact()
    r = grid_trace(2880, base_w=250.0, noise_w=15.0, seed=11)
    add_run(r, 100, 6, 2000.0)  # 180 s claimed run
    add_run(r, 400, 8, 2000.0)  # 240 s claimed run
    add_run(r, 700, 12, 2000.0)  # 360 s run: above the dur box, unverified
    table = _table([_episode(100, 6), _episode(400, 8)])

    reference_wh = _reference_wh(artefact, table)
    assert reference_wh == pytest.approx(100.0 + 2000.0 * 8 * STEP_S / 3600.0)

    res = replay_online([artefact], r, STEP_S)
    sig = artefact["signature_id"]
    e_online = sum(ep.energy_wh for ep in res.episodes[sig])
    # The verified population is the two claimed runs only: the 360 s run is
    # excluded from the episode list (it is counted unverified instead).
    assert e_online == pytest.approx(reference_wh, rel=0.02)
    assert res.unverified >= 1
    unverified_energy = res.energy_wh[sig] - e_online
    assert unverified_energy == pytest.approx(
        _lone_unverified_committed_wh(artefact, seed=11, start_i=700), rel=0.01
    )

    expected = round(100.0 * abs(e_online - reference_wh) / reference_wh, 1)
    assert validation_error_pct(artefact, r, STEP_S, reference_wh) == expected


def _lone_unverified_committed_wh(
    artefact: dict[str, Any], *, seed: int, start_i: int
) -> float:
    """The energy a lone 360 s above-the-box run commits while staying unverified.

    The trace uses the same seed and run position as the calling test so the
    measured on-step (edge samples carry the base noise) is identical.
    """
    r = grid_trace(2880, base_w=250.0, noise_w=15.0, seed=seed)
    add_run(r, start_i, 12, 2000.0)
    res = replay_online([artefact], r, STEP_S)
    sig = artefact["signature_id"]
    assert res.unverified == 1
    assert res.episodes.get(sig, []) == []
    return res.energy_wh[sig]


def test_validation_error_excludes_unverified_energy() -> None:
    """A run the detector commits but cannot verify never reaches the figure.

    The unverified run's duration (360 s) is above the artefact's dur box hi
    (300 s) but below the stuck-ON cap (1.5 x 300 s = 450 s), so the guard
    does not force it OFF first: the detector commits energy for it and counts
    it unverified, score.replay does not claim it, and validation_error_pct
    equals the value from the verified and claimed episodes only — never the
    value computed against OnlineReplay.energy_wh.
    """
    artefact = _kettle_artefact()
    r = grid_trace(2880, base_w=250.0, noise_w=15.0, seed=23)
    add_run(r, 100, 6, 2000.0)  # 180 s: claimed and verified
    add_run(r, 400, 12, 2000.0)  # 360 s: committed but unverified
    table = _table([_episode(100, 6)])

    reference_wh = _reference_wh(artefact, table)
    assert reference_wh == pytest.approx(100.0)

    res = replay_online([artefact], r, STEP_S)
    sig = artefact["signature_id"]
    e_verified = sum(ep.energy_wh for ep in res.episodes[sig])
    e_committed = res.energy_wh[sig]
    assert res.unverified == 1
    assert e_verified == pytest.approx(reference_wh, rel=0.02)
    assert e_committed - e_verified == pytest.approx(
        _lone_unverified_committed_wh(artefact, seed=23, start_i=400), rel=0.01
    )

    expected = round(100.0 * abs(e_verified - reference_wh) / reference_wh, 1)
    with_total = round(100.0 * abs(e_committed - reference_wh) / reference_wh, 1)
    assert expected != with_total
    assert validation_error_pct(artefact, r, STEP_S, reference_wh) == expected


def test_validation_error_none_for_preview_program_and_empty_reference() -> None:
    """None when the reference energy is 0 or the artefact is not switched.

    The preview case is the validated=False compile job (E_ref stays 0 because
    score.replay runs against the validation block of a preview-less split);
    here the helper contract is pinned directly plus the program kind.
    """
    artefact = _kettle_artefact()
    r = grid_trace(720, base_w=250.0, noise_w=15.0, seed=31)
    add_run(r, 50, 6, 2000.0)

    assert validation_error_pct(artefact, r, STEP_S, 0.0) is None
    assert validation_error_pct(artefact, r, STEP_S, -1.0) is None
    program_art = {**artefact, "kind": "program"}
    assert validation_error_pct(program_art, r, STEP_S, 100.0) is None


def test_rescore_outputs_validation_error_pct(tmp_path: Path) -> None:
    """The rescore stage_outputs entry keeps the key "share" and carries the figure.

    The whitelist rebuild must pass "validation_error_pct" through (None or
    float); renaming rescore's "share" to the compile-side "claimed_share"
    would silently null the merge and is pinned here (review concern
    L197@4e11826).
    """
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)
    buf_path = tmp_path / "buf_rescore_vep.sqlite"
    stage_dir = tmp_path / "stage_rescore_vep"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)
    first_weekday = datetime.fromtimestamp(
        day_bounds_ms[0] / 1000.0, tz=timezone.utc
    ).weekday()

    program_art: dict[str, Any] = {
        "schema_version": 1,
        "signature_id": "sig-program",
        "kind": "program",
        "label": "Dishwasher",
        "category": "appliance",
        "extraction": {"grid_s": 30, "edge_threshold_w": 120.0},
        "states": [],
        "session": {
            "energy_wh": {"lo": 1.0, "hi": 1e9},
            "dur_s": {"lo": 1.0, "hi": 1e9},
            "n_segments": {"lo": 1, "hi": 1000},
        },
        "time_prior": {},
        "constraints": {},
        "provenance": {},
        "scores": {},
    }
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "candidates": [_kettle_artefact(), program_art],
    }
    res = run_stage("rescore", job)
    out = stage_outputs("rescore", res)

    kettle_entry = out["per_signature"]["sig-kettle"]
    assert set(kettle_entry.keys()) == {"share", "n", "validation_error_pct"}
    assert isinstance(kettle_entry["share"], float)
    assert isinstance(kettle_entry["validation_error_pct"], float)
    program_entry = out["per_signature"]["sig-program"]
    assert set(program_entry.keys()) == {"share", "n", "validation_error_pct"}
    assert program_entry["validation_error_pct"] is None
