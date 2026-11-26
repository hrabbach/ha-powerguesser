"""Golden snapshot of the extract, compile_score and rescore stage outputs (Plan 04-13).

The snapshot was recorded from the UNCHANGED code and committed before any
production edit; it is never updated otherwise. Any change in what learning
extracts, compiles or scores fails ``test_extract_compile_and_rescore_unchanged_by_refactor``.

Deliberately regenerated exactly ONCE for Phase 8 D-01 (every compile_score
library dict now carries a ``fingerprint`` key). The regeneration is diff-checked
by ``test_candidates_without_fingerprint_unchanged``: the golden candidates minus
the ``fingerprint`` key must still equal the baseline recorded from the unchanged
compile code, so any change beyond that key fails. Every other part of the golden
(digest, digest_meta, episodes, rescore) stays byte-identical.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from custom_components.powerguesser.engine.learning.worker import run_stage
from tests.fixtures.learning.fake_llm import kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(AmberSnapshotExtension)


def _canon(obj: object) -> object:
    """Round-trip through sorted-key JSON so the snapshot holds plain JSON types."""
    return json.loads(json.dumps(obj, sort_keys=True))


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory):
    """Run extract, compile_score and rescore once on a fixed synthetic buffer.

    Same stage invocation shape as
    test_compile_score.py::test_rescore_replays_on_post_window_bounds with
    SynthSpec(days=22, oven=True, washer=True, spikes=True, gaps=True) so both
    switched and session episodes occur, and fixed values for every job field.
    """
    tmp = tmp_path_factory.mktemp("episode_golden")
    buf_path = tmp / "buf_golden.sqlite"
    stage_dir = tmp / "stage_golden"

    dt = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    total_days = 22
    all_bounds_ms = utc_day_bounds(end_ms, total_days)
    build_buffer(
        buf_path,
        SynthSpec(days=total_days, oven=True, washer=True, spikes=True, gaps=True),
        end_ms,
    )

    preview_days = 14
    preview_bounds = all_bounds_ms[: preview_days + 1]
    extract_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": preview_bounds,
        "first_weekday": datetime.fromtimestamp(
            preview_bounds[0] / 1000.0, tz=timezone.utc
        ).weekday(),
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }
    extract_res = run_stage("extract", extract_job)

    proposal = kettle_proposal(extract_res["digest"])
    (stage_dir / "proposal_1.json").write_text(json.dumps(proposal), encoding="utf-8")

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": "golden-run",
        "round": 1,
        "model": "golden-model",
        "proposal_path": str(stage_dir / "proposal_1.json"),
        "validated": False,
        "prior": [],
        "decisions": {},
    }
    compile_res = run_stage("compile_score", compile_job)
    candidates = json.loads(Path(compile_res["library"]).read_text(encoding="utf-8"))

    rescore_bounds = all_bounds_ms[preview_days:]
    rescore_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": rescore_bounds,
        "first_weekday": datetime.fromtimestamp(
            rescore_bounds[0] / 1000.0, tz=timezone.utc
        ).weekday(),
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "candidates": candidates,
    }
    run_stage("rescore", rescore_job)

    digest = json.loads((stage_dir / "digest.json").read_text(encoding="utf-8"))
    digest_meta = json.loads((stage_dir / "digest_meta.json").read_text(encoding="utf-8"))

    episodes = []
    with np.load(str(stage_dir / "episodes.npz"), allow_pickle=False) as data:
        for name in sorted(data.files):
            arr = data[name]
            episodes.append(
                {
                    "name": name,
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                }
            )

    rescore = json.loads((stage_dir / "rescore.json").read_text(encoding="utf-8"))

    # No captured field derives from wall-clock time — by inspection the digest
    # header/meta, candidate provenance and rescore.json carry only data-derived
    # values — so no key needs dropping here.
    snapshot_dict = {
        "digest": _canon(digest),
        "digest_meta": _canon(digest_meta),
        "episodes": episodes,
        "candidates": _canon(candidates),
        "rescore": _canon(rescore),
    }
    return SimpleNamespace(
        stage_dir=stage_dir,
        digest_meta=digest_meta,
        proposal=proposal,
        candidates=candidates,
        snapshot_dict=snapshot_dict,
    )


def test_extract_compile_and_rescore_unchanged_by_refactor(golden_run, snapshot) -> None:
    """The unchanged extract, compile_score and rescore outputs match the golden snapshot."""
    assert golden_run.snapshot_dict == snapshot


def test_candidates_without_fingerprint_unchanged(golden_run, snapshot) -> None:
    """D-01 pre-fingerprint baseline: the golden candidates with the "fingerprint"
    key removed equal this snapshot, recorded from unchanged compile code before
    any production edit.

    Phase 8 D-01 stamps a ``fingerprint`` on every compile_score library dict, so
    ``test_extract_compile_and_rescore_unchanged_by_refactor``'s golden changes by
    exactly that key. This baseline, recorded first, is what proves the golden diff
    is only the added ``fingerprint`` key and nothing else.
    """
    stripped = [
        {k: v for k, v in cand.items() if k != "fingerprint"}
        for cand in _canon(golden_run.candidates)
    ]
    assert stripped == snapshot


def test_golden_candidates_fingerprint_matches_artefact(golden_run) -> None:
    """D-01: every golden candidate's "fingerprint" equals SignatureArtefact.from_dict(candidate).fingerprint()."""
    from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact

    assert golden_run.candidates
    for cand in golden_run.candidates:
        assert cand["fingerprint"] == SignatureArtefact.from_dict(cand).fingerprint()
        assert cand["fingerprint"]["states"]


def test_fit_state_boxes_matches_compile_rules() -> None:
    """fit_state_boxes widens rise/fall/plateau by max(threshold_w/2, 0.05 x |p|), the dur box by 0.05 x p only."""
    from custom_components.powerguesser.engine.learning import compile as compile_mod

    fit_state_boxes = getattr(compile_mod, "fit_state_boxes", None)
    assert fit_state_boxes is not None, "compile.fit_state_boxes must exist (Plan 04-13)"

    table = {
        "rise_w": np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
        "fall_w": np.array([-5000.0, -4000.0, -3000.0, -2000.0, -1000.0]),
        "dur_s": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        "plateau_w": np.array([900.0, 1900.0, 2900.0, 3900.0, 4900.0]),
        "l1_share": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "l2_share": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        "l3_share": np.array([0.6, 0.5, 0.4, 0.3, 0.2]),
    }
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    state = fit_state_boxes(table, indices, 120.0, name="on")

    # rise p5/p50/p95 = 1200/3000/4800; widen by max(60, 0.05 x |p|)
    assert (state.rise_w.lo, state.rise_w.p50, state.rise_w.hi) == pytest.approx(
        (1140.0, 3000.0, 5040.0), abs=1e-9
    )
    # fall p5/p50/p95 = -4800/-3000/-1200
    assert (state.fall_w.lo, state.fall_w.p50, state.fall_w.hi) == pytest.approx(
        (-5040.0, -3000.0, -1140.0), abs=1e-9
    )
    # dur p5/p50/p95 = 120/300/480; the dur rule carries NO threshold_w term
    assert (state.dur_s.lo, state.dur_s.p50, state.dur_s.hi) == pytest.approx(
        (114.0, 300.0, 504.0), abs=1e-9
    )
    # plateau p5/p50/p95 = 1100/2900/4700
    assert (state.plateau_w.lo, state.plateau_w.p50, state.plateau_w.hi) == pytest.approx(
        (1040.0, 2900.0, 4935.0), abs=1e-9
    )
    # phase-share boxes are p5/p95 rounded to 2 decimals
    assert state.phase_share == {"l1": (0.12, 0.48), "l2": (0.3, 0.3), "l3": (0.22, 0.58)}
    assert state.name == "on"

    # Raising threshold_w widens rise/fall/plateau but leaves the dur box unchanged
    wide = fit_state_boxes(table, indices, 1000.0, name="on")
    assert (wide.rise_w.lo, wide.rise_w.hi) == pytest.approx((700.0, 5300.0), abs=1e-9)
    assert (wide.fall_w.lo, wide.fall_w.hi) == pytest.approx((-5300.0, -700.0), abs=1e-9)
    assert (wide.plateau_w.lo, wide.plateau_w.hi) == pytest.approx((600.0, 5200.0), abs=1e-9)
    assert (wide.dur_s.lo, wide.dur_s.p50, wide.dur_s.hi) == (
        state.dur_s.lo,
        state.dur_s.p50,
        state.dur_s.hi,
    )


def test_compile_uses_fit_state_boxes(golden_run) -> None:
    """compile_proposal's states for the golden kettle proposal equal fit_state_boxes on the same train indices."""
    from custom_components.powerguesser.engine.learning import compile as compile_mod
    from custom_components.powerguesser.engine.learning.artefact import State
    from custom_components.powerguesser.engine.learning.worker import load_episodes

    fit_state_boxes = getattr(compile_mod, "fit_state_boxes", None)
    assert fit_state_boxes is not None, "compile.fit_state_boxes must exist (Plan 04-13)"

    episodes = load_episodes(golden_run.stage_dir / "episodes.npz")
    cluster_map = golden_run.digest_meta["cluster_map"]
    threshold_w = float(golden_run.digest_meta["edge_threshold_w"])
    block_arr = episodes["block"]

    candidate = golden_run.candidates[0]
    device = golden_run.proposal["devices"][0]
    assert len(device["states"]) == len(candidate["states"])
    for st_prop, st_dict in zip(device["states"], candidate["states"]):
        cid = st_prop["cluster_ref"]
        raw_lab = cluster_map[cid]
        indices = np.array(
            [int(i) for i in np.where(episodes["cluster"] == int(raw_lab))[0] if block_arr[i] == 0],
            dtype=np.int64,
        )
        expected = fit_state_boxes(
            episodes, indices, threshold_w, name=str(st_prop["name"])
        )
        assert State.from_dict(st_dict) == expected
