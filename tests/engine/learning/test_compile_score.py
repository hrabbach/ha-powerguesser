"""Tests for proposal compilation, episode refitting, and replay scoring."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from custom_components.powerguesser.engine.learning.artefact import (
    Box,
    SignatureArtefact,
    State,
)
from custom_components.powerguesser.engine.learning.proposal import parse_proposal
from custom_components.powerguesser.engine.learning.worker import run_stage
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds


def test_compile_refits_kettle_from_episodes(tmp_path: Path) -> None:
    """A validated proposal is compiled into signature artefacts whose numbers come purely from real episodes."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_compile.sqlite"
    stage_dir = tmp_path / "stage_compile"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    extract_job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    extract_res = run_stage("extract", extract_job)
    digest = extract_res["digest"]

    # Build kettle proposal from the extracted digest
    proposal = kettle_proposal(digest)
    comp_json = chat_completion(json.dumps(proposal))
    message = comp_json["choices"][0]["message"]

    cluster_ids = [c["id"] for c in digest["clusters"]]
    valid_prop, errors, _ = parse_proposal(message, cluster_ids=cluster_ids, plug_ids=[])
    assert valid_prop is not None, f"Proposal validation failed: {errors}"

    proposal_path = stage_dir / "proposal_1.json"
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(valid_prop, f)

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": "test-run-1",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(proposal_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
    }

    compile_res = run_stage("compile_score", compile_job)
    assert compile_res["n_artefacts"] >= 1
    assert compile_res["unresolved"] == []

    lib_path = Path(compile_res["library"])
    assert lib_path.is_file()

    with open(lib_path, "r", encoding="utf-8") as f:
        artefact_dicts = json.load(f)

    kettle_artefact_dict = artefact_dicts[0]
    kettle_artefact = SignatureArtefact.from_dict(kettle_artefact_dict)

    state0 = kettle_artefact.states[0]
    # Check kettle rise_w.p50 is within 10% of 2000 W (1800 - 2200 W)
    assert 1800.0 <= state0.rise_w.p50 <= 2200.0, f"Expected ~2000 W rise, got {state0.rise_w.p50}"
    # Duration p50 between 60 and 300 s
    assert 60.0 <= state0.dur_s.p50 <= 300.0, f"Expected 60-300 s duration, got {state0.dur_s.p50}"


def test_compile_ignores_llm_numbers(tmp_path: Path) -> None:
    """Inserting arbitrary numbers into analysis, rationale, and label produces identical artefact fingerprints."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_ignore.sqlite"
    stage_dir = tmp_path / "stage_ignore"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    extract_job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    extract_res = run_stage("extract", extract_job)
    digest = extract_res["digest"]

    prop1 = kettle_proposal(digest)
    prop2 = kettle_proposal(digest)

    # Insert misleading numbers into text fields in prop2
    prop2["analysis"] = "Appliance consumes 2 kW with 180 s duration and 3500 W spikes."
    prop2["devices"][0]["rationale"] = "2 kW 180 s 3500 W nominal rating."
    prop2["devices"][0]["label"] = "Kettle 2000W 180s"

    p1_path = stage_dir / "proposal_1.json"
    p2_path = stage_dir / "proposal_2.json"
    with open(p1_path, "w", encoding="utf-8") as f:
        json.dump(prop1, f)
    with open(p2_path, "w", encoding="utf-8") as f:
        json.dump(prop2, f)

    job1 = {
        "stage_dir": str(stage_dir),
        "run_id": "run-a",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(p1_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
    }
    job2 = {
        "stage_dir": str(stage_dir),
        "run_id": "run-a",
        "round": 2,
        "model": "fake-model",
        "proposal_path": str(p2_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
    }

    res1 = run_stage("compile_score", job1)
    res2 = run_stage("compile_score", job2)

    with open(res1["library"], "r", encoding="utf-8") as f:
        art1_dict = json.load(f)[0]
    with open(res2["library"], "r", encoding="utf-8") as f:
        art2_dict = json.load(f)[0]

    art1 = SignatureArtefact.from_dict(art1_dict)
    art2 = SignatureArtefact.from_dict(art2_dict)

    # Fingerprint, time prior, and session are 100% identical
    assert art1.fingerprint() == art2.fingerprint()
    assert art1.time_prior == art2.time_prior
    assert art1.session == art2.session


def test_monotone_acceptance_boundary():
    """accept requires S_new >= S_best + 0.005 and conflict_share not higher."""
    from custom_components.powerguesser.engine.learning.score import accept

    incumbent = {"S": 0.500, "conflict_share": 0.10}

    # Exactly +0.005 -> True
    assert accept({"S": 0.505, "conflict_share": 0.10}, incumbent) is True

    # +0.0049 -> False
    assert accept({"S": 0.5049, "conflict_share": 0.10}, incumbent) is False

    # Equal S -> False
    assert accept({"S": 0.500, "conflict_share": 0.10}, incumbent) is False

    # S lower -> False
    assert accept({"S": 0.490, "conflict_share": 0.05}, incumbent) is False

    # Higher S but higher conflict share -> False
    assert accept({"S": 0.600, "conflict_share": 0.11}, incumbent) is False

    # First round (incumbent None) -> True
    assert accept({"S": 0.200, "conflict_share": 0.05}, None) is True


def test_empty_denominator_handling():
    """When block has 0 residual episode energy, S and conflict_share are 0.0."""
    from custom_components.powerguesser.engine.learning.score import ReplayResult, block_score

    empty_res = ReplayResult(
        per_signature={},
        conflict_wh=0.0,
        unique_wh=0.0,
        denominator_wh=0.0,
        claims={},
        conflicts_with={},
        conflict_episodes=set(),
    )
    s_score, conf_share = block_score(empty_res)
    assert s_score == 0.0
    assert conf_share == 0.0


def test_scorecard_no_validation_leakage(tmp_path: Path):
    """Train replay of member episodes is near-tautological by design (the boxes are percentiles of those same members, widened), so train scorecards describe coverage, not discrimination, and must never be used to tune constants or decide acceptance; only the validation S decides."""
    from custom_components.powerguesser.engine.learning.prompts import build_refine_request
    from custom_components.powerguesser.engine.llm_wire import Strategy

    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_leak.sqlite"
    stage_dir = tmp_path / "stage_leak"
    build_buffer(buf_path, SynthSpec(days=days, oven=True), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    extract_job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    extract_res = run_stage("extract", extract_job)
    digest = extract_res["digest"]

    prop = kettle_proposal(digest)
    prop_path = stage_dir / "proposal_1.json"
    with open(prop_path, "w", encoding="utf-8") as f:
        json.dump(prop, f)

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": "test-run-leak",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(prop_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
    }

    res = run_stage("compile_score", compile_job)

    # 1. Check scorecard keys: no key contains 'val'
    with open(res["scorecard"], "r", encoding="utf-8") as f:
        card_data = json.load(f)

    def _check_no_val_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "val" not in k.lower(), f"Key '{k}' in scorecard contains 'val'"
                _check_no_val_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                _check_no_val_keys(item)

    _check_no_val_keys(card_data)

    # 2. Check no validation numbers leak into scorecard or refine request
    with open(res["val"], "r", encoding="utf-8") as f:
        val_data = json.load(f)

    val_s_str = f"{val_data['S']:.3f}"
    card_raw = Path(res["scorecard"]).read_text(encoding="utf-8")

    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}
    refine_body = build_refine_request(
        model="fake-model",
        profile=profile,
        max_tokens=4000,
        digest=digest,
        previous=prop,
        scorecard=card_data,
    )
    refine_raw = json.dumps(refine_body)

    # Ensure validation score S does not leak if non-zero
    if val_data["S"] > 0:
        assert val_s_str not in card_raw
        assert val_s_str not in refine_raw


def test_rescore_replays_on_post_window_bounds(tmp_path: Path) -> None:
    """The rescore stage replays candidates on post-window whole days and writes rescore.json."""
    dt = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    total_days = 22
    all_bounds_ms = utc_day_bounds(end_ms, total_days)

    buf_path = tmp_path / "buf_rescore.sqlite"
    stage_dir = tmp_path / "stage_rescore"
    build_buffer(buf_path, SynthSpec(days=total_days), end_ms)

    # First compile kettle candidate from 14 preview days
    preview_days = 14
    preview_bounds = all_bounds_ms[: preview_days + 1]
    start_dt = datetime.fromtimestamp(preview_bounds[0] / 1000.0, tz=timezone.utc)
    extract_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": preview_bounds,
        "first_weekday": start_dt.weekday(),
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }
    extract_res = run_stage("extract", extract_job)
    prop = kettle_proposal(extract_res["digest"])
    prop_path = stage_dir / "proposal_1.json"
    prop_path.write_text(json.dumps(prop), encoding="utf-8")

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": "test-rescore-run",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(prop_path),
        "validated": False,
        "prior": [],
        "decisions": {},
    }
    compile_res = run_stage("compile_score", compile_job)
    with open(compile_res["library"], "r", encoding="utf-8") as f:
        candidates = json.load(f)
    assert len(candidates) == 1
    assert candidates[0]["scores"]["validated"] is False

    # Now run rescore on post-window days (days 14 to 22 = 8 days)
    rescore_bounds = all_bounds_ms[preview_days:]
    rescore_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": rescore_bounds,
        "first_weekday": datetime.fromtimestamp(rescore_bounds[0] / 1000.0, tz=timezone.utc).weekday(),
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "candidates": candidates,
    }
    rescore_out = run_stage("rescore", rescore_job)

    assert rescore_out["days"] == 8
    assert rescore_out["S"] > 0
    kettle_sig = candidates[0]["signature_id"]
    assert kettle_sig in rescore_out["per_signature"]
    stats = rescore_out["per_signature"][kettle_sig]
    assert stats["n"] > 0
    assert stats["share"] > 0

    # rescore.json was written atomically and matches output
    rescore_file = stage_dir / "rescore.json"
    assert rescore_file.exists()
    with open(rescore_file, "r", encoding="utf-8") as f:
        saved_dict = json.load(f)
    assert saved_dict == rescore_out


def test_rescore_with_zero_data_days_returns_zero_without_error(tmp_path: Path) -> None:
    """Rescore with zero data days in the bounds returns days 0 and per_signature {}."""
    stage_dir = tmp_path / "stage_zero"
    rescore_job = {
        "buffer_path": str(tmp_path / "nonexistent.sqlite"),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": [1_700_000_000_000],
        "candidates": [],
    }
    rescore_out = run_stage("rescore", rescore_job)
    assert rescore_out["days"] == 0
    assert rescore_out["S"] == 0.0
    assert rescore_out["conflict_share"] == 0.0
    assert rescore_out["per_signature"] == {}
    assert (stage_dir / "rescore.json").exists()



def _residual_wh(buf_path: Path, day_bounds_ms: list[int], days: list[int], grid_s: int = 30) -> float:
    """Independent reference: positive residual energy over the given whole days."""
    import numpy as np

    from custom_components.powerguesser.engine.learning.series import load_residual

    step_ms = grid_s * 1000
    total = 0.0
    for d in days:
        series = load_residual(
            str(buf_path), day_bounds_ms[d], day_bounds_ms[d + 1], step_ms, [], {}
        )
        r = series.residual[np.isfinite(series.residual)]
        total += float(np.sum(np.maximum(r, 0.0))) * grid_s / 3600.0
    return total


def test_shares_are_fractions_of_block_residual_energy(tmp_path: Path) -> None:
    """Train/val shares and S divide by the block's residual energy, like the digest (WR-12)."""
    import numpy as np

    from custom_components.powerguesser.engine.learning.score import replay
    from custom_components.powerguesser.engine.learning.worker import load_episodes

    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)
    buf_path = tmp_path / "buf_share.sqlite"
    stage_dir = tmp_path / "stage_share"
    synth = build_buffer(buf_path, SynthSpec(days=days), end_ms)

    extract_res = run_stage(
        "extract",
        {
            "buffer_path": str(buf_path),
            "stage_dir": str(stage_dir),
            "grid_s": 30,
            "day_bounds_ms": day_bounds_ms,
            "first_weekday": 0,
            "holdout_days": 0,
            "train_fraction": 0.7,
            "enabled_event_types": ["step", "cycle_shape", "continuous"],
            "metered_channels": [],
            "phase_channels": {},
        },
    )
    meta = extract_res["meta"]
    train_days = meta["days"]["train"]
    val_days = meta["days"]["val"]

    # The per-block residual energy is stored and uses the digest's residual definition
    block_res = meta["block_residual_wh"]
    ref_train = _residual_wh(buf_path, day_bounds_ms, train_days)
    ref_val = _residual_wh(buf_path, day_bounds_ms, val_days)
    assert block_res["train"] == pytest.approx(ref_train, rel=1e-6)
    assert block_res["val"] == pytest.approx(ref_val, rel=1e-6)
    assert extract_res["digest"]["header"]["residual_kwh"] == round(ref_train / 1000.0, 1)

    prop_path = stage_dir / "proposal_1.json"
    prop_path.write_text(json.dumps(kettle_proposal(extract_res["digest"])), encoding="utf-8")
    compile_res = run_stage(
        "compile_score",
        {
            "stage_dir": str(stage_dir),
            "run_id": "share-run",
            "round": 1,
            "model": "fake-model",
            "proposal_path": str(prop_path),
            "incumbent": None,
            "validated": True,
            "prior": [],
            "decisions": {},
        },
    )
    library = json.loads(Path(compile_res["library"]).read_text(encoding="utf-8"))
    kettle = library[0]
    arts = [SignatureArtefact.from_dict(kettle)]
    episodes = load_episodes(stage_dir / "episodes.npz")

    val_res = replay(arts, episodes, block=1, residual_wh=ref_val)
    assert val_res.denominator_wh == ref_val
    claimed_val = float(val_res.per_signature[kettle["signature_id"]]["claimed_wh"])
    assert kettle["scores"]["val"]["share"] == pytest.approx(claimed_val / ref_val, abs=0.0006)
    train_res = replay(arts, episodes, block=0, residual_wh=ref_train)
    claimed_train = float(train_res.per_signature[kettle["signature_id"]]["claimed_wh"])
    assert kettle["scores"]["train"]["share"] == pytest.approx(claimed_train / ref_train, abs=0.0006)

    # S is the unique-minus-conflict energy as a fraction of the validation residual
    val_json = json.loads((stage_dir / "val_1.json").read_text(encoding="utf-8"))
    expected_s = (val_res.unique_wh - val_res.conflict_wh) / ref_val
    assert val_json["S"] == pytest.approx(expected_s, abs=1e-5)
    assert compile_res["S"] == pytest.approx(expected_s, abs=1e-9)

    # Sanity against ground truth: kettle energy on validation days over residual energy
    val_ranges = [(day_bounds_ms[d], day_bounds_ms[d + 1]) for d in val_days]
    kettle_wh = sum(
        (e - s) / 3600.0 * 2.0  # 2000 W -> Wh per ms * 1000
        for s, e in synth.kettle_runs
        if any(lo <= s < hi for lo, hi in val_ranges)
    )
    assert kettle["scores"]["val"]["share"] == pytest.approx(kettle_wh / ref_val, rel=0.25)
    assert not np.isnan(kettle["scores"]["val"]["share"])


def test_rescore_shares_divide_by_rescore_residual(tmp_path: Path) -> None:
    """Rescore shares divide by the residual energy of the rescored days (WR-12)."""
    dt = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    all_bounds_ms = utc_day_bounds(end_ms, 22)
    buf_path = tmp_path / "buf_rs.sqlite"
    stage_dir = tmp_path / "stage_rs"
    synth = build_buffer(buf_path, SynthSpec(days=22), end_ms)

    preview = all_bounds_ms[:15]
    extract_res = run_stage(
        "extract",
        {
            "buffer_path": str(buf_path),
            "stage_dir": str(stage_dir),
            "grid_s": 30,
            "day_bounds_ms": preview,
            "first_weekday": 0,
            "holdout_days": 0,
            "train_fraction": 0.7,
            "enabled_event_types": ["step", "cycle_shape", "continuous"],
            "metered_channels": [],
            "phase_channels": {},
        },
    )
    prop_path = stage_dir / "proposal_1.json"
    prop_path.write_text(json.dumps(kettle_proposal(extract_res["digest"])), encoding="utf-8")
    compile_res = run_stage(
        "compile_score",
        {
            "stage_dir": str(stage_dir),
            "run_id": "rs-run",
            "round": 1,
            "model": "fake-model",
            "proposal_path": str(prop_path),
            "validated": False,
            "prior": [],
            "decisions": {},
        },
    )
    candidates = json.loads(Path(compile_res["library"]).read_text(encoding="utf-8"))

    bounds = all_bounds_ms[14:]
    out = run_stage(
        "rescore",
        {
            "buffer_path": str(buf_path),
            "stage_dir": str(stage_dir),
            "grid_s": 30,
            "day_bounds_ms": bounds,
            "first_weekday": 0,
            "enabled_event_types": ["step", "cycle_shape", "continuous"],
            "metered_channels": [],
            "phase_channels": {},
            "candidates": candidates,
        },
    )
    ref = _residual_wh(buf_path, bounds, list(range(len(bounds) - 1)))
    kettle_wh = sum(
        (e - s) / 3600.0 * 2.0 for s, e in synth.kettle_runs if bounds[0] <= s < bounds[-1]
    )
    share = out["per_signature"][candidates[0]["signature_id"]]["share"]
    assert share == pytest.approx(kettle_wh / ref, rel=0.25)


# --- D-20: switched-over-session sample de-duplication (score.replay) ---


def _dedup_table(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Minimal episode table with [start_i, end_i] spans for the D-20 de-dup tests."""
    import numpy as np

    n = len(rows)
    return {
        "start_i": np.array([r["start_i"] for r in rows], dtype=np.int64),
        "end_i": np.array([r["end_i"] for r in rows], dtype=np.int64),
        "rise_w": np.array([r.get("rise_w", 2000.0) for r in rows], dtype=np.float64),
        "fall_w": np.array([r.get("fall_w", -2000.0) for r in rows], dtype=np.float64),
        "dur_s": np.array([r.get("dur_s", 120.0) for r in rows], dtype=np.float64),
        "plateau_w": np.array([r.get("plateau_w", 2000.0) for r in rows], dtype=np.float64),
        "energy_wh": np.array([r["energy_wh"] for r in rows], dtype=np.float64),
        "kind": np.array([r["kind"] for r in rows], dtype=np.int32),
        "block": np.zeros(n, dtype=np.int32),
    }


def _session_artefact(sig_id: str) -> SignatureArtefact:
    """Program-kind artefact whose session box claims the 3000 Wh session episode."""
    return SignatureArtefact(
        schema_version=1,
        signature_id=sig_id,
        kind="program",
        label="Program",
        category="unknown",
        extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
        states=(),
        session={
            "energy_wh": {"lo": 0.0, "hi": 4000.0},
            "dur_s": {"lo": 0.0, "hi": 20000.0},
            "n_segments": {"lo": 1.0, "hi": 100.0},
        },
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={},
        scores={},
    )


def _kettle_artefact(sig_id: str) -> SignatureArtefact:
    """Switched-kind artefact whose state box claims the 200 Wh kettle episode."""
    st = State(
        name="on",
        rise_w=Box(lo=1500.0, p50=2000.0, hi=2500.0),
        fall_w=Box(lo=-2500.0, p50=-2000.0, hi=-1500.0),
        dur_s=Box(lo=60.0, p50=120.0, hi=300.0),
        plateau_w=Box(lo=1500.0, p50=2000.0, hi=2500.0),
        phase_share=None,
    )
    return SignatureArtefact(
        schema_version=1,
        signature_id=sig_id,
        kind="switched",
        label="Kettle",
        category="unknown",
        extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
        states=(st,),
        session=None,
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={},
        scores={},
    )


def _dedup_fixture() -> tuple[dict[str, np.ndarray], list[SignatureArtefact]]:
    """Session episode (samples 100..400, 3000 Wh) with a switched episode (200..206, 200 Wh) inside."""
    table = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 200, "end_i": 206, "energy_wh": 200.0},
    ])
    return table, [_session_artefact("sig-P"), _kettle_artefact("sig-K")]


def test_session_switched_overlap_counted_once() -> None:
    """D-20: a kettle episode inside a program session counts once — for the switched claim only."""
    from custom_components.powerguesser.engine.learning.score import replay

    table, arts = _dedup_fixture()
    res = replay(arts, table, block=0)

    assert res.per_signature["sig-K"] == {"n": 1, "claimed_wh": 200.0, "unique_wh": 200.0}
    assert res.per_signature["sig-P"]["n"] == 1
    assert res.per_signature["sig-P"]["claimed_wh"] == 2800.0
    assert res.per_signature["sig-P"]["unique_wh"] == 2800.0
    assert res.overlap_wh == 200.0
    assert res.unique_wh == 3000.0


def test_partial_overlap_proportional() -> None:
    """D-20: a switched episode half inside the session removes only its overlapping share."""
    from custom_components.powerguesser.engine.learning.score import replay

    table = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 350, "end_i": 450, "energy_wh": 200.0},
    ])
    arts = [_session_artefact("sig-P"), _kettle_artefact("sig-K")]
    res = replay(arts, table, block=0)

    # 50 of the switched episode's 100 samples lie inside (100, 400]
    assert res.overlap_wh == 100.0
    assert res.per_signature["sig-P"]["claimed_wh"] == 2900.0
    assert res.per_signature["sig-P"]["unique_wh"] == 2900.0
    assert res.per_signature["sig-K"]["claimed_wh"] == 200.0
    assert res.per_signature["sig-K"]["unique_wh"] == 200.0


def test_same_signature_or_unclaimed_session_untouched() -> None:
    """D-20: the subtraction only applies across different signatures on a claimed session."""
    from custom_components.powerguesser.engine.learning.score import replay

    # (a) Unclaimed session episode: the switched claim keeps its energy, no overlap.
    table = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 200, "end_i": 206, "energy_wh": 200.0},
    ])
    res = replay([_kettle_artefact("sig-K")], table, block=0)
    assert res.overlap_wh == 0.0
    assert res.per_signature["sig-K"]["unique_wh"] == 200.0
    assert res.unique_wh == 200.0

    # (b) Same signature claims both episodes: switched takes precedence over its own
    # session claim, so nothing is subtracted either.
    table_b = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 200, "end_i": 206, "energy_wh": 200.0},
    ])
    arts_b = [_session_artefact("sig-X"), _kettle_artefact("sig-X")]
    res_b = replay(arts_b, table_b, block=0)
    assert res_b.overlap_wh == 0.0
    assert res_b.per_signature["sig-X"]["n"] == 2
    assert res_b.per_signature["sig-X"]["claimed_wh"] == 3200.0
    assert res_b.per_signature["sig-X"]["unique_wh"] == 3200.0


def test_claims_unchanged_by_dedup() -> None:
    """D-20: claims, conflicts and n are identical to pre-change behaviour (identity matching relies on them)."""
    from custom_components.powerguesser.engine.learning.score import replay

    table = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 200, "end_i": 206, "energy_wh": 200.0},
    ])
    arts = [_session_artefact("sig-P"), _kettle_artefact("sig-K1"), _kettle_artefact("sig-K2")]
    res = replay(arts, table, block=0)

    # The switched episode is claimed by two signatures: existing conflict handling.
    assert res.claims["sig-K1"] == {1}
    assert res.claims["sig-K2"] == {1}
    assert res.claims["sig-P"] == {0}
    assert res.conflict_episodes == {1}
    assert res.conflict_wh == 200.0
    assert res.per_signature["sig-K1"]["n"] == 1
    assert res.per_signature["sig-K2"]["n"] == 1
    assert res.per_signature["sig-P"]["n"] == 1
    # A multi-claimant switched episode never triggers the session subtraction.
    assert res.overlap_wh == 0.0
    assert res.per_signature["sig-P"]["unique_wh"] == 3000.0


def test_block_score_not_above_one() -> None:
    """D-20: S never exceeds 1.0 when the residual equals the episode energy sum."""
    from custom_components.powerguesser.engine.learning.score import block_score, replay

    table, arts = _dedup_fixture()
    res = replay(arts, table, block=0, residual_wh=3000.0)
    s_score, _conflict_share = block_score(res)
    assert s_score <= 1.0
    assert s_score == pytest.approx(1.0)


# --- D-17: the re-learn proposal gate (Plan 05-06) ---


def _gate_table() -> dict[str, np.ndarray]:
    """A validation block (block=1) with two kettle episodes at nearby rises."""
    import numpy as np

    rows = [
        {"start_i": 100, "end_i": 140, "energy_wh": 60.0, "rise_w": 2000.0},
        {"start_i": 200, "end_i": 240, "energy_wh": 60.0, "rise_w": 2100.0},
    ]
    n = len(rows)
    return {
        "start_i": np.array([r["start_i"] for r in rows], dtype=np.int64),
        "end_i": np.array([r["end_i"] for r in rows], dtype=np.int64),
        "rise_w": np.array([r["rise_w"] for r in rows], dtype=np.float64),
        "fall_w": np.array([-2000.0] * n, dtype=np.float64),
        "dur_s": np.array([120.0] * n, dtype=np.float64),
        "plateau_w": np.array([2000.0] * n, dtype=np.float64),
        "energy_wh": np.array([r["energy_wh"] for r in rows], dtype=np.float64),
        "kind": np.zeros(n, dtype=np.int32),
        "block": np.ones(n, dtype=np.int32),
    }


def _gate_artefact(sig_id: str, rise_lo: float, rise_hi: float) -> SignatureArtefact:
    """A switched one-state artefact with the given rise box and loose other boxes."""
    st = State(
        name="on",
        rise_w=Box(lo=rise_lo, p50=(rise_lo + rise_hi) / 2.0, hi=rise_hi),
        fall_w=Box(lo=-2500.0, p50=-2000.0, hi=-1500.0),
        dur_s=Box(lo=60.0, p50=120.0, hi=300.0),
        plateau_w=Box(lo=1500.0, p50=2000.0, hi=2500.0),
        phase_share=None,
    )
    return SignatureArtefact(
        schema_version=1,
        signature_id=sig_id,
        kind="switched",
        label="Kettle",
        category="unknown",
        extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
        states=(st,),
        session=None,
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={},
        scores={},
    )


def test_proposal_gate_rejects_worse_and_passes_no_worse() -> None:
    """D-17: a proposal replaying worse than the active version on the same
    validation block fails the gate; an identical-claim proposal passes at
    equality; the basis names the learning-split validation block."""
    import dataclasses

    from custom_components.powerguesser.engine.learning.score import gate_proposal

    table = _gate_table()
    residual_wh = 500.0

    active = _gate_artefact("sig-K", 1500.0, 2500.0)  # claims both episodes
    narrower = _gate_artefact("sig-K", 1900.0, 2050.0)  # claims only the 2000 W episode
    gate = gate_proposal(narrower, active, table, residual_wh)

    assert gate["passed"] is False
    assert gate["active_S"] > gate["proposed_S"]
    assert gate["proposed_S"] > 0.0  # the narrower box still claims one episode
    assert gate["basis"] == "learning_split_val"

    # Identical claim: same boxes, only the state name differs -> equality passes.
    same_claim = _gate_artefact("sig-K", 1500.0, 2500.0)
    same_claim = dataclasses.replace(
        same_claim, states=(dataclasses.replace(same_claim.states[0], name="main"),)
    )
    gate_same = gate_proposal(same_claim, active, table, residual_wh)
    assert gate_same["passed"] is True
    assert gate_same["proposed_S"] == gate_same["active_S"]


def test_compile_score_gates_differing_matches_only_with_actives(tmp_path: Path) -> None:
    """compile_score writes proposal_gate only when the job carries actives:
    a differing matched artefact gated against a wider-box active fails; the
    same claim under a renamed state passes at equality; without actives the
    library entry carries no gate at all."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_gate.sqlite"
    stage_dir = tmp_path / "stage_gate"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    extract_res = run_stage(
        "extract",
        {
            "buffer_path": str(buf_path),
            "stage_dir": str(stage_dir),
            "grid_s": 30,
            "day_bounds_ms": day_bounds_ms,
            "first_weekday": start_dt.weekday(),
            "holdout_days": 0,
            "train_fraction": 0.7,
            "enabled_event_types": ["step", "cycle_shape", "continuous"],
            "metered_channels": [],
            "phase_channels": {},
        },
    )
    prop_path = stage_dir / "proposal_1.json"
    prop_path.write_text(json.dumps(kettle_proposal(extract_res["digest"])), encoding="utf-8")

    base_job = {
        "stage_dir": str(stage_dir),
        "run_id": "gate-run",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(prop_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
    }

    # Without actives: no gate entry (golden-shape jobs stay unchanged).
    res_plain = run_stage("compile_score", dict(base_job))
    with open(res_plain["library"], "r", encoding="utf-8") as f:
        plain_entry = json.load(f)[0]
    assert "proposal_gate" not in plain_entry

    # The active version claims at least as much (wider boxes) -> the proposal fails.
    wider_active = copy.deepcopy(plain_entry)
    for st in wider_active["states"]:
        for key in ("rise_w", "fall_w", "dur_s", "plateau_w"):
            st[key] = {"lo": 0.0, "p50": st[key]["p50"], "hi": 1e9}
    job_worse = dict(base_job, actives={plain_entry["signature_id"]: wider_active})
    res_worse = run_stage("compile_score", job_worse)
    with open(res_worse["library"], "r", encoding="utf-8") as f:
        worse_entry = json.load(f)[0]
    assert worse_entry["proposal_gate"]["passed"] is False
    assert worse_entry["proposal_gate"]["active_S"] > worse_entry["proposal_gate"]["proposed_S"]
    assert worse_entry["proposal_gate"]["basis"] == "learning_split_val"

    # The same claim under a renamed state -> equality passes the zero-tolerance gate.
    renamed_active = copy.deepcopy(plain_entry)
    renamed_active["states"][0]["name"] = "main"
    job_equal = dict(base_job, actives={plain_entry["signature_id"]: renamed_active})
    res_equal = run_stage("compile_score", job_equal)
    with open(res_equal["library"], "r", encoding="utf-8") as f:
        equal_entry = json.load(f)[0]
    assert equal_entry["proposal_gate"]["passed"] is True
    assert equal_entry["proposal_gate"]["proposed_S"] == equal_entry["proposal_gate"]["active_S"]


_EVIDENCE_NOW = "2026-11-25T07:00:00+00:00"


def _compile_kettle(
    tmp_path: Path,
    *,
    name: str,
    job_extra: dict | None = None,
    prior: list[dict] | None = None,
    drop_residual: bool = False,
    run_id: str | None = None,
) -> tuple[Path, list[dict]]:
    """Run the extract -> proposal -> compile_score recipe; return (stage_dir, library dicts)."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / f"buf_{name}.sqlite"
    stage_dir = tmp_path / f"stage_{name}"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    extract_job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }
    extract_res = run_stage("extract", extract_job)
    digest = extract_res["digest"]

    proposal = kettle_proposal(digest)
    message = chat_completion(json.dumps(proposal))["choices"][0]["message"]
    cluster_ids = [c["id"] for c in digest["clusters"]]
    valid_prop, errors, _ = parse_proposal(message, cluster_ids=cluster_ids, plug_ids=[])
    assert valid_prop is not None, f"Proposal validation failed: {errors}"

    proposal_path = stage_dir / "proposal_1.json"
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(valid_prop, f)

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": run_id or f"test-{name}",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(proposal_path),
        "incumbent": None,
        "validated": True,
        "prior": prior or [],
        "decisions": {},
    }
    if job_extra:
        compile_job.update(job_extra)
    if drop_residual:
        (stage_dir / "residual.npz").unlink()

    compile_res = run_stage("compile_score", compile_job)
    with open(compile_res["library"], "r", encoding="utf-8") as f:
        library = json.load(f)
    assert library
    return stage_dir, library


def test_compile_attaches_evidence_by_original_id(tmp_path: Path) -> None:
    """D-11: evidence is keyed by the ORIGINAL artefact id even when match_identities rewrites it."""
    _, first_library = _compile_kettle(
        tmp_path, name="ev1", job_extra={"with_evidence": True, "now_iso": _EVIDENCE_NOW}
    )
    prior_kettle = {**first_library[0], "signature_id": "prior-kettle-id"}
    prior_kettle.pop("evidence", None)

    _, second_library = _compile_kettle(
        tmp_path,
        name="ev2",
        job_extra={"with_evidence": True, "now_iso": _EVIDENCE_NOW},
        prior=[prior_kettle],
    )

    matched = [d for d in second_library if d["signature_id"] == "prior-kettle-id"]
    assert matched, [d["signature_id"] for d in second_library]
    assert matched[0]["evidence"]["n_runs"] > 0
    assert matched[0]["evidence"]["source"] == "run"


@pytest.mark.parametrize(
    "job_extra", [{"with_evidence": False}, {}], ids=["false", "absent"]
)
def test_compile_without_evidence_flag(tmp_path: Path, job_extra: dict) -> None:
    """D-11: only an explicit with_evidence True attaches evidence (golden/shadow jobs stay clean)."""
    _, library = _compile_kettle(
        tmp_path, name="noflag", job_extra={"now_iso": _EVIDENCE_NOW, **job_extra}
    )
    assert all("evidence" not in d for d in library)


def test_compile_without_residual_skips_evidence(tmp_path: Path) -> None:
    """A missing residual.npz skips the attach without error and without changing ids."""
    _, with_library = _compile_kettle(
        tmp_path,
        name="res1",
        job_extra={"with_evidence": True, "now_iso": _EVIDENCE_NOW},
        run_id="test-residual",
    )
    _, without_library = _compile_kettle(
        tmp_path,
        name="res2",
        job_extra={"with_evidence": True, "now_iso": _EVIDENCE_NOW},
        drop_residual=True,
        run_id="test-residual",
    )

    assert [d["signature_id"] for d in with_library] == [
        d["signature_id"] for d in without_library
    ]
    assert all("evidence" not in d for d in without_library)


def test_learned_candidate_has_no_fingerprint_documented_gap(tmp_path: Path) -> None:
    """D-21 pin: a real compile_score library dict carries no "fingerprint" key.

    Phase 8 previews want a fingerprint on learned candidates; adding one would
    change identity and dismiss behaviour, so it stays a documented v1.0 gap
    whose fix is a Phase 8 prerequisite. This test flips when Phase 8 fixes it.
    """
    _, library = _compile_kettle(tmp_path, name="fingerprint")
    assert all("fingerprint" not in d for d in library)


def test_dedup_share_inputs_exact() -> None:
    """D-20: pin the exact per-signature claimed/unique Wh feeding shares, scorecard and the D-21 confidence."""
    from custom_components.powerguesser.engine.learning.score import replay, scorecard

    table, arts = _dedup_fixture()
    res = replay(arts, table, block=0, residual_wh=4000.0)

    assert res.per_signature["sig-P"] == {"n": 1, "claimed_wh": 2800.0, "unique_wh": 2800.0}
    assert res.per_signature["sig-K"] == {"n": 1, "claimed_wh": 200.0, "unique_wh": 200.0}

    card = scorecard(res, {"P": arts[0], "K": arts[1]}, digest={})
    devices = {d["ref"]: d for d in card["devices"]}
    assert devices["P"]["claimed_share"] == 0.7
    assert devices["K"]["claimed_share"] == 0.05
    # The compile_score / rescore share rule is claimed_wh / denominator_wh.
    assert res.per_signature["sig-P"]["claimed_wh"] / res.denominator_wh == pytest.approx(0.7)
    assert res.per_signature["sig-K"]["claimed_wh"] / res.denominator_wh == pytest.approx(0.05)

    # Partial overlap: 3000 - 100 for the session claim, the switched claim stays whole.
    table_partial = _dedup_table([
        {"kind": 1, "start_i": 100, "end_i": 400, "energy_wh": 3000.0, "dur_s": 9000.0},
        {"kind": 0, "start_i": 350, "end_i": 450, "energy_wh": 200.0},
    ])
    res_partial = replay(arts, table_partial, block=0, residual_wh=4000.0)
    assert res_partial.per_signature["sig-P"]["claimed_wh"] == 2900.0
    assert res_partial.per_signature["sig-P"]["unique_wh"] == 2900.0
    assert res_partial.per_signature["sig-K"]["claimed_wh"] == 200.0
    assert res_partial.per_signature["sig-K"]["unique_wh"] == 200.0
