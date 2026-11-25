"""Tests for the HA-free evidence module and the worker evidence stage (EVID-01..03, 07).

Task 1 covers the engine statistics: rhythm, profile and evidence-quality hint
codes computed from a hand-built episode table and replay claim sets. The
posture is EVID-03/D-04: an unavailable value is ``None``, never 0.

Task 2 covers the worker ``evidence`` stage (candidate and ``device:<uuid>``
artefacts alike, D-07a), the evidence window contract shared with the run
attach (D-11) and rescore's explicit evidence opt-in.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning.evidence import (
    HINT_SHORT_COVERAGE_DAYS,
    MIN_RHYTHM_RUNS,
    TOD_MIN_CONCENTRATION,
    build_evidence,
)
from custom_components.powerguesser.engine.learning.proposal import parse_proposal
from custom_components.powerguesser.engine.learning.worker import (
    STAGES,
    run_stage,
    stage_outputs,
)
from tests.conftest import make_switched_candidate
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds

_NOW = "2026-11-25T07:00:00+00:00"

_BASE_MS = 1_700_000_000_000
_DAY_MS = 86_400_000
_STEP_MS = 30_000
_NAN = float("nan")


def day_bounds(days: int) -> list[int]:
    """UTC-midnight bounds for ``days`` whole days starting at _BASE_MS."""
    return [_BASE_MS + d * _DAY_MS for d in range(days + 1)]


def make_table(
    *,
    start_i: list[int],
    kind: list[int] | None = None,
    rise_w: list[float] | None = None,
    plateau_w: list[float] | None = None,
    dur_s: list[float] | None = None,
    energy_wh: list[float] | None = None,
    dow: list[int] | None = None,
    shares: tuple[list[float], list[float], list[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Build a minimal episode table carrying exactly the columns evidence reads."""
    n = len(start_i)
    zeros = [0.0] * n
    nan = [_NAN] * n
    l1, l2, l3 = shares if shares is not None else (nan, nan, nan)
    risers = rise_w if rise_w is not None else zeros
    return {
        "start_i": np.array(start_i, dtype=np.int64),
        "end_i": np.array([i + 10 for i in start_i], dtype=np.int64),
        "rise_w": np.array(risers, dtype=np.float64),
        "fall_w": np.array([-v for v in risers], dtype=np.float64),
        "dur_s": np.array(dur_s if dur_s is not None else zeros, dtype=np.float64),
        "plateau_w": np.array(plateau_w if plateau_w is not None else zeros, dtype=np.float64),
        "energy_wh": np.array(energy_wh if energy_wh is not None else zeros, dtype=np.float64),
        "kind": np.array(kind if kind is not None else [0] * n, dtype=np.int32),
        "dow": np.array(dow if dow is not None else [0] * n, dtype=np.int32),
        "l1_share": np.array(l1, dtype=np.float64),
        "l2_share": np.array(l2, dtype=np.float64),
        "l3_share": np.array(l3, dtype=np.float64),
    }


def evidence_for(
    table: dict[str, np.ndarray],
    rows: set[int],
    *,
    valid_days: list[int],
    bounds: list[int] | None = None,
) -> dict:
    """Run build_evidence for one signature claiming ``rows``."""
    if bounds is None:
        bounds = day_bounds(20)
    result = build_evidence(
        table,
        {"sig": set(rows)},
        start_ms=_BASE_MS,
        step_ms=_STEP_MS,
        day_bounds_ms=bounds,
        valid_days=valid_days,
        now_iso="2026-11-25T07:00:00+00:00",
        source="run",
    )
    return result["sig"]


def run_start_i(day: int, minute_of_day: int) -> int:
    """A start_i whose absolute start lands at ``minute_of_day`` on day ``day``."""
    return day * (_DAY_MS // _STEP_MS) + (minute_of_day * 60_000) // _STEP_MS


def test_rhythm_suppressed_below_min_runs() -> None:
    """4 claimed rows give rhythm None; 5 give a rhythm dict (D-02 boundary)."""
    bounds = day_bounds(10)
    valid = list(range(10))
    four = make_table(start_i=[run_start_i(d, 7 * 60) for d in range(4)])
    assert evidence_for(four, set(range(4)), valid_days=valid, bounds=bounds)["rhythm"] is None

    five = make_table(start_i=[run_start_i(d, 7 * 60) for d in range(MIN_RHYTHM_RUNS)])
    rhythm = evidence_for(five, set(range(MIN_RHYTHM_RUNS)), valid_days=valid, bounds=bounds)["rhythm"]
    assert isinstance(rhythm, dict)


def test_rhythm_wraps_midnight() -> None:
    """Runs at 23:50 and 00:10 give a center near midnight and a wrapping window (PITFALLS 17)."""
    bounds = day_bounds(10)
    starts = [run_start_i(d, 1430 if d % 2 == 0 else 10) for d in range(6)]
    table = make_table(start_i=starts)
    rhythm = evidence_for(table, set(range(6)), valid_days=list(range(10)), bounds=bounds)["rhythm"]

    assert rhythm is not None
    tod = rhythm["tod"]
    assert tod["center_min"] >= 1435 or tod["center_min"] <= 5
    # The circular window crosses midnight: lo is late evening, hi is early morning.
    assert tod["lo_min"] > tod["hi_min"]
    assert tod["lo_min"] <= 1435
    assert tod["hi_min"] >= 5


def test_rhythm_bimodal_has_low_concentration() -> None:
    """3 runs at 07:00 and 3 at 19:00 give a concentration below TOD_MIN_CONCENTRATION."""
    bounds = day_bounds(10)
    starts = [run_start_i(0, 420), run_start_i(1, 420), run_start_i(2, 420)]
    starts += [run_start_i(3, 1140), run_start_i(4, 1140), run_start_i(5, 1140)]
    table = make_table(start_i=starts)
    rhythm = evidence_for(table, set(range(6)), valid_days=list(range(10)), bounds=bounds)["rhythm"]

    assert rhythm is not None
    assert rhythm["tod"]["concentration"] < TOD_MIN_CONCENTRATION


def test_rates_use_observed_days() -> None:
    """runs_per_day and wh_per_day divide by observed days, not calendar days."""
    bounds = day_bounds(21)
    starts = [run_start_i(d, 6 * 60) for d in range(12)]
    table = make_table(
        start_i=starts,
        energy_wh=[120.0] * 12,
    )
    rows = set(range(12))

    ten = evidence_for(table, rows, valid_days=list(range(10)), bounds=bounds)["rhythm"]
    twenty = evidence_for(table, rows, valid_days=list(range(20)), bounds=bounds)["rhythm"]

    assert ten is not None and twenty is not None
    assert math.isclose(twenty["runs_per_day"] * 2, ten["runs_per_day"])
    assert math.isclose(ten["runs_per_week"], 7 * ten["runs_per_day"])

    profile_ten = evidence_for(table, rows, valid_days=list(range(10)), bounds=bounds)["profile"]
    profile_twenty = evidence_for(table, rows, valid_days=list(range(20)), bounds=bounds)["profile"]
    assert math.isclose(profile_twenty["wh_per_day"] * 2, profile_ten["wh_per_day"])


def test_gaps_and_weekdays() -> None:
    """gap_median_s and gap_p90_s come from consecutive starts; dow_counts sums to n_runs."""
    bounds = day_bounds(10)
    starts = [run_start_i(0, 6 * 60) + k * 60 for k in range(5)]  # every 30 min
    table = make_table(start_i=starts, dow=[0, 1, 2, 3, 4])
    rhythm = evidence_for(table, set(range(5)), valid_days=list(range(10)), bounds=bounds)["rhythm"]

    assert rhythm is not None
    assert math.isclose(rhythm["gap_median_s"], 1800.0)
    assert math.isclose(rhythm["gap_p90_s"], 1800.0)
    assert len(rhythm["dow_counts"]) == 7
    assert sum(rhythm["dow_counts"]) == 5


def test_profile_switched_and_session_step() -> None:
    """kind 0 rows use rise_w and kind 1 rows use plateau_w; wh_per_run is the median energy."""
    bounds = day_bounds(10)
    table = make_table(
        start_i=[run_start_i(d, 6 * 60) for d in range(4)],
        kind=[0, 0, 1, 1],
        rise_w=[2000.0, 2000.0, 0.0, 0.0],
        plateau_w=[2000.0, 2000.0, 1000.0, 1000.0],
        dur_s=[180.0, 180.0, 300.0, 300.0],
        energy_wh=[100.0, 200.0, 300.0, 400.0],
    )
    profile = evidence_for(table, set(range(4)), valid_days=list(range(10)), bounds=bounds)["profile"]

    assert profile["step_w"]["p10"] == 1000.0
    assert profile["step_w"]["p50"] == 1500.0
    assert profile["step_w"]["p90"] == 2000.0
    assert profile["dur_s"]["p50"] == 240.0
    assert profile["wh_per_run"] == 250.0


def test_phase_only_with_phase_channels() -> None:
    """All-NaN shares give phase None and phase_available False; finite shares pick the median argmax."""
    bounds = day_bounds(10)
    starts = [run_start_i(d, 6 * 60) for d in range(5)]
    rows = set(range(5))

    no_phase = evidence_for(make_table(start_i=starts), rows, valid_days=list(range(10)), bounds=bounds)
    assert no_phase["profile"]["phase"] is None
    assert no_phase["profile"]["phase_available"] is False

    with_phase = evidence_for(
        make_table(
            start_i=starts,
            shares=([0.2] * 5, [0.6] * 5, [0.2] * 5),
        ),
        rows,
        valid_days=list(range(10)),
        bounds=bounds,
    )
    assert with_phase["profile"]["phase"] == "l2"
    assert with_phase["profile"]["phase_available"] is True


def test_zero_runs_profile_is_none_not_zero() -> None:
    """An empty claim set gives n_runs 0, rhythm None and every profile value None."""
    bounds = day_bounds(10)
    table = make_table(start_i=[run_start_i(0, 6 * 60)])
    entry = evidence_for(table, set(), valid_days=list(range(10)), bounds=bounds)

    assert entry["n_runs"] == 0
    assert entry["occurrences"] == []
    assert entry["first_ms"] is None and entry["last_ms"] is None
    assert entry["rhythm"] is None
    profile = entry["profile"]
    assert profile["step_w"] is None
    assert profile["dur_s"] is None
    assert profile["wh_per_run"] is None
    assert profile["wh_per_day"] is None
    assert profile["phase"] is None
    assert profile["phase_available"] is False


def test_hint_thresholds() -> None:
    """few_runs, wide_step_spread and short_coverage fire one step either side of each threshold."""
    bounds = day_bounds(21)
    rows_of = lambda n: set(range(n))  # noqa: E731 - tiny local shorthand

    # few_runs: below 10 runs yes, at 10 no (steady step, wide coverage).
    steady = lambda n: make_table(  # noqa: E731
        start_i=[run_start_i(d, 6 * 60) for d in range(n)],
        rise_w=[1000.0] * n,
    )
    nine = evidence_for(steady(9), rows_of(9), valid_days=list(range(20)), bounds=bounds)
    ten = evidence_for(steady(10), rows_of(10), valid_days=list(range(20)), bounds=bounds)
    assert "few_runs" in nine["hints"]
    assert "few_runs" not in ten["hints"]

    # wide_step_spread: exactly 0.25 spread is not flagged, just above is.
    def spread_table(high: float) -> dict[str, np.ndarray]:
        return make_table(
            start_i=[run_start_i(d, 6 * 60) for d in range(20)],
            rise_w=[1000.0] * 15 + [high] * 5,
        )

    at_threshold = evidence_for(spread_table(1250.0), rows_of(20), valid_days=list(range(20)), bounds=bounds)
    above = evidence_for(spread_table(1260.0), rows_of(20), valid_days=list(range(20)), bounds=bounds)
    assert (at_threshold["profile"]["step_w"]["p90"] - at_threshold["profile"]["step_w"]["p10"]) / (
        at_threshold["profile"]["step_w"]["p50"]
    ) == 0.25
    assert "wide_step_spread" not in at_threshold["hints"]
    assert "wide_step_spread" in above["hints"]

    # short_coverage: 13 observed days yes, 14 no.
    thirteen = evidence_for(steady(10), rows_of(10), valid_days=list(range(13)), bounds=bounds)
    fourteen = evidence_for(steady(10), rows_of(10), valid_days=list(range(14)), bounds=bounds)
    assert "short_coverage" in thirteen["hints"]
    assert "short_coverage" not in fourteen["hints"]
    assert HINT_SHORT_COVERAGE_DAYS == 14


def test_evidence_is_json_safe() -> None:
    """Every case above survives a JSON round trip unchanged."""
    bounds = day_bounds(21)
    cases = [
        make_table(start_i=[run_start_i(0, 7 * 60)]),
        make_table(start_i=[run_start_i(d, 1430 if d % 2 == 0 else 10) for d in range(6)]),
        make_table(
            start_i=[run_start_i(d, 6 * 60) for d in range(4)],
            kind=[0, 0, 1, 1],
            rise_w=[2000.0, 2000.0, 0.0, 0.0],
            plateau_w=[2000.0, 2000.0, 1000.0, 1000.0],
            shares=([0.2] * 4, [0.6] * 4, [0.2] * 4),
        ),
    ]
    for table in cases:
        claims = {"sig": set(range(len(table["start_i"])))}
        result = build_evidence(
            table,
            claims,
            start_ms=_BASE_MS,
            step_ms=_STEP_MS,
            day_bounds_ms=bounds,
            valid_days=list(range(20)),
            now_iso="2026-11-25T07:00:00+00:00",
            source="run",
        )
        assert json.loads(json.dumps(result)) == result


def test_build_evidence_short_day_bounds() -> None:
    """One or zero day bounds with an empty table and empty claim set never raises."""
    for bounds in (day_bounds(0)[:1], []):
        result = build_evidence(
            {},
            {"a": set()},
            start_ms=_BASE_MS,
            step_ms=_STEP_MS,
            day_bounds_ms=bounds,
            valid_days=[],
            now_iso="2026-11-25T07:00:00+00:00",
            source="refresh",
        )
        entry = result["a"]
        assert entry["n_runs"] == 0
        assert entry["occurrences"] == []
        assert entry["first_ms"] is None and entry["last_ms"] is None
        assert entry["rhythm"] is None
        assert entry["profile"]["phase_available"] is False
        assert entry["window"]["observed_days"] == 0
        assert entry["window"]["n_days"] == 0
        if bounds:
            assert entry["window"]["start_ms"] == bounds[0]
            assert entry["window"]["end_ms"] == bounds[0]
        else:
            assert entry["window"]["start_ms"] is None
            assert entry["window"]["end_ms"] is None
        assert json.loads(json.dumps(result)) == result


# --------------------------------------------------------------------------
# Task 2: the worker evidence stage, the device co-replay, the run-versus-stage
# window equality and rescore evidence.
# --------------------------------------------------------------------------


def extract_compile(
    tmp_path: Path,
    *,
    days: int = 22,
    holdout_days: int = 0,
    name: str = "a",
) -> dict:
    """Run the extract -> proposal -> compile_score recipe and return its objects."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    day_bounds_ms = utc_day_bounds(end_ms, days)
    buf_path = tmp_path / f"buf_{name}.sqlite"
    stage_dir = tmp_path / f"stage_{name}"
    data = build_buffer(buf_path, SynthSpec(days=days), end_ms)
    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    extract_job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": holdout_days,
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
    proposal_path.write_text(json.dumps(valid_prop), encoding="utf-8")

    compile_job = {
        "stage_dir": str(stage_dir),
        "run_id": f"test-run-{name}",
        "round": 1,
        "model": "fake-model",
        "proposal_path": str(proposal_path),
        "incumbent": None,
        "validated": True,
        "prior": [],
        "decisions": {},
        "with_evidence": True,
        "now_iso": _NOW,
    }
    compile_res = run_stage("compile_score", compile_job)
    library = json.loads(Path(compile_res["library"]).read_text(encoding="utf-8"))
    assert library
    return {
        "data": data,
        "day_bounds_ms": day_bounds_ms,
        "extract_job": extract_job,
        "stage_dir": stage_dir,
        "digest": digest,
        "library": library,
        "kettle": library[0],
    }


def test_evidence_stage_registered(tmp_path: Path) -> None:
    """"evidence" is in STAGES and stage_outputs("evidence") is exactly {result, n}."""
    assert "evidence" in STAGES

    outputs = stage_outputs(
        "evidence",
        {"stage_dir": str(tmp_path), "result": str(tmp_path / "evidence.json"), "n": 3},
    )
    assert set(outputs) == {"result", "n"}
    assert outputs["result"] == str(tmp_path / "evidence.json")
    assert outputs["n"] == 3

    with pytest.raises(ValueError):
        stage_outputs("not-a-stage", {})
    with pytest.raises(ValueError):
        run_stage("not-a-stage", {})


def test_evidence_stage_end_to_end(tmp_path: Path) -> None:
    """The stage replays the sent candidates and writes evidence.json for each of them."""
    ctx = extract_compile(tmp_path, days=22, name="e2e")
    kettle = ctx["kettle"]
    empty = make_switched_candidate("sig-empty", "Nothing", rise_w=50.0)
    stage_dir = tmp_path / "stage_evidence_e2e"

    res = run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(stage_dir),
            "candidates": [kettle, empty],
            "now_iso": _NOW,
        },
    )

    evidence_map = json.loads((stage_dir / "evidence.json").read_text(encoding="utf-8"))
    assert set(evidence_map) == {kettle["signature_id"], "sig-empty"}

    kettle_ev = evidence_map[kettle["signature_id"]]
    assert kettle_ev["source"] == "refresh"
    bounds = ctx["day_bounds_ms"]
    expected_runs = sum(
        1 for start_ms, _end_ms in ctx["data"].kettle_runs if bounds[0] <= start_ms < bounds[-1]
    )
    assert 0.8 * expected_runs <= kettle_ev["n_runs"] <= expected_runs

    assert evidence_map["sig-empty"]["n_runs"] == 0

    outputs = stage_outputs("evidence", res)
    assert json.loads(json.dumps(outputs)) == outputs


def test_evidence_stage_isolates_malformed_entry(tmp_path: Path) -> None:
    """WR-02: one unusable sent artefact is skipped, not fatal for the whole batch."""
    ctx = extract_compile(tmp_path, days=22, name="malformed")
    kettle = ctx["kettle"]
    malformed = make_switched_candidate("sig-broken", "Broken", rise_w=50.0)
    # A stored artefact from another version: a required key is gone, so
    # SignatureArtefact.from_dict raises KeyError.
    malformed.pop("scores")
    stage_dir = tmp_path / "stage_evidence_malformed"

    res = run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(stage_dir),
            "candidates": [kettle, malformed],
            "now_iso": _NOW,
        },
    )

    evidence_map = json.loads((stage_dir / "evidence.json").read_text(encoding="utf-8"))
    # The good candidate is still computed, and the malformed id still gets the
    # documented n_runs-0 entry so the supervisor's evidence-needed predicate
    # clears for it instead of looping on a permanently malformed artefact.
    assert set(evidence_map) == {kettle["signature_id"], "sig-broken"}
    assert evidence_map[kettle["signature_id"]]["n_runs"] > 0
    malformed_ev = evidence_map["sig-broken"]
    assert malformed_ev["source"] == "refresh"
    assert malformed_ev["n_runs"] == 0
    assert malformed_ev["occurrences"] == []
    assert stage_outputs("evidence", res)["n"] == 2


def test_evidence_stage_isolates_nested_and_non_mapping_entries(tmp_path: Path) -> None:
    """WR-01: a nested malformed state and a non-mapping entry are skipped, not fatal.

    Two shapes escape the round-1 ``(KeyError, TypeError, ValueError)`` guard:
    a non-mapping entry raises ``AttributeError`` (``d.get`` on a list) and a
    one-element ``states[].phase_share`` raises ``IndexError`` in
    ``State.from_dict``. Neither may fail the whole evidence stage.
    """
    ctx = extract_compile(tmp_path, days=22, name="hostile")
    kettle = ctx["kettle"]
    # phase_share values must be two-element sequences, so a one-element value
    # raises IndexError inside State.from_dict (a hand-edit / other version).
    nested = make_switched_candidate("sig-index", "Index", rise_w=50.0)
    nested["states"][0]["phase_share"] = {"l1": [0.1]}
    stage_dir = tmp_path / "stage_evidence_hostile"

    res = run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(stage_dir),
            "candidates": [kettle, nested, [1, 2, 3]],
            "now_iso": _NOW,
        },
    )

    evidence_map = json.loads((stage_dir / "evidence.json").read_text(encoding="utf-8"))
    # The good candidate is computed, the nested-malformed id gets the
    # documented n_runs-0 entry, and the non-mapping entry (no id to key on) is
    # dropped without failing the batch.
    assert set(evidence_map) == {kettle["signature_id"], "sig-index"}
    assert evidence_map[kettle["signature_id"]]["n_runs"] > 0
    assert evidence_map["sig-index"]["n_runs"] == 0
    assert stage_outputs("evidence", res)["n"] == 2


def test_evidence_stage_device_artefact_matches_candidate_replay(tmp_path: Path) -> None:
    """D-07a: a version-shaped device artefact gets the identical replay and does not disturb the candidate."""
    ctx = extract_compile(tmp_path, days=22, name="device")
    kettle = ctx["kettle"]
    device = {
        k: v
        for k, v in kettle.items()
        if k not in ("identity", "fingerprint", "first_label", "evidence")
    }
    device["signature_id"] = "device:test-uuid"

    first_dir = tmp_path / "stage_evidence_first"
    run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(first_dir),
            "candidates": [kettle],
            "now_iso": _NOW,
        },
    )
    first_map = json.loads((first_dir / "evidence.json").read_text(encoding="utf-8"))

    second_dir = tmp_path / "stage_evidence_second"
    run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(second_dir),
            "candidates": [kettle, device],
            "now_iso": _NOW,
        },
    )
    second_map = json.loads((second_dir / "evidence.json").read_text(encoding="utf-8"))

    assert set(second_map) == {kettle["signature_id"], "device:test-uuid"}
    kettle_ev = first_map[kettle["signature_id"]]
    device_ev = second_map["device:test-uuid"]

    assert device_ev["n_runs"] > 0
    for key in ("n_runs", "first_ms", "last_ms", "occurrences"):
        assert device_ev[key] == kettle_ev[key]
    assert device_ev["rhythm"] is None or isinstance(device_ev["rhythm"], dict)
    assert isinstance(device_ev["profile"], dict)
    assert isinstance(device_ev["hints"], list)

    # Co-replaying the device copy changes nothing about the kettle entry.
    assert second_map[kettle["signature_id"]] == kettle_ev


@pytest.mark.parametrize("holdout_days", [0, 3])
def test_run_and_stage_evidence_agree_on_window(tmp_path: Path, holdout_days: int) -> None:
    """Run-source and stage-source evidence of the same artefact share one window (contract pin)."""
    ctx = extract_compile(tmp_path, days=22, holdout_days=holdout_days, name=f"win{holdout_days}")
    kettle = ctx["kettle"]
    run_ev = kettle["evidence"]
    assert run_ev["source"] == "run"

    stage_dir = tmp_path / f"stage_win{holdout_days}"
    run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "stage_dir": str(stage_dir),
            "candidates": [kettle],
            "now_iso": _NOW,
        },
    )
    stage_ev = json.loads((stage_dir / "evidence.json").read_text(encoding="utf-8"))[
        kettle["signature_id"]
    ]
    assert stage_ev["source"] == "refresh"

    for key in ("n_runs", "first_ms", "last_ms", "occurrences"):
        assert stage_ev[key] == run_ev[key]
    for key in ("start_ms", "end_ms", "observed_days", "n_days"):
        assert stage_ev["window"][key] == run_ev["window"][key]

    if holdout_days:
        meta = json.loads((ctx["stage_dir"] / "digest_meta.json").read_text(encoding="utf-8"))
        holdout = meta["days"]["holdout"]
        assert holdout, "expected a non-empty holdout with holdout_days 3"
        bounds = np.array(ctx["day_bounds_ms"], dtype=np.int64)
        newest_day = int(np.searchsorted(bounds, run_ev["last_ms"], side="right") - 1)
        assert newest_day in holdout


def test_rescore_per_signature_carries_evidence(tmp_path: Path) -> None:
    """Rescore emits per-signature evidence only for a job that opts in with with_evidence True."""
    ctx = extract_compile(tmp_path, days=22, name="rescore")
    kettle = ctx["kettle"]
    sig = kettle["signature_id"]

    with_dir = tmp_path / "stage_rescore_with"
    with_res = run_stage(
        "rescore",
        {
            **ctx["extract_job"],
            "stage_dir": str(with_dir),
            "candidates": [kettle],
            "with_evidence": True,
            "now_iso": _NOW,
        },
    )
    assert with_res["per_signature"][sig]["evidence"]["source"] == "rescore"
    on_disk = json.loads((with_dir / "rescore.json").read_text(encoding="utf-8"))
    assert on_disk["per_signature"][sig]["evidence"]["source"] == "rescore"
    assert stage_outputs("rescore", with_res)["per_signature"][sig]["evidence"]["source"] == "rescore"

    without_dir = tmp_path / "stage_rescore_without"
    without_res = run_stage(
        "rescore",
        {
            **ctx["extract_job"],
            "stage_dir": str(without_dir),
            "candidates": [kettle],
        },
    )
    assert all("evidence" not in entry for entry in without_res["per_signature"].values())
    without_disk = json.loads((without_dir / "rescore.json").read_text(encoding="utf-8"))
    assert all("evidence" not in entry for entry in without_disk["per_signature"].values())
    assert all(
        "evidence" not in entry
        for entry in stage_outputs("rescore", without_res)["per_signature"].values()
    )


def test_evidence_stage_too_few_day_bounds_emits_empty_entries(tmp_path: Path) -> None:
    """Fewer than 2 day bounds still writes one n_runs-0 entry per sent candidate (no refresh loop)."""
    ctx = extract_compile(tmp_path, days=22, name="short")
    kettle = ctx["kettle"]
    empty = make_switched_candidate("sig-empty-2", "Nothing", rise_w=50.0)
    stage_dir = tmp_path / "stage_evidence_short"

    res = run_stage(
        "evidence",
        {
            **ctx["extract_job"],
            "day_bounds_ms": ctx["day_bounds_ms"][:1],
            "stage_dir": str(stage_dir),
            "candidates": [kettle, empty],
            "now_iso": _NOW,
        },
    )

    assert stage_outputs("evidence", res)["n"] == 2
    evidence_map = json.loads((stage_dir / "evidence.json").read_text(encoding="utf-8"))
    assert set(evidence_map) == {kettle["signature_id"], "sig-empty-2"}
    for entry in evidence_map.values():
        assert entry["n_runs"] == 0
        assert entry["occurrences"] == []
        assert entry["window"]["n_days"] == 0
