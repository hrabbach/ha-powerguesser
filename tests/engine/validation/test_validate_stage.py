"""Validate stage edge tests (Plan 05-02, Task 2).

Determinism: two runs on identical inputs produce byte-identical report.json
and the report serialises under allow_nan=False. Plug gaps: with the kettle
plug NULLed for 12 h of one holdout day, both energy sums leave out exactly
the samples the plug no longer covers (METERED_HOLD_MS hold semantics), and
each equals the test's own recomputation over the same ok mask (A6).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from custom_components.powerguesser.engine.capture.constants import HOLDOUT_PAD_STEPS
from custom_components.powerguesser.engine.capture.exclusion import (
    holdout_exclusion_mask,
)
from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.learning.digest import canonical_json
from custom_components.powerguesser.engine.learning.series import load_residual
from custom_components.powerguesser.engine.learning.split import (
    calendar_holdout_start,
    data_days,
)
from custom_components.powerguesser.engine.learning.worker import (
    measurable_artefacts,
    run_stage,
)
from custom_components.powerguesser.engine.validation.constants import (
    CLASS_MAX_MEDIAN_PCT,
    CLASS_MAX_PEER_PCT,
    CLASS_MIN_PASSING,
    MAX_DEVICE_ERROR_PCT,
    MIN_COVERED_HOLDOUT_DAYS,
    MIN_EVENTS_FIGURE,
    MIN_EVENTS_PASS,
    MIN_USABLE_DAYS,
    OVERLAP_MIN_TICKS,
)
from tests.fixtures.detect_traces import switched_artefact
from tests.fixtures.learning.synth import (
    ENTITY_KETTLE,
    ENTITY_KETTLE_SESSION,
    ENTITY_WASHER,
    SynthSpec,
    build_buffer,
    utc_day_bounds,
)
from tests.fixtures.programs import program_artefact

GRID_S = 30
ENABLED_EVENT_TYPES = ["step", "cycle_shape", "continuous"]
END_MS = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
VALIDATION_TIME_MS = END_MS
SIG_ID = "sig-shadow-kettle"


def _kettle_artefact() -> dict[str, Any]:
    return switched_artefact(
        SIG_ID,
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )


def _validate_job(tmp_path: Path, tag: str, days: int = 35) -> tuple[dict[str, Any], Path]:
    """Build a buffer with a plugged kettle and the validate job over it."""
    buf_path = tmp_path / f"buf_{tag}.sqlite"
    stage_dir = tmp_path / f"stage_{tag}"
    data = build_buffer(buf_path, SynthSpec(days=days, kettle_plug=True), END_MS)
    day_bounds_ms = utc_day_bounds(END_MS, days)
    kettle_ch = data.channel_ids[ENTITY_KETTLE]
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, 7)
    shadow = {
        "schema_version": 1,
        "created": "2026-11-13T00:00:00+00:00",
        "run_id": "test-shadow",
        "grid_s": GRID_S,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "eligible": [ENTITY_KETTLE],
        "train_cutoff_ms": int(day_bounds_ms[first_holdout]),
        "holdout_days": 7,
        "artefacts": [_kettle_artefact()],
        "links": [
            {
                "entity_id": ENTITY_KETTLE,
                "signature_id": SIG_ID,
                "kind": "switched",
                "overlap": 0.8,
                "channel": kettle_ch,
            }
        ],
        "not_discovered": [],
        "plugs": {ENTITY_KETTLE: {"train_mean_w": 12.5}},
    }
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": GRID_S,
        "day_bounds_ms": day_bounds_ms,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "phase_channels": {},
        "shadow": shadow,
        "validation_time_ms": VALIDATION_TIME_MS,
    }
    return job, buf_path


def test_validate_stage_deterministic_and_nan_free(tmp_path: Path) -> None:
    """Two validate runs on identical inputs give byte-identical, NaN-free reports."""
    job1, _buf = _validate_job(tmp_path, "det1")
    job2, _buf2 = _validate_job(tmp_path, "det2")

    report1 = run_stage("validate", job1)
    report2 = run_stage("validate", job2)

    path1 = Path(job1["stage_dir"]) / "report.json"
    path2 = Path(job2["stage_dir"]) / "report.json"
    assert path1.read_bytes() == path2.read_bytes()

    # No NaN/inf anywhere: the report must serialize under allow_nan=False.
    json.dumps(report1, allow_nan=False)
    json.dumps(report2, allow_nan=False)

    # The kettle figures are present on the untouched block.
    assert report1["status"] == "ok"
    kettle = next(d for d in report1["devices"] if d["entity_id"] == ENTITY_KETTLE)
    assert kettle["energy_true_wh"] > 0
    assert kettle["energy_est_wh"] > 0
    assert kettle["energy_error_pct"] is not None
    assert kettle["nde"] is not None
    assert kettle["mae_w"] is not None


def test_validate_stage_plug_gap_alignment(tmp_path: Path) -> None:
    """A 12 h plug gap removes the same samples from the reference and the estimate.

    The kettle plug is NULLed for 12 h of one holdout day. Beyond the
    METERED_HOLD_MS hold the plug series is NaN, so the ok mask shrinks; the
    stage's energy_true_wh and energy_est_wh must each equal the test's own
    sum over that same ok mask, recomputed independently from load_residual
    and replay_online(collect_power=True).
    """
    job, buf_path = _validate_job(tmp_path, "gap")
    day_bounds_ms = job["day_bounds_ms"]
    kettle_ch = job["shadow"]["links"][0]["channel"]
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, 7)
    gap_day = first_holdout + 2  # a middle holdout day
    gap_start = int(day_bounds_ms[gap_day]) + 12 * 3600 * 1000  # 12:00
    gap_end = gap_start + 12 * 3600 * 1000  # 24:00

    with sqlite3.connect(str(buf_path)) as conn:
        conn.execute(
            "UPDATE samples SET w = NULL WHERE ch = ? AND ts >= ? AND ts < ?",
            (kettle_ch, gap_start, gap_end),
        )
        conn.commit()

    report = run_stage("validate", job)
    assert report["status"] == "ok"
    kettle = next(d for d in report["devices"] if d["entity_id"] == ENTITY_KETTLE)

    # Independent recomputation over the same ok mask (A6).
    start_ms = int(day_bounds_ms[0])
    end_ms = int(day_bounds_ms[-1])
    step_ms = GRID_S * 1000
    step_s = float(GRID_S)
    series = load_residual(str(buf_path), start_ms, end_ms, step_ms, [kettle_ch], {})
    r = series.gross
    y = series.plugs[kettle_ch]
    valid = data_days(r, start_ms, step_ms, day_bounds_ms)
    train_cutoff_ms = int(job["shadow"]["train_cutoff_ms"])
    valid_set = set(valid)
    holdout_idx = [
        d
        for d in range(first_holdout, n_days)
        if d in valid_set and int(day_bounds_ms[d]) >= train_cutoff_ms
    ]
    holdout_set = set(holdout_idx)
    bounds = np.array(day_bounds_ms, dtype=np.int64)
    sample_day = (
        np.searchsorted(bounds, start_ms + np.arange(len(r), dtype=np.int64) * step_ms, side="right")
        - 1
    )
    holdout_mask = np.isin(sample_day, list(holdout_set))
    ok = holdout_mask & np.isfinite(r) & np.isfinite(y)

    # The gap is real: some holdout samples lost their plug reading.
    assert not bool(np.all(np.isfinite(y[holdout_mask])))

    r_masked = np.where(holdout_mask, r, np.nan)
    online = replay_online(
        job["shadow"]["artefacts"],
        r_masked,
        step_s,
        suppress_alignment=True,
        collect_power=True,
    )
    expected_true_wh = float(np.sum(y[ok])) * step_s / 3600.0
    expected_est_wh = float(np.sum(online.energy_step_wh[SIG_ID][ok]))
    assert kettle["energy_true_wh"] == round(expected_true_wh, 1)
    assert kettle["energy_est_wh"] == round(expected_est_wh, 1)


def test_validate_stage_refuses_short_history(tmp_path: Path) -> None:
    """Fewer than 28 usable days: the stage refuses with a reason, no figures (D-12)."""
    job, _buf = _validate_job(tmp_path, "short", days=20)
    report = run_stage("validate", job)
    assert report["status"] == "refused"
    assert report["reason"] == "insufficient_data"
    assert report["devices"] == []
    assert report["classes"] == []


def test_validate_stage_every_eligible_plug_accounted(tmp_path: Path) -> None:
    """Every eligible plug lands in exactly one of devices, refusals, not_discovered (D-03).

    The kettle is linked to a switched artefact and is measured; the washer is
    linked to a session-kind artefact and is named in refusals as
    unsupported_kind. Nothing is silently dropped, nothing is double-counted.
    """
    buf_path = tmp_path / "buf_accounted.sqlite"
    stage_dir = tmp_path / "stage_accounted"
    spec = SynthSpec(days=35, kettle_plug=True, washer=True)
    data = build_buffer(buf_path, spec, END_MS)
    day_bounds_ms = utc_day_bounds(END_MS, 35)
    kettle_ch = data.channel_ids[ENTITY_KETTLE]
    washer_ch = data.channel_ids[ENTITY_WASHER]
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, 7)

    washer_artefact = switched_artefact(
        "sig-shadow-washer",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )
    washer_artefact["kind"] = "session"
    shadow = {
        "schema_version": 1,
        "created": "2026-11-13T00:00:00+00:00",
        "run_id": "test-shadow",
        "grid_s": GRID_S,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "eligible": sorted([ENTITY_KETTLE, ENTITY_WASHER]),
        "train_cutoff_ms": int(day_bounds_ms[first_holdout]),
        "holdout_days": 7,
        "artefacts": [_kettle_artefact(), washer_artefact],
        "links": [
            {
                "entity_id": ENTITY_KETTLE,
                "signature_id": SIG_ID,
                "kind": "switched",
                "overlap": 0.8,
                "channel": kettle_ch,
            },
            {
                "entity_id": ENTITY_WASHER,
                "signature_id": "sig-shadow-washer",
                "kind": "session",
                "overlap": 0.7,
                "channel": washer_ch,
            },
        ],
        "not_discovered": [],
        "plugs": {
            ENTITY_KETTLE: {"train_mean_w": 12.5},
            ENTITY_WASHER: {"train_mean_w": 12.5},
        },
    }
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": GRID_S,
        "day_bounds_ms": day_bounds_ms,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "phase_channels": {},
        "shadow": shadow,
        "validation_time_ms": VALIDATION_TIME_MS,
    }

    report = run_stage("validate", job)
    assert report["status"] == "ok"

    device_ids = [d["entity_id"] for d in report["devices"]]
    refusal_ids = [r["entity_id"] for r in report["refusals"]]
    undiscovered_ids = [e["entity_id"] for e in report["not_discovered"]]
    all_ids = device_ids + refusal_ids + undiscovered_ids
    assert sorted(all_ids) == shadow["eligible"]
    assert len(all_ids) == len(set(all_ids))

    # the washer: only a refusal, named with its reason and audit trail
    assert ENTITY_WASHER not in device_ids
    washer_refusal = next(r for r in report["refusals"] if r["entity_id"] == ENTITY_WASHER)
    assert washer_refusal["reason"] == "unsupported_kind"
    assert washer_refusal["link_overlap"] == 0.7

    # the kettle: measured, with the class derived from its artefact
    assert device_ids == [ENTITY_KETTLE]
    kettle = report["devices"][0]
    assert kettle["class"] == "switched_1.5_3kw_1_level"
    assert isinstance(kettle["unvalidated"], bool)
    assert kettle["unvalidated"] == (not kettle["passes"])
    # one peer in the class: the class bar cannot pass, so only the D-07 bar can clear it
    klass = next(c for c in report["classes"] if c["class"] == kettle["class"])
    assert klass["peers"] == 1
    assert klass["passes"] is False


# --- Plan 06-07 (DET-02, D-24): program devices through the same validation ---

WASHER_SIG_ID = "sig-shadow-washer"
WASHER_STAGES = [(2000, 30), (200, 80), (500, 10)]
WASHER_CLASS = "program_1.5_3kw_multi_level"


def _washer_program_artefact(*, with_stages: bool = True) -> dict[str, Any]:
    """The shadow program artefact of the synthetic 3-stage washer."""
    artefact = program_artefact(WASHER_SIG_ID, WASHER_STAGES)
    if not with_stages:
        # A program the detector cannot track: no stage template at all.
        artefact["session"] = {**artefact["session"], "stages": []}
    return artefact


def _washer_validate_job(
    tmp_path: Path,
    tag: str,
    *,
    daily: bool,
    with_stages: bool = True,
) -> tuple[dict[str, Any], int]:
    """Build a washer buffer plus the validate job that shadow-links it as a program."""
    days = 35
    buf_path = tmp_path / f"buf_{tag}.sqlite"
    stage_dir = tmp_path / f"stage_{tag}"
    data = build_buffer(
        buf_path,
        SynthSpec(days=days, washer=True, washer_daily=daily),
        END_MS,
    )
    day_bounds_ms = utc_day_bounds(END_MS, days)
    washer_ch = data.channel_ids[ENTITY_WASHER]
    first_holdout = calendar_holdout_start(days, 7)
    shadow = {
        "schema_version": 1,
        "created": "2026-11-13T00:00:00+00:00",
        "run_id": "test-shadow-washer",
        "grid_s": GRID_S,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "eligible": [ENTITY_WASHER],
        "train_cutoff_ms": int(day_bounds_ms[first_holdout]),
        "holdout_days": 7,
        "artefacts": [_washer_program_artefact(with_stages=with_stages)],
        "links": [
            {
                "entity_id": ENTITY_WASHER,
                "signature_id": WASHER_SIG_ID,
                "kind": "program",
                "overlap": 0.9,
                "channel": washer_ch,
            }
        ],
        "not_discovered": [],
        "plugs": {ENTITY_WASHER: {"train_mean_w": 12.5}},
    }
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": GRID_S,
        "day_bounds_ms": day_bounds_ms,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "phase_channels": {},
        "shadow": shadow,
        "validation_time_ms": VALIDATION_TIME_MS,
    }
    return job, washer_ch


def test_validate_stage_measures_program_washer(tmp_path: Path) -> None:
    """A plug-metered washer program is measured by the stage tracker (DET-02, D-24).

    The washer runs daily, so all 7 holdout days hold one run: enough for a
    figure (>= MIN_EVENTS_FIGURE) but not for the 8-event pass bar, so the
    honest v0.x verdict is "unvalidated" through the unchanged P5 bars.
    """
    job, _washer_ch = _washer_validate_job(tmp_path, "prog", daily=True)
    report = run_stage("validate", job)

    assert report["status"] == "ok"
    washer = next(d for d in report["devices"] if d["entity_id"] == ENTITY_WASHER)
    assert washer["class"] == WASHER_CLASS
    assert washer["n_events"] == 7
    for key in ("energy_error_pct", "nde", "mae_w", "bias_pct"):
        assert isinstance(washer[key], float), key
    assert washer["baseline"]["nde"] is not None
    assert washer["baseline"]["energy_error_pct"] is not None
    assert len(washer["per_day"]) == 7
    assert washer["energy_true_wh"] > 0
    # One peer only: the class bar cannot pass, and 7 events miss the 8 pass bar.
    assert washer["passes"] is False
    assert "events_below_pass_bar" in washer["reasons"]
    assert washer["unvalidated"] is True

    # D-24: the thresholds block is the unchanged Phase 5 constant set.
    assert report["thresholds"] == {
        "min_usable_days": MIN_USABLE_DAYS,
        "min_covered_holdout_days": MIN_COVERED_HOLDOUT_DAYS,
        "min_events_figure": MIN_EVENTS_FIGURE,
        "min_events_pass": MIN_EVENTS_PASS,
        "max_device_error_pct": MAX_DEVICE_ERROR_PCT,
        "class_min_passing": CLASS_MIN_PASSING,
        "class_max_median_pct": CLASS_MAX_MEDIAN_PCT,
        "class_max_peer_pct": CLASS_MAX_PEER_PCT,
    }
    assert not [r for r in report["refusals"] if r["entity_id"] == ENTITY_WASHER]


def test_validate_stage_program_every_second_day_refused(tmp_path: Path) -> None:
    """A washer used every second day gets no figure: insufficient_data (D-24, SC4).

    Four holdout runs are below MIN_EVENTS_FIGURE, so the report refuses the
    device and the release table reads "not validated" — the honest outcome.
    """
    job, _washer_ch = _washer_validate_job(tmp_path, "alt", daily=False)
    report = run_stage("validate", job)

    assert report["status"] == "ok"
    assert all(d["entity_id"] != ENTITY_WASHER for d in report["devices"])
    refusal = next(r for r in report["refusals"] if r["entity_id"] == ENTITY_WASHER)
    assert refusal["reason"] == "insufficient_data"
    assert refusal["n_events"] == 4
    assert refusal["link_overlap"] == 0.9
    assert "energy_error_pct" not in refusal
    assert "nde" not in refusal


def test_validate_stage_program_without_stages_reads_zero_energy(tmp_path: Path) -> None:
    """A program link the detector cannot track is reported with 0 Wh, never dropped.

    The artefact carries no stage template (P5 D-03 edge): the stage is still
    measured — it keeps its class-less figure — but the estimate is zero.
    """
    job, _washer_ch = _washer_validate_job(
        tmp_path, "nostages", daily=True, with_stages=False
    )
    report = run_stage("validate", job)

    assert report["status"] == "ok"
    washer = next(d for d in report["devices"] if d["entity_id"] == ENTITY_WASHER)
    assert washer["energy_est_wh"] == 0.0
    assert washer["energy_true_wh"] > 0
    assert washer["class"] is None
    assert washer["n_events"] == 7
    assert washer["passes"] is False
    assert washer["unvalidated"] is True
    assert not [r for r in report["refusals"] if r["entity_id"] == ENTITY_WASHER]


# --- Plan 06-08 (DET-03, D-08, D-28): the overlap block and the synthetic injection ---

OVERLAP_DAYS = 35


def _overlap_validate_job(
    tmp_path: Path, tag: str
) -> tuple[dict[str, Any], Path, int, int]:
    """Build a kettle-during-wash buffer plus the validate job linking both plugs.

    A switched kettle artefact and a program washer artefact are linked to their
    plugs, and the washer runs daily with a kettle boil five minutes into every
    wash, so the two plugs overlap by ground truth (SC2's canonical case).
    """
    buf_path = tmp_path / f"buf_{tag}.sqlite"
    stage_dir = tmp_path / f"stage_{tag}"
    data = build_buffer(
        buf_path,
        SynthSpec(
            days=OVERLAP_DAYS,
            washer=True,
            washer_daily=True,
            kettle_plug=True,
            kettle_during_washer=True,
        ),
        END_MS,
    )
    day_bounds_ms = utc_day_bounds(END_MS, OVERLAP_DAYS)
    kettle_ch = data.channel_ids[ENTITY_KETTLE]
    washer_ch = data.channel_ids[ENTITY_WASHER]
    first_holdout = calendar_holdout_start(OVERLAP_DAYS, 7)
    shadow = {
        "schema_version": 1,
        "created": "2026-11-13T00:00:00+00:00",
        "run_id": "test-shadow-overlap",
        "grid_s": GRID_S,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "eligible": sorted([ENTITY_KETTLE, ENTITY_WASHER]),
        "train_cutoff_ms": int(day_bounds_ms[first_holdout]),
        "holdout_days": 7,
        "artefacts": [_kettle_artefact(), _washer_program_artefact()],
        "links": [
            {
                "entity_id": ENTITY_KETTLE,
                "signature_id": SIG_ID,
                "kind": "switched",
                "overlap": 0.8,
                "channel": kettle_ch,
            },
            {
                "entity_id": ENTITY_WASHER,
                "signature_id": WASHER_SIG_ID,
                "kind": "program",
                "overlap": 0.9,
                "channel": washer_ch,
            },
        ],
        "not_discovered": [],
        "plugs": {
            ENTITY_KETTLE: {"train_mean_w": 12.5},
            ENTITY_WASHER: {"train_mean_w": 12.5},
        },
    }
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": GRID_S,
        "day_bounds_ms": day_bounds_ms,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "phase_channels": {},
        "shadow": shadow,
        "validation_time_ms": VALIDATION_TIME_MS,
    }
    return job, buf_path, kettle_ch, washer_ch


def test_validate_stage_reports_kettle_during_wash_overlap(tmp_path: Path) -> None:
    """SC2 report half: an ok report shows the kettle-during-wash overlap (D-08).

    The report carries the per-device solo/overlap errors, the joint vs refused
    overlap edges, the energy left in the remainder and exactly one
    (kettle, washer) pair row with both devices' errors.
    """
    job, _buf, _kettle_ch, _washer_ch = _overlap_validate_job(tmp_path, "overlap")
    report = run_stage("validate", job)

    assert report["status"] == "ok"
    assert "overlap_synthetic" not in report
    block = report["overlap"]
    assert isinstance(block["definition"], str) and block["definition"]
    assert isinstance(block["edges_definition"], str) and block["edges_definition"]
    assert isinstance(block["energy_left_definition"], str)
    assert block["n_overlap_ticks"] >= OVERLAP_MIN_TICKS
    assert block["n_solo_ticks"] >= 0
    assert isinstance(block["edges"]["attributed_jointly"], int)
    assert block["edges"]["attributed_jointly"] >= 0
    assert isinstance(block["edges"]["refused"], int)
    assert block["edges"]["refused"] >= 0
    assert block["energy_left_in_remainder_wh"] is not None

    pairs = block["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair["a"], pair["b"]) == (ENTITY_KETTLE, ENTITY_WASHER)
    assert pair["n_ticks"] >= OVERLAP_MIN_TICKS
    assert isinstance(pair["a_error_pct"], float)
    assert isinstance(pair["b_error_pct"], float)

    per_device = {entry["entity_id"]: entry for entry in block["per_device"]}
    assert set(per_device) == {ENTITY_KETTLE, ENTITY_WASHER}
    kettle_overlap = per_device[ENTITY_KETTLE]["overlap"]
    washer_overlap = per_device[ENTITY_WASHER]["overlap"]
    assert kettle_overlap["n_ticks"] >= OVERLAP_MIN_TICKS
    assert washer_overlap["n_ticks"] >= OVERLAP_MIN_TICKS
    assert isinstance(kettle_overlap["energy_error_pct"], float)
    assert isinstance(washer_overlap["energy_error_pct"], float)
    # The overlap block is pure data: the report still serialises nan-free.
    json.dumps(report, allow_nan=False)


def test_validate_stage_synthetic_overlap_is_labelled_and_isolated(
    tmp_path: Path,
) -> None:
    """D-28: the synthetic-overlap flag adds a labelled block and changes nothing
    else — devices, classes, refusals and the real overlap block are identical."""
    job, _buf, _kettle_ch, _washer_ch = _overlap_validate_job(tmp_path, "synth")
    plain = run_stage("validate", dict(job))

    flagged_job = dict(job)
    flagged_job["stage_dir"] = str(tmp_path / "stage_synth_flag")
    flagged_job["synthetic_overlap"] = True
    flagged = run_stage("validate", flagged_job)

    assert plain["status"] == "ok"
    assert flagged["status"] == "ok"
    for key in ("devices", "classes", "refusals", "not_discovered", "overlap"):
        assert json.dumps(flagged[key], sort_keys=True) == json.dumps(
            plain[key], sort_keys=True
        ), key
    assert "overlap_synthetic" not in plain

    block = flagged["overlap_synthetic"]
    assert block["synthetic"] is True
    assert block["label"].startswith("SYNTHETIC:")
    injected = block["injected"]
    assert injected["shift_s"] != 0
    assert injected["a"] in block["label"] and injected["b"] in block["label"]
    assert f"{injected['shift_s']} s" in block["label"]
    # The synthetic block carries the same keys as the real overlap block.
    assert set(block) >= set(plain["overlap"])
    assert block["n_overlap_ticks"] >= OVERLAP_MIN_TICKS
    assert len(block["pairs"]) == 1
    json.dumps(flagged, allow_nan=False)


def test_validate_stage_synthetic_overlap_needs_two_plugs(tmp_path: Path) -> None:
    """With fewer than two measured plugs the flag refuses without figures."""
    job, _buf = _validate_job(tmp_path, "synth_one")
    flagged_job = dict(job)
    flagged_job["synthetic_overlap"] = True
    report = run_stage("validate", flagged_job)

    assert report["status"] == "ok"
    assert report["overlap_synthetic"] == {
        "synthetic": True,
        "status": "refused",
        "reason": "fewer_than_two_plugs",
    }
    # The real block is still there and unchanged.
    assert "overlap" in report


def test_worker_synthetic_overlap_refuses_zero_shift() -> None:
    """A computed shift of 0 samples would only duplicate a natural overlap, so
    the synthetic block refuses with reason zero_shift and no figures
    (review concern L209@8b3be9c)."""
    from custom_components.powerguesser.engine.learning.worker import (
        _synthetic_overlap_block,
    )

    n = 40
    masks = {"sensor.a": np.zeros(n, dtype=bool), "sensor.b": np.zeros(n, dtype=bool)}
    masks["sensor.a"][0:5] = True
    masks["sensor.b"][0:5] = True  # first ON run starts in the same sample

    block = _synthetic_overlap_block(
        ys={"sensor.a": np.full(n, 100.0), "sensor.b": np.full(n, 100.0)},
        sig_of={"sensor.a": "sa", "sensor.b": "sb"},
        masks=masks,
        r_masked=np.full(n, np.nan),
        artefacts=[],
        step_s=float(GRID_S),
        phases=None,
        suppress_alignment=True,
    )

    assert block == {"synthetic": True, "status": "refused", "reason": "zero_shift"}


# --- Plan 06-09 (REV-04, D-13): device-target plug sessions ---

SESSION_DAYS = 20


def _session_validate_job(
    tmp_path: Path,
    tag: str,
    *,
    window_days: int = 15,
    extra_others: list[dict[str, Any]] | None = None,
    own_artefact: dict[str, Any] | None = None,
    second_own_artefact: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, Path]:
    """Build a kettle_session buffer plus the validate_sessions job over it."""
    buf_path = tmp_path / f"buf_{tag}.sqlite"
    stage_dir = tmp_path / f"stage_{tag}"
    data = build_buffer(
        buf_path, SynthSpec(days=SESSION_DAYS, kettle_session=True), END_MS
    )
    day_bounds_ms = utc_day_bounds(END_MS, SESSION_DAYS)
    session_ch = data.channel_ids[ENTITY_KETTLE_SESSION]
    n_days = len(day_bounds_ms) - 1
    w_start = day_bounds_ms[n_days - window_days]
    w_end = day_bounds_ms[-1]
    artefact = own_artefact if own_artefact is not None else _kettle_artefact()
    sessions = [
        {
            "session_id": "sess-1",
            "device_uuid": "dev-kettle",
            "artefact": dict(artefact),
            "artefact_v": 1,
            "plug_channel": session_ch,
            "window_start_ms": w_start,
            "window_end_ms": w_end,
        }
    ]
    if second_own_artefact is not None:
        sessions.append(
            {
                "session_id": "sess-2",
                "device_uuid": "dev-bad",
                "artefact": dict(second_own_artefact),
                "artefact_v": 1,
                "plug_channel": session_ch,
                "window_start_ms": w_start,
                "window_end_ms": w_end,
            }
        )
    others = [dict(_kettle_artefact())]
    if extra_others:
        others.extend(extra_others)
    job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": GRID_S,
        "day_bounds_ms": day_bounds_ms,
        "enabled_event_types": list(ENABLED_EVENT_TYPES),
        "phase_channels": {},
        "metered_channels": [],
        "others": others,
        "sessions": sessions,
    }
    return job, session_ch, buf_path


def _read_sessions(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the stage's sessions.json."""
    path = Path(job["stage_dir"]) / "sessions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _zero_session_samples_before(buf_path: Path, channel: int, ts_ms: int) -> None:
    """Set a session channel's samples before ts_ms to 0 W.

    The session baseline is the plug's mean over the earlier window days. For a
    stationary duty cycle that constant is nearly optimal, so the device can
    never beat it — the honest v0.x limitation the plan's flagged assumptions
    name. This fixture makes the plug's earlier window readings 0 W (the device
    was idle before the holdout) so the constant baseline is deliberately weak
    and the passing path is reachable.
    """
    with sqlite3.connect(str(buf_path)) as conn:
        conn.execute(
            "UPDATE samples SET w = 0 WHERE ch = ? AND ts < ? AND w IS NOT NULL",
            (channel, ts_ms),
        )
        conn.commit()


def test_validate_sessions_passes_kettle_session(tmp_path: Path) -> None:
    """A device-target session that clears the P5 device bar passes (D-13)."""
    job, session_ch, buf_path = _session_validate_job(tmp_path, "sess_pass")
    day_bounds_ms = job["day_bounds_ms"]
    n_days = len(day_bounds_ms) - 1
    _zero_session_samples_before(buf_path, session_ch, day_bounds_ms[n_days - 7])
    out = run_stage("validate_sessions", job)

    assert out["n"] == 1
    entries = _read_sessions(job)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "ok"
    assert entry["session_id"] == "sess-1"
    assert entry["device_uuid"] == "dev-kettle"
    assert entry["artefact_v"] == 1
    assert entry["passes"] is True
    assert entry["n_events"] >= MIN_EVENTS_PASS
    assert entry["usable_holdout_days"] == 7
    assert isinstance(entry["energy_error_pct"], float)
    assert entry["energy_error_pct"] <= MAX_DEVICE_ERROR_PCT
    assert entry["nde"] is not None and entry["mae_w"] is not None
    assert entry["window_start"] and entry["window_end"]
    assert entry["holdout_start"] and entry["holdout_end"]
    assert entry["skipped_competitors"] == []
    json.dumps(entries, allow_nan=False)


def test_validate_sessions_refuses_without_baseline(tmp_path: Path) -> None:
    """A window with no earlier baseline day is refused insufficient_data (edge)."""
    job, _ch, _buf = _session_validate_job(tmp_path, "sess_nobase", window_days=7)
    run_stage("validate_sessions", job)

    entry = _read_sessions(job)[0]
    assert entry["status"] == "refused"
    assert entry["reason"] == "insufficient_data"
    assert entry["passes"] is False
    assert "energy_error_pct" not in entry


def test_validate_sessions_baseline_day_with_null_plug_is_excluded(
    tmp_path: Path,
) -> None:
    """A whole-day plug outage never poisons the baseline: the figure is still made."""
    job, session_ch, buf_path = _session_validate_job(tmp_path, "sess_null")
    day_bounds_ms = job["day_bounds_ms"]
    n_days = len(day_bounds_ms) - 1
    base_day = n_days - 12  # an earlier (baseline) window day
    with sqlite3.connect(str(buf_path)) as conn:
        conn.execute(
            "UPDATE samples SET w = NULL WHERE ch = ? AND ts >= ? AND ts < ?",
            (session_ch, day_bounds_ms[base_day], day_bounds_ms[base_day + 1]),
        )
        conn.commit()

    run_stage("validate_sessions", job)

    entry = _read_sessions(job)[0]
    # The day is excluded from the baseline rather than poisoning it: the
    # figure is still computed (a NaN baseline would have leaked into it).
    assert entry["status"] == "ok"
    assert isinstance(entry["energy_error_pct"], float)
    assert entry["nde"] is not None
    assert entry["usable_holdout_days"] == 7


def test_validate_sessions_skips_corrupt_competitor(tmp_path: Path) -> None:
    """A competitor that fails compile_model is skipped, never aborts the stage."""
    bad_program = program_artefact("sig-bad-program", [(2000, 30)])
    bad_program["session"] = {**bad_program["session"], "stages": []}
    job, _ch, _buf = _session_validate_job(
        tmp_path, "sess_skip", extra_others=[bad_program]
    )
    run_stage("validate_sessions", job)

    entry = _read_sessions(job)[0]
    assert entry["status"] == "ok"
    assert entry["skipped_competitors"] == ["sig-bad-program"]
    assert entry["energy_error_pct"] is not None


def test_validate_sessions_refuses_own_model_error_and_judges_others(
    tmp_path: Path,
) -> None:
    """A session whose own artefact fails is refused model_error; others are judged."""
    bad_own = program_artefact("sig-bad-own", [(2000, 30)])
    bad_own["session"] = {**bad_own["session"], "stages": []}
    job, _ch, _buf = _session_validate_job(
        tmp_path, "sess_own", second_own_artefact=bad_own
    )
    run_stage("validate_sessions", job)

    entries = _read_sessions(job)
    assert len(entries) == 2
    by_id = {entry["session_id"]: entry for entry in entries}
    assert by_id["sess-1"]["status"] == "ok"
    assert by_id["sess-1"]["energy_error_pct"] is not None
    assert by_id["sess-2"]["status"] == "refused"
    assert by_id["sess-2"]["reason"] == "model_error"


def test_validate_sessions_empty_without_sessions(tmp_path: Path) -> None:
    """With no sessions the stage writes an empty list and touches nothing."""
    job, _ch, _buf = _session_validate_job(tmp_path, "sess_none")
    job["sessions"] = []
    out = run_stage("validate_sessions", job)
    assert out["n"] == 0
    assert _read_sessions(job) == []


# --- Plan 08-04 (D-05): capture windows are dropped from every holdout metric ---


def _find_kettle_run_ms(
    buf_path: Path, channel: int, lo_ms: int, hi_ms: int
) -> tuple[int, int] | None:
    """Return the first contiguous ON (>= 1500 W) block's (start_ms, end_ms)."""
    with sqlite3.connect(str(buf_path)) as conn:
        rows = conn.execute(
            "SELECT ts, w FROM samples WHERE ch = ? AND ts >= ? AND ts < ? ORDER BY ts",
            (channel, lo_ms, hi_ms),
        ).fetchall()
    start: int | None = None
    end: int | None = None
    for ts, w in rows:
        if w is not None and w >= 1500:
            if start is None:
                start = int(ts)
            end = int(ts)
        elif start is not None:
            return start, end if end is not None else start
    if start is not None:
        return start, end if end is not None else start
    return None


def _holdout_masks(job: dict[str, Any]):
    """Recompute the stage's holdout grid/masks from the job's buffer (without cap)."""
    day_bounds_ms = job["day_bounds_ms"]
    start_ms = int(day_bounds_ms[0])
    end_ms = int(day_bounds_ms[-1])
    step_ms = GRID_S * 1000
    kettle_ch = int(job["shadow"]["links"][0]["channel"])
    series = load_residual(str(job["buffer_path"]), start_ms, end_ms, step_ms, [kettle_ch], {})
    r = series.gross
    y = series.plugs[kettle_ch]
    valid = data_days(r, start_ms, step_ms, day_bounds_ms)
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, 7)
    train_cutoff_ms = int(job["shadow"]["train_cutoff_ms"])
    valid_set = set(valid)
    holdout_idx = [
        d
        for d in range(first_holdout, n_days)
        if d in valid_set and int(day_bounds_ms[d]) >= train_cutoff_ms
    ]
    holdout_set = set(holdout_idx)
    bounds = np.array(day_bounds_ms, dtype=np.int64)
    grid_ts = start_ms + np.arange(len(r), dtype=np.int64) * step_ms
    sample_day = np.searchsorted(bounds, grid_ts, side="right") - 1
    holdout_mask = np.isin(sample_day, list(holdout_set))
    return series, r, y, holdout_mask, grid_ts


def _kettle_capture_window(job: dict[str, Any]) -> list[int]:
    """A short capture window covering the start of a holdout kettle run."""
    day_bounds_ms = job["day_bounds_ms"]
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, 7)
    kettle_ch = int(job["shadow"]["links"][0]["channel"])
    run = _find_kettle_run_ms(
        Path(job["buffer_path"]), kettle_ch, day_bounds_ms[first_holdout], day_bounds_ms[-1]
    )
    assert run is not None, "no kettle run in the holdout to capture"
    on_ms, _off_ms = run
    return [on_ms, on_ms + 2 * GRID_S * 1000]


def test_validate_excludes_capture_windows(tmp_path: Path) -> None:
    """A capture window over a holdout kettle run drops the event and reports the count."""
    job, _buf = _validate_job(tmp_path, "excl")
    window = _kettle_capture_window(job)
    job["capture_windows"] = [window]

    with_report = run_stage("validate", job)
    job_without, _buf2 = _validate_job(tmp_path, "excl_plain")
    without_report = run_stage("validate", job_without)

    assert with_report["status"] == "ok"
    excluded = with_report["holdout"]["excluded_capture"]
    assert excluded["windows"] == 1
    assert excluded["samples"] > 0
    assert "excluded_capture" not in without_report["holdout"]

    with_kettle = next(d for d in with_report["devices"] if d["entity_id"] == ENTITY_KETTLE)
    without_kettle = next(
        d for d in without_report["devices"] if d["entity_id"] == ENTITY_KETTLE
    )
    assert with_kettle["n_events"] == without_kettle["n_events"] - 1
    assert with_kettle["energy_true_wh"] < without_kettle["energy_true_wh"]


def test_validate_without_capture_windows_is_byte_identical(tmp_path: Path) -> None:
    """No key and an empty list both leave the report byte-identical and key-free."""
    job, _buf = _validate_job(tmp_path, "ident")
    report_no_key = run_stage("validate", job)
    job_empty = {
        **job,
        "stage_dir": str(tmp_path / "stage_ident_empty"),
        "capture_windows": [],
    }
    report_empty = run_stage("validate", job_empty)

    assert canonical_json(report_no_key) == canonical_json(report_empty)
    assert b"excluded_capture" not in canonical_json(report_no_key)
    assert "excluded_capture" not in report_no_key["holdout"]


def test_validate_does_not_mask_the_replayed_residual(tmp_path: Path) -> None:
    """The replay runs on the un-masked residual; only the metric mask loses samples."""
    job, _buf = _validate_job(tmp_path, "resid")
    window = _kettle_capture_window(job)
    job["capture_windows"] = [window]
    report = run_stage("validate", job)

    series, r, y, holdout_mask, grid_ts = _holdout_masks(job)
    cap = holdout_exclusion_mask(grid_ts, [window], HOLDOUT_PAD_STEPS * (GRID_S * 1000))
    # The stage's own replay construction, run on the residual WITHOUT the cap.
    r_masked = np.where(holdout_mask, r, np.nan)
    online = replay_online(
        measurable_artefacts(job["shadow"]),
        r_masked,
        float(GRID_S),
        suppress_alignment=True,
        collect_power=True,
    )
    sig_id = str(job["shadow"]["links"][0]["signature_id"])
    step_energy = online.energy_step_wh[sig_id]
    ok_with = holdout_mask & ~cap & np.isfinite(r) & np.isfinite(y)
    expected = float(np.sum(step_energy[ok_with]))

    entry = next(d for d in report["devices"] if d["entity_id"] == ENTITY_KETTLE)
    # If the implementation had masked r_masked, the replay would differ and this
    # equality (against the un-masked replay) would fail.
    assert entry["energy_est_wh"] == round(expected, 1)


def test_validate_sessions_applies_capture_exclusion(tmp_path: Path) -> None:
    """A capture window drops a session event; without the key the bytes are unchanged."""
    job, session_ch, buf_path = _session_validate_job(tmp_path, "sess_cap")
    day_bounds_ms = job["day_bounds_ms"]
    n_days = len(day_bounds_ms) - 1
    _zero_session_samples_before(buf_path, session_ch, day_bounds_ms[n_days - 7])

    # Byte-identical without the key: two runs on the same job produce the same bytes.
    plain_job = {**job, "stage_dir": str(tmp_path / "stage_sess_cap_plain")}
    run_stage("validate_sessions", plain_job)
    plain_bytes = (Path(plain_job["stage_dir"]) / "sessions.json").read_bytes()
    run_stage("validate_sessions", job)
    assert (Path(job["stage_dir"]) / "sessions.json").read_bytes() == plain_bytes
    plain_entry = _read_sessions(job)[0]

    run = _find_kettle_run_ms(
        buf_path, session_ch, day_bounds_ms[n_days - 7], day_bounds_ms[-1]
    )
    assert run is not None
    on_ms, _off_ms = run
    capped_job = {
        **job,
        "stage_dir": str(tmp_path / "stage_sess_cap_out"),
        "capture_windows": [[on_ms, on_ms + 2 * GRID_S * 1000]],
    }
    run_stage("validate_sessions", capped_job)
    capped_entry = _read_sessions(capped_job)[0]

    assert capped_entry["n_events"] == plain_entry["n_events"] - 1
    assert capped_entry["energy_true_wh"] < plain_entry["energy_true_wh"]


def test_validate_sessions_does_not_mask_the_replayed_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture window drops the metric samples but never NaNs the replay input.

    The replay input keeps the pre-exclusion holdout mask (Pitfall 5); only the
    ``ok``/``n_events`` metric mask loses the capture window. If the window had
    masked the replay residual, the replay input would differ from the un-masked
    run inside the capture window.
    """
    from custom_components.powerguesser.engine.learning import worker as worker_mod

    job, session_ch, buf_path = _session_validate_job(tmp_path, "sess_resid")
    day_bounds_ms = job["day_bounds_ms"]
    n_days = len(day_bounds_ms) - 1
    _zero_session_samples_before(buf_path, session_ch, day_bounds_ms[n_days - 7])

    run = _find_kettle_run_ms(
        buf_path, session_ch, day_bounds_ms[n_days - 7], day_bounds_ms[-1]
    )
    assert run is not None
    on_ms, _off_ms = run
    window = [on_ms, on_ms + 2 * GRID_S * 1000]

    inputs: list[np.ndarray] = []
    real = worker_mod.replay_online

    def _spy(artefacts: Any, r: Any, *args: Any, **kwargs: Any) -> Any:
        inputs.append(np.asarray(r, dtype=float).copy())
        return real(artefacts, r, *args, **kwargs)

    monkeypatch.setattr(worker_mod, "replay_online", _spy)

    plain_job = {**job, "stage_dir": str(tmp_path / "stage_sess_resid_plain")}
    run_stage("validate_sessions", plain_job)
    capped_job = {
        **job,
        "stage_dir": str(tmp_path / "stage_sess_resid_cap"),
        "capture_windows": [window],
    }
    run_stage("validate_sessions", capped_job)

    assert len(inputs) == 2
    r_plain, r_capped = inputs[0], inputs[1]
    grid_ts = day_bounds_ms[0] + np.arange(len(r_plain), dtype=np.int64) * (GRID_S * 1000)
    cap = holdout_exclusion_mask(grid_ts, [window], HOLDOUT_PAD_STEPS * GRID_S * 1000)
    assert np.count_nonzero(cap) > 0
    # The capture window is real data in the replay input (finite, not a NaN hole)...
    assert np.isfinite(r_capped[cap]).any()
    # ...and the input is identical to the un-capped run.
    assert np.array_equal(r_plain, r_capped, equal_nan=True)

