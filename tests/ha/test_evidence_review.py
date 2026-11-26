"""Phase 7 tracer: a learning run's evidence reaches the candidate menu.

EVID-01 (recent occurrences + summary) and EVID-04 (evidence as concise text
in the options-flow Review) end to end, plus the pure formatter/engine
boundary tests (D-03, D-06, D-08, D-09, D-11, D-19).
"""

import json
import copy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util

from custom_components.powerguesser import review_actions
from custom_components.powerguesser.config_flow import _escape_markdown
from custom_components.powerguesser.const import (
    BUFFER_DIR,
    DOMAIN,
    EVIDENCE_MAX_FAILURES,
    LEARNING_DIR,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.engine.learning.evidence import (
    MAX_OCCURRENCES,
    MIN_RHYTHM_RUNS,
    TOD_MIN_CONCENTRATION,
    build_evidence,
)
from custom_components.powerguesser.review_actions import (
    EVIDENCE_FAILED_TEXT,
    EVIDENCE_PENDING_TEXT,
    evidence_state,
    evidence_text,
    format_duration,
    format_energy_wh,
    format_gap,
    overlay_captured,
)
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    mock_llm,
    proposal_responder,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec

_DAY_MS = 86_400_000
_BASE_MS = 1_700_000_000_000


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _occurrences(n: int) -> list[dict[str, Any]]:
    """n occurrences, newest first (descending start_ms)."""
    return [
        {
            "start_ms": _BASE_MS - i * _DAY_MS,
            "dur_s": 660.0,
            "step_w": 2150.0,
            "energy_wh": 390.0,
            "phase": "l2",
        }
        for i in range(n)
    ]


def _evidence(
    n_runs: int, *, observed_days: int = 20, n_days: int = 30
) -> dict[str, Any]:
    """A ready-shape evidence dict with n_runs runs (occurrences capped)."""
    occurrences = _occurrences(min(n_runs, MAX_OCCURRENCES))
    return {
        "v": 1,
        "computed_at": "2026-11-24T10:00:00+00:00",
        "source": "run",
        "window": {
            "start_ms": _BASE_MS - (n_days - 1) * _DAY_MS,
            "end_ms": _BASE_MS,
            "observed_days": observed_days,
            "n_days": n_days,
        },
        "n_runs": n_runs,
        "first_ms": _BASE_MS - (n_runs - 1) * _DAY_MS if n_runs else None,
        "last_ms": _BASE_MS if n_runs else None,
        "occurrences": occurrences,
        "rhythm": None,
        "profile": None,
        "hints": [],
    }


def _rhythm(
    *,
    runs_per_day: float = 2.0,
    runs_per_week: float = 14.0,
    lo_min: int = 390,
    hi_min: int = 540,
    concentration: float = 0.8,
    dow_counts: list[int] | None = None,
    gap_median_s: float | None = 86_400.0,
) -> dict[str, Any]:
    """A hand-built rhythm dict in the engine's contract."""
    return {
        "runs_per_day": runs_per_day,
        "runs_per_week": runs_per_week,
        "active_days": 18,
        "tod": {
            "center_min": 465,
            "lo_min": lo_min,
            "hi_min": hi_min,
            "concentration": concentration,
        },
        "dow_counts": dow_counts if dow_counts is not None else [3, 3, 3, 3, 3, 3, 3],
        "gap_median_s": gap_median_s,
        "gap_p90_s": gap_median_s,
    }


def _profile(
    *,
    step_p50: float | None = 2150.0,
    dur_p50: float | None = 660.0,
    wh_per_run: float | None = 390.0,
    wh_per_day: float | None = 780.0,
    phase: str | None = "l2",
) -> dict[str, Any]:
    """A hand-built profile dict in the engine's contract."""

    def _pct(p50: float | None) -> dict[str, float] | None:
        if p50 is None:
            return None
        return {"p10": p50 * 0.9, "p50": p50, "p90": p50 * 1.1}

    return {
        "step_w": _pct(step_p50),
        "dur_s": _pct(dur_p50),
        "wh_per_run": wh_per_run,
        "wh_per_day": wh_per_day,
        "phase": phase,
        "phase_available": phase is not None,
    }


def _ready(
    n_runs: int,
    *,
    observed_days: int = 20,
    n_days: int = 30,
    rhythm: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    hints: list[str] | None = None,
) -> dict[str, Any]:
    """A ready evidence dict with explicit rhythm/profile/hints sections."""
    ev = _evidence(n_runs, observed_days=observed_days, n_days=n_days)
    ev["rhythm"] = rhythm
    ev["profile"] = profile
    ev["hints"] = hints if hints is not None else []
    return ev


def _build(
    n: int, *, minutes: list[int] | None = None, minute_of_day: int = 400
) -> dict[str, Any]:
    """A real build_evidence result for n runs, one run per local day."""
    if minutes is None:
        minutes = [minute_of_day] * n
    step_ms = 30_000
    # start_ms = _BASE_MS + start_i * step_ms; one run per day at the wanted minute.
    start_i = np.array(
        [day * 2880 + minutes[day] * 2 for day in range(n)], dtype=np.int64
    )
    table: dict[str, np.ndarray] = {
        "start_i": start_i,
        "end_i": start_i + 10,
        "rise_w": np.full(n, 2150.0),
        "fall_w": np.full(n, -2150.0),
        "dur_s": np.full(n, 660.0),
        "plateau_w": np.full(n, 2150.0),
        "energy_wh": np.full(n, 390.0),
        "kind": np.zeros(n, dtype=np.int32),
        "hour": np.zeros(n, dtype=np.int32),
        "dow": np.array([day % 7 for day in range(n)], dtype=np.int32),
        "l1_share": np.full(n, np.nan),
        "l2_share": np.full(n, np.nan),
        "l3_share": np.full(n, np.nan),
    }
    n_days = max(n, 1)
    day_bounds_ms = [_BASE_MS + day * _DAY_MS for day in range(n_days + 1)]
    result = build_evidence(
        table,
        {"sig": set(range(n))},
        start_ms=_BASE_MS,
        step_ms=step_ms,
        day_bounds_ms=day_bounds_ms,
        valid_days=list(range(n_days)),
        now_iso="2026-11-24T10:00:00+00:00",
        source="run",
    )
    return result["sig"]


def _assert_no_accuracy(block: str) -> None:
    """No percentage and no accuracy wording outside the fixed header (D-05)."""
    assert "%" not in block
    for line in block.split("\n"):
        if line == "**Evidence** (heuristics, not accuracy)":
            continue
        assert "accura" not in line.lower(), line



# ---------------------------------------------------------------------------
# Tracer: engine -> compile_score -> candidates Store -> candidate menu
# ---------------------------------------------------------------------------


async def test_tracer_run_evidence_reaches_candidate_menu(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """A real run stores evidence and the candidate menu renders it in local time."""
    await hass.config.async_set_time_zone("Europe/Berlin")
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    assert supervisor.run_task is not None
    await supervisor.run_task

    candidates = supervisor.candidates
    assert candidates
    for cand in candidates:
        evidence = cand["evidence"]
        assert evidence["v"] == 1
        assert evidence["source"] == "run"
        assert evidence["n_runs"] >= len(evidence["occurrences"])
    assert any(c["evidence"]["n_runs"] >= 1 for c in candidates)

    # T-07-03: evidence is attached after the scorecard is written, so it never
    # reaches the LLM refinement request built from the scorecard.
    last_run = supervisor.coordinator.data["last_run"]
    stage_dir = Path(
        hass.config.path(BUFFER_DIR, LEARNING_DIR, str(last_run["run_id"]))
    )
    scorecard_text = (stage_dir / "scorecard_1.json").read_text(encoding="utf-8")
    assert '"evidence"' not in scorecard_text

    target = next(c for c in candidates if c["evidence"]["n_runs"] >= 1)

    review = await async_open_options_step(hass, entry, "review")
    assert review["type"] is FlowResultType.FORM
    cand_menu = await hass.config_entries.options.async_configure(
        review["flow_id"], {"candidate": target["signature_id"]}
    )
    assert cand_menu["type"] is FlowResultType.MENU
    assert cand_menu["step_id"] == "candidate"

    placeholders = cand_menu["description_placeholders"]
    block = placeholders["evidence"]
    lines = block.split("\n")
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "times in Europe/Berlin" in block
    assert "Recent runs, newest first:" in block
    assert any(line.startswith("- ") for line in lines)
    assert len([line for line in lines if line.strip()]) <= 12
    assert lines[1] == ""
    assert placeholders["candidate"] == _escape_markdown(
        supervisor.candidate_label(target)
    )


# ---------------------------------------------------------------------------
# Pure formatter and engine boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [0, 5, 6])
def test_sections_are_blank_line_separated(n_runs: int) -> None:
    """Every section is its own paragraph; only bullets may be adjacent."""
    full = _ready(
        n_runs,
        rhythm=_rhythm() if n_runs >= MIN_RHYTHM_RUNS else None,
        profile=_profile() if n_runs else None,
        hints=["few_runs"] if n_runs < 10 else [],
    )
    # The 30-run block (the 12-line cap) has the same structure.
    big = _ready(
        30, rhythm=_rhythm(), profile=_profile(), hints=["short_coverage"]
    )
    for ev in (_evidence(n_runs), full, big):
        lines = evidence_text(ev, "ready").split("\n")
        for prev, cur in zip(lines, lines[1:]):
            if prev.strip() and cur.strip():
                assert prev.startswith("- ") and cur.startswith("- "), (prev, cur)
        for marker in ("Rhythm:", "Profile:", "Hints (heuristic):"):
            idx = next(i for i, line in enumerate(lines) if line.startswith(marker))
            assert lines[idx - 1] == "", (marker, lines)

    # The "none" block (a device without an active artefact) is separated too.
    none_lines = evidence_text(None, "none").split("\n")
    for prev, cur in zip(none_lines, none_lines[1:]):
        if prev.strip() and cur.strip():
            assert prev.startswith("- ") and cur.startswith("- "), (prev, cur)

    lines = evidence_text(_evidence(n_runs), "ready").split("\n")
    assert lines[0] == "**Evidence** (heuristics, not accuracy)"
    assert lines[1] == ""
    assert lines[3] == ""  # the line after the summary line (index 2)


def test_occurrence_list_boundaries() -> None:
    """5 runs: 5 bullets, no more-line; 6: newest 5 + more; 0: none; capped: true total."""
    five = evidence_text(_evidence(5), "ready").split("\n")
    assert len([ln for ln in five if ln.startswith("- ")]) == 5
    assert not any("…and" in ln for ln in five)

    six = evidence_text(_evidence(6), "ready").split("\n")
    bullets = [ln for ln in six if ln.startswith("- ") and "…and" not in ln]
    assert len(bullets) == 5
    assert "- …and 1 more" in six

    zero = evidence_text(_evidence(0), "ready").split("\n")
    assert "Recent runs: none" in zero
    assert zero[2].startswith("0 runs")

    # The stored occurrence list is capped at MAX_OCCURRENCES (25); the tail must
    # count the true total (n_runs), not len(occurrences) (which would read 20).
    capped = evidence_text(_evidence(30), "ready").split("\n")
    assert len([ln for ln in capped if ln.startswith("- ") and "…and" not in ln]) == 5
    assert "- …and 25 more" in capped


def test_overlay_on_zero_run_evidence_lists_the_captures() -> None:
    """A capture-only group (zero detected runs) lists its captures, not "none"."""
    ev = _evidence(0)
    assert "Recent runs: none" in evidence_text(ev, "ready")

    out = overlay_captured(ev, [_run(_BASE_MS)], 30_000)
    assert out["n_runs"] == 0
    assert out["n_captured_unmatched"] == 1
    block = evidence_text(out, "ready")
    assert "Recent runs: none" not in block
    assert "Recent runs, newest first:" in block
    assert "0 runs, 1 captured" in block
    assert len(_bullets(block)) == 1


def test_energy_and_duration_formatting() -> None:
    """The formatting contract: rounded Wh/kWh, minute durations, n/a for None."""
    assert format_energy_wh(999.4) == "999 Wh"
    assert format_energy_wh(999.6) == "1.00 kWh"
    assert format_energy_wh(None) == "n/a"
    assert format_duration(59) == "<1 min"
    assert format_duration(90) == "2 min"
    assert format_duration(3720) == "1 h 2 min"
    assert format_duration(None) == "n/a"


def test_bullet_missing_step_reads_bare_n_a() -> None:
    """A missing step_w (e.g. NaN rise_w) prints "step n/a", not "step +n/a W"."""
    ev = _evidence(1)
    ev["occurrences"][0]["step_w"] = None
    block = evidence_text(ev, "ready")
    assert "step n/a," in block
    assert "+n/a" not in block


def test_candidate_without_evidence_shows_pending_text() -> None:
    """A candidate with no evidence yet shows the D-12 pending text."""
    assert evidence_text(None, "pending") == EVIDENCE_PENDING_TEXT
    assert evidence_text({}, "pending") == EVIDENCE_PENDING_TEXT
    assert evidence_text(None, "failed") == EVIDENCE_FAILED_TEXT


def test_build_evidence_caps_stored_occurrences() -> None:
    """MAX_OCCURRENCES caps stored occurrences while n_runs stays the total."""
    n = 30
    base_i = 1000
    step_ms = 30_000
    table: dict[str, np.ndarray] = {
        "start_i": np.arange(base_i, base_i + n, dtype=np.int64),
        "end_i": np.arange(base_i, base_i + n, dtype=np.int64) + 10,
        "rise_w": np.full(n, 2150.0),
        "fall_w": np.full(n, -2150.0),
        "dur_s": np.full(n, 660.0),
        "plateau_w": np.full(n, 2150.0),
        "energy_wh": np.full(n, 390.0),
        "kind": np.zeros(n, dtype=np.int32),
        "hour": np.zeros(n, dtype=np.int32),
        "dow": np.zeros(n, dtype=np.int32),
        "l1_share": np.full(n, np.nan),
        "l2_share": np.full(n, np.nan),
        "l3_share": np.full(n, np.nan),
    }
    day_bounds_ms = [_BASE_MS + d * _DAY_MS for d in range(n + 1)]
    claims = {
        "sig-many": set(range(n)),
        "sig-25": set(range(MAX_OCCURRENCES)),
        "sig-none": set(),
    }

    result = build_evidence(
        table,
        claims,
        start_ms=_BASE_MS,
        step_ms=step_ms,
        day_bounds_ms=day_bounds_ms,
        valid_days=list(range(n)),
        now_iso="2026-11-24T10:00:00+00:00",
        source="run",
    )

    many = result["sig-many"]
    assert many["n_runs"] == n
    assert len(many["occurrences"]) == MAX_OCCURRENCES == 25
    starts = [occ["start_ms"] for occ in many["occurrences"]]
    assert starts == sorted(starts, reverse=True)
    assert many["occurrences"][0]["start_ms"] == many["last_ms"]
    assert many["occurrences"][-1]["start_ms"] == starts[-1]
    # first_ms is the oldest run's start even though it is not stored.
    assert many["first_ms"] == _BASE_MS + base_i * step_ms
    assert many["last_ms"] == _BASE_MS + (base_i + n - 1) * step_ms
    assert many["occurrences"][-1]["start_ms"] != many["first_ms"]

    assert result["sig-25"]["n_runs"] == MAX_OCCURRENCES
    assert len(result["sig-25"]["occurrences"]) == MAX_OCCURRENCES

    assert result["sig-none"]["n_runs"] == 0
    assert result["sig-none"]["occurrences"] == []
    assert result["sig-none"]["first_ms"] is None

    json.dumps(result)


# ---------------------------------------------------------------------------
# Full evidence text: rhythm, profile, heuristic hints, states (Plan 07-05)
# ---------------------------------------------------------------------------


def test_rhythm_line_present_and_suppressed() -> None:
    """A build_evidence result with 5+ runs renders the rhythm line; 4 runs the short form."""
    five = _build(5, minute_of_day=400)
    assert five["rhythm"] is not None
    block = evidence_text(five, "ready")
    assert "Rhythm: about " in block
    assert "runs per day (" in block
    assert "per week)" in block
    assert f"(from 5 runs over {five['window']['observed_days']} observed days)" in block

    four = _build(4, minute_of_day=400)
    assert four["rhythm"] is None
    assert "Rhythm: too few runs yet (4 of 5 needed)" in evidence_text(four, "ready")


def test_rhythm_time_window_and_no_fixed_time() -> None:
    """A concentrated time of day renders mostly HH:MM-HH:MM (wrapping); a spread one does not."""
    fixed = _build(6, minute_of_day=400)
    assert fixed["rhythm"]["tod"]["concentration"] >= TOD_MIN_CONCENTRATION
    tod = fixed["rhythm"]["tod"]
    lo = f"{tod['lo_min'] // 60:02d}:{tod['lo_min'] % 60:02d}"
    hi = f"{tod['hi_min'] // 60:02d}:{tod['hi_min'] % 60:02d}"
    assert f"mostly {lo}-{hi}" in evidence_text(fixed, "ready")

    wrapped = _build(6, minutes=[1430, 1430, 1430, 10, 10, 10])
    assert wrapped["rhythm"]["tod"]["lo_min"] > wrapped["rhythm"]["tod"]["hi_min"]
    assert "mostly 23:50-00:10" in evidence_text(wrapped, "ready")

    spread = _build(6, minutes=[180, 900, 180, 900, 180, 900])
    assert spread["rhythm"]["tod"]["concentration"] < TOD_MIN_CONCENTRATION
    assert "no fixed time of day" in evidence_text(spread, "ready")


def test_rhythm_weekday_pattern() -> None:
    """A dominant weekday renders 'most often' with the picked abbreviations; a flat week does not."""
    dominant = _rhythm(dow_counts=[0, 0, 0, 0, 5, 0, 0])
    assert "most often Fri" in evidence_text(_ready(5, rhythm=dominant), "ready")

    flat = _rhythm(dow_counts=[3, 3, 3, 3, 3, 3, 3])
    assert "spread over the week" in evidence_text(_ready(5, rhythm=flat), "ready")


def test_profile_line_never_zero() -> None:
    """A missing value prints n/a (never 0); the line is always present."""
    full = evidence_text(_ready(5, profile=_profile()), "ready")
    assert (
        "Profile: step ~2150 W, ~11 min, ~390 Wh per run, ~780 Wh per day, phase L2"
        in full
    )

    missing = evidence_text(_ready(0, profile=None), "ready")
    assert (
        "Profile: step n/a, n/a, n/a per run, n/a per day, phase n/a" in missing
    )
    assert "step ~0" not in missing

    no_phase = evidence_text(_ready(5, profile=_profile(phase=None)), "ready")
    assert "phase n/a" in no_phase


def test_hints_line_wording() -> None:
    """Each hint code maps to its phrase; an empty list says none triggered."""
    block = evidence_text(
        _ready(4, hints=["few_runs", "wide_step_spread", "short_coverage"]), "ready"
    )
    assert (
        "Hints (heuristic): few runs (4); step size varies widely; "
        "short coverage (20 observed days)" in block
    )

    assert (
        "Hints (heuristic): none triggered"
        in evidence_text(_ready(20, hints=[]), "ready")
    )


def test_no_accuracy_wording_or_percent() -> None:
    """No percentage and no accuracy wording except the fixed header (D-05)."""
    ready = _ready(
        30, rhythm=_rhythm(), profile=_profile(), hints=["short_coverage"]
    )
    for block in (
        evidence_text(ready, "ready"),
        evidence_text(None, "pending"),
        evidence_text(None, "failed"),
        evidence_text(None, "none"),
    ):
        _assert_no_accuracy(block)


def test_block_line_cap() -> None:
    """A 30-run evidence renders exactly 12 non-blank lines with the markup unescaped."""
    block = evidence_text(
        _ready(30, rhythm=_rhythm(), profile=_profile(), hints=["short_coverage"]),
        "ready",
    )
    non_blank = [line for line in block.splitlines() if line.strip()]
    assert len(non_blank) == 12
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert any(line.startswith("- ") for line in block.split("\n"))
    assert "\\*" not in block
    assert "\\-" not in block


def test_evidence_state_transitions() -> None:
    """ready on a current-version dict, failed at the cap, pending below it."""
    assert evidence_state(_evidence(5), 0) == "ready"
    assert evidence_state({"v": 2}, 0) == "pending"
    assert evidence_state(None, 0) == "pending"
    assert evidence_state(None, EVIDENCE_MAX_FAILURES - 1) == "pending"
    assert evidence_state(None, EVIDENCE_MAX_FAILURES) == "failed"
    # A ready value wins over a reached cap.
    assert evidence_state(_evidence(5), EVIDENCE_MAX_FAILURES) == "ready"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "n/a"),
        (600.0, "10 min"),
        (3000.0, "50 min"),
        (3900.0, "1 h"),
        (7200.0, "2 h"),
        (86400.0, "24 h"),
        (172800.0, "2 d 0 h"),
        (180000.0, "2 d 2 h"),
    ],
)
def test_format_gap(seconds: float | None, expected: str) -> None:
    """Gaps render as minutes, hours or days+hours."""
    assert format_gap(seconds) == expected


async def test_review_states(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """No evidence -> pending, the cap -> failed, evidence -> the block (D-12)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    seed_candidates(
        hass_storage, [make_switched_candidate("sig-kettle", "Kettle")]
    )
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning

    def _set_failures(value: int) -> None:
        data = dict(supervisor.coordinator.data or {})
        data["evidence_failures"] = value
        supervisor.coordinator.async_set_updated_data(data)

    async def _evidence_placeholder() -> str:
        review = await async_open_options_step(hass, entry, "review")
        assert review["type"] is FlowResultType.FORM
        menu = await hass.config_entries.options.async_configure(
            review["flow_id"], {"candidate": "sig-kettle"}
        )
        assert menu["type"] is FlowResultType.MENU
        assert menu["step_id"] == "candidate"
        return menu["description_placeholders"]["evidence"]

    _set_failures(0)
    assert await _evidence_placeholder() == EVIDENCE_PENDING_TEXT

    _set_failures(EVIDENCE_MAX_FAILURES)
    assert await _evidence_placeholder() == EVIDENCE_FAILED_TEXT

    cand = next(
        c for c in supervisor.candidates if c["signature_id"] == "sig-kettle"
    )
    cand["evidence"] = _evidence(5)
    _set_failures(0)
    block = await _evidence_placeholder()
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "5 runs" in block


async def test_evidence_times_in_berlin(hass: HomeAssistant) -> None:
    """Occurrence times render in the HA time zone (Europe/Berlin, D-03)."""
    await hass.config.async_set_time_zone("Europe/Berlin")
    start_ms = int(
        datetime(2026, 11, 1, 5, 52, tzinfo=timezone.utc).timestamp() * 1000
    )
    ev = _evidence(1)
    ev["occurrences"] = [
        {
            "start_ms": start_ms,
            "dur_s": 660.0,
            "step_w": 2150.0,
            "energy_wh": 390.0,
            "phase": "l2",
        }
    ]
    block = evidence_text(ev, "ready")
    assert "times in Europe/Berlin" in block
    assert "06:52" in block


# ---------------------------------------------------------------------------
# Device evidence: the same block for confirmed devices and unconfirmed rows
# ---------------------------------------------------------------------------


async def _seed_supervisor_candidates(
    supervisor: Any, candidates: list[dict[str, Any]]
) -> None:
    """Put candidates into the supervisor and its Store (no learning run)."""
    record = {
        "run_id": "run-evidence-review",
        "created": "2026-11-16T12:00:00Z",
        "window_end_ms": 0,
        "validated": True,
        "candidates": candidates,
    }
    await supervisor._candidates_store.async_save(record)
    supervisor._candidates_record = record
    supervisor.candidates = candidates


def _set_failures(supervisor: Any, value: int) -> None:
    """Set the coordinator's evidence failure counter (the D-12a source)."""
    data = dict(supervisor.coordinator.data or {})
    data["evidence_failures"] = value
    supervisor.coordinator.async_set_updated_data(data)


async def _open_candidate_menu(
    hass: HomeAssistant, entry: Any, cand_id: str
) -> dict[str, Any]:
    review = await async_open_options_step(hass, entry, "review")
    assert review["type"] is FlowResultType.FORM
    menu = await hass.config_entries.options.async_configure(
        review["flow_id"], {"candidate": cand_id}
    )
    assert menu["type"] is FlowResultType.MENU
    assert menu["step_id"] == "candidate"
    return menu


async def _open_device_menu(
    hass: HomeAssistant, entry: Any, dev_uuid: str
) -> dict[str, Any]:
    review = await async_open_options_step(hass, entry, "review")
    assert review["type"] is FlowResultType.FORM
    menu = await hass.config_entries.options.async_configure(
        review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )
    assert menu["type"] is FlowResultType.MENU
    assert menu["step_id"] == "device"
    return menu


async def _confirmed_uuid(entry: Any, title: str) -> str:
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == title)
    return sub.data["device_uuid"]


@pytest.mark.evidence_enabled
async def test_device_menu_shows_replayed_evidence(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """A confirmed device shows the stage's replay of its active artefact (D-07a)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor = entry.runtime_data.learning
    await _seed_supervisor_candidates(
        supervisor, [make_switched_candidate("sig-kettle", "Kettle")]
    )
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = await _confirmed_uuid(entry, "Kettle")

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task

    evidence = supervisor.device_evidence(dev_uuid)
    assert isinstance(evidence, dict)
    assert evidence["source"] == "refresh"

    menu = await _open_device_menu(hass, entry, dev_uuid)
    block = menu["description_placeholders"]["evidence"]
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    summary = block.split("\n")[2]
    assert summary.startswith(str(evidence["n_runs"]))
    assert "days observed" in block
    lines = block.split("\n")
    assert any(line.startswith("Rhythm:") for line in lines)
    assert any(line.startswith("Profile:") for line in lines)
    assert any(line.startswith("Hints (heuristic):") for line in lines)
    assert EVIDENCE_PENDING_TEXT not in block
    assert "Runs: n/a" not in block


async def test_device_menu_right_after_confirm_uses_candidate_evidence(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A just-confirmed device reads the confirmed-from candidate's evidence (D-07a)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    cand = make_switched_candidate("sig-kettle", "Kettle")
    cand["evidence"] = _evidence(7)
    seed_candidates(hass_storage, [cand])
    entry = await async_setup_via_flow(hass)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = await _confirmed_uuid(entry, "Kettle")

    menu = await _open_device_menu(hass, entry, dev_uuid)
    block = menu["description_placeholders"]["evidence"]
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "7 runs" in block
    assert EVIDENCE_PENDING_TEXT not in block


async def test_device_pending_and_failed_states(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A device without evidence shows pending, then failed at the cap (D-12)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    seed_candidates(
        hass_storage, [make_switched_candidate("sig-kettle", "Kettle")]
    )
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = await _confirmed_uuid(entry, "Kettle")

    _set_failures(supervisor, 0)
    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert menu["description_placeholders"]["evidence"] == EVIDENCE_PENDING_TEXT

    _set_failures(supervisor, EVIDENCE_MAX_FAILURES)
    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert menu["description_placeholders"]["evidence"] == EVIDENCE_FAILED_TEXT


async def test_device_without_artefact_prints_na(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device with no active artefact (nothing to replay) shows every value n/a."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    seed_candidates(
        hass_storage, [make_switched_candidate("sig-kettle", "Kettle")]
    )
    entry = await async_setup_via_flow(hass)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = await _confirmed_uuid(entry, "Kettle")
    monkeypatch.setattr(
        entry.runtime_data.devices,
        "active_artefact_and_version",
        lambda uuid: (None, 0),
    )

    menu = await _open_device_menu(hass, entry, dev_uuid)
    block = menu["description_placeholders"]["evidence"]
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "Runs: n/a" in block
    assert "Recent runs: n/a" in block
    assert "Rhythm: n/a" in block
    assert "Profile: step n/a, n/a, n/a per run, n/a per day, phase n/a" in block
    assert "Hints (heuristic): n/a" in block
    assert "0 runs" not in block
    assert EVIDENCE_PENDING_TEXT not in block
    assert EVIDENCE_FAILED_TEXT not in block
    _assert_no_accuracy(block)


async def test_unconfirmed_device_row_uses_device_evidence(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A deleted device's review row reads its device evidence, never the pending text."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    seed_candidates(
        hass_storage, [make_switched_candidate("sig-kettle", "Kettle")]
    )
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = await _confirmed_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    rec = devices.record(dev_uuid)
    assert rec is not None

    # Delete means unconfirm (D-05): the record stays, the row returns.
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    )
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()
    assert devices.record(dev_uuid)["status"] == "unconfirmed"

    supervisor._device_evidence[dev_uuid] = {
        **_evidence(9),
        "active_v": int(rec.get("active_v") or 0),
    }
    # A later learning run no longer holding the signature: the row comes from
    # devices.unconfirmed_candidates(), not from the supervisor's candidates.
    supervisor.candidates = []
    assert "sig-kettle" not in {
        c.get("signature_id") for c in supervisor.candidates
    }

    menu = await _open_candidate_menu(hass, entry, "sig-kettle")
    block = menu["description_placeholders"]["evidence"]
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "9 runs" in block
    assert EVIDENCE_PENDING_TEXT not in block


def test_unconfirmed_row_uuid_tie_break() -> None:
    """An unconfirmed row resolves to the record it was built from, then the lowest uuid."""
    a_artefact = {"states": [{"name": "on"}]}
    b_artefact = {"states": [{"name": "b"}]}
    sources = [
        {
            "uuid": "a-uuid",
            "signature_id": "sig-x",
            "status": "unconfirmed",
            "active_v": 1,
            "artefact": a_artefact,
        },
        {
            "uuid": "b-uuid",
            "signature_id": "sig-x",
            "status": "unconfirmed",
            "active_v": 1,
            "artefact": b_artefact,
        },
    ]
    evidences = {
        "a-uuid": {**_evidence(3), "active_v": 1},
        "b-uuid": {**_evidence(9), "active_v": 1},
    }

    def _active(uuid: str) -> tuple[Any, int]:
        for source in sources:
            if source["uuid"] == uuid:
                return dict(source["artefact"]), int(source["active_v"])
        return None, 0

    hub = SimpleNamespace(
        learning=SimpleNamespace(
            candidates=[],
            coordinator=SimpleNamespace(data={"evidence_failures": 0}),
            device_evidence=lambda uuid: evidences.get(str(uuid)),
        ),
        devices=SimpleNamespace(
            evidence_sources=lambda: list(sources),
            active_artefact_and_version=_active,
        ),
    )

    row_b = {"signature_id": "sig-x", "states": b_artefact["states"]}
    evidence, state = review_actions.evidence_view(hub, cand=row_b)
    assert state == "ready"
    assert evidence["n_runs"] == 9

    # Both records carry identical states: the lowest uuid (a-uuid) wins.
    sources[1]["artefact"] = {"states": [{"name": "on"}]}
    row_tie = {"signature_id": "sig-x", "states": [{"name": "on"}]}
    evidence, state = review_actions.evidence_view(hub, cand=row_tie)
    assert state == "ready"
    assert evidence["n_runs"] == 3


def test_strings_identical_and_placeholders() -> None:
    """Both string files are byte-identical and both menus carry both placeholders."""
    root = (
        Path(__file__).resolve().parents[2]
        / "custom_components"
        / "powerguesser"
    )
    strings_bytes = (root / "strings.json").read_bytes()
    en_bytes = (root / "translations" / "en.json").read_bytes()
    assert strings_bytes == en_bytes

    step = json.loads(strings_bytes)["options"]["step"]
    assert step["candidate"]["description"] == "{candidate}\n\n{evidence}"
    assert step["device"]["description"] == "{candidate}\n\n{evidence}"


# ---------------------------------------------------------------------------
# EVID-05: evidence marks user-captured occurrences (Plan 08-07)
# ---------------------------------------------------------------------------
_GRID_S = 30
_DAY = 86_400_000


def _pin_grid(entry: Any) -> None:
    enabled, disabled = gate(float(_GRID_S), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(_GRID_S),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


async def _setup_capture(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    spec: SynthSpec | None = None,
) -> tuple[Any, Any]:
    """Set up a loaded entry with a synthetic buffer and pinned 30 s grid."""
    if candidates:
        seed_candidates(hass_storage, candidates)
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, spec or SynthSpec(days=3))
    return entry, data


async def _capture_window(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "run",
    target_type: str = "new_device",
    target: str = "Kettle",
) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso_utc(start_ms),
            "end": _iso_utc(end_ms),
            "label": label,
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
    )


def _run(
    rise_ms: int,
    *,
    capture_id: str = "c0",
    verdict: str = "clean",
    step_w: float = 2000.0,
    dur_s: float = 180.0,
    energy_wh: float = 100.0,
    grid_s: float = 30.0,
) -> dict[str, Any]:
    """One captured_runs entry shape."""
    return {
        "capture_id": capture_id,
        "rise_ms": rise_ms,
        "fall_ms": rise_ms + int(dur_s * 1000),
        "step_w": step_w,
        "dur_s": dur_s,
        "energy_wh": energy_wh,
        "grid_s": grid_s,
        "verdict": verdict,
    }


def _occ(start_ms: int) -> dict[str, Any]:
    return {
        "start_ms": start_ms,
        "dur_s": 660.0,
        "step_w": 2150.0,
        "energy_wh": 390.0,
        "phase": None,
    }


def _occ_evidence(starts: list[int]) -> dict[str, Any]:
    """A ready evidence dict whose occurrences are the given starts (unsorted)."""
    ev = _evidence(len(starts))
    ev["occurrences"] = [_occ(s) for s in starts]
    ev["n_runs"] = len(starts)
    return ev


def _bullets(block: str) -> list[str]:
    return [ln for ln in block.split("\n") if ln.startswith("- ") and "…and" not in ln]


# --- overlay_captured unit tests (Edge EVID-05) -----------------------------


def test_overlay_adjacency_boundary() -> None:
    """One grid step (inclusive) marks; one ms more appends; dedupe prefers the clean capture."""
    s = 1_700_000_000_000
    step = 30_000
    ev = _occ_evidence([s])

    marked = overlay_captured(ev, [_run(s + step)], step)
    assert marked["occurrences"][0]["captured"] is True
    assert marked["n_captured"] == 1
    assert marked["n_captured_unmatched"] == 0

    unmatched = overlay_captured(ev, [_run(s + step + 1)], step)
    assert unmatched["n_captured"] == 1
    assert unmatched["n_captured_unmatched"] == 1
    appended = [o for o in unmatched["occurrences"] if o.get("detected") is False]
    assert len(appended) == 1
    assert appended[0]["start_ms"] == s + step + 1
    detected = [o for o in unmatched["occurrences"] if o.get("detected") is not False]
    assert detected[0].get("captured") is not True

    # Two captures of the same run mark one occurrence and count once.
    dup = overlay_captured(
        ev, [_run(s + step, capture_id="a"), _run(s + step, capture_id="b")], step
    )
    assert dup["n_captured"] == 1
    assert dup["n_captured_unmatched"] == 0

    # Two unmatched captures 10 s apart are appended once.
    two = overlay_captured(
        ev,
        [_run(s + step + 1_000, capture_id="a"), _run(s + step + 11_000, capture_id="b")],
        step,
    )
    assert two["n_captured_unmatched"] == 1

    # The single appended entry carries the clean capture's values whichever order.
    contaminated = _run(
        s + step + 1_000, capture_id="a", verdict="contaminated",
        step_w=9999.0, dur_s=11.0, energy_wh=22.0,
    )
    clean = _run(
        s + step + 1_000, capture_id="b", verdict="clean",
        step_w=2000.0, dur_s=180.0, energy_wh=100.0,
    )
    for runs in ([contaminated, clean], [clean, contaminated]):
        out = overlay_captured(ev, runs, step)
        assert out["n_captured_unmatched"] == 1
        app = [o for o in out["occurrences"] if o.get("detected") is False]
        assert len(app) == 1
        assert app[0]["step_w"] == 2000.0
        assert app[0]["dur_s"] == 180.0
        assert app[0]["energy_wh"] == 100.0


def test_overlay_marks_across_grid_phase() -> None:
    """A rise offset by half a step or one step minus 1 ms still marks the occurrence."""
    s = 1_700_000_000_000
    step = 30_000
    ev = _occ_evidence([s])
    for offset in (15_000, step - 1):
        out = overlay_captured(ev, [_run(s + offset)], step)
        assert out["n_captured"] == 1
        assert out["n_captured_unmatched"] == 0
        assert out["occurrences"][0]["captured"] is True


def test_overlay_without_captures_is_identical() -> None:
    """No captures returns the evidence unchanged and the same rendered text."""
    ev = _ready(6)
    assert overlay_captured(ev, [], 30_000) is ev
    assert evidence_text(overlay_captured(ev, [], 30_000), "ready") == evidence_text(
        ev, "ready"
    )
    # A none-verdict capture has no pair and never reaches the overlay.
    assert "captured" not in evidence_text(overlay_captured(ev, [], 30_000), "ready")


def test_overlay_ordering_and_more_count() -> None:
    """Detected and appended runs merge newest first; the tail counts every occurrence."""
    base = 1_700_000_000_000
    starts = [base - i * _DAY for i in range(7)]
    ev = _occ_evidence(starts)
    step = 30_000
    newer = starts[0] + 3_600_000
    between = starts[3] - 3_600_000  # between the 4th and 5th newest (index 5)
    out = overlay_captured(
        ev, [_run(newer, capture_id="A"), _run(between, capture_id="B")], step
    )
    ordered = [o["start_ms"] for o in out["occurrences"]]
    assert ordered == sorted(ordered, reverse=True)
    assert ordered[0] == newer
    assert ordered.index(between) == 5
    assert len(out["occurrences"]) == 9
    assert out["n_captured_unmatched"] == 2

    block = evidence_text(out, "ready")
    assert "- …and 4 more" in block
    assert _bullets(block)[0].startswith("- ")


# --- HA end-to-end evidence tests ------------------------------------------


async def test_preview_card_evidence_is_ready(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A 2-capture preview's menu shows the capture evidence block, both runs marked."""
    entry, data = await _setup_capture(hass, energy_manager, std_prefs, hass_storage)
    on_ms, off_ms = data.kettle_runs[-1]
    first = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    await _capture_window(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="Kettle",
    )

    menu = await _open_candidate_menu(hass, entry, sig)
    block = menu["description_placeholders"]["evidence"]
    assert block.startswith("**Evidence** (heuristics, not accuracy)")
    assert "2 runs, 2 captured" in block
    bullets = _bullets(block)
    assert len(bullets) == 2
    assert all(ln.endswith(" (captured)") for ln in bullets)
    assert EVIDENCE_PENDING_TEXT not in block


async def test_learned_candidate_marks_captured_occurrence(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """One attached capture marks exactly its occurrence; the stored evidence is untouched."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup_capture(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]
    await _capture_window(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="sig-kettle",
    )
    rise = hub.captures.captured_runs("candidate", "sig-kettle")[0]["rise_ms"]

    live = next(c for c in hub.learning.candidates if c["signature_id"] == "sig-kettle")
    starts = [rise + 4 * _DAY, rise + 3 * _DAY, rise + 2 * _DAY, rise + 20_000,
              rise - 1 * _DAY, rise - 2 * _DAY]
    live["evidence"] = _occ_evidence(starts)
    stored_before = copy.deepcopy(live["evidence"])

    menu = await _open_candidate_menu(hass, entry, "sig-kettle")
    block = menu["description_placeholders"]["evidence"]
    assert "6 runs, 1 captured" in block
    bullets = _bullets(block)
    assert bullets[3].endswith(" (captured)")
    for i in (0, 1, 2, 4):
        assert not bullets[i].endswith(" (captured)")

    # The overlay works on a copy: the stored evidence dict is unchanged.
    assert live["evidence"] == stored_before
    assert all("captured" not in occ for occ in live["evidence"]["occurrences"])


async def test_unmatched_capture_is_appended(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A captured rise matching no occurrence is appended as a not-matched bullet."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup_capture(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]
    await _capture_window(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="sig-kettle",
    )
    rise = hub.captures.captured_runs("candidate", "sig-kettle")[0]["rise_ms"]
    live = next(c for c in hub.learning.candidates if c["signature_id"] == "sig-kettle")
    live["evidence"] = _occ_evidence([rise - 100 * _DAY, rise - 101 * _DAY])

    menu = await _open_candidate_menu(hass, entry, "sig-kettle")
    bullets = _bullets(menu["description_placeholders"]["evidence"])
    appended = [ln for ln in bullets if ln.endswith(" (captured, not matched by the detector)")]
    assert len(appended) == 1
    assert bullets[0] == appended[0]


async def test_device_evidence_marks_captures(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """After confirming a preview, the device evidence marks the captures that resolve to it."""
    entry, data = await _setup_capture(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    runs = data.kettle_runs[-3:]
    first = await _capture_window(
        hass, start_ms=runs[0][0] - 45_000, end_ms=runs[0][1] + 50_000
    )
    sig = first["target"]["id"]
    for on_ms, off_ms in runs[1:]:
        await _capture_window(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="Kettle",
        )

    await async_confirm_candidate(hass, entry, sig, "Kettle")
    uuid = await _confirmed_uuid(entry, "Kettle")

    device_runs = hub.captures.captured_runs("device", uuid)
    assert len(device_runs) == 3
    hub.learning._device_evidence[uuid] = {
        **_occ_evidence([r["rise_ms"] for r in device_runs]),
        "active_v": int(hub.devices.record(uuid)["active_v"]),
    }

    menu = await _open_device_menu(hass, entry, uuid)
    block = menu["description_placeholders"]["evidence"]
    assert "3 runs, 3 captured" in block
    bullets = _bullets(block)
    assert len(bullets) == 3
    assert all(ln.endswith(" (captured)") for ln in bullets)


async def test_evidence_with_captures_has_only_fixed_words(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A capture label never reaches the evidence block (only numbers, dates, fixed words)."""
    entry, data = await _setup_capture(hass, energy_manager, std_prefs, hass_storage)
    on_ms, off_ms = data.kettle_runs[-1]
    label = "*bold* [x](y)"
    first = await _capture_window(
        hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, label=label
    )
    sig = first["target"]["id"]

    menu = await _open_candidate_menu(hass, entry, sig)
    block = menu["description_placeholders"]["evidence"]
    assert "*bold*" not in block
    assert "[x](y)" not in block
    assert "\\*" not in block


