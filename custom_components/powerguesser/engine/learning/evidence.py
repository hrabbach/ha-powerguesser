"""HA-free per-candidate evidence over the episode table and replay claims.

This module is the engine half of Phase 7's candidate evidence (D-09, D-11):
it turns the objects ``compile_score`` and ``rescore`` already hold - the
episode table and the per-signature claim sets returned by ``score.replay()``
- into the dict a candidate carries under ``"evidence"`` and the Review
renders as markdown.

It imports numpy and the standard library only (never ``homeassistant``),
reads no wall-clock time (the caller passes ``now_iso``) and returns plain
JSON-safe Python values.

``rhythm`` is a circular time-of-day summary (suppressed below
``MIN_RHYTHM_RUNS`` runs), ``profile`` the step/duration/energy percentiles and
the phase, and ``hints`` a list of evidence-quality *codes* (never wording).
Every unavailable value is ``None``, never 0 (D-04); hint wording lives in the
HA-side formatter (Plan 07-05). Evidence counts only the signature's own
claimed rows: merged aliases are not pooled (D-20).
"""

from collections.abc import Mapping, Sequence
import math

import numpy as np

# A bump makes every stored evidence dict stale (the evidence stage recomputes).
EVIDENCE_VERSION = 1

# Stored occurrences are capped while n_runs keeps the true total (EVID-01).
MAX_OCCURRENCES = 25

# EVID-02 / D-02: the rhythm line is suppressed below this many runs; at the
# boundary 4 runs give None and 5 give a rhythm dict.
MIN_RHYTHM_RUNS = 5

# EVID-02 / PITFALLS 17: below this circular concentration the time of day has
# no fixed centre, so the renderer says "no fixed time of day".
TOD_MIN_CONCENTRATION = 0.3

# EVID-07 / D-05: hint thresholds (Claude's discretion per CONTEXT, values from
# RESEARCH). Each is tested one step either side in test_evidence.py.
HINT_FEW_RUNS = 10
HINT_WIDE_SPREAD = 0.25
HINT_SHORT_COVERAGE_DAYS = 14


def _row_phase(table: Mapping[str, np.ndarray], idx: int) -> str | None:
    """Return the phase channel carrying the largest finite share of a row.

    The phase shares are NaN when phase channels are not mapped, so an
    unmapped row yields None (never a fabricated "l1").
    """
    best_share: float | None = None
    best_name: str | None = None
    for key, name in (("l1_share", "l1"), ("l2_share", "l2"), ("l3_share", "l3")):
        column = table.get(key)
        if column is None:
            continue
        value = float(column[idx])
        if not math.isfinite(value):
            continue
        if best_share is None or value > best_share:
            best_share = value
            best_name = name
    return best_name


def _finite_or_none(value: float) -> float | None:
    """Return the float, or None for NaN/inf (an unknown value, never 0)."""
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    """p10/p50/p90 of the given values, or None when no value is available."""
    if len(values) == 0:
        return None
    p10, p50, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _local_day_index(start_ms: int, bounds: Sequence[int]) -> int:
    """The local day containing ``start_ms``, clamped into the bounds range."""
    if len(bounds) < 2:
        return 0
    idx = int(np.searchsorted(np.asarray(bounds, dtype=np.int64), start_ms, side="right")) - 1
    return max(0, min(len(bounds) - 2, idx))


def _minute_of_day(start_ms: int, bounds: Sequence[int], day_index: int) -> float:
    """Minutes since local midnight, measured against the local-midnight day bounds."""
    if not bounds:
        return 0.0
    return (int(start_ms) - int(bounds[day_index])) / 60_000.0


def _build_rhythm(
    starts_ms: list[int],
    minutes: np.ndarray,
    days: np.ndarray,
    dow: np.ndarray,
    valid_days: Sequence[int],
) -> dict | None:
    """Rhythm summary, or None below MIN_RHYTHM_RUNS (EVID-02, D-02)."""
    n_runs = len(starts_ms)
    if n_runs < MIN_RHYTHM_RUNS:
        return None

    observed_days = int(len(valid_days))
    runs_per_day = float(n_runs) / observed_days if observed_days > 0 else None
    runs_per_week = 7.0 * runs_per_day if runs_per_day is not None else None

    valid_set = {int(d) for d in valid_days}
    active_days = len({int(d) for d in days if int(d) in valid_set})

    angle = 2.0 * math.pi * minutes / 1440.0
    cos_mean = float(np.mean(np.cos(angle)))
    sin_mean = float(np.mean(np.sin(angle)))
    concentration = float(math.hypot(cos_mean, sin_mean))
    center = (math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0) * 4.0
    offset = ((minutes - center + 720.0) % 1440.0) - 720.0
    lo, hi = np.percentile(offset, [10.0, 90.0])

    sorted_starts = sorted(starts_ms)
    gaps_s = np.array(
        [(b - a) / 1000.0 for a, b in zip(sorted_starts, sorted_starts[1:])],
        dtype=np.float64,
    )

    return {
        "runs_per_day": runs_per_day,
        "runs_per_week": runs_per_week,
        "active_days": active_days,
        "tod": {
            "center_min": int(round(center)) % 1440,
            "lo_min": int(round(center + float(lo))) % 1440,
            "hi_min": int(round(center + float(hi))) % 1440,
            "concentration": concentration,
        },
        "dow_counts": [int(np.sum(dow == slot)) for slot in range(7)],
        "gap_median_s": float(np.median(gaps_s)) if len(gaps_s) else None,
        "gap_p90_s": float(np.percentile(gaps_s, 90.0)) if len(gaps_s) else None,
    }


def _build_hints(n_runs: int, step_w: dict[str, float] | None, observed_days: int) -> list[str]:
    """Evidence-quality hint codes only, never wording (EVID-07, D-05)."""
    hints: list[str] = []
    if n_runs < HINT_FEW_RUNS:
        hints.append("few_runs")
    if n_runs >= 2 and step_w is not None and step_w["p50"] > 0:
        if (step_w["p90"] - step_w["p10"]) / step_w["p50"] > HINT_WIDE_SPREAD:
            hints.append("wide_step_spread")
    if observed_days < HINT_SHORT_COVERAGE_DAYS:
        hints.append("short_coverage")
    return hints


def build_evidence(
    table: Mapping[str, np.ndarray],
    claims: Mapping[str, set[int]],
    *,
    start_ms: int,
    step_ms: int,
    day_bounds_ms: Sequence[int] | np.ndarray,
    valid_days: Sequence[int],
    now_iso: str | None,
    source: str,
) -> dict[str, dict]:
    """Build one evidence dict per claimed signature id.

    ``table`` is the ``compute_features`` episode table and ``claims`` maps a
    signature id to the row indices ``score.replay()`` claimed for it. Every
    key of ``claims`` gets an entry, including an empty claim set (n_runs 0,
    no occurrences). ``valid_days`` are the replayed residual's valid days
    (``split.data_days``); ``day_bounds_ms`` are the local midnights of the
    replayed residual, so ``window`` describes the same window whichever
    source (run, stage or rescore) last wrote the evidence.
    """
    bounds = [int(b) for b in day_bounds_ms]
    n_days = max(0, len(bounds) - 1)
    window = {
        "start_ms": bounds[0] if bounds else None,
        "end_ms": bounds[-1] if bounds else None,
        "observed_days": int(len(valid_days)),
        "n_days": int(n_days),
    }
    observed_days = int(len(valid_days))

    result: dict[str, dict] = {}
    for sig_id, raw_rows in claims.items():
        rows = [int(r) for r in raw_rows]
        occurrences: list[dict] = []
        first_ms: int | None = None
        last_ms: int | None = None
        starts_ms: list[int] = []
        minutes: list[float] = []
        day_indices: list[int] = []
        step_values: list[float] = []
        dur_values: list[float] = []
        energy_values: list[float] = []

        for idx in rows:
            run_start_ms = int(start_ms) + int(table["start_i"][idx]) * int(step_ms)
            kind = int(table["kind"][idx])
            step_w = (
                _finite_or_none(table["plateau_w"][idx])
                if kind == 1
                else _finite_or_none(table["rise_w"][idx])
            )
            dur_value = _finite_or_none(table["dur_s"][idx])
            energy_value = _finite_or_none(table["energy_wh"][idx])
            occurrences.append(
                {
                    "start_ms": run_start_ms,
                    "dur_s": dur_value,
                    "step_w": step_w,
                    "energy_wh": energy_value,
                    "phase": _row_phase(table, idx),
                }
            )
            if first_ms is None or run_start_ms < first_ms:
                first_ms = run_start_ms
            if last_ms is None or run_start_ms > last_ms:
                last_ms = run_start_ms
            starts_ms.append(run_start_ms)
            day_index = _local_day_index(run_start_ms, bounds)
            day_indices.append(day_index)
            minutes.append(_minute_of_day(run_start_ms, bounds, day_index))
            if step_w is not None:
                step_values.append(step_w)
            if dur_value is not None:
                dur_values.append(dur_value)
            if energy_value is not None:
                energy_values.append(energy_value)

        occurrences.sort(key=lambda occ: occ["start_ms"], reverse=True)

        # EVID-03 / D-04: unavailable values are None, never 0. step_w uses
        # rise_w for switched rows and plateau_w for session rows.
        step_w_profile = _percentiles(np.array(step_values, dtype=np.float64))
        dur_profile = _percentiles(np.array(dur_values, dtype=np.float64))
        wh_per_run = float(np.median(energy_values)) if energy_values else None
        wh_per_day = (
            float(np.sum(energy_values)) / observed_days
            if (energy_values and observed_days > 0)
            else None
        )

        # Phase only when the phase channels exist (all-NaN shares -> None).
        phase_medians: dict[str, float] = {}
        if rows:
            for key, name in (("l1_share", "l1"), ("l2_share", "l2"), ("l3_share", "l3")):
                column = table.get(key)
                if column is None:
                    continue
                shares = np.array([float(column[i]) for i in rows], dtype=np.float64)
                shares = shares[np.isfinite(shares)]
                if len(shares):
                    phase_medians[name] = float(np.median(shares))
        phase_available = bool(phase_medians)
        phase = max(phase_medians, key=phase_medians.__getitem__) if phase_available else None

        dow_column = table.get("dow")
        dow = (
            np.array([int(dow_column[i]) for i in rows], dtype=np.int64)
            if (rows and dow_column is not None)
            else np.empty(0, dtype=np.int64)
        )
        rhythm = _build_rhythm(
            starts_ms,
            np.array(minutes, dtype=np.float64),
            np.array(day_indices, dtype=np.int64),
            dow,
            valid_days,
        )

        result[str(sig_id)] = {
            "v": EVIDENCE_VERSION,
            "computed_at": now_iso,
            "source": source,
            "window": dict(window),
            "n_runs": len(rows),
            "first_ms": first_ms,
            "last_ms": last_ms,
            "occurrences": occurrences[:MAX_OCCURRENCES],
            "rhythm": rhythm,
            "profile": {
                "step_w": step_w_profile,
                "dur_s": dur_profile,
                "wh_per_run": wh_per_run,
                "wh_per_day": wh_per_day,
                "phase": phase,
                "phase_available": phase_available,
            },
            "hints": _build_hints(len(rows), step_w_profile, observed_days),
        }

    return result
