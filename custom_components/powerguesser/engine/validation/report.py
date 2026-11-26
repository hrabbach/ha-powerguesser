"""assemble_report: the run_validation report builder (Plan 05-02; D-03, D-06..D-12).

Rolling-window interaction (review concern, dual calendar windows): the
validate stage derives the holdout as a calendar window over the CURRENT
90-day buffer window, so it can include days newer than the shadow set's own
holdout. Only days starting at or after the shadow's train_cutoff_ms count,
so every counted day is unseen by the shadow artefacts, and train_cutoff is
echoed in the report.

assemble_report is pure and the ONLY writer of the report's metrics field
(cross-AI review Round 1): validate collects figures, this module gates,
judges and assembles. A refusal is report content (status refused + reason),
never an exception, and carries no figure (D-12, Pattern 5, T-05-04).

Plan 06-09 (REV-04, D-13): assemble_sessions is the SECOND writer surface.
It judges every device-target plug-session result with the identical D-07
device bar (``_device_verdict``) and forms the report's separate ``sessions``
list, which never enters devices, classes or class peers (D-32). A refused
session result keeps its reason and carries no figure.

Pure stdlib; HA-free (tests/engine/test_no_ha_imports.py).
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
import statistics
from typing import Any

from ..learning.split import MIN_DAY_COVERAGE
from .classes import class_passes, derive_class
from .constants import (
    CLASS_MAX_MEDIAN_PCT,
    CLASS_MAX_PEER_PCT,
    CLASS_MIN_PASSING,
    MAX_DEVICE_ERROR_PCT,
    MIN_COVERED_HOLDOUT_DAYS,
    MIN_EVENTS_FIGURE,
    MIN_EVENTS_PASS,
    MIN_USABLE_DAYS,
    SHADOW_HOLDOUT_DAYS,
)

_BASELINE_DEFINITION = (
    "training-mean constant power of the plug over the shadow train block"
)

# DET-02/D-24: the artefact kinds assemble_report measures. A switched and a
# program link are judged by the identical D-07/D-08 bars; only a link of
# another kind is refused. Mirrors worker.MEASURED_KINDS (which lives in the
# learning package so this module keeps importing nothing heavy).
_MEASURED_KINDS: tuple[str, ...] = ("switched", "program")

_MEASURES_DEFINITION = (
    "the masked shadow model trained up to train_cutoff and replayed on the "
    "holdout block"
)

_METRICS_DEFINITIONS = {
    "nde": (
        "NDE = sum((y - y_hat)^2) / sum(y^2) over the holdout samples where "
        "both the residual and the plug reading are finite"
    ),
    "mae_w": "MAE = mean |y - y_hat| in watts over the same samples",
    "unvalidated": (
        "false only when the plug device passes the device bar (energy error "
        "<= 20 %, beats the naive baseline on NDE and energy, >= 8 plug ON "
        "sessions across all 7 holdout days) or its class passes the class bar "
        "(>= 2 passing peers, median peer energy error <= 25 %, no peer above "
        "35 %); true otherwise"
    ),
    "link_overlap": (
        "the link's train-block ON overlap score, so the match behind every "
        "figure or refusal can be audited"
    ),
    "measures": _MEASURES_DEFINITION,
}


def _thresholds() -> dict[str, float | int]:
    """The current pass bars and data gates (initial values, CONTEXT Ideas)."""
    return {
        "min_usable_days": MIN_USABLE_DAYS,
        "min_covered_holdout_days": MIN_COVERED_HOLDOUT_DAYS,
        "min_events_figure": MIN_EVENTS_FIGURE,
        "min_events_pass": MIN_EVENTS_PASS,
        "max_device_error_pct": MAX_DEVICE_ERROR_PCT,
        "class_min_passing": CLASS_MIN_PASSING,
        "class_max_median_pct": CLASS_MAX_MEDIAN_PCT,
        "class_max_peer_pct": CLASS_MAX_PEER_PCT,
    }


def _json_safe(value: Any) -> Any:
    """Replace NaN/inf with None so the report serializes under allow_nan=False.

    canonical_json does not enforce this (digest.py), so the structural
    replacement lives here, the single report writer (T-05-04).
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _base_report(
    *,
    shadow: Mapping,
    holdout_days_info: Mapping,
    usable_days: int,
    shadow_created: str | None,
    shadow_age_days: float | None,
    shadow_stale: bool | None,
    excluded_capture: Mapping | None = None,
) -> dict[str, Any]:
    """The fields every report carries, ok or refused alike (D-27).

    ``excluded_capture`` (D-05, Plan 08-04) is written into the report's
    ``holdout`` dict as ``{"windows", "samples"}`` when capture windows were
    passed; without it the report is byte-identical to before.
    """
    holdout: dict[str, Any] = {
        "start_ms": int(holdout_days_info.get("start_ms", 0) or 0),
        "end_ms": int(holdout_days_info.get("end_ms", 0) or 0),
        "days": [
            dict(d)
            for d in holdout_days_info.get("days", [])
            if isinstance(d, Mapping)
        ],
    }
    if excluded_capture is not None:
        holdout["excluded_capture"] = {
            "windows": int(excluded_capture.get("windows", 0) or 0),
            "samples": int(excluded_capture.get("samples", 0) or 0),
        }
    return {
        "train_cutoff_ms": int(shadow.get("train_cutoff_ms", 0) or 0),
        "usable_days": int(usable_days),
        "holdout": holdout,
        "baseline": _BASELINE_DEFINITION,
        "metrics": dict(_METRICS_DEFINITIONS),
        "shadow_created": shadow_created,
        "shadow_age_days": shadow_age_days,
        "shadow_stale": shadow_stale,
    }


def _refusal_reason(
    *,
    shadow: Mapping,
    holdout_days_info: Mapping,
    usable_days: int,
    grid_s: int,
    enabled_event_types: Sequence[str],
) -> str | None:
    """The D-12 whole-report gates, in order; None when the report can proceed."""
    if int(shadow.get("grid_s", grid_s) or grid_s) != int(grid_s):
        return "grid_mismatch"
    shadow_types = sorted(str(t) for t in (shadow.get("enabled_event_types") or []))
    if shadow_types != sorted(str(t) for t in enabled_event_types):
        return "event_types_mismatch"
    if int(usable_days) < MIN_USABLE_DAYS:
        return "insufficient_data"
    covered = sum(
        1
        for d in holdout_days_info.get("days", [])
        if isinstance(d, Mapping) and float(d.get("coverage", 0.0) or 0.0) >= MIN_DAY_COVERAGE
    )
    if covered < MIN_COVERED_HOLDOUT_DAYS:
        return "insufficient_data"
    return None


def _beats(device: float | None, baseline: float | None) -> bool:
    """The device figure is strictly better than the baseline's (lower is better)."""
    return device is not None and baseline is not None and device < baseline


def _iso_ms(ms: Any) -> str | None:
    """ISO 8601 UTC string for a millisecond epoch, or None when not a number."""
    if isinstance(ms, bool) or not isinstance(ms, (int, float)):
        return None
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()


def _device_verdict(entry: Mapping) -> tuple[bool, list[str], dict[str, bool]]:
    """The D-07 device bar: all four conditions must hold (D-21 A3)."""
    error = entry.get("energy_error_pct")
    baseline = entry.get("baseline") if isinstance(entry.get("baseline"), Mapping) else {}
    base_error = baseline.get("energy_error_pct")
    base_nde = baseline.get("nde")
    device_nde = entry.get("nde")
    n_events = int(entry.get("n_events", 0) or 0)
    usable_holdout_days = int(entry.get("usable_holdout_days", 0) or 0)

    beats = {
        "nde": _beats(
            float(device_nde) if device_nde is not None else None,
            float(base_nde) if base_nde is not None else None,
        ),
        "energy": _beats(
            float(error) if error is not None else None,
            float(base_error) if base_error is not None else None,
        ),
    }
    reasons: list[str] = []
    if n_events < MIN_EVENTS_PASS:
        reasons.append("events_below_pass_bar")
    if usable_holdout_days < SHADOW_HOLDOUT_DAYS:
        reasons.append("holdout_incomplete")
    if error is None or float(error) > MAX_DEVICE_ERROR_PCT:
        reasons.append("error_above_bar")
    if not beats["nde"] or not beats["energy"]:
        reasons.append("baseline_not_beaten")
    return (not reasons, reasons, beats)


def assemble_report(
    device_results: Sequence[Mapping],
    *,
    shadow: Mapping,
    holdout_days_info: Mapping,
    usable_days: int,
    grid_s: int,
    enabled_event_types: Sequence[str],
    plug_classes: Mapping[str, str],
    shadow_created: str | None,
    shadow_age_days: float | None,
    shadow_stale: bool | None,
    overlap: Mapping | None = None,
    overlap_synthetic: Mapping | None = None,
    excluded_capture: Mapping | None = None,
) -> dict[str, Any]:
    """Build the full run_validation report (the only metrics writer).

    device_results holds one entry per shadow link, switched and program links
    with their holdout figures and links of any other kind without
    (assemble_report turns the latter into unsupported_kind refusals; D-03:
    nothing is silently dropped). Program devices are judged by the unchanged
    P5 bars and classed from their stage template (DET-02, D-24).

    ``overlap`` is the D-08 overlap block (Plan 06-08) and
    ``overlap_synthetic`` the D-28 synthetic-overlap block; both are copied
    verbatim into ok reports only, so a refused report carries neither and the
    synthetic block never reaches devices, classes, refusals or a verdict.

    ``excluded_capture`` (D-05, Plan 08-04) is written into the report's
    ``holdout`` dict as ``{"windows", "samples"}`` on ok and refused reports
    alike: ``windows`` counts the capture windows whose padded span overlaps
    the holdout, not every capture record; ``samples`` counts the holdout
    samples removed. Without it the report carries no ``excluded_capture`` key
    and stays byte-identical (REPORT_SCHEMA_VERSION unchanged).
    """
    base = _base_report(
        shadow=shadow,
        holdout_days_info=holdout_days_info,
        usable_days=usable_days,
        shadow_created=shadow_created,
        shadow_age_days=shadow_age_days,
        shadow_stale=shadow_stale,
        excluded_capture=excluded_capture,
    )
    not_discovered = [
        dict(e) for e in (shadow.get("not_discovered") or []) if isinstance(e, Mapping)
    ]

    reason = _refusal_reason(
        shadow=shadow,
        holdout_days_info=holdout_days_info,
        usable_days=usable_days,
        grid_s=grid_s,
        enabled_event_types=enabled_event_types,
    )
    if reason is not None:
        report = {
            "status": "refused",
            "reason": reason,
            **base,
            "devices": [],
            "classes": [],
            "refusals": [],
            "not_discovered": not_discovered,
        }
        return _json_safe(report)

    devices: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for result in device_results:
        entity_id = str(result.get("entity_id", ""))
        link_overlap = float(result.get("overlap", 0.0) or 0.0)
        kind = str(result.get("kind", ""))
        n_events = int(result.get("n_events", 0) or 0)
        if kind not in _MEASURED_KINDS:
            # D-03: a link to a shadow artefact of any other kind is reported,
            # not measured (Plan 05-03's shadow_link links artefacts of every
            # kind). Switched and program links are measured alike (DET-02),
            # by the unchanged P5 bars (D-24).
            refusals.append(
                {
                    "entity_id": entity_id,
                    "reason": "unsupported_kind",
                    "kind": kind,
                    "n_events": n_events,
                    "link_overlap": link_overlap,
                }
            )
            continue
        if n_events < MIN_EVENTS_FIGURE:
            refusals.append(
                {
                    "entity_id": entity_id,
                    "reason": "insufficient_data",
                    "n_events": n_events,
                    "link_overlap": link_overlap,
                }
            )
            continue

        true_wh = float(result.get("energy_true_wh", 0.0) or 0.0)
        est_wh = float(result.get("energy_est_wh", 0.0) or 0.0)
        klass = derive_class(
            result.get("artefact") or {},
            previous=(plug_classes.get(entity_id) if isinstance(plug_classes, Mapping) else None),
        )
        passes, reasons, beats = _device_verdict(result)
        devices.append(
            {
                "entity_id": entity_id,
                "signature_id": str(result.get("signature_id", "")),
                "channel": int(result.get("channel", 0) or 0),
                "class": klass,
                "n_events": n_events,
                "link_overlap": link_overlap,
                "energy_true_wh": round(true_wh, 1),
                "energy_est_wh": round(est_wh, 1),
                "energy_error_pct": result.get("energy_error_pct"),
                "nde": result.get("nde"),
                "mae_w": result.get("mae_w"),
                "baseline": dict(result.get("baseline") or {}),
                "beats_baseline": beats,
                "bias_pct": (
                    round(100.0 * (est_wh - true_wh) / true_wh, 1) if true_wh > 0.0 else None
                ),
                "usable_holdout_days": int(result.get("usable_holdout_days", 0) or 0),
                "passes": passes,
                "reasons": reasons,
                "per_day": [dict(d) for d in result.get("per_day", []) if isinstance(d, Mapping)],
            }
        )

    # D-08 class bar over every peer of the class that has a figure (A2).
    by_class: dict[str, list[dict[str, Any]]] = {}
    for entry in devices:
        if entry["class"] is not None:
            by_class.setdefault(entry["class"], []).append(entry)
    classes: list[dict[str, Any]] = []
    class_pass: dict[str, bool] = {}
    for klass in sorted(by_class):
        peers = by_class[klass]
        peer_errors = [e["energy_error_pct"] for e in peers]
        n_passing = sum(1 for e in peers if e["passes"])
        passes = class_passes(peer_errors, n_passing)
        class_pass[klass] = passes
        ndes = [e["nde"] for e in peers if e["nde"] is not None]
        maes = [e["mae_w"] for e in peers if e["mae_w"] is not None]
        base_ndes = [
            e["baseline"].get("nde")
            for e in peers
            if isinstance(e["baseline"], Mapping) and e["baseline"].get("nde") is not None
        ]
        errors = [float(e) for e in peer_errors if e is not None]
        classes.append(
            {
                "class": klass,
                "peers": len(peers),
                "passing": n_passing,
                "median_error_pct": round(statistics.median(errors), 1) if errors else None,
                "max_error_pct": round(max(errors), 1) if errors else None,
                "median_nde": round(statistics.median(ndes), 4) if ndes else None,
                "median_mae_w": round(statistics.median(maes), 1) if maes else None,
                "median_baseline_nde": (
                    round(statistics.median(base_ndes), 4) if base_ndes else None
                ),
                "n_events": sum(int(e["n_events"]) for e in peers),
                "peer_entities": sorted(e["entity_id"] for e in peers),
                "passes": passes,
            }
        )

    # VAL-03 / D-19: one filterable flag per figure-bearing plug device.
    for entry in devices:
        entry["unvalidated"] = not (
            entry["passes"] or class_pass.get(entry["class"] or "", False)
        )

    report = {
        "status": "ok",
        "reason": None,
        **base,
        "thresholds": _thresholds(),
        "devices": devices,
        "classes": classes,
        "refusals": refusals,
        "not_discovered": not_discovered,
        "plug_classes": {e["entity_id"]: e["class"] for e in devices if e["class"] is not None},
    }
    # Plan 06-08 (D-08/D-28): the overlap blocks are copied in verbatim and only
    # into an ok report, so a refusal carries neither and the synthetic block
    # never reaches devices, classes or a verdict.
    if overlap is not None:
        report["overlap"] = dict(overlap)
    if overlap_synthetic is not None:
        report["overlap_synthetic"] = dict(overlap_synthetic)
    return _json_safe(report)


def assemble_sessions(session_results: Sequence[Mapping]) -> list[dict[str, Any]]:
    """Judge device-target plug-session results (Plan 06-09; REV-04, D-13).

    One entry per validated session, judged by the identical P5 D-07 device bar
    (``_device_verdict``) so a session must clear the same figures a plug device
    does. A refused session result (model_error, insufficient_data) keeps its
    reason and carries no figure and no pass. The entries are the report's
    separate ``sessions`` list: they never enter devices, classes or a class's
    peers (D-32). Pure data, stdlib, HA-free.
    """
    sessions: list[dict[str, Any]] = []
    for result in session_results:
        if not isinstance(result, Mapping):
            continue
        entry: dict[str, Any] = {
            "session_id": str(result.get("session_id", "")),
            "device_uuid": str(result.get("device_uuid", "")),
            "plug_channel": int(result.get("plug_channel", 0) or 0),
            "artefact_v": int(result.get("artefact_v", 0) or 0),
            "window_start": _iso_ms(result.get("window_start_ms")),
            "window_end": _iso_ms(result.get("window_end_ms")),
            "holdout_start": _iso_ms(result.get("holdout_start_ms")),
            "holdout_end": _iso_ms(result.get("holdout_end_ms")),
            "skipped_competitors": [
                str(name) for name in (result.get("skipped_competitors") or [])
            ],
        }
        if str(result.get("status", "")) != "ok":
            reason = str(result.get("reason") or "insufficient_data")
            entry["status"] = "refused"
            entry["reason"] = reason
            entry["passes"] = False
            entry["reasons"] = [reason]
            sessions.append(entry)
            continue
        passes, reasons, beats = _device_verdict(result)
        base = result.get("baseline")
        entry.update(
            {
                "status": "ok",
                "reason": None,
                "energy_true_wh": round(float(result.get("energy_true_wh", 0.0) or 0.0), 1),
                "energy_est_wh": round(float(result.get("energy_est_wh", 0.0) or 0.0), 1),
                "energy_error_pct": result.get("energy_error_pct"),
                "nde": result.get("nde"),
                "mae_w": result.get("mae_w"),
                "baseline": dict(base) if isinstance(base, Mapping) else {},
                "beats_baseline": beats,
                "n_events": int(result.get("n_events", 0) or 0),
                "usable_holdout_days": int(result.get("usable_holdout_days", 0) or 0),
                "passes": passes,
                "reasons": reasons,
            }
        )
        sessions.append(entry)
    return _json_safe(sessions)
