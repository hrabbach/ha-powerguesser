"""assemble_report: data gates, device/class pass bars, refusals and provenance (Plan 05-02, Task 3).

Pure-input tests over D-12 whole-report gates (insufficient_data, grid_mismatch,
event_types_mismatch), the D-07 device bar, the D-08 class bar, the VAL-03
unvalidated flag, the D-27 provenance fields, the link_overlap audit trail and
the single-writer metrics field.
"""

import json
from typing import Any

import pytest

from custom_components.powerguesser.engine.validation.report import assemble_report
from tests.fixtures.detect_traces import switched_artefact
from tests.fixtures.programs import program_artefact

KETTLE_CLASS = "switched_1.5_3kw_1_level"
EVENT_TYPES = ["step", "cycle_shape", "continuous"]


def _artefact() -> dict[str, Any]:
    """The 2000 W kettle shadow artefact behind every linked figure."""
    return switched_artefact(
        "sig-shadow-kettle",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )


def _device(
    entity_id: str,
    *,
    n_events: int = 9,
    error: float = 12.0,
    device_nde: float = 0.05,
    baseline_nde: float = 0.10,
    baseline_error: float = 20.0,
    overlap: float = 0.8,
    usable_holdout_days: int = 7,
    energy_true_wh: float = 100.0,
) -> dict[str, Any]:
    """One linked switched device with holdout figures and its artefact."""
    est = energy_true_wh * (1.0 + error / 100.0)
    return {
        "entity_id": entity_id,
        "channel": 1,
        "signature_id": f"sig-{entity_id}",
        "kind": "switched",
        "overlap": overlap,
        "artefact": _artefact(),
        "n_events": n_events,
        "energy_true_wh": energy_true_wh,
        "energy_est_wh": est,
        "energy_error_pct": error,
        "nde": device_nde,
        "mae_w": 50.0,
        "baseline": {"energy_error_pct": baseline_error, "nde": baseline_nde},
        "usable_holdout_days": usable_holdout_days,
        "per_day": [
            {
                "day_start_ms": 1_728_000_000_000 + i * 86_400_000,
                "coverage": 1.0,
                "n_events": max(1, n_events // 7),
                "energy_true_wh": round(energy_true_wh / 7, 1),
                "energy_est_wh": round(est / 7, 1),
                "energy_error_pct": error,
            }
            for i in range(usable_holdout_days)
        ],
    }


def _non_switched(entity_id: str, kind: str = "session", overlap: float = 0.7) -> dict[str, Any]:
    """A link to a non-switched shadow artefact: no figures (D-03)."""
    return {
        "entity_id": entity_id,
        "channel": 2,
        "signature_id": f"sig-{entity_id}",
        "kind": kind,
        "overlap": overlap,
    }


PROGRAM_CLASS = "program_1.5_3kw_multi_level"


def _program_artefact(*, stages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The 3-stage washer shadow artefact behind a program linkage (DET-02)."""
    artefact = program_artefact(f"sig-program", [(2000, 30), (200, 80), (500, 10)])
    if stages is not None:
        artefact["session"] = {**artefact["session"], "stages": stages}
    return artefact


def _program_device(
    entity_id: str,
    *,
    n_events: int = 7,
    error: float | None = 35.0,
    energy_est_wh: float = 135.0,
    energy_true_wh: float = 100.0,
    artefact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One linked program device with holdout figures and its stage template."""
    return {
        "entity_id": entity_id,
        "channel": 3,
        "signature_id": f"sig-{entity_id}",
        "kind": "program",
        "overlap": 0.9,
        "artefact": _program_artefact() if artefact is None else artefact,
        "n_events": n_events,
        "energy_true_wh": energy_true_wh,
        "energy_est_wh": energy_est_wh,
        "energy_error_pct": error,
        "nde": 0.08,
        "mae_w": 55.0,
        "baseline": {"energy_error_pct": 60.0, "nde": 0.12},
        "usable_holdout_days": 7,
        "per_day": [
            {
                "day_start_ms": 1_735_200_000_000 + i * 86_400_000,
                "coverage": 1.0,
                "n_events": 1,
                "energy_true_wh": round(energy_true_wh / 7, 1),
                "energy_est_wh": round(energy_est_wh / 7, 1),
                "energy_error_pct": error,
            }
            for i in range(7)
        ],
    }


def _report(
    device_results: list[dict[str, Any]],
    *,
    usable_days: int = 35,
    covered_holdout_days: int = 7,
    shadow_grid_s: int = 30,
    shadow_event_types: list[str] | None = None,
    not_discovered: list[dict[str, Any]] | None = None,
    plug_classes: dict[str, str] | None = None,
    shadow_stale: bool = False,
) -> dict[str, Any]:
    """assemble_report over a 35-day window with sane defaults."""
    return assemble_report(
        device_results,
        shadow={
            "grid_s": shadow_grid_s,
            "enabled_event_types": shadow_event_types
            if shadow_event_types is not None
            else list(EVENT_TYPES),
            "train_cutoff_ms": 1_728_000_000_000,
            "holdout_days": 7,
            "not_discovered": not_discovered or [],
        },
        holdout_days_info={
            "start_ms": 1_735_200_000_000,
            "end_ms": 1_735_200_000_000 + 7 * 86_400_000,
            "days": [
                {"day_start_ms": 1_735_200_000_000 + i * 86_400_000, "coverage": 1.0}
                for i in range(covered_holdout_days)
            ],
        },
        usable_days=usable_days,
        grid_s=30,
        enabled_event_types=list(EVENT_TYPES),
        plug_classes=plug_classes or {},
        shadow_created="2026-11-13T00:00:00+00:00",
        shadow_age_days=7.0,
        shadow_stale=shadow_stale,
    )


def _kettle(report: dict[str, Any]) -> dict[str, Any]:
    return next(d for d in report["devices"] if d["entity_id"] == "sensor.kettle_power")


def _walk_pct_numbers(value: Any) -> list[float]:
    """Every numeric value whose field name ends in _pct (the refusal audit)."""
    if isinstance(value, dict):
        out: list[float] = []
        for k, v in value.items():
            if isinstance(k, str) and k.endswith("_pct") and isinstance(v, int | float) and not isinstance(v, bool):
                out.append(float(v))
            out.extend(_walk_pct_numbers(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_walk_pct_numbers(v))
        return out
    return []


def test_report_refuses_fewer_than_five_events() -> None:
    """n_events 3: a refusal entry, no devices entry, the overlap audit trail kept."""
    report = _report([_device("sensor.kettle_power", n_events=3)])
    kettle_refusal = next(
        r for r in report["refusals"] if r["entity_id"] == "sensor.kettle_power"
    )
    assert kettle_refusal["reason"] == "insufficient_data"
    assert kettle_refusal["n_events"] == 3
    assert kettle_refusal["link_overlap"] == 0.8
    assert all(d["entity_id"] != "sensor.kettle_power" for d in report["devices"])


def test_report_figure_below_pass_bar_events() -> None:
    """5-7 events: a figure, passes false, reason events_below_pass_bar (D-12)."""
    report = _report([_device("sensor.kettle_power", n_events=6)])
    kettle = _kettle(report)
    assert kettle["n_events"] == 6
    assert kettle["passes"] is False
    assert "events_below_pass_bar" in kettle["reasons"]
    assert kettle["unvalidated"] is True


def test_report_device_passes_bar() -> None:
    """9 events, 12 % error, baseline beaten on both figures, 7 usable days: passes (D-07)."""
    report = _report([_device("sensor.kettle_power")])
    kettle = _kettle(report)
    assert kettle["passes"] is True
    assert kettle["reasons"] == []
    assert kettle["beats_baseline"] == {"nde": True, "energy": True}
    assert kettle["unvalidated"] is False
    assert kettle["class"] == KETTLE_CLASS


def test_report_device_error_above_bar() -> None:
    """25 % error: fails, reason error_above_bar."""
    report = _report([_device("sensor.kettle_power", error=25.0)])
    kettle = _kettle(report)
    assert kettle["passes"] is False
    assert "error_above_bar" in kettle["reasons"]
    assert kettle["unvalidated"] is True


def test_report_device_baseline_not_beaten() -> None:
    """A baseline NDE below the device's: reason baseline_not_beaten."""
    report = _report([_device("sensor.kettle_power", device_nde=0.20, baseline_nde=0.10)])
    kettle = _kettle(report)
    assert kettle["passes"] is False
    assert "baseline_not_beaten" in kettle["reasons"]
    assert kettle["beats_baseline"]["nde"] is False


def test_report_device_holdout_incomplete() -> None:
    """Fewer than 7 usable holdout days: reason holdout_incomplete (A3)."""
    report = _report([_device("sensor.kettle_power", usable_holdout_days=6)])
    kettle = _kettle(report)
    assert kettle["passes"] is False
    assert "holdout_incomplete" in kettle["reasons"]
    assert kettle["unvalidated"] is True


def test_report_class_bar_marks_unvalidated() -> None:
    """D-08 + VAL-03: two passing peers carry a non-passing peer to unvalidated false."""
    devices = [
        _device("sensor.kettle_power", error=10.0, device_nde=0.05, baseline_nde=0.10),
        _device("sensor.tea_power", error=18.0, device_nde=0.08, baseline_nde=0.12),
        _device("sensor.coffee_power", error=30.0, device_nde=0.20, baseline_nde=0.10),
    ]
    report = _report(devices)
    klass = next(c for c in report["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["peers"] == 3
    assert klass["passing"] == 2
    assert klass["median_error_pct"] == 18.0
    assert klass["max_error_pct"] == 30.0
    assert klass["passes"] is True
    # every peer, including the non-passing one, reads unvalidated false
    for entry in report["devices"]:
        assert entry["unvalidated"] is False
    # the metrics field defines the flag
    assert "unvalidated" in report["metrics"]


def test_report_class_bar_fails_with_one_passing_peer() -> None:
    """One passing peer cannot carry the class; other peers stay unvalidated."""
    devices = [
        _device("sensor.kettle_power", error=10.0),
        _device("sensor.coffee_power", error=30.0),
    ]
    report = _report(devices)
    klass = next(c for c in report["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["passes"] is False
    by_entity = {d["entity_id"]: d for d in report["devices"]}
    assert by_entity["sensor.kettle_power"]["unvalidated"] is False  # its own D-07 pass
    assert by_entity["sensor.coffee_power"]["unvalidated"] is True


def test_report_class_bar_fails_above_peer_cap() -> None:
    """A peer at 36 % fails the class even with two passing peers."""
    devices = [
        _device("sensor.kettle_power", error=10.0),
        _device("sensor.tea_power", error=18.0),
        _device("sensor.coffee_power", error=36.0),
    ]
    report = _report(devices)
    klass = next(c for c in report["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["passes"] is False


def test_report_refuses_short_history() -> None:
    """Fewer than 28 usable days: whole-report refusal, no figures anywhere (D-12)."""
    report = _report([_device("sensor.kettle_power")], usable_days=27)
    assert report["status"] == "refused"
    assert report["reason"] == "insufficient_data"
    assert report["devices"] == []
    assert report["classes"] == []
    assert "plug_classes" not in report
    # a refusal carries a reason, never a percentage figure (D-12)
    assert _walk_pct_numbers(report) == []


def test_report_refuses_few_covered_holdout_days() -> None:
    """Fewer than 5 holdout days at >= 50 % coverage: insufficient_data (D-12)."""
    report = _report([_device("sensor.kettle_power")], covered_holdout_days=4)
    assert report["status"] == "refused"
    assert report["reason"] == "insufficient_data"
    assert report["devices"] == []


def test_report_refuses_grid_mismatch() -> None:
    """A shadow set learned on another grid is refused, never measured (D-12)."""
    report = _report([_device("sensor.kettle_power")], shadow_grid_s=60)
    assert report["status"] == "refused"
    assert report["reason"] == "grid_mismatch"
    assert report["devices"] == []


def test_report_refuses_event_types_mismatch() -> None:
    """Figures never mix two event-type configurations (D-26 (1), T-05-26)."""
    report = _report(
        [ _device("sensor.kettle_power")],
        shadow_event_types=["step", "short_event"],
    )
    assert report["status"] == "refused"
    assert report["reason"] == "event_types_mismatch"
    assert report["devices"] == []


def test_report_event_types_order_is_not_a_mismatch() -> None:
    """The same event types in a different order are not a mismatch."""
    report = _report(
        [_device("sensor.kettle_power")],
        shadow_event_types=list(reversed(EVENT_TYPES)),
    )
    assert report["status"] == "ok"


def test_report_metrics_single_writer() -> None:
    """The report's metrics field carries exactly the five definitions."""
    report = _report([_device("sensor.kettle_power")])
    assert set(report["metrics"].keys()) == {"nde", "mae_w", "unvalidated", "link_overlap", "measures"}


def test_report_link_overlap_on_linked_entries() -> None:
    """Every figure or refusal derived from a link carries its overlap; not_discovered none."""
    report = _report(
        [_device("sensor.kettle_power", overlap=0.8), _non_switched("sensor.washer_power")],
        not_discovered=[{"entity_id": "sensor.oven_power", "channel": 3, "n_train_sessions": 4}],
    )
    kettle = _kettle(report)
    assert kettle["link_overlap"] == 0.8
    refusal = next(r for r in report["refusals"] if r["entity_id"] == "sensor.washer_power")
    assert refusal["reason"] == "unsupported_kind"
    assert refusal["link_overlap"] == 0.7
    undiscovered = report["not_discovered"]
    assert undiscovered == [{"entity_id": "sensor.oven_power", "channel": 3, "n_train_sessions": 4}]
    assert "link_overlap" not in undiscovered[0]


def test_report_states_what_it_measured() -> None:
    """D-27: provenance on every report; stale is flagged, never refused, figures kept."""
    refused = _report([_device("sensor.kettle_power")], usable_days=20)
    assert refused["shadow_created"] == "2026-11-13T00:00:00+00:00"
    assert refused["shadow_age_days"] == 7.0
    assert refused["shadow_stale"] is False

    stale = _report([_device("sensor.kettle_power")], shadow_stale=True)
    assert stale["status"] == "ok"
    assert stale["shadow_stale"] is True
    assert len(stale["devices"]) == 1

    assert (
        stale["metrics"]["measures"]
        == "the masked shadow model trained up to train_cutoff and replayed on the holdout block"
    )


def test_report_bias_pct_signed_and_none_for_zero_reference() -> None:
    """bias_pct = 100 x (est - true) / true, signed; None when true <= 0."""
    over = _report([_device("sensor.kettle_power", error=12.0, energy_true_wh=100.0)])
    assert _kettle(over)["bias_pct"] == pytest.approx(12.0, abs=0.2)
    zero_dev = _device("sensor.kettle_power", energy_true_wh=0.0)
    zero_dev["energy_error_pct"] = None  # the worker refuses the figure at 0 reference
    zero = _report([zero_dev])
    assert _kettle(zero)["bias_pct"] is None
    assert _kettle(zero)["energy_error_pct"] is None


def test_report_plug_classes_only_when_ok() -> None:
    """An ok report maps every figure-bearing plug to its class for the Store (D-06)."""
    report = _report([_device("sensor.kettle_power")])
    assert report["plug_classes"] == {"sensor.kettle_power": KETTLE_CLASS}


def test_report_replaces_non_finite_with_none() -> None:
    """NaN/inf never appear in the report (T-05-04): serialize under allow_nan=False."""
    bad = _device("sensor.kettle_power")
    bad["mae_w"] = float("inf")
    bad["baseline"]["nde"] = float("nan")
    report = _report([bad])
    json.dumps(report, allow_nan=False)
    assert _kettle(report)["mae_w"] is None
    assert _kettle(report)["baseline"]["nde"] is None


def test_report_sorts_classes_and_carries_peer_entities() -> None:
    """classes are sorted by name and name every peer entity (A2)."""
    report = _report(
        [_device("sensor.kettle_power", error=10.0), _device("sensor.heater_power", error=10.0)]
    )
    names = [c["class"] for c in report["classes"]]
    assert names == sorted(names)
    klass = next(c for c in report["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["peer_entities"] == ["sensor.heater_power", "sensor.kettle_power"]


# --- Plan 06-07 (DET-02, D-24): the program kind is measured like a switched one ---


def test_report_measures_program_device() -> None:
    """A program link with figures is measured, never refused as unsupported_kind.

    It carries the same fields as a switched device, is classed from its stage
    template, and is judged by the unchanged D-07 bar (7 events, one peer).
    """
    report = _report([_program_device("sensor.washer_power", error=12.0)])
    washer = next(d for d in report["devices"] if d["entity_id"] == "sensor.washer_power")
    assert washer["class"] == PROGRAM_CLASS
    assert washer["energy_true_wh"] == 100.0
    assert washer["energy_est_wh"] == 135.0
    assert washer["energy_error_pct"] == 12.0
    assert washer["passes"] is False
    assert "events_below_pass_bar" in washer["reasons"]
    assert washer["unvalidated"] is True
    assert all(r["entity_id"] != "sensor.washer_power" for r in report["refusals"])


def test_report_refuses_unknown_kind_only() -> None:
    """Only a kind outside switched/program is refused with unsupported_kind (DET-02)."""
    report = _report(
        [
            _program_device("sensor.washer_power"),
            _non_switched("sensor.mystery_power", kind="mystery", overlap=0.55),
        ]
    )
    assert [d["entity_id"] for d in report["devices"]] == ["sensor.washer_power"]
    refusal = next(r for r in report["refusals"] if r["entity_id"] == "sensor.mystery_power")
    assert refusal["reason"] == "unsupported_kind"
    assert refusal["kind"] == "mystery"
    assert refusal["link_overlap"] == 0.55


def test_report_program_without_stages_has_no_class_and_zero_energy() -> None:
    """A program template the detector cannot track reports 0 Wh and no class.

    P5 D-08: with no usable stage template there is no class, so the device can
    never be counted under an unrelated passing class; the entry itself stays,
    with its own figures (energy_est_wh 0.0 here).
    """
    report = _report(
        [
            _program_device(
                "sensor.washer_power",
                error=100.0,
                energy_est_wh=0.0,
                artefact=_program_artefact(stages=[]),
            )
        ]
    )
    washer = next(d for d in report["devices"] if d["entity_id"] == "sensor.washer_power")
    assert washer["energy_est_wh"] == 0.0
    assert washer["class"] is None
    assert report["plug_classes"] == {}
    assert report["classes"] == []
    assert washer["unvalidated"] is True
    assert not [r for r in report["refusals"] if r["entity_id"] == "sensor.washer_power"]
