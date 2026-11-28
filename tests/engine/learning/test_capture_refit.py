"""Tests for the capture_refit worker stage (Plan 09-05 Task 2, CAP-13, D-09).

The stage refits a confirmed switched device on the union of the train episodes
it already claims and its captured runs, and judges the result with the unchanged
D-17 gate on validation days that exclude every captured run (the shared
``_gate_excluding_captures`` path). A program device is skipped; a refit whose
states do not move produces no artefact; an emptied validation block is
``insufficient_validation``.

The run dir is a hand-built stage dir (episodes.npz + digest_meta.json +
residual.npz), which makes every claimed/unclaimed/anchored row exact.
"""

from pathlib import Path
from types import SimpleNamespace

import json

import numpy as np
import pytest

from custom_components.powerguesser.engine.capture.constants import HOLDOUT_PAD_STEPS
from custom_components.powerguesser.engine.learning import worker
from custom_components.powerguesser.engine.learning.artefact import (
    Box,
    SignatureArtefact,
    State,
    detection_differs,
)
from custom_components.powerguesser.engine.learning.worker import run_stage

_STEP_MS = 30_000
_START_MS = 0
_THRESHOLD_W = 120.0
_SNAPSHOT_AT = "2026-11-27T00:00:00+00:00"
_FIELDS = (
    "start_i",
    "end_i",
    "kind",
    "block",
    "rise_w",
    "fall_w",
    "dur_s",
    "plateau_w",
    "energy_wh",
    "hour",
    "dow",
    "l1_share",
    "l2_share",
    "l3_share",
)
_INT64 = {"start_i", "end_i"}
_INT32 = {"kind", "block", "hour", "dow"}


def _row(start_i: int, block: int, *, rise=2000.0, fall=-2000.0, dur=180.0, plateau=2000.0,
         energy=100.0, hour=8, dow=0, kind=0) -> dict:
    return {
        "start_i": start_i,
        "end_i": start_i + 6,
        "kind": kind,
        "block": block,
        "rise_w": rise,
        "fall_w": fall,
        "dur_s": dur,
        "plateau_w": plateau,
        "energy_wh": energy,
        "hour": hour,
        "dow": dow,
        "l1_share": float("nan"),
        "l2_share": float("nan"),
        "l3_share": float("nan"),
    }


def _table(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in _FIELDS:
        dtype = np.int64 if key in _INT64 else (np.int32 if key in _INT32 else np.float64)
        out[key] = np.asarray([r[key] for r in rows], dtype=dtype)
    return out


# Rows 0-3 are the claimed train kettle runs; 4/5/8 and the session 7 sit in the
# validation block; 6 is an unclaimed train load at 3500 W.
_ROWS = [
    _row(10, 0),
    _row(20, 0),
    _row(30, 0),
    _row(40, 0),
    _row(100, 1),
    _row(110, 1),
    _row(200, 0, rise=3500.0, fall=-3500.0, dur=300.0, plateau=3500.0, hour=20),
    _row(300, 1, kind=1, dur=9000.0, hour=12),
    _row(400, 1, dur=400.0, hour=9),
]
_EPISODES = _table(_ROWS)
# The narrow kettle the fit over rows 0-3 reproduces exactly (no_change baseline).
_NARROW = SignatureArtefact(
    schema_version=1,
    signature_id="sig-main",
    kind="switched",
    label="Kettle",
    category="kettle",
    extraction={"grid_s": 30, "edge_threshold_w": _THRESHOLD_W, "merge_samples": 2, "pair_tol": 0.25},
    states=(
        worker.fit_state_boxes(_EPISODES, np.array([0, 1, 2, 3], dtype=np.int64), _THRESHOLD_W, name="on"),
    ),
    session=None,
    time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
    constraints={"max_concurrent": 1},
    provenance={},
    scores={},
)


def _wide_current() -> SignatureArtefact:
    """A kettle whose duration box is wider than any fit of rows 0-3, 6 can produce.

    Its validation claim on row 8 (dur 400) is one the refit loses, so the gate
    reads it as worse_than_active.
    """
    return SignatureArtefact(
        schema_version=1,
        signature_id="sig-wide",
        kind="switched",
        label="Kettle",
        category="kettle",
        extraction={"grid_s": 30, "edge_threshold_w": _THRESHOLD_W, "merge_samples": 2, "pair_tol": 0.25},
        states=(
            State(
                name="on",
                rise_w=Box(lo=1900.0, p50=2000.0, hi=2100.0),
                fall_w=Box(lo=-2100.0, p50=-2000.0, hi=-1900.0),
                dur_s=Box(lo=100.0, p50=180.0, hi=500.0),
                plateau_w=Box(lo=1900.0, p50=2000.0, hi=2100.0),
                phase_share=None,
            ),
        ),
        session=None,
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={},
        scores={},
    )


def _program_current() -> SignatureArtefact:
    data = _NARROW.to_dict()
    data["kind"] = "program"
    return SignatureArtefact.from_dict(data)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> SimpleNamespace:
    stage_dir = tmp_path_factory.mktemp("capture_refit") / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez(stage_dir / "episodes.npz", **_EPISODES)
    meta = {
        "edge_threshold_w": _THRESHOLD_W,
        "block_residual_wh": {"train": 1000.0, "val": 1000.0, "holdout": 0.0},
        "days": {"train": [0], "val": [1], "holdout": [], "dropped": []},
    }
    (stage_dir / "digest_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(
        stage_dir / "residual.npz",
        r=np.zeros(500, dtype=np.float64),
        start_ms=np.array([_START_MS], dtype=np.int64),
        step_ms=np.array([_STEP_MS], dtype=np.int64),
        day_bounds_ms=np.array([0, 86_400_000, 172_800_000], dtype=np.int64),
        suppress_alignment=np.array([1], dtype=np.int8),
    )
    return SimpleNamespace(stage_dir=stage_dir)


def _capture(capture_id: str, row: int | None) -> dict:
    """A capture anchored on a table row, or an unanchored one (row None)."""
    if row is None:
        return {
            "capture_id": capture_id,
            "rise_ms": 10**12,
            "fall_ms": 10**12 + 1,
            "features": {
                "rise_w": 1500.0,
                "fall_w": -1500.0,
                "dur_s": 120.0,
                "plateau_w": 1500.0,
                "hour": 7,
                "dow": 1,
            },
        }
    return {
        "capture_id": capture_id,
        "rise_ms": _START_MS + _ROWS[row]["start_i"] * _STEP_MS,
        "fall_ms": _START_MS + _ROWS[row]["end_i"] * _STEP_MS,
        "features": {
            "rise_w": float(_ROWS[row]["rise_w"]),
            "fall_w": float(_ROWS[row]["fall_w"]),
            "dur_s": float(_ROWS[row]["dur_s"]),
            "plateau_w": float(_ROWS[row]["plateau_w"]),
            "hour": int(_ROWS[row]["hour"]),
            "dow": int(_ROWS[row]["dow"]),
        },
    }


def _device(current: SignatureArtefact, captures: list[dict], *, uuid: str = "dev-1") -> dict:
    return {
        "device_uuid": uuid,
        "signature_id": current.signature_id,
        "current": current.to_dict(),
        "captures": captures,
    }


def _job(run_dir: SimpleNamespace, devices: list[dict], *, capture_windows=None) -> dict:
    return {
        "stage_dir": str(run_dir.stage_dir),
        "validated": False,
        "captures_snapshot_at": _SNAPSHOT_AT,
        "capture_windows": [] if capture_windows is None else capture_windows,
        "devices": devices,
    }


def _run(run_dir, devices, *, capture_windows=None) -> dict:
    out = run_stage("capture_refit", _job(run_dir, devices, capture_windows=capture_windows))
    result = json.loads(Path(out["result"]).read_text(encoding="utf-8"))
    return {"out": out, "devices": result["devices"]}


def _overlaps(row_start: int, row_end: int, windows, pad_ms: int) -> bool:
    for window in windows:
        if row_start < int(window[1]) + pad_ms and row_end >= int(window[0]) - pad_ms:
            return True
    return False


# --------------------------------------------------------------------------- #
def test_capture_refit_union_without_double_count(run_dir, monkeypatch) -> None:
    """A capture mapped to a row the current artefact already claims adds no second
    copy; a capture on an unclaimed row adds it."""
    unions: list[list[int]] = []
    original = worker._refit_from_rows

    def spy(current, table, union_idx, threshold_w, *, provenance):
        unions.append(sorted(int(i) for i in union_idx))
        return original(current, table, union_idx, threshold_w, provenance=provenance)

    monkeypatch.setattr(worker, "_refit_from_rows", spy)

    _run(run_dir, [
        _device(_NARROW, [_capture("cap-claimed", 0)], uuid="a"),
        _device(_NARROW, [_capture("cap-unclaimed", 6)], uuid="b"),
    ])

    assert unions[0] == [0, 1, 2, 3], unions
    assert unions[1] == [0, 1, 2, 3, 6], unions


def test_unmapped_capture_uses_features_in_fit_only(run_dir, monkeypatch) -> None:
    """An unmapped capture (no row within ANCHOR_TOL_STEPS) contributes its features
    row to the fit table only, is counted in n_unmapped, and never enters a
    replayed/gate table."""
    fit_tables: list = []
    gate_tables: list = []
    original_fit = worker._refit_from_rows
    original_gate = worker.gate_proposal

    def fit_spy(current, table, union_idx, threshold_w, *, provenance):
        fit_tables.append((table, sorted(int(i) for i in union_idx)))
        return original_fit(current, table, union_idx, threshold_w, provenance=provenance)

    def gate_spy(proposed, active, table, residual_wh):
        gate_tables.append(table)
        return original_gate(proposed, active, table, residual_wh)

    monkeypatch.setattr(worker, "_refit_from_rows", fit_spy)
    monkeypatch.setattr(worker, "gate_proposal", gate_spy)

    res = _run(run_dir, [_device(_NARROW, [_capture("cap-off", None)])])
    entry = res["devices"][0]

    assert entry["n_unmapped"] == 1
    fit_table, union_idx = fit_tables[0]
    n_base = len(_EPISODES["start_i"])
    # The appended fit row sits past the episode rows and carries the capture's rise.
    assert len(fit_table["rise_w"]) == n_base + 1
    assert float(fit_table["rise_w"][n_base]) == 1500.0
    assert float(fit_table["dur_s"][n_base]) == 120.0
    assert int(fit_table["hour"][n_base]) == 7
    assert union_idx == [0, 1, 2, 3, n_base]
    # The replayed table the gate judges on has no appended row.
    assert gate_tables
    assert len(gate_tables[0]["rise_w"]) == n_base
    assert len(gate_tables[0]["block"]) == n_base


def test_fit_only_table_drops_a_non_finite_required_feature_row() -> None:
    """A capture whose required numeric feature is non-finite is dropped, not NaN-padded.

    The earlier coercion appended NaN for a non-numeric feature, which is not row-local:
    the NaN rise row entered the fit union, ``_refit_from_rows`` sent it to state 0, and
    ``fit_state_boxes`` turned that whole state's boxes NaN. Such a row is now dropped
    entirely, so it can neither poison a state's boxes nor reach capture_refit.json; the
    remaining (usable) rows are appended exactly as before.
    """
    episodes = _table([_row(10, 0)])
    good = _capture("cap-good", None)
    non_numeric = _capture("cap-bad", None)
    non_numeric["features"] = {**non_numeric["features"], "rise_w": "not-a-number"}
    nan_valued = _capture("cap-nan", None)
    nan_valued["features"] = {**nan_valued["features"], "rise_w": float("nan")}
    feature_rows = [good, non_numeric, nan_valued]

    table = worker._fit_only_table(episodes, feature_rows)  # must not raise

    n_base = len(episodes["start_i"])
    # Only the usable row is appended; both bad rows are dropped.
    assert len(table["rise_w"]) == n_base + 1
    assert float(table["rise_w"][n_base]) == good["features"]["rise_w"]
    assert float(table["dur_s"][n_base]) == good["features"]["dur_s"]
    for key in ("rise_w", "fall_w", "dur_s", "plateau_w"):
        assert np.all(np.isfinite(table[key])), key


def test_fit_only_table_coerces_a_bad_hour_dow_to_zero() -> None:
    """A non-numeric stored hour/dow becomes 0, not a ValueError (WR-02 class).

    ``_fit_only_table`` called ``int(raw)`` unguarded for hour and dow, so a
    hand-edited or partially written capture with, say, ``hour="late"`` raised out of
    ``capture_refit`` and dropped every device's refit. It defaults to 0, the same
    coercion ``examples._int_or_zero`` applies.
    """
    episodes = _table([_row(10, 0)])
    features = {**_capture("cap-bad", None)["features"], "hour": "late", "dow": None}
    feature_rows = [{"features": features}]

    table = worker._fit_only_table(episodes, feature_rows)  # must not raise

    assert len(table["hour"]) == len(episodes["start_i"]) + 1
    assert int(table["hour"][-1]) == 0
    assert int(table["dow"][-1]) == 0


def test_corrupt_capture_row_is_dropped_and_never_passes_the_gate(run_dir) -> None:
    """A corrupt capture row is dropped from the refit fit, so it can never back-door
    the D-17 gate as a NaN-boxed proposal nor write NaN into capture_refit.json.

    Regression for the follow-up to the NaN coercion: appending a non-numeric
    ``rise_w`` as NaN was not row-local. The NaN rise row entered the fit union,
    ``_refit_from_rows`` assigned it to state 0 (NaN fails every containment test and
    ``min`` over all-NaN keys returns index 0), and ``fit_state_boxes`` made that whole
    state's rise/fall/dur/plateau boxes NaN. The NaN-boxed artefact claims nothing, so
    its ``proposed_S`` is 0; with an active artefact that also scores 0 on the
    capture-excluded validation block the unchanged ``proposed_S >= active_S`` gate
    PASSED it, and ``canonical_json`` wrote the NaN boxes into capture_refit.json.

    Dropping the row leaves the fit at the active artefact's own claimed rows, which
    reproduce it exactly (no_change), so no artefact — NaN-boxed or otherwise — is
    proposed.
    """
    # Exclude the block-1 rows the active kettle matches (rows 4 and 5). What remains
    # is row 7 (a program) and row 8 (dur 400, outside the kettle's dur box), so the
    # active artefact scores 0 on the capture-excluded block.
    windows = [[_START_MS + 100 * _STEP_MS, _START_MS + 116 * _STEP_MS]]
    corrupt = _capture("cap-corrupt", None)
    corrupt["features"] = {**corrupt["features"], "rise_w": "not-a-number"}

    res = _run(run_dir, [_device(_NARROW, [corrupt])], capture_windows=windows)
    entry = res["devices"][0]

    # The unmapped (corrupt) capture is still counted, but dropped from the fit.
    assert entry["n_unmapped"] == 1
    assert entry["reason"] == "no_change"
    assert entry["passed"] is not True
    assert entry["artefact"] is None

    raw = (run_dir.stage_dir / "capture_refit.json").read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw


def test_capture_refit_uses_unchanged_gate(run_dir, monkeypatch) -> None:
    """The gate is called with a table that has no block-1 row overlapping a padded
    capture window, and the entry's gate fields equal its return values."""
    windows = [[_START_MS + 100 * _STEP_MS, _START_MS + 106 * _STEP_MS]]  # row 4's span
    gate_tables: list = []
    gate_results: list = []
    original = worker.gate_proposal

    def spy(proposed, active, table, residual_wh):
        out = original(proposed, active, table, residual_wh)
        gate_tables.append(table)
        gate_results.append(out)
        return out

    monkeypatch.setattr(worker, "gate_proposal", spy)

    res = _run(
        run_dir,
        [_device(_NARROW, [_capture("cap-6", 6)])],
        capture_windows=windows,
    )
    entry = res["devices"][0]

    assert gate_tables and gate_results
    pad_ms = HOLDOUT_PAD_STEPS * _STEP_MS
    for table in gate_tables:
        for i in np.where(table["block"] == 1)[0]:
            assert not _overlaps(
                int(_START_MS + table["start_i"][i] * _STEP_MS),
                int(_START_MS + table["end_i"][i] * _STEP_MS),
                windows,
                pad_ms,
            )
    assert int(gate_tables[0]["block"][4]) == -1
    assert int(gate_tables[0]["block"][5]) == 1

    assert entry["passed"] == gate_results[0]["passed"]
    assert entry["proposed_S"] == gate_results[0]["proposed_S"]
    assert entry["active_S"] == gate_results[0]["active_S"]
    assert entry["basis"] == gate_results[0]["basis"] == "learning_split_val"
    assert entry["reason"] in {"gated", "worse_than_active"}


def test_capture_refit_insufficient_validation(run_dir, monkeypatch) -> None:
    """Every block-1 row inside a padded window: reason insufficient_validation,
    passed False, proposed_S/active_S None, artefact None, and the gate is not
    called."""
    # One window spanning rows 4, 5, 7 and 8 (all block-1 rows).
    windows = [[_START_MS + 100 * _STEP_MS, _START_MS + 406 * _STEP_MS]]
    gate_calls: list = []
    original = worker.gate_proposal

    def spy(*args, **kwargs):
        gate_calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "gate_proposal", spy)

    res = _run(
        run_dir,
        [_device(_NARROW, [_capture("cap-6", 6)])],
        capture_windows=windows,
    )
    entry = res["devices"][0]

    assert gate_calls == []
    assert entry["reason"] == "insufficient_validation"
    assert entry["passed"] is False
    assert entry["proposed_S"] is None
    assert entry["active_S"] is None
    assert entry["artefact"] is None
    assert entry["capture_ids"] == ["cap-6"]
    assert entry["n_unmapped"] == 0


def test_program_device_skipped(run_dir, monkeypatch) -> None:
    """A program device produces no artefact and reason program_not_supported."""
    gate_calls: list = []
    monkeypatch.setattr(worker, "gate_proposal", lambda *a, **k: gate_calls.append(a))

    res = _run(run_dir, [_device(_program_current(), [_capture("cap-6", 6)])])
    entry = res["devices"][0]

    assert entry["reason"] == "program_not_supported"
    assert entry["artefact"] is None
    assert gate_calls == []


def test_no_change_no_proposal(run_dir, monkeypatch) -> None:
    """A refit whose states do not differ from the active version yields no artefact
    (reason no_change) and calls no gate."""
    gate_calls: list = []
    monkeypatch.setattr(worker, "gate_proposal", lambda *a, **k: gate_calls.append(a))

    res = _run(run_dir, [_device(_NARROW, [])])
    entry = res["devices"][0]

    assert entry["reason"] == "no_change"
    assert entry["artefact"] is None
    assert gate_calls == []


def test_capture_refit_provenance(run_dir) -> None:
    """A gated artefact keeps signature_id, kind and label, carries capture
    provenance and the gate result under proposal_gate, and scores with the same
    keys as refit's."""
    res = _run(run_dir, [_device(_NARROW, [_capture("cap-b", 6), _capture("cap-a", 6)])])
    entry = res["devices"][0]

    assert entry["reason"] == "gated"
    assert entry["passed"] is True
    artefact = entry["artefact"]
    assert artefact is not None
    assert artefact["signature_id"] == _NARROW.signature_id
    assert artefact["kind"] == "switched"
    assert artefact["label"] == "Kettle"
    assert artefact["provenance"] == {
        "source": "capture",
        "capture_ids": ["cap-a", "cap-b"],
        "captures_snapshot_at": _SNAPSHOT_AT,
    }
    assert entry["capture_ids"] == ["cap-a", "cap-b"]
    assert artefact["proposal_gate"] == {
        "passed": entry["passed"],
        "proposed_S": entry["proposed_S"],
        "active_S": entry["active_S"],
        "basis": entry["basis"],
    }
    assert set(artefact["scores"]) == {
        "train",
        "val",
        "validated",
        "conflicts",
        "validation_error_pct",
        "validation_basis",
    }
    # A capture-informed refit is a real differing proposal for this device.
    assert detection_differs(_NARROW.to_dict(), artefact)


def test_capture_refit_worse_than_active(run_dir) -> None:
    """A refit the wider active version beats on the validation block is recorded
    as worse_than_active and keeps its artefact for diagnosis."""
    res = _run(run_dir, [_device(_wide_current(), [_capture("cap-6", 6)])])
    entry = res["devices"][0]

    assert entry["passed"] is False
    assert entry["reason"] == "worse_than_active"
    assert entry["active_S"] > entry["proposed_S"]
    assert entry["artefact"] is not None
    assert entry["artefact"]["proposal_gate"]["passed"] is False


def test_stage_registered(run_dir) -> None:
    """"capture_refit" is in STAGES; stage_outputs reports the three keys; the
    dispatcher runs the stage."""
    assert "capture_refit" in worker.STAGES
    outputs = worker.stage_outputs(
        "capture_refit",
        {"stage_dir": "/tmp/x", "result": "/tmp/x/capture_refit.json", "n_devices": 3},
    )
    assert outputs == {
        "stage_dir": "/tmp/x",
        "result": "/tmp/x/capture_refit.json",
        "n_devices": 3,
    }
    res = _run(run_dir, [])
    assert res["out"]["n_devices"] == 0
    assert (run_dir.stage_dir / "capture_refit.json").is_file()
