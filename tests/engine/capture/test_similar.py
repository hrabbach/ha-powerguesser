"""Engine tests for the D-11 similarity and near-twin computation (CAP-12, D-11).

Plan 09-02 Task 1: ``similar.similar`` scores every visible candidate and
confirmed device against a capture's snapped pair, orders the top list stably
and reports the near-twin flag over the full scored pool, never the display list.
"""

import numpy as np

from custom_components.powerguesser.engine.capture import similar as similar_mod
from custom_components.powerguesser.engine.capture.config import CaptureConfig
from custom_components.powerguesser.engine.learning import compile as compile_mod
from tests.conftest import make_switched_candidate
from tests.fixtures.learning.synth import SynthSpec, build_buffer

END_MS = 1_700_000_000_000
GRID_S = 30.0


def _artefact_dict(sig_id, label, state):
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "switched",
        "label": label,
        "category": "appliance",
        "extraction": {
            "grid_s": 30,
            "edge_threshold_w": 120.0,
            "merge_samples": 2,
            "pair_tol": 0.25,
        },
        "states": [state],
        "session": None,
        "time_prior": {"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        "constraints": {"max_concurrent": 1},
        "provenance": {},
        "scores": {},
    }


def _learned(sig_id, label, rise_w=2000.0, dur_s=180.0, n=6):
    """A learned switched artefact from compile.fit_state_boxes over n episodes."""
    table = {
        "rise_w": np.full(n, rise_w),
        "fall_w": np.full(n, -rise_w),
        "dur_s": np.full(n, dur_s),
        "plateau_w": np.full(n, rise_w),
        "l1_share": np.full(n, np.nan),
        "l2_share": np.full(n, np.nan),
        "l3_share": np.full(n, np.nan),
    }
    state = compile_mod.fit_state_boxes(table, np.arange(n), 120.0, name="on")
    return _artefact_dict(sig_id, label, state.to_dict())


def _box(rise, dur):
    rbox = {"lo": float(rise[0]), "p50": float(rise[1]), "hi": float(rise[2])}
    dbox = {"lo": float(dur[0]), "p50": float(dur[1]), "hi": float(dur[2])}
    return {
        "name": "on",
        "rise_w": rbox,
        "fall_w": rbox,
        "dur_s": dbox,
        "plateau_w": rbox,
        "phase_share": None,
    }


def _capture(rise_ms, fall_ms, *, rise_w=2000.0, dur_s=180.0, energy_wh=100.0):
    return {
        "capture_id": "c1",
        "result": {
            "verdict": "clean",
            "rise_ms": rise_ms,
            "fall_ms": fall_ms,
            "step_w": rise_w,
            "fall_w": -rise_w,
            "plateau_w": rise_w,
            "dur_s": dur_s,
            "energy_wh": energy_wh,
            "grid_s": GRID_S,
            "threshold_w": 120.0,
        },
        "features": {
            "rise_w": rise_w,
            "fall_w": -rise_w,
            "dur_s": dur_s,
            "plateau_w": rise_w,
            "energy_wh": energy_wh,
            "hour": 7,
            "dow": 0,
            "l1_share": None,
            "l2_share": None,
            "l3_share": None,
        },
    }


def _entry(eid, kind, name, artefact):
    return {"id": eid, "kind": kind, "name": name, "artefact": artefact}


def test_kettle_capture_is_near_twin_of_learned_kettle(tmp_path) -> None:
    """A kettle capture is a near twin of a learned kettle, not of a 9000 W heater."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=22), END_MS)
    on_ms, off_ms = data.kettle_runs[-1]
    capture = _capture(on_ms, off_ms)

    kettle = _learned("sig-kettle", "Kettle")
    heater = make_switched_candidate("sig-heater", "Heater", rise_w=9000.0)

    out = similar_mod.similar(
        capture,
        [
            _entry("sig-kettle", "candidate", "Kettle", kettle),
            _entry("sig-heater", "candidate", "Heater", heater),
        ],
    )
    assert [e["id"] for e in out["top"]][0] == "sig-kettle"
    assert out["top"][0]["claims"] is True
    assert out["top"][0]["step_w"] == 2000.0
    assert out["top"][0]["dur_s"] == 180.0
    assert out["near_twin"] is True
    assert out["twin"] == {"id": "sig-kettle", "kind": "candidate", "name": "Kettle"}

    alone = similar_mod.similar(
        capture, [_entry("sig-heater", "candidate", "Heater", heater)]
    )
    assert alone["near_twin"] is False
    assert alone["twin"] is None


def test_similar_skips_program_and_orders_stably() -> None:
    """Program artefacts never appear; equal scores order by name then id, capped at top_k."""
    capture = _capture(END_MS, END_MS + 180_000)
    program = {**_learned("p1", "Program"), "kind": "program"}
    pool = [
        _entry("p1", "candidate", "Program", program),
        _entry("s1", "candidate", "Same", _learned("s1", "Same")),
        _entry("s0", "candidate", "Same", _learned("s0", "Same")),
        _entry("aaa", "candidate", "Alpha", _learned("aaa", "Alpha")),
        _entry("zzz", "candidate", "Zulu", _learned("zzz", "Zulu")),
        _entry("mmm", "candidate", "Mike", _learned("mmm", "Mike")),
    ]
    out = similar_mod.similar(capture, pool, config=CaptureConfig(similar_top_k=4))
    assert all(e["kind"] == "candidate" for e in out["top"])
    assert [e["id"] for e in out["top"]] == ["aaa", "mmm", "s0", "s1"]
    assert "p1" not in [e["id"] for e in out["top"]]


def test_similar_uses_d02_tolerance_both_ways() -> None:
    """Claims fire when the artefact box contains the point OR its p50s sit in the capture box."""
    capture = _capture(END_MS, END_MS + 180_000)

    # The box misses the capture point, but the p50s sit inside the capture's D-02 box.
    tolerance = _artefact_dict("tol", "Tol", _box((2100, 2150, 2200), (200, 205, 210)))
    out = similar_mod.similar(capture, [_entry("tol", "candidate", "Tol", tolerance)])
    assert out["top"][0]["claims"] is True

    # The box contains the capture point (replay claims) though the p50 dur is far outside.
    contains = _artefact_dict("wide", "Wide", _box((1900, 2000, 2100), (180, 1000, 2000)))
    out2 = similar_mod.similar(capture, [_entry("wide", "candidate", "Wide", contains)])
    assert out2["top"][0]["claims"] is True


def test_near_twin_from_full_pool() -> None:
    """near_twin and twin read the full scored pool, not the displayed top list (D-11)."""
    capture = _capture(END_MS, END_MS + 180_000)

    # Three artefacts whose nearest state sits just outside EPS (d ~ 1.2, no claim).
    far1 = _entry(
        "f1", "candidate", "Far1", _artefact_dict("f1", "Far1", _box((2200, 2000, 2400), (270, 274.3, 280)))
    )
    far2 = _entry(
        "f2", "candidate", "Far2", _artefact_dict("f2", "Far2", _box((2200, 2000, 2400), (270, 274.3, 280)))
    )
    far3 = _entry(
        "f3", "candidate", "Far3", _artefact_dict("f3", "Far3", _box((2200, 2000, 2400), (270, 274.3, 280)))
    )
    # A wide-box artefact that claims the capture but scores far lower.
    wide = _entry(
        "wide", "candidate", "Wide", _artefact_dict("wide", "Wide", _box((1500, 2000, 2500), (100, 500, 600)))
    )

    out = similar_mod.similar(capture, [far1, far2, far3, wide])
    assert [e["id"] for e in out["top"]] == ["f1", "f2", "f3"]
    assert all(e["claims"] is False for e in out["top"])
    assert out["near_twin"] is True
    assert out["twin"] == {"id": "wide", "kind": "candidate", "name": "Wide"}

    # Without the claiming artefact, no twin remains.
    out2 = similar_mod.similar(capture, [far1, far2, far3])
    assert out2["near_twin"] is False
    assert out2["twin"] is None
