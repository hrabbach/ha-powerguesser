"""Tests for the offline tuning harness (scripts/tune_offline.py, Plan 06-07).

The harness is dev-only: it reads a COPY of the reference-install buffer
read-only and prints measurements for the constants that ship as conservative
initial values this phase (D-22). Every case here runs on a synthetic buffer
and asserts hand-computable expectations, so a broken decimation or a silent
no-op cannot pass on merely in-range numbers.

Script loading follows tests/scripts/test_llm_spike.py (importlib by path).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from custom_components.powerguesser.engine.buffer import (
    KIND_GROSS,
    KIND_METERED,
    ROLE_METERED,
    BufferDB,
    ChannelSpec,
    GrossLayoutEntry,
)
from custom_components.powerguesser.engine.learning.digest import dbscan, session_features
from custom_components.powerguesser.engine.learning.series import load_residual
from custom_components.powerguesser.engine.learning.worker import build_episode_table
from tests.fixtures.learning.synth import (
    ENTITY_WASHER,
    SynthSpec,
    build_buffer,
)
from tests.fixtures.programs import program_artefact

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "tune_offline.py"

DAY_MS = 86_400_000
GRID_S = 30
EVENT_TYPES = ["step", "cycle_shape", "continuous"]
END_MS = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
GRID_ENTITY = "sensor.grid_power"
WASHER_SIG_ID = "sig-washer"
# The synthetic washer program: 2000 W x 30, 200 W x 80, 500 W x 10 samples.
WASHER_STAGES = [(2000, 30), (200, 80), (500, 10)]
WASHER_RUN_SAMPLES = 120
BASE_W = 250.0


def _load_harness():
    """Load scripts/tune_offline.py as a module via spec_from_file_location."""
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Script {_SCRIPT_PATH} does not exist yet")
    spec = importlib.util.spec_from_file_location("tune_offline", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec from {_SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(mod, argv: list[str], capsys) -> dict[str, Any]:
    """Run one subcommand and parse the single JSON document it prints."""
    rc = mod.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    return json.loads(out)


def _build_aligned_grid_buffer(
    path: Path,
    *,
    day_start_ms: int,
    run_start_i: int,
    run_samples: int = 20,
    run_w: float = 2000.0,
) -> None:
    """One UTC day of exact 30 s samples with a single rectangular run.

    The buffer holds the day's 2880 samples plus one sample exactly at the
    next UTC midnight, so the harness window ('the last whole days before the
    newest sample') is exactly this day. ``run_start_i`` is an even index, so
    both 60 s decimations see clean steps.
    """
    db = BufferDB(str(path))
    db.open()
    id_map = db.ensure_channels(
        [ChannelSpec(GRID_ENTITY, "grid", KIND_GROSS)],
        [GrossLayoutEntry("grid", "direct", (GRID_ENTITY,))],
    )
    ch = id_map[(GRID_ENTITY, "grid")]
    n = 2881
    values = np.zeros(n, dtype=np.float64)
    values[run_start_i : run_start_i + run_samples] = run_w
    rows = [(day_start_ms + i * 30_000, int(values[i]), ch) for i in range(n)]
    for i in range(0, len(rows), 20_000):
        db.write_batch([(ts, c, w) for ts, w, c in rows[i : i + 20_000]])
    db.close_clean()


def _program_run_values(n_samples: int, starts: Sequence[int]) -> np.ndarray:
    """The washer program's sample values over ``n_samples`` (0 W between runs)."""
    template = np.concatenate(
        [np.full(count, float(level)) for level, count in WASHER_STAGES]
    )
    values = np.zeros(n_samples, dtype=np.float64)
    for start_i in starts:
        if start_i >= n_samples:
            continue
        end = min(start_i + len(template), n_samples)
        values[start_i:end] = template[: end - start_i]
    return values


def _build_program_buffer(
    path: Path,
    *,
    first_ts_ms: int,
    n_samples: int,
    run_starts: Sequence[int],
) -> None:
    """Exact 30 s grid + plug channels with washer runs at ``run_starts`` (0-based).

    The gross grid is the base load plus the plug's value (no wet noise), and
    the washer plug channel carries the program levels, so the residual the
    harness reads is a clean multi-stage washer run on a flat base.
    """
    db = BufferDB(str(path))
    db.open()
    id_map = db.ensure_channels(
        [
            ChannelSpec(GRID_ENTITY, "grid", KIND_GROSS),
            ChannelSpec(ENTITY_WASHER, ROLE_METERED, KIND_METERED),
        ],
        [GrossLayoutEntry("grid", "direct", (GRID_ENTITY,))],
    )
    grid_ch = id_map[(GRID_ENTITY, "grid")]
    washer_ch = id_map[(ENTITY_WASHER, ROLE_METERED)]
    washer = _program_run_values(n_samples, run_starts)
    rows: list[tuple[int, int, int]] = []
    for i in range(n_samples):
        rows.append((first_ts_ms + i * 30_000, grid_ch, int(BASE_W + washer[i])))
        rows.append((first_ts_ms + i * 30_000, washer_ch, int(washer[i])))
    rows.sort(key=lambda r: r[0])
    for i in range(0, len(rows), 20_000):
        db.write_batch(rows[i : i + 20_000])
    db.close_clean()


def _write_artefacts(path: Path, artefacts: Sequence[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(list(artefacts)), encoding="utf-8")
    return path


def _washer_artefact() -> dict[str, Any]:
    return program_artefact(WASHER_SIG_ID, WASHER_STAGES)


def _stage1_dur_lo() -> float:
    """Stage-1 dur_s.lo of the washer artefact: the D-03 lag cap at fraction 1.0."""
    return float(_washer_artefact()["session"]["stages"][0]["dur_s"]["lo"])


def _expected_session_cluster_count(buf: Path, *, days: int, grid_s: int) -> int:
    """The untouched digest path's cluster count over the same session rows."""
    step_ms = grid_s * 1000
    start_ms = END_MS - days * DAY_MS
    day_bounds_ms = [start_ms + i * DAY_MS for i in range(days + 1)]
    conn_series = load_residual(str(buf), start_ms, END_MS, step_ms, [], {})
    table, _rle, _threshold = build_episode_table(
        conn_series.residual,
        conn_series.phases,
        start_ms,
        step_ms,
        day_bounds_ms,
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).weekday(),
        EVENT_TYPES,
    )
    sess_idx = np.where(table["kind"] == 1)[0]
    X = session_features({k: table[k][sess_idx] for k in ("energy_wh", "dur_s", "peak_w", "n_segments")})
    labels = dbscan(X)
    return len({int(v) for v in labels} - {-1})


# --- interval-60s (folded todo: evaluate detection at 60 s sampling) -------


def test_interval_60s_decimations_agree_on_aligned_steps(tmp_path: Path, capsys) -> None:
    """One 2000 W, 10-minute run aligned to 60 s: every variant sees 2 edges, 1 pair.

    The run occupies 20 samples of the 30 s grid (333.33 Wh) and is aligned to
    an even index, so both 60 s decimations are exact and their paired energy
    must match the 30 s figure within 1 %.
    """
    buf = tmp_path / "buf_60s.sqlite"
    day_start = END_MS - DAY_MS
    run_start_i = 100  # even -> 60 s aligned
    _build_aligned_grid_buffer(buf, day_start_ms=day_start, run_start_i=run_start_i)
    before = _sha(buf)

    payload = _run(_load_harness(), ["interval-60s", "--buffer", str(buf), "--days", "1"], capsys)

    variants = {v["variant"]: v for v in payload["variants"]}
    assert set(variants) == {"30s", "60s_every_second", "60s_mean"}
    for variant in variants.values():
        assert variant["edges"] == 2
        assert variant["pairs"] == 1
        assert 0.0 < variant["explained_share"] <= 1.0

    expected_wh = 20 * 2000.0 * GRID_S / 3600.0
    assert variants["30s"]["paired_energy_wh"] == pytest.approx(expected_wh, rel=1e-6)
    for name in ("60s_every_second", "60s_mean"):
        assert variants[name]["step_s"] == 60.0
        assert variants[name]["paired_energy_wh"] == pytest.approx(
            variants["30s"]["paired_energy_wh"], rel=0.01
        )
    assert _sha(buf) == before


# --- session-scaling (folded todo: recheck CR-03 session feature scaling) ---


def test_session_scaling_reports_nine_settings_and_matches_untouched_dbscan(
    tmp_path: Path, capsys
) -> None:
    """Nine multiplier settings over the washer buffer; 1.0 x 1.0 is the digest path.

    Sizes never exceed the session count, and the identity setting reproduces
    digest.dbscan's own cluster count over the untouched session_features of
    the same sessions, so the harness cannot silently no-op.
    """
    days = 35
    buf = tmp_path / "buf_sessions.sqlite"
    build_buffer(buf, SynthSpec(days=days, washer=True), END_MS)
    before = _sha(buf)

    payload = _run(
        _load_harness(),
        ["session-scaling", "--buffer", str(buf), "--days", str(days), "--grid-s", str(GRID_S)],
        capsys,
    )

    settings = payload["settings"]
    assert len(settings) == 9
    assert {(s["m_dur"], s["m_rise"]) for s in settings} == {
        (m_dur, m_rise) for m_dur in (0.5, 1.0, 2.0) for m_rise in (0.5, 1.0, 2.0)
    }
    n_sessions = payload["n_sessions"]
    assert n_sessions >= 1
    for setting in settings:
        assert setting["n_sessions"] == n_sessions
        assert setting["n_clusters"] == len(setting["sizes"])
        assert sum(setting["sizes"]) <= n_sessions
        assert isinstance(setting["median_spread_ln_energy"], float)
        assert isinstance(setting["median_spread_ln_duration"], float)

    identity = next(s for s in settings if s["m_dur"] == 1.0 and s["m_rise"] == 1.0)
    assert identity["n_clusters"] == _expected_session_cluster_count(
        buf, days=days, grid_s=GRID_S
    )
    assert _sha(buf) == before


# --- lag-cap (D-03 recognition lag per PROGRAM_PROMOTE_FRACTION) ------------


def test_lag_cap_reports_lag_and_energy_per_fraction(tmp_path: Path, capsys) -> None:
    """Seven daily washer runs: lag, misses and energy per promote fraction (D-03).

    The stage-1 template is 30 samples (900 s) long, so a fraction of 1.0
    promotes after the D-03 cap (stage-1 dur_s.lo = 720 s) plus at most one
    grid step, while 0.5 promotes earlier and 1.5 (36 samples = 1080 s) can
    never complete inside stage 1. One of the seven rises is smeared across two
    30 s samples by the fixture's 1 s-to-grid averaging, so it never matches
    the stage-1 entry box and the run is missed — the harness reports that
    rather than hiding it. The shipped PROGRAM_PROMOTE_FRACTION is never
    touched.
    """
    days = 7
    buf = tmp_path / "buf_lag.sqlite"
    build_buffer(buf, SynthSpec(days=days, washer=True, washer_daily=True), END_MS)
    before = _sha(buf)
    artefacts = _write_artefacts(tmp_path / "washer.json", [_washer_artefact()])

    payload = _run(
        _load_harness(),
        [
            "lag-cap",
            "--buffer",
            str(buf),
            "--days",
            str(days),
            "--artefacts",
            str(artefacts),
            "--plug-entity",
            ENTITY_WASHER,
        ],
        capsys,
    )

    assert payload["plug_entity"] == ENTITY_WASHER
    assert payload["program_keys"] == [WASHER_SIG_ID]
    assert payload["stage1_dur_lo_s"] == _stage1_dur_lo()
    fractions = {f["fraction"]: f for f in payload["fractions"]}
    assert set(fractions) == {0.5, 1.0, 1.5}
    for entry in payload["fractions"]:
        assert entry["runs_plug"] == 7
        assert entry["runs_truncated"] == 0
        assert entry["runs_detected"] + entry["runs_missed"] == entry["runs_plug"]

    # The shipped fraction: every detected run is recognised within the D-03 bound.
    assert fractions[1.0]["runs_detected"] >= 6
    assert fractions[1.0]["lag_s_median"] is not None
    assert fractions[1.0]["lag_s_max"] <= _stage1_dur_lo() + GRID_S
    # A smaller fraction cannot lag longer than a larger one on the same data.
    assert fractions[0.5]["lag_s_median"] < fractions[1.0]["lag_s_median"]
    # 1.5 needs 36 in-box samples but stage 1 only lasts 30: the cap costs every run.
    assert fractions[1.5]["runs_detected"] == 0
    assert fractions[1.5]["runs_missed"] == 7
    assert fractions[1.5]["lag_s_median"] is None
    assert fractions[1.5]["lag_s_max"] is None
    # Energy error is measured over the complete runs against the plug.
    assert fractions[1.0]["energy_error_pct"] is not None
    assert fractions[1.0]["energy_error_pct"] < 20.0

    # The harness never assigns an engine constant, and reads the buffer only.
    program_module = sys.modules.get("engine.detect.program")
    assert program_module is not None
    assert program_module.PROGRAM_PROMOTE_FRACTION == 1.0
    assert _sha(buf) == before


def test_lag_cap_detects_clean_runs_within_the_cap(tmp_path: Path, capsys) -> None:
    """Seven aligned runs on a flat base: all detected, lag inside the D-03 bound.

    The buffer the tests build themselves has no smeared rise, so the harness
    must recognise every run at the shipped fraction — the crisp form of the
    must_have bound (lag <= stage-1 dur_s.lo + one grid step).
    """
    days = 7
    window_start = END_MS - days * DAY_MS
    n_samples = days * 2880 + 1
    run_starts = [day * 2880 + 1080 for day in range(days)]  # 09:00 each day
    buf = tmp_path / "buf_clean.sqlite"
    _build_program_buffer(
        buf, first_ts_ms=window_start, n_samples=n_samples, run_starts=run_starts
    )
    before = _sha(buf)
    artefacts = _write_artefacts(tmp_path / "washer.json", [_washer_artefact()])

    payload = _run(
        _load_harness(),
        [
            "lag-cap",
            "--buffer",
            str(buf),
            "--days",
            str(days),
            "--artefacts",
            str(artefacts),
            "--plug-entity",
            ENTITY_WASHER,
            "--fractions",
            "0.5,1.0",
        ],
        capsys,
    )

    fractions = {f["fraction"]: f for f in payload["fractions"]}
    assert set(fractions) == {0.5, 1.0}
    for entry in payload["fractions"]:
        assert entry["runs_plug"] == 7
        assert entry["runs_truncated"] == 0
        assert entry["runs_detected"] == 7
        assert entry["runs_missed"] == 0
        assert entry["lag_s_max"] <= _stage1_dur_lo() + GRID_S
        assert entry["energy_error_pct"] < 5.0
    # The promotion threshold is the D-03 formula, not a shipped constant.
    assert fractions[0.5]["promote_steps"] == 12
    assert fractions[1.0]["promote_steps"] == 24
    assert fractions[0.5]["lag_s_median"] <= fractions[1.0]["lag_s_median"]
    assert _sha(buf) == before


def test_lag_cap_counts_runs_cut_by_the_window_edge_as_truncated(
    tmp_path: Path, capsys
) -> None:
    """A run still ON at the window end, or already ON at its start, is truncated.

    Six complete runs plus one that crosses the window end, and one that began
    before the window: both edge runs are counted in ``runs_truncated`` and in
    none of runs_plug / runs_detected / runs_missed or the lag statistics.
    """
    days = 7
    window_start = END_MS - days * DAY_MS
    # Samples: one hour before the window start through the final sample exactly
    # at the window end, so the newest sample's whole-day window is [start, end).
    lead_samples = 120
    n_samples = lead_samples + days * 2880 + 1
    samples_start = window_start - lead_samples * 30_000
    runs = [
        lead_samples // 2,  # ON at the window start: began before the window
        *[lead_samples + day * 2880 + 1080 for day in range(6)],  # days 0..5, 09:00
        lead_samples + 6 * 2880 + 2760,  # day 6, 23:00 -> still ON at the window end
    ]
    buf = tmp_path / "buf_lag_edges.sqlite"
    _build_program_buffer(buf, first_ts_ms=samples_start, n_samples=n_samples, run_starts=runs)
    before = _sha(buf)
    artefacts = _write_artefacts(tmp_path / "washer.json", [_washer_artefact()])

    payload = _run(
        _load_harness(),
        [
            "lag-cap",
            "--buffer",
            str(buf),
            "--days",
            str(days),
            "--artefacts",
            str(artefacts),
            "--plug-entity",
            ENTITY_WASHER,
            "--fractions",
            "1.0",
        ],
        capsys,
    )

    entry = payload["fractions"][0]
    assert entry["fraction"] == 1.0
    assert entry["runs_truncated"] == 2
    assert entry["runs_plug"] == 6
    assert entry["runs_detected"] + entry["runs_missed"] == 6
    assert entry["runs_detected"] == 6
    # The truncated runs contribute no lag at all.
    assert entry["lag_s_max"] is not None
    assert entry["lag_s_max"] <= _stage1_dur_lo() + GRID_S
    assert _sha(buf) == before


# --- capture-snap (D-09 post-ship snap measurement, Plan 08-04) -------------


def _build_snap_buffer(path: Path) -> None:
    build_buffer(path, SynthSpec(days=10), END_MS)


def test_capture_snap_reports_margins(tmp_path: Path, capsys) -> None:
    """Per-margin correct/multi/none fractions for both bands; seeded and read-only."""
    buf = tmp_path / "buf_snap.sqlite"
    _build_snap_buffer(buf)
    before = _sha(buf)

    argv = [
        "capture-snap",
        "--buffer",
        str(buf),
        "--days",
        "9",
        "--margins",
        "60,120",
        "--jitter-s",
        "60",
    ]
    payload = _run(_load_harness(), argv, capsys)

    assert set(payload["bands"]) >= {"kettle_like", "long_heavy"}
    kettle = payload["bands"]["kettle_like"]
    assert set(kettle) == {"60", "120"}
    for entry in kettle.values():
        assert set(entry) == {"n", "correct", "multi", "none"}
        assert 0.0 <= entry["correct"] <= 1.0
        assert 0.0 <= entry["multi"] <= 1.0
        assert 0.0 <= entry["none"] <= 1.0
    assert kettle["120"]["n"] >= 1
    assert kettle["120"]["correct"] >= 0.98

    # Seeded: two runs print the identical document.
    payload2 = _run(_load_harness(), argv, capsys)
    assert payload2 == payload
    assert _sha(buf) == before


def test_capture_snap_tolerance_spread(tmp_path: Path, capsys) -> None:
    """The kettle-like band carries the rise and duration spread relative to the median."""
    buf = tmp_path / "buf_snap_tol.sqlite"
    _build_snap_buffer(buf)

    payload = _run(
        _load_harness(),
        [
            "capture-snap",
            "--buffer",
            str(buf),
            "--days",
            "9",
            "--margins",
            "120",
            "--jitter-s",
            "60",
        ],
        capsys,
    )

    tolerance = payload["tolerance"]
    assert set(tolerance) == {"rise_frac", "dur_frac"}
    for key in ("rise_frac", "dur_frac"):
        assert set(tolerance[key]) == {"p5", "p50", "p95"}
        assert tolerance[key]["p50"] == pytest.approx(1.0, abs=1e-6)
        assert (
            tolerance[key]["p5"]
            <= tolerance[key]["p50"]
            <= tolerance[key]["p95"]
        )


def test_capture_snap_press_histogram(tmp_path: Path, capsys) -> None:
    """--captures adds the live press-to-edge delays; window records are ignored."""
    buf = tmp_path / "buf_snap_hist.sqlite"
    _build_snap_buffer(buf)
    captures = tmp_path / "captures.json"
    captures.write_text(
        json.dumps(
            {
                "data": {
                    "captures": [
                        {
                            "mode": "live",
                            "state": "done",
                            "result": {"press_to_rise_s": 30.0, "press_to_fall_s": -15.0},
                        },
                        {
                            "mode": "live",
                            "state": "done",
                            "result": {"press_to_rise_s": 45.0, "press_to_fall_s": 10.0},
                        },
                        {
                            "mode": "window",
                            "state": "done",
                            "result": {"press_to_rise_s": 999.0, "press_to_fall_s": 999.0},
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    payload = _run(
        _load_harness(),
        [
            "capture-snap",
            "--buffer",
            str(buf),
            "--days",
            "9",
            "--margins",
            "120",
            "--jitter-s",
            "60",
            "--captures",
            str(captures),
        ],
        capsys,
    )

    press_to_edge = payload["press_to_edge"]
    assert press_to_edge["n"] == 2
    assert press_to_edge["rise_s"] == [30.0, 45.0]
    assert press_to_edge["fall_s"] == [-15.0, 10.0]


def test_capture_snap_opens_buffer_read_only(tmp_path: Path, capsys) -> None:
    """The buffer's bytes and mtime are unchanged after the run."""
    buf = tmp_path / "buf_snap_ro.sqlite"
    _build_snap_buffer(buf)
    before_bytes = buf.read_bytes()
    before_mtime = buf.stat().st_mtime

    _run(
        _load_harness(),
        [
            "capture-snap",
            "--buffer",
            str(buf),
            "--days",
            "9",
            "--margins",
            "120",
            "--jitter-s",
            "60",
        ],
        capsys,
    )

    assert buf.read_bytes() == before_bytes
    assert buf.stat().st_mtime == before_mtime
