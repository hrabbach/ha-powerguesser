"""Unit tests for episode detection, sessions, RLE, and worker integration."""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pytest

from custom_components.powerguesser.engine.buffer import BufferDB
from custom_components.powerguesser.engine.intervals import EventType
from custom_components.powerguesser.engine.learning.episodes import (
    compute_features,
    detect_edges,
    extract_sessions,
    pair_switched,
    suppress_alignment_artefacts,
)
from custom_components.powerguesser.engine.learning.series import ResidualSeries, load_residual
from custom_components.powerguesser.engine.learning.worker import extract, load_episodes
from tests.fixtures.learning.synth import (
    ENTITY_GRID,
    ENTITY_L1,
    ENTITY_L2,
    ENTITY_L3,
    ENTITY_WASHER,
    SynthSpec,
    build_buffer,
    utc_day_bounds,
)


def test_split_step_merges_to_single_edge() -> None:
    """A 2000 W step split across two samples (+900 then +1100) yields one +2000 W edge."""
    # Base 250 W, then 1150 W (+900), then 2250 W (+1100), then flat 2250 W
    r = np.array([250.0, 1150.0, 2250.0, 2250.0, 2250.0], dtype=np.float64)
    idx, delta = detect_edges(r, threshold=120.0)

    assert len(idx) == 1
    assert idx[0] == 0
    assert delta[0] == 2000.0


def test_alignment_artefact_suppression() -> None:
    """One-sample +400 W excursion is dropped when short_event is disabled, kept when enabled."""
    # 250 W -> 650 W (+400) -> 250 W (-400) -> 250 W
    # Edge indices: idx 0 (+400), idx 1 (-400)
    idx = np.array([0, 1], dtype=np.int64)
    delta = np.array([400.0, -400.0], dtype=np.float64)

    # When short_event is disabled (not in enabled_types)
    clean_idx, clean_delta = suppress_alignment_artefacts(
        idx, delta, enabled_event_types=["step", "cycle_shape"]
    )
    assert len(clean_idx) == 0
    assert len(clean_delta) == 0

    # When short_event is enabled
    kept_idx, kept_delta = suppress_alignment_artefacts(
        idx, delta, enabled_event_types=["short_event", "step", "cycle_shape"]
    )
    assert len(kept_idx) == 2
    np.testing.assert_array_equal(kept_idx, idx)
    np.testing.assert_array_equal(kept_delta, delta)


def test_no_episode_or_session_spans_nan_gap() -> None:
    """Episodes and sessions never pair or span across a NaN sample."""
    # Switched rise before NaN, fall after NaN
    # idx 0: 250 -> 2250 (+2000), idx 1: NaN, idx 2: 2250, idx 3: 250 (-2000)
    r = np.array([250.0, 2250.0, np.nan, 2250.0, 250.0], dtype=np.float64)
    idx = np.array([0, 3], dtype=np.int64)
    delta = np.array([2000.0, -2000.0], dtype=np.float64)

    table = pair_switched(r, idx, delta, step_s=30.0)
    if isinstance(table, tuple):
        table = table[0]
    assert len(table["rise_w"]) == 0

    # Sessions across NaN
    r_sess = np.array([1000.0, 1000.0, np.nan, 1000.0, 1000.0], dtype=np.float64)
    sess_table, rle_list = extract_sessions(r_sess, step_s=30.0, threshold=120.0)
    for s_start, s_end in zip(sess_table["start_i"], sess_table["end_i"], strict=True):
        assert not np.isnan(r_sess[s_start : s_end + 1]).any()


def test_oven_cycle_shape_session_rle() -> None:
    """Thermostat cycling (oven) yields a session with <= 60 segments, 10 W multiples, only when enabled."""
    cycle = np.tile([2200.0] * 4 + [250.0] * 4, 8)
    r = np.concatenate([[250.0] * 10, cycle, [250.0] * 10])

    sess_disabled, _ = extract_sessions(r, step_s=30.0, threshold=120.0, enabled=False)
    assert len(sess_disabled["dur_s"]) == 0

    sess, rle_list = extract_sessions(r, step_s=30.0, threshold=120.0, enabled=True)
    assert len(sess["dur_s"]) >= 1
    assert sess["kind"][0] == 1

    for rle in rle_list:
        assert len(rle) <= 60
        for level_w, n_samples in rle:
            assert level_w % 10 == 0


def test_phase_shares_on_episodes() -> None:
    """With phase channels kettle on L1 gets l1 ~ 1.0, l2/l3 ~ 0; without phases shares are NaN."""
    # Fake episodes table with 1 episode at start_i = 10
    table = {
        "start_i": np.array([10], dtype=np.int64),
        "end_i": np.array([16], dtype=np.int64),
        "rise_w": np.array([2000.0], dtype=np.float64),
        "fall_w": np.array([-2000.0], dtype=np.float64),
        "dur_s": np.array([180.0], dtype=np.float64),
        "plateau_w": np.array([2000.0], dtype=np.float64),
        "energy_wh": np.array([100.0], dtype=np.float64),
        "kind": np.array([0], dtype=np.int32),
    }
    day_bounds_ms = [0, 86_400_000]

    # Without phases
    feat_no_phase = compute_features(table, 0, 30_000, day_bounds_ms, 0, {})
    assert np.isnan(feat_no_phase["l1_share"][0])
    assert np.isnan(feat_no_phase["l2_share"][0])
    assert np.isnan(feat_no_phase["l3_share"][0])

    # With phases: L1 has 2000 W rise, L2 and L3 flat
    l1 = np.zeros(20, dtype=np.float64)
    l1[11:] = 2000.0
    l2 = np.zeros(20, dtype=np.float64)
    l3 = np.zeros(20, dtype=np.float64)
    phases = {"l1": l1, "l2": l2, "l3": l3}

    feat_phase = compute_features(table, 0, 30_000, day_bounds_ms, 0, phases)
    assert feat_phase["l1_share"][0] == pytest.approx(1.0, abs=0.01)
    assert feat_phase["l2_share"][0] == pytest.approx(0.0, abs=0.01)
    assert feat_phase["l3_share"][0] == pytest.approx(0.0, abs=0.01)


def test_plug_channel_lifetime_rule(tmp_path: Path) -> None:
    """Metered plug channel counts as 0 W before its first row (NULL rules: test_series_residual)."""
    buf_path = tmp_path / "buf.sqlite"
    end_ms = 1793491200000
    spec = SynthSpec(days=3, washer=True, plug_start_day=1)
    data = build_buffer(buf_path, spec, end_ms)

    washer_ch = data.channel_ids[ENTITY_WASHER]
    bounds = utc_day_bounds(end_ms, 3)

    series = load_residual(
        str(buf_path),
        bounds[0],
        bounds[-1],
        30_000,
        metered_channels=[washer_ch],
        phase_channels={},
    )
    step_ms = 30_000
    day0_samples = int(86_400_000 / step_ms)
    assert np.all(np.isfinite(series.residual[:day0_samples]))
    assert washer_ch in series.plugs


def test_train_only_digest_kettle_count_and_npz_roundtrip(tmp_path: Path) -> None:
    """Digest uses train days only; kettle n matches train count; npz roundtrips with allow_pickle=False."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buffer_path = tmp_path / "buffer.sqlite"
    stage_dir = tmp_path / "stage"
    synth_data = build_buffer(buffer_path, SynthSpec(days=days), end_ms)

    job = {
        "schema": 1,
        "buffer_path": str(buffer_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    result = extract(job)
    digest = result["digest"]
    meta = result["meta"]

    train_kettle_runs = [
        r for r in synth_data.kettle_runs if day_bounds_ms[0] <= r[0] < day_bounds_ms[5]
    ]

    clusters = digest["clusters"]
    kettle = next(
        c
        for c in clusters
        if c["kind"] == "switched" and 1800 <= c["rise_w"]["p50"] <= 2200
    )
    expected_n = len(train_kettle_runs)
    assert abs(kettle["n"] - expected_n) <= max(2, int(0.15 * expected_n))

    npz_path = stage_dir / "episodes.npz"
    ep_data = load_episodes(npz_path)
    for col_name, arr in ep_data.items():
        assert arr.dtype != object, f"Column {col_name} has object dtype"

    days_meta = meta["days"]
    assert "train" in days_meta
    assert "val" in days_meta
    assert "holdout" in days_meta
    assert "dropped" in days_meta


def test_dropped_days_and_failure_on_too_few_days(tmp_path: Path) -> None:
    """Mostly-NULL days appear under dropped; extract fails when fewer than 2 data days."""
    buf_path = tmp_path / "buf_sparse.sqlite"
    end_ms = 1793491200000
    bounds = utc_day_bounds(end_ms, 2)

    synth_data = build_buffer(buf_path, SynthSpec(days=2), end_ms)
    grid_ch = synth_data.channel_ids[ENTITY_GRID]

    # Set grid samples to NULL so resample_zoh sees all NaNs for both days
    db = BufferDB(str(buf_path))
    db.open()
    db._conn.execute("UPDATE samples SET w = NULL WHERE ch = ?", (grid_ch,))
    db._conn.commit()
    db.close_clean()

    stage_dir = tmp_path / "stage_fail"
    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": bounds,
        "first_weekday": 0,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    with pytest.raises(
        ValueError,
        match=r"need at least 2 data days, have 0 \(2 dropped below 50 % coverage\)",
    ):
        extract(job)
