"""Tests for digest format, plug reference examples, determinism, and privacy."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(AmberSnapshotExtension)

from custom_components.powerguesser.engine.learning import digest as digest_module
from custom_components.powerguesser.engine.learning.episodes import plug_sessions
from custom_components.powerguesser.engine.learning.worker import extract
from tests.fixtures.learning.synth import (
    ENTITY_GRID,
    ENTITY_WASHER,
    SynthSpec,
    build_buffer,
    utc_day_bounds,
)


def test_plug_sessions_and_digest_reference_lines(tmp_path: Path) -> None:
    """Washer runs appear as p1 line; washer energy excluded from cN residual clusters."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_washer.sqlite"
    stage_dir = tmp_path / "stage_washer"
    synth_data = build_buffer(buf_path, SynthSpec(days=days, washer=True), end_ms)

    washer_ch = synth_data.channel_ids[ENTITY_WASHER]

    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [washer_ch],
        "phase_channels": {},
    }

    res = extract(job)
    digest = res["digest"]
    meta = res["meta"]

    # Must contain plug reference lines
    assert "plugs" in digest
    plugs = digest["plugs"]
    assert len(plugs) == 1
    p1 = plugs[0]
    assert p1["id"] == "p1"
    assert p1["n"] >= 1
    assert p1["peak_w"] >= 1800  # Washer heat step is 2000 W
    assert len(p1["hour"]) == 24
    assert len(p1["dow"]) == 7

    # Meta records mapping of p1 to channel id
    assert meta["plugs"] == {"p1": washer_ch}


def test_hour_and_dow_histograms_on_all_clusters(tmp_path: Path) -> None:
    """Every cluster carries 24-int hour and 7-int dow histograms."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_hist.sqlite"
    stage_dir = tmp_path / "stage_hist"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
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

    res = extract(job)
    clusters = res["digest"]["clusters"]
    assert len(clusters) > 0
    for c in clusters:
        assert len(c["hour"]) == 24
        assert all(isinstance(v, int) for v in c["hour"])
        assert len(c["dow"]) == 7
        assert all(isinstance(v, int) for v in c["dow"])


def test_oven_sessions_form_session_cluster_with_exemplars(tmp_path: Path) -> None:
    """Train sessions are clustered too (CR-03): the oven appears as a session cluster."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_oven.sqlite"
    stage_dir = tmp_path / "stage_oven"
    build_buffer(buf_path, SynthSpec(days=days, oven=True), end_ms)

    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
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

    res = extract(job)
    clusters = res["digest"]["clusters"]
    sessions = [c for c in clusters if c["kind"] == "session"]
    assert sessions, clusters
    # The oven: 30-40 min thermostat cycling around 2200 W
    oven = [c for c in sessions if c["dur_min"]["p50"] >= 20]
    assert oven, sessions
    assert oven[0]["n"] >= digest_module.MIN_SAMPLES
    assert 1 <= len(oven[0]["exemplars"]) <= 3
    assert all("+2200x" in ex for ex in oven[0]["exemplars"])
    for c in clusters:
        if c["kind"] == "switched":
            assert "exemplars" not in c

    # Session cluster labels never mix episode kinds, and only train episodes are labelled
    episodes = res["episodes"]
    for cid, raw in res["meta"]["cluster_map"].items():
        members = np.where(episodes["cluster"] == raw)[0]
        kinds = set(int(k) for k in episodes["kind"][members])
        assert len(kinds) == 1, (cid, kinds)
        assert set(int(b) for b in episodes["block"][members]) == {0}


def test_co_occurrence_detected() -> None:
    """Clusters whose episodes start within 2 samples list co-occurrence share."""
    # Synthetic cluster episodes where c01 and c02 always start at the same time
    starts = np.array([10, 10, 50, 50, 100, 100], dtype=np.int64)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    id_map = {0: "c01", 1: "c02"}

    conc = digest_module.compute_co_occurrence(starts, labels, id_map)
    # c01 should co-occur 100% (1.0) with c02
    assert "c01" in conc
    assert any(entry["id"] == "c02" and entry["share"] == 1.0 for entry in conc["c01"])


def test_max_clusters_cap_and_other_summary() -> None:
    """When >40 clusters exist, only top 40 listed and remainder summarised as other."""
    assert digest_module.MAX_CLUSTERS == 40


def test_digest_determinism(tmp_path: Path) -> None:
    """Two extracts of the same buffer produce byte-identical digest.json and matching sha256."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_det.sqlite"
    stage_dir_1 = tmp_path / "stage_1"
    stage_dir_2 = tmp_path / "stage_2"
    build_buffer(buf_path, SynthSpec(days=days), end_ms)

    job1 = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir_1),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    job2 = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir_2),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }

    res1 = extract(job1)
    res2 = extract(job2)

    bytes1 = (stage_dir_1 / "digest.json").read_bytes()
    bytes2 = (stage_dir_2 / "digest.json").read_bytes()

    assert bytes1 == bytes2
    assert res1["meta"]["input_digest"] == res2["meta"]["input_digest"]


def test_digest_privacy_contract(tmp_path: Path) -> None:
    """Canonical digest contains no 13-digit timestamps, sensor. entity ids, ISO dates, or forbidden keys."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_priv.sqlite"
    stage_dir = tmp_path / "stage_priv"
    synth_data = build_buffer(buf_path, SynthSpec(days=days, washer=True), end_ms)
    washer_ch = synth_data.channel_ids[ENTITY_WASHER]

    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [washer_ch],
        "phase_channels": {},
    }

    extract(job)
    raw_text = (stage_dir / "digest.json").read_text(encoding="utf-8")
    parsed = json.loads(raw_text)

    # 1. No integer >= 10^12 (epoch ms timestamp)
    int_pattern = re.compile(r"\b\d{12,}\b")
    assert not int_pattern.findall(raw_text), "Found 12+ digit integer in digest"

    # 2. No sensor. substring
    assert "sensor." not in raw_text

    # 3. No ISO date/time string (YYYY-MM-DD or HH:MM:SS)
    iso_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}")
    assert not iso_pattern.findall(raw_text), "Found ISO date/time in digest"

    # 4. No forbidden keys: entity_id, name, ts
    forbidden_keys = {"entity_id", "name", "ts"}

    def _check_keys(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k not in forbidden_keys, f"Forbidden key '{k}' found in digest"
                _check_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                _check_keys(item)

    _check_keys(parsed)


def test_digest_snapshot(tmp_path: Path, snapshot) -> None:
    """Digest of SynthSpec(days=22, oven=True, washer=True, spikes=True, gaps=True) matches snapshot."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 22
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buf_path = tmp_path / "buf_snap.sqlite"
    stage_dir = tmp_path / "stage_snap"
    synth_data = build_buffer(
        buf_path,
        SynthSpec(days=days, oven=True, washer=True, spikes=True, gaps=True),
        end_ms,
    )
    washer_ch = synth_data.channel_ids[ENTITY_WASHER]

    job = {
        "schema": 1,
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": 6,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [washer_ch],
        "phase_channels": {},
    }

    res = extract(job)
    digest = res["digest"]

    # syrupy snapshot assertion
    assert digest == snapshot
