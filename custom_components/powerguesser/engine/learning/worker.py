"""Learning worker process orchestrator and stage execution."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

try:
    import resource
except ImportError:
    resource = None

import numpy as np

from .artefact import SignatureArtefact, detection_differs
from .compile import MIN_TRAIN_EPISODES, compile_proposal, fit_state_boxes
from .digest import (
    build_digest,
    canonical_json,
    dbscan,
    session_features,
    switched_features,
)
from .episodes import (
    PLUG_GAP_S,
    PLUG_ON_W,
    compute_features,
    detect_edges,
    edge_threshold,
    extract_sessions,
    pair_switched,
    plug_sessions,
    suppress_alignment_artefacts,
)
from .identity import match_identities
from .score import accept, block_score, gate_proposal, replay, scorecard
from .series import load_residual
from .split import (
    MIN_DAY_COVERAGE,
    block_of,
    calendar_holdout_start,
    data_days,
    split_days,
)
from ..detect.online import compile_model
from ..detect.replay import replay_online, validation_error_pct
from ..validation import overlap as overlap_metrics
from ..validation.constants import (
    LINK_MIN_OVERLAP,
    MIN_USABLE_DAYS,
    OVERLAP_MIN_TICKS,
    SHADOW_HOLDOUT_DAYS,
    SHADOW_MAX_AGE_DAYS,
)
from ..validation.link import assign_links, jaccard, on_mask
from ..validation.metrics import baseline, energy_error_pct, mae, nde
from ..validation.report import assemble_report, assemble_sessions

# DET-02 (this plan): the shadow artefact kinds the validate stage MEASURES.
# A switched and a program shadow artefact are replayed together over the
# gross residual and judged by the unchanged P5 bars (D-24); a link of any
# other kind is reported as an unsupported_kind refusal (D-03), never dropped.
# assemble_report applies the same vocabulary on its side.
MEASURED_KINDS: tuple[str, ...] = ("switched", "program")

STAGES: tuple[str, ...] = (
    "extract",
    "compile_score",
    "rescore",
    "refit",
    "validate",
    "validate_sessions",
    "shadow_link",
)

# Plan 06-09 (REV-04): synthetic plug-line keys start at this base so they can
# never collide with real buffer channel ids (SQLite rowids of a small channels
# table). learning.py builds the same keys and _plug_names maps them to labels.
SESSION_KEY_BASE = 1_000_000


def _write_atomic(target_path: Path, data: bytes) -> None:
    """Write data to target_path atomically using a temporary file in the same directory.

    The temp name carries the pid, so two workers writing the same stage dir never
    share (and steal) one temp file (WR-08).
    """
    tmp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def _write_atomic_npz(target_path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write numpy arrays to target_path atomically using a temporary file in the same directory."""
    tmp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def residual_energy_wh(
    r: np.ndarray,
    start_ms: int,
    step_ms: int,
    day_bounds_ms: Sequence[int],
    days: Sequence[int],
) -> float:
    """Positive residual energy (Wh) over the samples of the given whole days.

    This is the digest's residual definition and the denominator of every candidate share
    and of S, so "% of residual energy" means what the digest means (WR-12).
    """
    if len(r) == 0 or not days:
        return 0.0
    day_idx = np.searchsorted(
        np.array(day_bounds_ms, dtype=np.int64),
        start_ms + np.arange(len(r), dtype=np.int64) * step_ms,
        side="right",
    ) - 1
    sel = r[np.isin(day_idx, list(days))]
    sel = sel[np.isfinite(sel)]
    return float(np.sum(np.maximum(sel, 0.0)) * (step_ms / 1000.0) / 3600.0)


def load_episodes(path: str | Path) -> dict[str, np.ndarray]:
    """Load episodes table and metadata from npz file without pickling."""
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_episode_table(
    r: np.ndarray,
    phases: Mapping[str, np.ndarray],
    start_ms: int,
    step_ms: int,
    day_bounds_ms: Sequence[int],
    first_weekday: int,
    enabled_event_types: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[list[list[int]]], float]:
    """Build the episode table shared by extract, rescore (and refit in Plan 04-12).

    The edges -> alignment suppression -> pair_switched -> extract_sessions ->
    combine -> compute_features sequence, moved as-is from the two stages that
    duplicated it; block assignment deliberately stays in each caller (extract:
    split_days/block_of; rescore: its own day_indices/valid_set rule). Returns
    (table without "block", all_rle, threshold_w). The returned table transiently
    carries the session-only clustering columns peak_w and n_segments (switched
    rows hold zeros); each caller pops them, so the published episodes table is
    unchanged.
    """
    step_s = step_ms / 1000.0

    finite_d = np.diff(r[np.isfinite(r)])
    threshold = edge_threshold(finite_d)
    idx, delta = detect_edges(r, threshold)
    idx, delta = suppress_alignment_artefacts(
        idx, delta, enabled_event_types=enabled_event_types
    )
    switched_table = pair_switched(r, idx, delta, step_s)

    sess_table, sess_rle = extract_sessions(
        r, step_s, threshold, enabled_event_types=enabled_event_types
    )

    combined: dict[str, list[float | int]] = {
        "start_i": list(switched_table["start_i"]) + list(sess_table["start_i"]),
        "end_i": list(switched_table["end_i"]) + list(sess_table["end_i"]),
        "rise_w": list(switched_table["rise_w"]) + list(sess_table["rise_w"]),
        "fall_w": list(switched_table["fall_w"]) + list(sess_table["fall_w"]),
        "dur_s": list(switched_table["dur_s"]) + list(sess_table["dur_s"]),
        "plateau_w": list(switched_table["plateau_w"]) + list(sess_table["plateau_w"]),
        "energy_wh": list(switched_table["energy_wh"]) + list(sess_table["energy_wh"]),
        "kind": list(switched_table["kind"]) + list(sess_table["kind"]),
        # Session-only clustering features (extract pops these two columns
        # right after the call; switched rows carry zeros)
        "peak_w": [0.0] * len(switched_table["start_i"]) + list(sess_table["peak_w"]),
        "n_segments": [0] * len(switched_table["start_i"]) + list(sess_table["n_segments"]),
    }

    # Combined RLE list: switched episodes have []
    n_switched = len(switched_table["start_i"])
    all_rle: list[list[list[int]]] = [[] for _ in range(n_switched)] + sess_rle

    table = {
        k: np.array(v, dtype=np.int64 if k in ("start_i", "end_i") else (np.int32 if k == "kind" else np.float64))
        for k, v in combined.items()
    }

    # Add hour, dow, phase shares
    table = compute_features(
        table,
        start_ms,
        step_ms,
        day_bounds_ms,
        first_weekday,
        phases,
    )

    return table, all_rle, threshold


def extract(job: Mapping) -> dict:
    """Execute extract stage: load residual, detect episodes, cluster, and save digest."""
    buffer_path = job["buffer_path"]
    stage_dir = Path(job["stage_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = job["day_bounds_ms"]
    first_weekday = int(job.get("first_weekday", 0))
    holdout_days = int(job.get("holdout_days", 0))
    holdout_calendar_days = int(job.get("holdout_calendar_days", 0))
    train_fraction = float(job.get("train_fraction", 0.7))
    enabled_event_types = job.get("enabled_event_types", ["step", "cycle_shape", "continuous"])
    metered_channels = job.get("metered_channels", [])
    phase_channels = job.get("phase_channels", {})
    # Plan 06-09 (REV-04, D-25/D-12): the plug sessions overlapping this window.
    # Every session plug (candidate and device targets) is subtracted only
    # inside its own window; only a candidate-target window becomes a labelled
    # plug reference line (D-12).
    session_windows_raw = job.get("session_windows", []) or []
    window_map: dict[int, list[tuple[int, int]]] = {}
    window_by_key: dict[int, dict] = {}
    for window in session_windows_raw:
        if not isinstance(window, Mapping) or window.get("channel") is None:
            continue
        ch = int(window["channel"])
        key = int(window.get("key", 0) or 0)
        window_map.setdefault(ch, []).append(
            (int(window.get("start_ms", 0) or 0), int(window.get("end_ms", 0) or 0))
        )
        window_by_key[key] = dict(window)

    start_ms = day_bounds_ms[0]
    end_ms = day_bounds_ms[-1]
    step_ms = grid_s * 1000

    series = load_residual(
        buffer_path,
        start_ms,
        end_ms,
        step_ms,
        metered_channels,
        phase_channels,
        session_windows=window_map,
    )
    r = series.residual
    step_s = float(grid_s)

    # Day sufficiency and train/val/holdout split
    total_days = len(day_bounds_ms) - 1
    valid_days = data_days(r, start_ms, step_ms, day_bounds_ms)
    n_dropped = total_days - len(valid_days)

    # D-12/RESEARCH Pitfall 5: the shadow extract counts its own valid days.
    # A shortfall the HA usable-days pre-filter missed is a normal result,
    # never the ValueError split_days raises below 2 remaining days; nothing
    # is written on this path.
    if holdout_calendar_days > 0 and len(valid_days) < MIN_USABLE_DAYS:
        return {
            "stage_dir": str(stage_dir),
            "insufficient_data": True,
        }

    if holdout_calendar_days > 0:
        split = split_days(
            valid_days,
            train_fraction=train_fraction,
            holdout_from_day=calendar_holdout_start(total_days, holdout_calendar_days),
            n_dropped=n_dropped,
        )
    else:
        split = split_days(
            valid_days,
            train_fraction=train_fraction,
            holdout_days=holdout_days,
            n_dropped=n_dropped,
        )

    # Episode table through the shared helper (block assignment below stays here).
    # The helper ferries the session-only clustering columns (peak_w,
    # n_segments) out of extract_sessions; the published episodes table carries
    # neither, so pop them before anything downstream. They keep the combined
    # table's row alignment (switched rows first, sessions after), so index
    # them with the same global row indices as the rest of the table.
    table, all_rle, threshold = build_episode_table(
        r,
        series.phases,
        start_ms,
        step_ms,
        day_bounds_ms,
        first_weekday,
        enabled_event_types,
    )
    sess_peak_w = table.pop("peak_w").astype(np.float64)
    sess_n_segments = table.pop("n_segments").astype(np.int64)

    # Tag each episode with block id (0=train, 1=val, 2=holdout, -1=none)
    start_ts_arr = start_ms + table["start_i"] * step_ms
    blocks = block_of(start_ts_arr, day_bounds_ms, split)
    table["block"] = blocks

    # Pre-cluster TRAIN block episodes only
    train_mask = blocks == 0
    train_indices = np.where(train_mask)[0]
    n_total = len(table["start_i"])
    cluster_labels = np.full(n_total, -1, dtype=np.int32)

    train_days_count = len(split.train)
    # Residual energy per block: the digest's residual and the share denominator (WR-12)
    block_residual_wh = {
        "train": residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, split.train),
        "val": residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, split.val),
        "holdout": residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, split.holdout),
    }
    train_residual_kwh = block_residual_wh["train"] / 1000.0

    # Extract plug sessions for labelled reference lines (LRN-02)
    plugs_data: dict[int, dict[str, object]] = {}
    for m_ch in metered_channels:
        if m_ch in series.plugs:
            p_series = series.plugs[m_ch]
            p_table, p_rle = plug_sessions(p_series, step_s)
            p_start_ts = start_ms + p_table["start_i"] * step_ms
            p_blocks = block_of(p_start_ts, day_bounds_ms, split)
            # Train block only
            train_p_idx = np.where(p_blocks == 0)[0]
            if len(train_p_idx) > 0:
                sub_p_table = {k: p_table[k][train_p_idx] for k in p_table}
                sub_p_table = compute_features(
                    sub_p_table,
                    start_ms,
                    step_ms,
                    day_bounds_ms,
                    first_weekday,
                    {},
                )
                sub_p_rle = [p_rle[k] for k in train_p_idx]
                plugs_data[m_ch] = {"sessions": sub_p_table, "rle": sub_p_rle}
            else:
                empty_p = {k: np.empty(0, dtype=p_table[k].dtype) for k in p_table}
                empty_p["hour"] = np.empty(0, dtype=np.int32)
                empty_p["dow"] = np.empty(0, dtype=np.int32)
                plugs_data[m_ch] = {"sessions": empty_p, "rle": []}

    # Plan 06-09 (REV-04, D-12): a candidate-target session window becomes a
    # labelled plug reference line, built from the windowed session series like
    # the metered plugs above (train-block days only). Device-target windows are
    # subtracted from the residual but never become a reference line.
    if window_by_key:
        n_points = len(series.residual)
        grid = start_ms + np.arange(n_points, dtype=np.int64) * step_ms
        for key, window in window_by_key.items():
            if str(window.get("target_kind")) != "candidate":
                continue
            ch = int(window["channel"])
            masked_sessions = series.sessions.get(ch)
            if masked_sessions is None:
                continue
            win_start = int(window.get("start_ms", 0) or 0)
            win_end = int(window.get("end_ms", 0) or 0)
            in_win = (grid >= win_start) & (grid < win_end)
            p_series = np.where(in_win, masked_sessions, np.nan)
            p_table, p_rle = plug_sessions(p_series, step_s)
            p_start_ts = start_ms + p_table["start_i"] * step_ms
            p_blocks = block_of(p_start_ts, day_bounds_ms, split)
            train_p_idx = np.where(p_blocks == 0)[0]
            if len(train_p_idx) > 0:
                sub_p_table = {k: p_table[k][train_p_idx] for k in p_table}
                sub_p_table = compute_features(
                    sub_p_table,
                    start_ms,
                    step_ms,
                    day_bounds_ms,
                    first_weekday,
                    {},
                )
                sub_p_rle = [p_rle[k] for k in train_p_idx]
                plugs_data[key] = {"sessions": sub_p_table, "rle": sub_p_rle}
            else:
                empty_p = {k: np.empty(0, dtype=p_table[k].dtype) for k in p_table}
                empty_p["hour"] = np.empty(0, dtype=np.int32)
                empty_p["dow"] = np.empty(0, dtype=np.int32)
                plugs_data[key] = {"sessions": empty_p, "rle": []}

    has_phases = bool(phase_channels and len(series.phases) > 0)

    if len(train_indices) > 0:
        # Cluster switched episodes among train
        train_switched_idx = [i for i in train_indices if table["kind"][i] == 0]
        if train_switched_idx:
            sub_table = {k: table[k][train_switched_idx] for k in table}
            X = switched_features(sub_table)
            sub_labels = dbscan(X)
            for local_i, global_i in enumerate(train_switched_idx):
                cluster_labels[global_i] = sub_labels[local_i]

        # Cluster sessions among train separately (plan 03-01: session clusters on
        # ln energy, ln duration, peak, n_segments), labels offset past the switched ones.
        # Sessions follow the switched rows in the combined table, in sess_table order.
        train_sess_idx = [i for i in train_indices if table["kind"][i] == 1]
        if train_sess_idx:
            Xs = session_features(
                {
                    "energy_wh": table["energy_wh"][train_sess_idx],
                    "dur_s": table["dur_s"][train_sess_idx],
                    "peak_w": sess_peak_w[train_sess_idx],
                    "n_segments": sess_n_segments[train_sess_idx],
                }
            )
            s_labels = dbscan(Xs)
            offset = int(cluster_labels.max()) + 1 if (cluster_labels >= 0).any() else 0
            for local_i, global_i in enumerate(train_sess_idx):
                if s_labels[local_i] >= 0:
                    cluster_labels[global_i] = int(s_labels[local_i]) + offset

        train_table = {k: table[k][train_indices] for k in table}
        train_labels = cluster_labels[train_indices]
        train_all_rle = [all_rle[k] for k in train_indices]
        digest, meta = build_digest(
            train_table,
            train_labels,
            grid_s=grid_s,
            threshold_w=threshold,
            data_days=train_days_count,
            residual_kwh=train_residual_kwh,
            phases_mapped=has_phases,
            plugs_data=plugs_data,
            all_rle=train_all_rle,
        )
    else:
        digest, meta = build_digest(
            {k: np.empty(0) for k in table},
            np.empty(0, dtype=np.int32),
            grid_s=grid_s,
            threshold_w=threshold,
            data_days=train_days_count,
            residual_kwh=train_residual_kwh,
            phases_mapped=has_phases,
            plugs_data=plugs_data,
        )

    # Encode RLE CSR-style for episodes.npz
    rle_offsets: list[int] = [0]
    rle_levels: list[int] = []
    rle_counts: list[int] = []
    for rle in all_rle:
        for lvl, cnt in rle:
            rle_levels.append(int(lvl))
            rle_counts.append(int(cnt))
        rle_offsets.append(len(rle_levels))

    rle_offset_arr = np.array(rle_offsets, dtype=np.int64)
    rle_level_w_arr = np.array(rle_levels, dtype=np.int32)
    rle_n_arr = np.array(rle_counts, dtype=np.int32)

    episodes_arrays = {
        **table,
        "cluster": cluster_labels,
        "rle_offset": rle_offset_arr,
        "rle_level_w": rle_level_w_arr,
        "rle_n": rle_n_arr,
    }

    # Meta records days split details
    dropped_days = [d for d in range(total_days) if d not in valid_days]
    meta["days"] = {
        "train": list(split.train),
        "val": list(split.val),
        "holdout": list(split.holdout),
        "dropped": dropped_days,
    }
    meta["block_residual_wh"] = block_residual_wh

    # D-16 residual cache (Plan 04-07): the gridded residual the replay helper
    # needs to compute validation_error_pct on the validation days, written
    # atomically like episodes.npz and deleted with the stage dir at the next
    # run. Phases are included only when all three are present.
    suppress_alignment = 1 if "short_event" not in enabled_event_types else 0
    residual_arrays: dict[str, np.ndarray] = {
        "r": np.asarray(r, dtype=np.float64),
        "start_ms": np.array([int(start_ms)], dtype=np.int64),
        "step_ms": np.array([int(step_ms)], dtype=np.int64),
        "day_bounds_ms": np.array([int(b) for b in day_bounds_ms], dtype=np.int64),
        "suppress_alignment": np.array([suppress_alignment], dtype=np.int8),
    }
    if has_phases:
        for ph_name in ("l1", "l2", "l3"):
            if ph_name in series.phases:
                residual_arrays[ph_name] = np.asarray(
                    series.phases[ph_name], dtype=np.float64
                )

    _write_atomic(stage_dir / "digest.json", canonical_json(digest))
    _write_atomic(stage_dir / "digest_meta.json", canonical_json(meta))
    _write_atomic_npz(stage_dir / "episodes.npz", episodes_arrays)
    _write_atomic_npz(stage_dir / "residual.npz", residual_arrays)

    return {
        "stage_dir": str(stage_dir),
        "digest": digest,
        "meta": meta,
        "episodes": episodes_arrays,
        "n_episodes": n_total,
        "n_clusters": len(meta["clusters"]),
    }


def compile_score(job: Mapping) -> dict:
    """Execute compile_score stage: compile proposal into signature artefacts."""
    stage_dir = Path(job.get("stage_dir", "."))
    episodes_path = job.get("episodes_path", stage_dir / "episodes.npz")
    proposal_path = Path(job["proposal_path"])

    episodes = load_episodes(episodes_path)

    meta_path = stage_dir / "digest_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(proposal_path, "r", encoding="utf-8") as f:
        proposal = json.load(f)

    run_id = str(job.get("run_id", "run-1"))
    round_no = int(job.get("round", 1))
    model = str(job.get("model", "model"))
    input_digest = str(meta.get("input_digest", ""))
    threshold_w = float(meta.get("edge_threshold_w", 120))
    extraction = {
        "grid_s": meta.get("grid_s", 30),
        "edge_threshold_w": threshold_w,
        "merge_samples": 2,
        "pair_tol": 0.25,
    }

    # Build cluster index mapping cluster id ("c01", ...) to row indices in episodes table
    cluster_map = meta.get("cluster_map", {})
    cluster_index: dict[str, np.ndarray] = {}
    if cluster_map and "cluster" in episodes:
        for cid, raw_lab in cluster_map.items():
            cluster_index[cid] = np.where(episodes["cluster"] == int(raw_lab))[0]
    elif "cluster" in episodes:
        unique_labels = sorted(set(episodes["cluster"]) - {-1})
        for i, l_id in enumerate(unique_labels[:40]):
            cid = f"c{i + 1:02d}"
            cluster_index[cid] = np.where(episodes["cluster"] == l_id)[0]

    artefacts, unresolved = compile_proposal(
        proposal,
        episodes,
        cluster_index,
        run_id=run_id,
        round_no=round_no,
        model=model,
        input_digest=input_digest,
        extraction=extraction,
        threshold_w=threshold_w,
    )

    # Replay on train (block 0) and validation (block 1); shares and S are fractions of
    # each block's residual energy (WR-12)
    block_residual = meta["block_residual_wh"]
    train_res = replay(artefacts, episodes, block=0, residual_wh=float(block_residual["train"]))
    train_denom = train_res.denominator_wh

    digest_path = stage_dir / "digest.json"
    with open(digest_path, "r", encoding="utf-8") as f:
        digest = json.load(f)

    artefacts_by_ref = {
        str(art.provenance.get("device_ref", "")): art
        for art in artefacts
        if art.provenance.get("device_ref")
    }
    scorecard_dict = scorecard(train_res, artefacts_by_ref, digest, round_no=round_no)
    scorecard_path = stage_dir / f"scorecard_{round_no}.json"
    _write_atomic(scorecard_path, canonical_json(scorecard_dict))

    val_res = replay(artefacts, episodes, block=1, residual_wh=float(block_residual["val"]))
    val_S, val_conf = block_score(val_res)
    incumbent = job.get("incumbent")
    accepted = True if incumbent is None else accept((val_S, val_conf), incumbent)

    val_dict = {
        "round": round_no,
        "S": round(float(val_S), 5),
        "conflict_share": round(float(val_conf), 5),
        "accepted": accepted,
        "per_signature": {
            sig: {
                "claimed_share": round(float(stats["claimed_wh"]) / val_res.denominator_wh, 4)
                if val_res.denominator_wh > 0
                else 0.0,
                "n": int(stats["n"]),
            }
            for sig, stats in val_res.per_signature.items()
        },
    }
    val_path = stage_dir / f"val_{round_no}.json"
    _write_atomic(val_path, canonical_json(val_dict))

    # Fill each artefact's scores
    validated_flag = bool(job.get("validated", True))

    # D-16 (Plan 04-07): the detector-vs-extraction energy error over the
    # learning split's validation days. Runs only here in the niced worker
    # subprocess, never in HA (T-04-19). The residual cache from extract is
    # masked to the validation days; the reference is the summed offline
    # energy of the validation episodes score.replay claims for the artefact
    # (switched rows only), so both sides cover the same episode population.
    # Preview runs and a missing residual cache give None (with basis None).
    val_error_pct: dict[str, float | None] = {}
    residual_path = stage_dir / "residual.npz"
    if validated_flag and residual_path.is_file():
        with np.load(str(residual_path), allow_pickle=False) as residual_data:
            r_full = np.asarray(residual_data["r"], dtype=np.float64)
            res_start_ms = int(residual_data["start_ms"][0])
            res_step_ms = int(residual_data["step_ms"][0])
            res_bounds = np.asarray(
                residual_data["day_bounds_ms"], dtype=np.int64
            )
            res_suppress = bool(int(residual_data["suppress_alignment"][0]))
            res_phase_keys = [
                k for k in ("l1", "l2", "l3") if k in residual_data.files
            ]
            res_phases = (
                {
                    k: np.asarray(residual_data[k], dtype=np.float64)
                    for k in res_phase_keys
                }
                if res_phase_keys
                else None
            )
        val_days = [int(d) for d in (meta.get("days", {}).get("val") or [])]
        sample_day = (
            np.searchsorted(
                res_bounds,
                res_start_ms + np.arange(len(r_full), dtype=np.int64) * res_step_ms,
                side="right",
            )
            - 1
        )
        r_val = np.where(np.isin(sample_day, val_days), r_full, np.nan)
        res_step_s = res_step_ms / 1000.0
        for art in artefacts:
            if art.kind != "switched":
                # Program devices keep no learning-split figure by design: the
                # program holdout figure comes from the shadow validate stage
                # (Plan 06-07), never from this per-run split (RESEARCH
                # Pitfall 1). Do not route a program figure in here.
                val_error_pct[art.signature_id] = None
                continue
            claimed_rows = [
                int(i)
                for i in val_res.claims.get(art.signature_id, set())
                if int(episodes["kind"][int(i)]) == 0
            ]
            reference_wh = (
                float(np.sum(episodes["energy_wh"][claimed_rows]))
                if claimed_rows
                else 0.0
            )
            val_error_pct[art.signature_id] = validation_error_pct(
                art.to_dict(),
                r_val,
                res_step_s,
                reference_wh,
                phases=res_phases,
                suppress_alignment=res_suppress,
            )

    updated_artefacts: list[SignatureArtefact] = []
    for art in artefacts:
        sig_id = art.signature_id
        t_stats = train_res.per_signature.get(sig_id, {})
        t_claimed = float(t_stats.get("claimed_wh", 0.0))
        t_share = t_claimed / train_denom if train_denom > 0 else 0.0

        v_stats = val_res.per_signature.get(sig_id, {})
        v_claimed = float(v_stats.get("claimed_wh", 0.0))
        v_share = v_claimed / val_res.denominator_wh if val_res.denominator_wh > 0 else 0.0

        conflicts_cnt = len(train_res.conflicts_with.get(sig_id, set()))

        err_pct = val_error_pct.get(sig_id)
        scores = {
            "train": {"share": round(t_share, 3), "n": int(t_stats.get("n", 0))},
            "val": {"share": round(v_share, 3), "n": int(v_stats.get("n", 0))} if validated_flag else None,
            "validated": validated_flag,
            "conflicts": conflicts_cnt,
            "validation_error_pct": err_pct,
            "validation_basis": "learning_split" if err_pct is not None else None,
        }
        updated_art = SignatureArtefact(
            schema_version=art.schema_version,
            signature_id=art.signature_id,
            kind=art.kind,
            label=art.label,
            category=art.category,
            extraction=art.extraction,
            states=art.states,
            session=art.session,
            time_prior=art.time_prior,
            constraints=art.constraints,
            provenance=art.provenance,
            scores=scores,
        )
        updated_artefacts.append(updated_art)

    prior = job.get("prior", [])
    decisions = job.get("decisions", {})
    matched_dicts = match_identities(
        updated_artefacts, prior, decisions, episodes, owned=job.get("owned", [])
    )

    # D-17 (Plan 05-06): the proposal gate. The HA side passes the frozen
    # active artefact of every confirmed device under its signature id (the
    # job's "actives"); a matched artefact whose states differ from the active
    # version replays both sides alone on this run's validation block and
    # carries the result in its "proposal_gate". Without actives nothing
    # changes (the shadow pass and preview runs pass no actives). A differing
    # artefact without a usable active artefact stays ungated: learning.py
    # records it as not_gated and offers nothing (a device confirmed while
    # the run was in flight).
    actives = job.get("actives") or {}
    if actives:
        val_residual = float(meta["block_residual_wh"]["val"])
        for matched in matched_dicts:
            sig_id = matched.get("signature_id")
            active_raw = actives.get(str(sig_id)) if sig_id else None
            if not isinstance(active_raw, Mapping):
                continue
            if not detection_differs(active_raw, matched):
                continue
            try:
                matched["proposal_gate"] = gate_proposal(
                    SignatureArtefact.from_dict(matched),
                    SignatureArtefact.from_dict(active_raw),
                    episodes,
                    val_residual,
                )
            except (KeyError, TypeError, ValueError):
                continue

    # Rank by train claimed share descending, ties by signature_id ascending
    matched_dicts.sort(
        key=lambda d: (-d.get("scores", {}).get("train", {}).get("share", 0.0), d["signature_id"])
    )

    lib_path = stage_dir / f"library_{round_no}.json"
    _write_atomic(lib_path, canonical_json(matched_dicts))

    return {
        "stage_dir": str(stage_dir),
        "library": str(lib_path),
        "scorecard": str(scorecard_path),
        "val": str(val_path),
        "S": float(val_S),
        "conflict_share": float(val_conf),
        "accepted": accepted,
        "n_artefacts": len(matched_dicts),
        "unresolved": unresolved,
    }


def rescore(job: Mapping) -> dict:
    """Execute rescore stage: replay candidate artefacts across post-window validation days."""
    buffer_path = job["buffer_path"]
    stage_dir = Path(job.get("stage_dir", "."))
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = job["day_bounds_ms"]
    first_weekday = int(job.get("first_weekday", 0))
    enabled_event_types = job.get("enabled_event_types", ["step", "cycle_shape", "continuous"])
    metered_channels = job.get("metered_channels", [])
    phase_channels = job.get("phase_channels", {})
    candidates_raw = job.get("candidates", [])

    if len(day_bounds_ms) < 2:
        rescore_dict: dict[str, Any] = {
            "S": 0.0,
            "conflict_share": 0.0,
            "per_signature": {},
            "days": 0,
        }
        rescore_path = stage_dir / "rescore.json"
        _write_atomic(rescore_path, canonical_json(rescore_dict))
        return rescore_dict

    start_ms = day_bounds_ms[0]
    end_ms = day_bounds_ms[-1]
    step_ms = grid_s * 1000
    step_s = float(grid_s)

    series = load_residual(
        buffer_path,
        start_ms,
        end_ms,
        step_ms,
        metered_channels,
        phase_channels,
    )
    r = series.residual

    total_days = len(day_bounds_ms) - 1
    valid_days = data_days(r, start_ms, step_ms, day_bounds_ms)

    if len(valid_days) == 0:
        rescore_dict = {
            "S": 0.0,
            "conflict_share": 0.0,
            "per_signature": {},
            "days": 0,
        }
        rescore_path = stage_dir / "rescore.json"
        _write_atomic(rescore_path, canonical_json(rescore_dict))
        return rescore_dict

    # Episode table through the shared helper (block assignment below stays
    # here). The transient session-only clustering columns are not used by
    # rescore; pop them so the table matches the previous inline sequence.
    table, all_rle, _threshold_w = build_episode_table(
        r,
        series.phases,
        start_ms,
        step_ms,
        day_bounds_ms,
        first_weekday,
        enabled_event_types,
    )
    table.pop("peak_w", None)
    table.pop("n_segments", None)

    start_ts_arr = start_ms + table["start_i"] * step_ms
    day_indices = (
        np.searchsorted(np.array(day_bounds_ms, dtype=np.int64), start_ts_arr, side="right") - 1
    )
    valid_set = set(valid_days)
    blocks = np.where(np.isin(day_indices, list(valid_set)), 1, -1).astype(np.int32)
    table["block"] = blocks

    rle_offsets: list[int] = [0]
    rle_levels: list[int] = []
    rle_counts: list[int] = []
    for rle_item in all_rle:
        for lvl, cnt in rle_item:
            rle_levels.append(int(lvl))
            rle_counts.append(int(cnt))
        rle_offsets.append(len(rle_levels))

    table["rle_offset"] = np.array(rle_offsets, dtype=np.int64)
    table["rle_level_w"] = np.array(rle_levels, dtype=np.int32)
    table["rle_n"] = np.array(rle_counts, dtype=np.int32)

    artefacts = [SignatureArtefact.from_dict(c) for c in candidates_raw]
    residual_wh = residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, valid_days)
    val_res = replay(artefacts, table, block=1, residual_wh=residual_wh)
    val_S, val_conf = block_score(val_res)

    # D-16 (Plan 04-07): same computation as compile_score, with the
    # post-window valid days as the validation days and the artefact replayed
    # alone. Program artefacts and an empty reference give None.
    sample_day = (
        np.searchsorted(
            np.array(day_bounds_ms, dtype=np.int64),
            start_ms + np.arange(len(r), dtype=np.int64) * step_ms,
            side="right",
        )
        - 1
    )
    r_val = np.where(np.isin(sample_day, list(valid_set)), r, np.nan)
    suppress_flag = "short_event" not in enabled_event_types
    res_phases = series.phases if series.phases else None

    denom = val_res.denominator_wh
    per_signature: dict[str, dict[str, Any]] = {}
    for art in artefacts:
        sig_id = art.signature_id
        stats = val_res.per_signature.get(sig_id, {})
        claimed = float(stats.get("claimed_wh", 0.0))
        share = claimed / denom if denom > 0 else 0.0
        claimed_rows = [
            int(i)
            for i in val_res.claims.get(sig_id, set())
            if int(table["kind"][int(i)]) == 0
        ]
        reference_wh = (
            float(np.sum(table["energy_wh"][claimed_rows])) if claimed_rows else 0.0
        )
        per_signature[sig_id] = {
            "share": round(share, 3),
            "n": int(stats.get("n", 0)),
            "validation_error_pct": validation_error_pct(
                art.to_dict(),
                r_val,
                step_s,
                reference_wh,
                phases=res_phases,
                suppress_alignment=suppress_flag,
            ),
        }

    rescore_dict = {
        "S": round(float(val_S), 5),
        "conflict_share": round(float(val_conf), 5),
        "per_signature": per_signature,
        "days": len(valid_days),
    }

    rescore_path = stage_dir / "rescore.json"
    _write_atomic(rescore_path, canonical_json(rescore_dict))
    return rescore_dict


def refit(job: Mapping) -> dict:
    """Execute the refit stage: locally retrain a device on the union of its own
    and its merged candidates' train episodes (D-13).

    The episode table comes from the shared build_episode_table helper (Plan
    04-13) with extract's split_days/block_of block marking on a fresh 70/30
    split over the current buffer; every refitted state comes from
    compile.fit_state_boxes on that same table (no copy of either sequence).
    The union = the kind-0 train episodes claimed by the current artefact or any
    extra (merged-candidate) artefact; each union episode is assigned to the
    current artefact's state whose rise box contains its rise_w, else the state
    with the nearest rise p50. A state with fewer than MIN_TRAIN_EPISODES union
    episodes keeps its current boxes, so a retrain never drops a state.

    The refitted artefact keeps the current signature_id, kind and label, with
    provenance {"source": "union", "aliases": [extra signature ids]}. Scores are
    computed like compile_score: train/val shares of block residual energy, val
    null and validated false when the job is not validated, and the D-16
    validation_error_pct via replay.validation_error_pct on the validation days.
    current_S and new_S are block 1 replay scores of the current and refitted
    parameters on the same fresh validation days; the refit is accepted only
    when new_S >= current_S (no worse, D-13 — no ACCEPT_MARGIN).
    """
    buffer_path = job["buffer_path"]
    stage_dir = Path(job["stage_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = job["day_bounds_ms"]
    first_weekday = int(job.get("first_weekday", 0))
    holdout_days = int(job.get("holdout_days", 0))
    train_fraction = float(job.get("train_fraction", 0.7))
    enabled_event_types = job.get("enabled_event_types", ["step", "cycle_shape", "continuous"])
    metered_channels = job.get("metered_channels", [])
    phase_channels = job.get("phase_channels", {})
    validated = bool(job.get("validated", False))

    start_ms = day_bounds_ms[0]
    end_ms = day_bounds_ms[-1]
    step_ms = grid_s * 1000

    series = load_residual(
        buffer_path,
        start_ms,
        end_ms,
        step_ms,
        metered_channels,
        phase_channels,
    )
    r = series.residual
    step_s = float(grid_s)

    total_days = len(day_bounds_ms) - 1
    valid_days = data_days(r, start_ms, step_ms, day_bounds_ms)
    n_dropped = total_days - len(valid_days)
    split = split_days(
        valid_days,
        train_fraction=train_fraction,
        holdout_days=holdout_days,
        n_dropped=n_dropped,
    )

    # Episode table through the shared helper (block assignment below stays
    # here, exactly like extract). The transient session-only clustering
    # columns are not used by refit; pop them like rescore does.
    table, _all_rle, threshold_w = build_episode_table(
        r,
        series.phases,
        start_ms,
        step_ms,
        day_bounds_ms,
        first_weekday,
        enabled_event_types,
    )
    table.pop("peak_w", None)
    table.pop("n_segments", None)

    start_ts_arr = start_ms + table["start_i"] * step_ms
    blocks = block_of(start_ts_arr, day_bounds_ms, split)
    table["block"] = blocks

    block_residual_wh = {
        "train": residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, split.train),
        "val": residual_energy_wh(r, start_ms, step_ms, day_bounds_ms, split.val),
    }

    current = SignatureArtefact.from_dict(job["current"])
    extra = [SignatureArtefact.from_dict(a) for a in job.get("extra", [])]
    artefacts = [current, *extra]

    # Union of kind-0 train episodes claimed by any of the artefacts.
    union_res = replay(artefacts, table, block=0, residual_wh=float(block_residual_wh["train"]))
    union_idx = sorted(
        int(i)
        for art in artefacts
        for i in union_res.claims.get(art.signature_id, set())
        if int(table["kind"][int(i)]) == 0
    )

    # Assign each union episode to the current artefact's state whose rise box
    # contains its rise_w, else the state with the nearest rise p50.
    per_state: list[list[int]] = [[] for _ in current.states]
    for idx in union_idx:
        rise = float(table["rise_w"][idx])
        containing = [
            si
            for si, st in enumerate(current.states)
            if st.rise_w.lo <= rise <= st.rise_w.hi
        ]
        if containing:
            si = containing[0]
        else:
            si = min(
                range(len(current.states)),
                key=lambda s: abs(float(current.states[s].rise_w.p50) - rise),
            )
        per_state[si].append(idx)

    new_states: list[object] = []
    for si, st in enumerate(current.states):
        indices = per_state[si]
        if len(indices) >= MIN_TRAIN_EPISODES:
            new_states.append(
                fit_state_boxes(
                    table,
                    np.array(indices, dtype=np.int64),
                    threshold_w,
                    name=st.name,
                )
            )
        else:
            # Too few union episodes: keep the current boxes (never drop a state).
            new_states.append(st)

    # Time prior from the union with the compile.py formula.
    union_arr = np.array(union_idx, dtype=np.int64)
    n_union = len(union_idx)
    if n_union > 0:
        hour_counts = np.bincount(table["hour"][union_arr].astype(int), minlength=24)[:24]
        hour_smoothed = (hour_counts + 1.0) / (n_union + 24.0)
        dow_counts = np.bincount(table["dow"][union_arr].astype(int), minlength=7)[:7]
        dow_smoothed = (dow_counts + 1.0) / (n_union + 7.0)
        time_prior: dict[str, list[float]] = {
            "hour": [round(float(x), 4) for x in hour_smoothed],
            "dow": [round(float(x), 4) for x in dow_smoothed],
        }
    else:
        time_prior = dict(current.time_prior)

    new_art = SignatureArtefact(
        schema_version=current.schema_version,
        signature_id=current.signature_id,
        kind=current.kind,
        label=current.label,
        category=current.category,
        extraction=current.extraction,
        states=tuple(new_states),
        session=current.session,
        time_prior=time_prior,
        constraints=current.constraints,
        provenance={
            "source": "union",
            "aliases": [a.signature_id for a in extra],
        },
        scores={},
    )

    # Scores like compile_score, replaying the refitted artefact alone.
    train_res = replay([new_art], table, block=0, residual_wh=float(block_residual_wh["train"]))
    train_stats = train_res.per_signature.get(new_art.signature_id, {})
    train_denom = train_res.denominator_wh
    t_share = (
        float(train_stats.get("claimed_wh", 0.0)) / train_denom if train_denom > 0 else 0.0
    )
    conflicts_cnt = len(union_res.conflicts_with.get(current.signature_id, set()))

    val_res = replay([new_art], table, block=1, residual_wh=float(block_residual_wh["val"]))
    v_stats = val_res.per_signature.get(new_art.signature_id, {})
    val_denom = val_res.denominator_wh
    v_share = float(v_stats.get("claimed_wh", 0.0)) / val_denom if val_denom > 0 else 0.0

    # D-16 (Plan 04-07): same computation as compile_score, with the fresh
    # split's validation days and the in-memory residual (refit loads r itself,
    # so no residual.npz round trip is needed).
    err_pct: float | None = None
    if validated:
        val_days = [int(d) for d in split.val]
        sample_day = (
            np.searchsorted(
                np.array(day_bounds_ms, dtype=np.int64),
                start_ms + np.arange(len(r), dtype=np.int64) * step_ms,
                side="right",
            )
            - 1
        )
        r_val = np.where(np.isin(sample_day, val_days), r, np.nan)
        claimed_rows = [
            int(i)
            for i in val_res.claims.get(new_art.signature_id, set())
            if int(table["kind"][int(i)]) == 0
        ]
        reference_wh = (
            float(np.sum(table["energy_wh"][claimed_rows])) if claimed_rows else 0.0
        )
        err_pct = validation_error_pct(
            new_art.to_dict(),
            r_val,
            step_s,
            reference_wh,
            phases=series.phases if series.phases else None,
            suppress_alignment="short_event" not in enabled_event_types,
        )

    new_art = SignatureArtefact(
        schema_version=new_art.schema_version,
        signature_id=new_art.signature_id,
        kind=new_art.kind,
        label=new_art.label,
        category=new_art.category,
        extraction=new_art.extraction,
        states=new_art.states,
        session=new_art.session,
        time_prior=new_art.time_prior,
        constraints=new_art.constraints,
        provenance=new_art.provenance,
        scores={
            "train": {"share": round(t_share, 3), "n": int(train_stats.get("n", 0))},
            "val": (
                {"share": round(v_share, 3), "n": int(v_stats.get("n", 0))}
                if validated
                else None
            ),
            "validated": validated,
            "conflicts": conflicts_cnt,
            "validation_error_pct": err_pct,
            "validation_basis": "learning_split" if err_pct is not None else None,
        },
    )

    # Keep-or-reject (D-13): same validation days, no-worse rule, no margin.
    cur_val = replay([current], table, block=1, residual_wh=float(block_residual_wh["val"]))
    current_S, _cur_conf = block_score(cur_val)
    new_S, _new_conf = block_score(val_res)
    accepted = bool(new_S >= current_S)

    refit_dict = {
        "accepted": accepted,
        "current_S": round(float(current_S), 5),
        "new_S": round(float(new_S), 5),
        "artefact": new_art.to_dict(),
    }
    refit_path = stage_dir / "refit.json"
    _write_atomic(refit_path, canonical_json(refit_dict))

    return {
        "stage_dir": str(stage_dir),
        "result": str(refit_path),
        "accepted": accepted,
        "current_S": float(refit_dict["current_S"]),
        "new_S": float(refit_dict["new_S"]),
    }


def measurable_artefacts(shadow: Mapping) -> list[Mapping]:
    """The shadow artefacts the validate stage replays (DET-02, D-03).

    Every switched and program artefact in the set is replayed, so the two
    kinds compete for the same edges exactly as before (the program tracker
    joins the same online step). An artefact of another kind is never replayed
    (assemble_report refuses its link with unsupported_kind), and a measured
    artefact the detector cannot compile — a program whose stage template is
    missing — is skipped here too: its link is still reported, with a zero
    power series, so it measures 0 Wh rather than failing the whole stage
    (P5 D-03: nothing is silently dropped, no refit of another kind).
    """
    out: list[Mapping] = []
    for artefact in shadow.get("artefacts", []):
        if not isinstance(artefact, Mapping) or artefact.get("kind") not in MEASURED_KINDS:
            continue
        try:
            compile_model(str(artefact.get("signature_id", "")), artefact)
        except (KeyError, TypeError, ValueError):
            continue
        out.append(artefact)
    return out


def overlap_block(
    ys: Mapping[str, np.ndarray],
    ests: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    online: Any,
    step_s: float,
) -> dict[str, Any]:
    """Assemble the D-08 overlap block from plug ground truth and replay counters.

    ``masks`` holds each measured plug's ON ticks over the holdout (plug ground
    truth, D-08: an ON tick is a holdout sample where the plug reads above
    PLUG_ON_W), ``ests`` the replay's per-step power for the linked signature and
    ``online`` the collect_power replay whose cumulative edge counters are read
    over the overlap samples. The result is plain data handed to
    assemble_report, the single report writer (Plan 06-08).

    The per-device solo/overlap errors, the pair rows and the energy left in the
    remainder follow the report's refuse-never-0 rule: a figure over fewer than
    OVERLAP_MIN_TICKS ticks is None, never 0. The tick counts and the edge
    counts keep their numeric value (0 is a fact for a count).
    """
    solo_by_entity, overlap_by_entity, n_on = overlap_metrics.tick_masks(masks)
    n_overlap_ticks = int(np.count_nonzero(n_on >= 2))
    n_solo_ticks = int(np.count_nonzero(n_on == 1))
    per_device = [
        {
            "entity_id": entity_id,
            **overlap_metrics.split_errors(
                ys.get(entity_id),
                ests.get(entity_id),
                solo_by_entity[entity_id],
                overlap_by_entity[entity_id],
                step_s,
            ),
        }
        for entity_id in sorted(masks)
    ]
    energy_left: float | None = None
    if n_overlap_ticks >= OVERLAP_MIN_TICKS:
        energy_left = round(
            overlap_metrics.remainder_energy_wh(ys, ests, overlap_by_entity, step_s), 1
        )
    return {
        "definition": overlap_metrics.OVERLAP_DEFINITION,
        "n_overlap_ticks": n_overlap_ticks,
        "n_solo_ticks": n_solo_ticks,
        "per_device": per_device,
        "edges": overlap_metrics.edge_counts(
            getattr(online, "joint_attributed_cum", None),
            getattr(online, "overlap_refused_cum", None),
            n_on >= 2,
        ),
        "edges_definition": overlap_metrics.EDGES_DEFINITION,
        "energy_left_in_remainder_wh": energy_left,
        "energy_left_definition": overlap_metrics.ENERGY_LEFT_DEFINITION,
        "pairs": overlap_metrics.pair_table(ys, ests, masks, step_s),
    }


def _synthetic_overlap_block(
    *,
    ys: Mapping[str, np.ndarray],
    sig_of: Mapping[str, str],
    masks: Mapping[str, np.ndarray],
    r_masked: np.ndarray,
    artefacts: Sequence[Mapping],
    step_s: float,
    phases: Mapping[str, np.ndarray] | None,
    suppress_alignment: bool,
) -> dict[str, Any]:
    """Build the D-28 synthetic-overlap block, or a labelled refusal.

    With fewer than two measured plugs that have holdout ON ticks the block is
    ``{synthetic: true, status: refused, reason: fewer_than_two_plugs}``; when
    the two busiest plugs' first ON runs start in the same sample the computed
    shift is 0 (the copy would only duplicate a natural overlap), so the block
    refuses with ``zero_shift``. Otherwise plug B (the second busiest) is
    time-shifted onto plug A's first ON run and added to the holdout-masked
    residual (NaN as 0 W), which is replayed once more; B's ground truth is its
    real series plus that shifted copy (both are in the injected residual) so
    B's real runs are not read as over-estimation, and A's ground truth is
    unchanged. Every figure is computed by the same ``overlap_block`` as the
    real block, so the two cannot drift apart.
    """
    entity_ids = sorted(
        entity_id for entity_id, mask in masks.items() if int(np.count_nonzero(mask)) > 0
    )
    if len(entity_ids) < 2:
        return {
            "synthetic": True,
            "status": "refused",
            "reason": "fewer_than_two_plugs",
        }
    ranked = sorted(
        entity_ids,
        key=lambda entity_id: (-int(np.count_nonzero(masks[entity_id])), entity_id),
    )
    entity_a, entity_b = ranked[0], ranked[1]
    start_a = int(np.flatnonzero(masks[entity_a])[0])
    start_b = int(np.flatnonzero(masks[entity_b])[0])
    shift = start_a - start_b
    if shift == 0:
        return {"synthetic": True, "status": "refused", "reason": "zero_shift"}

    n = int(len(np.asarray(r_masked)))
    b_shifted = overlap_metrics.shifted_copy(ys[entity_b], shift)
    r_injected = np.asarray(r_masked, dtype=np.float64) + b_shifted

    injected = replay_online(
        artefacts,
        r_injected,
        step_s,
        phases=phases,
        suppress_alignment=suppress_alignment,
        collect_power=True,
    )

    ys_injected = {key: np.asarray(value, dtype=np.float64) for key, value in ys.items()}
    ys_injected[entity_b] = ys_injected[entity_b] + b_shifted
    finite_injected = np.isfinite(r_injected)
    masks_injected = {
        key: finite_injected & np.isfinite(value) & (value > PLUG_ON_W)
        for key, value in ys_injected.items()
    }
    ests_injected: dict[str, np.ndarray] = {}
    for key in ys_injected:
        power = injected.power_w.get(sig_of.get(key, ""))
        if power is None:
            power = np.zeros(n, dtype=np.float64)
        ests_injected[key] = power

    shift_s = shift * int(round(step_s))
    return {
        "synthetic": True,
        "label": (
            f"SYNTHETIC: {entity_b} time-shifted by {shift_s} s and added to "
            f"the measured residual to force overlap with {entity_a}; not a "
            "real measurement"
        ),
        "injected": {"a": entity_a, "b": entity_b, "shift_s": shift_s},
        **overlap_block(ys_injected, ests_injected, masks_injected, injected, step_s),
    }


def validate(job: Mapping) -> dict:
    """Execute the validate stage: replay stored shadow artefacts over the holdout block.

    The estimate comes from replay_online over the gross residual (every masked
    plug added back, series.gross), masked to NaN outside the holdout days, with
    every switched and program shadow artefact competing for edges (RESEARCH
    Pattern 3, A6; DET-02): energy_est_wh is the detector's committed per-step
    energy for the linked artefact summed over exactly the samples where the
    residual and the plug are both finite. The naive baseline is the plug's
    train_mean_w constant (A8). The holdout is the last shadow["holdout_days"]
    calendar days before today's local midnight, gaps included, restricted to
    valid days that start at or after the shadow set's train_cutoff_ms (D-11,
    D-18, RESEARCH Pitfall 3). Every shadow link becomes a device_result —
    switched and program links with figures, links of any other kind without —
    and engine.validation.report.assemble_report applies the D-12 data gates,
    the D-07/D-08 pass bars and the VAL-03 unvalidated flag, so report.json
    carries exactly one metrics object (the one assemble_report builds). Data
    refusals are report content, never exceptions (Pattern 5).

    Plan 06-08 adds the D-08 overlap block (per-device solo/overlap errors, the
    joint vs refused overlap edges, the energy left in the remainder and the
    device-pair table) and, under job["synthetic_overlap"], the D-28
    ``"overlap_synthetic"`` block: a labelled time-shifted plug copy replayed
    once more so SC2 can be demonstrated when natural overlap is rare. Both are
    handed to assemble_report as plain data and never feed devices, classes or a
    verdict (D-28).
    """
    buffer_path = job["buffer_path"]
    stage_dir = Path(job.get("stage_dir", "."))
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = job["day_bounds_ms"]
    enabled_event_types = job.get("enabled_event_types", ["step", "cycle_shape", "continuous"])
    phase_channels = job.get("phase_channels", {})
    shadow = job.get("shadow") or {}

    # D-27: the report says what it measured — the shadow set's age at
    # validation time (the HA side supplies the clock as validation_time_ms so
    # identical jobs stay byte-identical; the worker never reads the clock).
    created_raw = shadow.get("created")
    shadow_created = created_raw if isinstance(created_raw, str) else None
    shadow_age_days: float | None = None
    shadow_stale: bool | None = None
    validation_time_ms = int(job.get("validation_time_ms") or 0)
    if shadow_created and validation_time_ms:
        try:
            created_dt = datetime.fromisoformat(shadow_created)
        except ValueError:
            created_dt = None
        if created_dt is not None:
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            created_ms = int(round(created_dt.timestamp() * 1000))
            shadow_age_days = round((validation_time_ms - created_ms) / 86_400_000, 1)
            shadow_stale = shadow_age_days > SHADOW_MAX_AGE_DAYS

    report_path = stage_dir / "report.json"

    if len(day_bounds_ms) < 2:
        report = assemble_report(
            [],
            shadow=shadow,
            holdout_days_info={"start_ms": 0, "end_ms": 0, "days": []},
            usable_days=0,
            grid_s=grid_s,
            enabled_event_types=enabled_event_types,
            plug_classes=job.get("plug_classes", {}),
            shadow_created=shadow_created,
            shadow_age_days=shadow_age_days,
            shadow_stale=shadow_stale,
        )
        _write_atomic(report_path, canonical_json(report))
        return report

    holdout_days = int(shadow.get("holdout_days", 7) or 7)
    # Every shadow link becomes a device_result: switched and program links
    # with their holdout figures, links of any other kind without
    # (assemble_report turns those into unsupported_kind refusals; D-03:
    # nothing is silently dropped).
    links = [
        link
        for link in shadow.get("links", [])
        if isinstance(link, Mapping) and link.get("channel") is not None
    ]
    measured_links = [
        link for link in links if str(link.get("kind", "")) in MEASURED_KINDS
    ]
    channels = sorted({int(link["channel"]) for link in measured_links})

    start_ms = int(day_bounds_ms[0])
    end_ms = int(day_bounds_ms[-1])
    step_ms = grid_s * 1000
    step_s = float(grid_s)

    series = load_residual(
        buffer_path,
        start_ms,
        end_ms,
        step_ms,
        channels,
        phase_channels,
    )
    r = series.gross
    valid = data_days(r, start_ms, step_ms, day_bounds_ms)
    n_days = len(day_bounds_ms) - 1
    first_holdout = calendar_holdout_start(n_days, holdout_days)
    train_cutoff_ms = int(
        shadow.get("train_cutoff_ms", day_bounds_ms[first_holdout])
        or day_bounds_ms[first_holdout]
    )
    valid_set = set(valid)
    holdout_idx = [
        d
        for d in range(first_holdout, n_days)
        if d in valid_set and int(day_bounds_ms[d]) >= train_cutoff_ms
    ]
    holdout_set = set(holdout_idx)

    bounds = np.array(day_bounds_ms, dtype=np.int64)
    sample_day = (
        np.searchsorted(
            bounds,
            start_ms + np.arange(len(r), dtype=np.int64) * step_ms,
            side="right",
        )
        - 1
    )
    if holdout_set:
        holdout_mask = np.isin(sample_day, list(holdout_set))
        r_masked = np.where(holdout_mask, r, np.nan)
    else:
        holdout_mask = np.zeros(len(r), dtype=bool)
        r_masked = np.full_like(r, np.nan)

    days_report = []
    for d in holdout_idx:
        m = sample_day == d
        n_total = int(np.sum(m))
        n_finite = int(np.sum(np.isfinite(r[m]))) if n_total else 0
        days_report.append(
            {
                "day_start_ms": int(day_bounds_ms[d]),
                "coverage": round(n_finite / n_total, 3) if n_total else 0.0,
            }
        )

    artefacts = measurable_artefacts(shadow)
    suppress_flag = "short_event" not in enabled_event_types
    phases = series.phases if series.phases else None
    online = replay_online(
        artefacts,
        r_masked,
        step_s,
        phases=phases,
        suppress_alignment=suppress_flag,
        collect_power=True,
    )

    plugs_meta = shadow.get("plugs", {})
    artefact_of = {
        str(a.get("signature_id", "")): a
        for a in shadow.get("artefacts", [])
        if isinstance(a, Mapping)
    }
    device_results: list[dict[str, Any]] = []
    for link in links:
        entity_id = str(link.get("entity_id", ""))
        channel = int(link["channel"])
        sig_id = str(link.get("signature_id", ""))
        kind = str(link.get("kind", ""))
        overlap = float(link.get("overlap", 0.0) or 0.0)
        if kind not in MEASURED_KINDS or not entity_id or not sig_id:
            # D-03: a link of any other kind is reported as unsupported_kind by
            # assemble_report and nothing is replayed for it. A measured link
            # whose artefact the detector could not compile (see
            # measurable_artefacts) lands on the same figures path below, where
            # its power and energy series default to zeros: it reads 0 Wh
            # instead of disappearing (P5 D-03).
            if entity_id and sig_id:
                device_results.append(
                    {
                        "entity_id": entity_id,
                        "channel": channel,
                        "signature_id": sig_id,
                        "kind": kind,
                        "overlap": overlap,
                    }
                )
            continue
        y = series.plugs.get(channel)
        if y is None:
            device_results.append(
                {
                    "entity_id": entity_id,
                    "channel": channel,
                    "signature_id": sig_id,
                    "kind": kind,
                    "overlap": overlap,
                }
            )
            continue
        ok = holdout_mask & np.isfinite(r) & np.isfinite(y)
        # A6/Pattern 3: both energy sums and every metric use exactly the
        # samples where the residual and the plug both have a reading, so a
        # plug gap removes the same samples from the reference and the
        # estimate. A linked signature without a collected series yields a
        # zero series, never a KeyError.
        power = online.power_w.get(sig_id)
        if power is None:
            power = np.zeros(len(y), dtype=np.float64)
        step_energy = online.energy_step_wh.get(sig_id)
        if step_energy is None:
            step_energy = np.zeros(len(y), dtype=np.float64)
        y_ok = y[ok]
        power_ok = power[ok]
        energy_true_wh = float(np.sum(y_ok)) * step_s / 3600.0
        energy_est_wh = float(np.sum(step_energy[ok]))
        table, _rle = plug_sessions(y, step_s)
        start_rows = table.get("start_i")
        day_of_event: np.ndarray = np.empty(0, dtype=np.int64)
        if start_rows is not None and len(start_rows):
            idx = np.asarray(start_rows, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(sample_day))]
            day_of_event = sample_day[idx]
        n_events = int(np.sum(np.isin(day_of_event, list(holdout_set)))) if len(day_of_event) else 0

        per_day = []
        for d in holdout_idx:
            m = sample_day == d
            n_total = int(np.sum(m))
            n_finite = int(np.sum(np.isfinite(r[m]))) if n_total else 0
            ok_d = m & ok
            true_d = float(np.sum(y[ok_d])) * step_s / 3600.0
            est_d = float(np.sum(step_energy[ok_d]))
            per_day.append(
                {
                    "day_start_ms": int(day_bounds_ms[d]),
                    "coverage": round(n_finite / n_total, 3) if n_total else 0.0,
                    "n_events": (
                        int(np.sum(day_of_event == d)) if len(day_of_event) else 0
                    ),
                    "energy_true_wh": round(true_d, 1),
                    "energy_est_wh": round(est_d, 1),
                    "energy_error_pct": energy_error_pct(true_d, est_d),
                }
            )
        mean_w = 0.0
        if isinstance(plugs_meta, Mapping):
            plug_meta = plugs_meta.get(entity_id)
            if isinstance(plug_meta, Mapping):
                mean_w = float(plug_meta.get("train_mean_w", 0.0) or 0.0)
        device_results.append(
            {
                "entity_id": entity_id,
                "channel": channel,
                "signature_id": sig_id,
                "kind": kind,
                "overlap": overlap,
                "artefact": artefact_of.get(sig_id, {}),
                "n_events": n_events,
                "energy_true_wh": round(energy_true_wh, 1),
                "energy_est_wh": round(energy_est_wh, 1),
                "energy_error_pct": energy_error_pct(energy_true_wh, energy_est_wh),
                "nde": nde(y_ok, power_ok),
                "mae_w": mae(y_ok, power_ok),
                "baseline": baseline(y_ok, mean_w, step_s),
                "usable_holdout_days": len(holdout_idx),
                "per_day": per_day,
            }
        )

    # Plan 06-08 (D-08/D-28): the overlap block. Its judge is the plug ground
    # truth - an ON tick is a holdout sample where a linked plug reads above
    # PLUG_ON_W - and its edge counters are the replay's cumulative arrays. It
    # is built here, where every linked plug series and the replay's per-step
    # power are in hand, and handed to assemble_report (the single report
    # writer) as plain data.
    overlap_ys: dict[str, np.ndarray] = {}
    overlap_ests: dict[str, np.ndarray] = {}
    overlap_masks: dict[str, np.ndarray] = {}
    sig_of: dict[str, str] = {}
    for link in measured_links:
        entity_id = str(link.get("entity_id", ""))
        channel = int(link["channel"])
        y = series.plugs.get(channel)
        if not entity_id or y is None:
            continue
        ok_holdout = holdout_mask & np.isfinite(r) & np.isfinite(y)
        power = online.power_w.get(str(link.get("signature_id", "")))
        if power is None:
            power = np.zeros(len(y), dtype=np.float64)
        overlap_ys[entity_id] = y
        overlap_ests[entity_id] = power
        overlap_masks[entity_id] = ok_holdout & (y > PLUG_ON_W)
        sig_of[entity_id] = str(link.get("signature_id", ""))
    overlap_report = overlap_block(
        overlap_ys, overlap_ests, overlap_masks, online, step_s
    )
    overlap_synthetic: dict[str, Any] | None = None
    if bool(job.get("synthetic_overlap")):
        overlap_synthetic = _synthetic_overlap_block(
            ys=overlap_ys,
            sig_of=sig_of,
            masks=overlap_masks,
            r_masked=r_masked,
            artefacts=artefacts,
            step_s=step_s,
            phases=phases,
            suppress_alignment=suppress_flag,
        )

    holdout_info = {
        "start_ms": (
            int(day_bounds_ms[holdout_idx[0]])
            if holdout_idx
            else int(day_bounds_ms[first_holdout])
        ),
        "end_ms": (
            int(day_bounds_ms[holdout_idx[-1] + 1])
            if holdout_idx
            else int(day_bounds_ms[-1])
        ),
        "days": days_report,
    }
    report = assemble_report(
        device_results,
        shadow=shadow,
        holdout_days_info=holdout_info,
        usable_days=len(valid),
        grid_s=grid_s,
        enabled_event_types=enabled_event_types,
        plug_classes=job.get("plug_classes", {}),
        shadow_created=shadow_created,
        shadow_age_days=shadow_age_days,
        shadow_stale=shadow_stale,
        overlap=overlap_report,
        overlap_synthetic=overlap_synthetic,
    )
    _write_atomic(report_path, canonical_json(report))
    return report


def _day_coverage(sample_day: np.ndarray, day: int, *arrays: np.ndarray) -> float:
    """Fraction of one day's samples where every array is finite (0 when empty)."""
    mask = sample_day == day
    n_total = int(np.count_nonzero(mask))
    if n_total == 0:
        return 0.0
    finite = mask
    for arr in arrays:
        finite = finite & np.isfinite(arr)
    return float(np.count_nonzero(finite)) / n_total


def validate_sessions(job: Mapping) -> dict:
    """Execute the validate_sessions stage: judge device-target plug sessions (REV-04, D-13).

    For every device-target plug session the session's own device is replayed
    over the production residual (metered plugs subtracted, the session plug
    left in exactly once) with every other confirmed device competing for
    edges, on the last up to SHADOW_HOLDOUT_DAYS whole local days inside the
    session window, and judged against the session plug's readings with the
    unchanged P5 bars (D-13). The holdout day must carry MIN_DAY_COVERAGE
    finite residual AND finite plug samples; the baseline is the NaN-ignoring
    mean of the plug over the earlier window days that carry MIN_DAY_COVERAGE
    finite plug samples (a day the plug was unrecorded never poisons it, review
    L231). No such holdout or baseline day refuses the session insufficient_data.

    Compile isolation (review L233): every competitor artefact is compiled up
    front; one that fails compile_model is dropped from the replay and named in
    each session entry's skipped_competitors, and a session whose own artefact
    fails is refused model_error while every other session is still judged — one
    unusable artefact never aborts the stage. The per-session verdicts are
    written with report.assemble_sessions (the second report writer surface).
    """
    buffer_path = job["buffer_path"]
    stage_dir = Path(job.get("stage_dir", "."))
    stage_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = stage_dir / "sessions.json"

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = [int(b) for b in job["day_bounds_ms"]]
    enabled_event_types = job.get(
        "enabled_event_types", ["step", "cycle_shape", "continuous"]
    )
    phase_channels = job.get("phase_channels", {})
    metered_channels = [int(c) for c in (job.get("metered_channels") or [])]
    others = [
        a for a in (job.get("others") or []) if isinstance(a, Mapping)
    ]
    sessions = [
        s for s in (job.get("sessions") or []) if isinstance(s, Mapping)
    ]

    if len(day_bounds_ms) < 2 or not sessions:
        _write_atomic(sessions_path, canonical_json([]))
        return {"stage_dir": str(stage_dir), "sessions": [], "n": 0}

    step_ms = grid_s * 1000
    step_s = float(grid_s)
    start_ms = day_bounds_ms[0]
    end_ms = day_bounds_ms[-1]
    bounds = np.array(day_bounds_ms, dtype=np.int64)

    # Every session's plug channel is added to the metered read so its series is
    # available; the per-session residual then adds it (and every other session
    # plug) back, so the residual is the production residual (metered plugs
    # subtracted) with the session plugs left in.
    session_chs = sorted(
        {
            int(s["plug_channel"])
            for s in sessions
            if s.get("plug_channel") is not None
        }
    )
    read_channels = sorted(set(metered_channels) | set(session_chs))
    series = load_residual(
        buffer_path, start_ms, end_ms, step_ms, read_channels, phase_channels
    )
    residual_base = series.residual.copy()
    for s_ch in session_chs:
        plug_series = series.plugs.get(s_ch)
        if plug_series is not None:
            residual_base = residual_base + np.where(
                np.isnan(plug_series), 0.0, plug_series
            )
    grid_ts = start_ms + np.arange(len(residual_base), dtype=np.int64) * step_ms
    sample_day = (
        np.searchsorted(bounds, grid_ts, side="right") - 1
    )
    phases = series.phases if series.phases else None
    suppress_flag = "short_event" not in enabled_event_types

    # Compile the competitors once. A failing one is dropped from every replay
    # and named in each session entry's skipped_competitors (review L233).
    valid_others: list[Mapping] = []
    skipped: list[str] = []
    for art in others:
        try:
            compile_model(str(art.get("signature_id", "")), art)
        except (KeyError, TypeError, ValueError):
            skipped.append(str(art.get("signature_id", "")))
            continue
        valid_others.append(art)
    other_sigs = {str(a.get("signature_id", "")) for a in valid_others}

    results: list[dict[str, Any]] = []
    n_days = len(day_bounds_ms) - 1
    for session in sessions:
        session_id = str(session.get("session_id", ""))
        device_uuid = str(session.get("device_uuid", ""))
        plug_channel = int(session.get("plug_channel") or 0)
        artefact = session.get("artefact")
        artefact = artefact if isinstance(artefact, Mapping) else {}
        artefact_v = int(session.get("artefact_v") or 0)
        w_start = int(session.get("window_start_ms", 0) or 0)
        w_end = int(session.get("window_end_ms", 0) or 0)
        base: dict[str, Any] = {
            "session_id": session_id,
            "device_uuid": device_uuid,
            "plug_channel": plug_channel,
            "artefact_v": artefact_v,
            "window_start_ms": w_start,
            "window_end_ms": w_end,
            "skipped_competitors": list(skipped),
        }
        own_sig = str(artefact.get("signature_id", ""))
        try:
            compile_model(own_sig, artefact)
        except (KeyError, TypeError, ValueError):
            results.append({**base, "status": "refused", "reason": "model_error"})
            continue
        plug = series.plugs.get(plug_channel)
        if plug is None:
            results.append(
                {**base, "status": "refused", "reason": "insufficient_data"}
            )
            continue

        in_window = (grid_ts >= w_start) & (grid_ts < w_end)
        plug_w = np.where(in_window, plug, np.nan)
        residual_w = np.where(in_window, residual_base, np.nan)

        # Whole local days fully inside the window.
        inside = [
            d
            for d in range(n_days)
            if day_bounds_ms[d] >= w_start and day_bounds_ms[d + 1] <= w_end
        ]
        covered = [
            d
            for d in inside
            if _day_coverage(sample_day, d, residual_w, plug_w) >= MIN_DAY_COVERAGE
        ]
        holdout_idx = covered[-SHADOW_HOLDOUT_DAYS:]
        holdout_set = set(holdout_idx)
        earlier = [d for d in inside if d not in holdout_set]
        baseline_days = [
            d
            for d in earlier
            if _day_coverage(sample_day, d, plug_w) >= MIN_DAY_COVERAGE
        ]
        if not holdout_idx or not baseline_days:
            results.append(
                {**base, "status": "refused", "reason": "insufficient_data"}
            )
            continue

        holdout_mask = np.isin(sample_day, list(holdout_set))
        baseline_mask = np.isin(sample_day, baseline_days)
        mean_w = (
            float(np.nanmean(plug_w[baseline_mask]))
            if bool(np.any(baseline_mask))
            else float("nan")
        )

        pool = list(valid_others)
        if own_sig not in other_sigs:
            pool = [*pool, artefact]
        r_session = np.where(holdout_mask, residual_w, np.nan)
        online = replay_online(
            pool,
            r_session,
            step_s,
            phases=phases,
            suppress_alignment=suppress_flag,
            collect_power=True,
        )
        power = online.power_w.get(own_sig)
        if power is None:
            power = np.zeros(len(plug_w), dtype=np.float64)
        step_energy = online.energy_step_wh.get(own_sig)
        if step_energy is None:
            step_energy = np.zeros(len(plug_w), dtype=np.float64)

        ok = holdout_mask & np.isfinite(residual_base) & np.isfinite(plug_w)
        y_ok = plug_w[ok]
        power_ok = power[ok]
        energy_true_wh = float(np.sum(y_ok)) * step_s / 3600.0
        energy_est_wh = float(np.sum(step_energy[ok]))

        p_table, _p_rle = plug_sessions(plug_w, step_s)
        start_rows = p_table.get("start_i")
        day_of_event: np.ndarray = np.empty(0, dtype=np.int64)
        if start_rows is not None and len(start_rows):
            idx = np.asarray(start_rows, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(sample_day))]
            day_of_event = sample_day[idx]
        n_events = (
            int(np.sum(np.isin(day_of_event, list(holdout_set))))
            if len(day_of_event)
            else 0
        )

        results.append(
            {
                **base,
                "status": "ok",
                "reason": None,
                "holdout_start_ms": day_bounds_ms[holdout_idx[0]],
                "holdout_end_ms": day_bounds_ms[holdout_idx[-1] + 1],
                "usable_holdout_days": len(holdout_idx),
                "energy_true_wh": round(energy_true_wh, 1),
                "energy_est_wh": round(energy_est_wh, 1),
                "energy_error_pct": energy_error_pct(energy_true_wh, energy_est_wh),
                "nde": nde(y_ok, power_ok),
                "mae_w": mae(y_ok, power_ok),
                "baseline": baseline(y_ok, mean_w, step_s),
                "n_events": n_events,
            }
        )

    entries = assemble_sessions(results)
    _write_atomic(sessions_path, canonical_json(entries))
    return {"stage_dir": str(stage_dir), "sessions": entries, "n": len(entries)}


def shadow_link(job: Mapping) -> dict:
    """Execute the shadow_link stage: link shadow artefacts to plugs (D-03, A12).

    Runs once per shadow set, after the shadow compile rounds, while the
    train-block plug data still exists in the buffer (RESEARCH: linking at
    shadow time, not validation time). Every shadow artefact of every kind is
    scored against every masked plug by the Jaccard overlap of their ON masks
    on the shadow run's TRAIN-block days only (holdout overlap never counts);
    assignment is one-to-one greedy at or above LINK_MIN_OVERLAP, and every
    plug left over is reported not_discovered. Each plug's train-block mean
    power (the naive validation baseline, A8) is recorded under plugs. The
    result is part of the powerguesser.shadow Store contract (Plan 05-01):
    plugs entries carry only train_mean_w.
    """
    buffer_path = job["buffer_path"]
    stage_dir = Path(job["stage_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid_s = int(job.get("grid_s", 30))
    day_bounds_ms = job["day_bounds_ms"]
    masked_channels = job.get("masked_channels", [])
    step_ms = grid_s * 1000
    step_s = float(grid_s)
    start_ms = int(day_bounds_ms[0])
    end_ms = int(day_bounds_ms[-1])
    gap_samples = int(PLUG_GAP_S // grid_s)

    with open(stage_dir / "digest_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(job["library_path"], "r", encoding="utf-8") as f:
        library = json.load(f)

    days_meta = meta.get("days", {})
    train_days = [int(d) for d in days_meta.get("train", [])]
    val_days = [int(d) for d in days_meta.get("val", [])]
    episodes = load_episodes(stage_dir / "episodes.npz")

    # Train-block sample mask over the gridded window.
    n_samples = int(round((end_ms - start_ms) / step_ms))
    allowed = np.zeros(n_samples, dtype=bool)
    bounds = np.array(day_bounds_ms, dtype=np.int64)
    for d in train_days:
        i0 = max(0, int(round((int(bounds[d]) - start_ms) / step_ms)))
        i1 = min(n_samples, int(round((int(bounds[d + 1]) - start_ms) / step_ms)))
        allowed[i0:i1] = True

    # Artefact ON masks from the train-block replay claims.
    artefacts = [SignatureArtefact.from_dict(d) for d in library]
    block_residual_wh = float(meta.get("block_residual_wh", {}).get("train", 0.0))
    claims = replay(artefacts, episodes, block=0, residual_wh=block_residual_wh).claims
    artefact_masks: dict[str, np.ndarray] = {}
    for art in artefacts:
        rows = sorted(claims.get(art.signature_id, set()))
        spans = [
            (int(episodes["start_i"][i]), int(episodes["end_i"][i])) for i in rows
        ]
        artefact_masks[art.signature_id] = on_mask(
            spans, n_samples, allowed=allowed, gap_samples=gap_samples
        )

    # Plug ON masks from plug_sessions spans starting on train days, plus the
    # train-block mean power (the naive baseline constant, A8).
    eligible = [str(m["entity_id"]) for m in masked_channels]
    channel_of = {str(m["entity_id"]): int(m["channel"]) for m in masked_channels}
    train_day_set = set(train_days)
    plug_masks: dict[str, np.ndarray] = {}
    train_mean: dict[str, float] = {}
    n_train_sessions: dict[str, int] = {}
    series = load_residual(buffer_path, start_ms, end_ms, step_ms, list(channel_of.values()), {})
    for entity_id in eligible:
        channel = channel_of[entity_id]
        p = np.asarray(series.plugs.get(channel, np.full(n_samples, np.nan)), dtype=np.float64)
        p_table, _p_rle = plug_sessions(p, step_s)
        n_spans = len(p_table["start_i"])
        train_spans = [
            (int(p_table["start_i"][i]), int(p_table["end_i"][i]))
            for i in range(n_spans)
            if int(np.searchsorted(bounds, start_ms + int(p_table["start_i"][i]) * step_ms, side="right")) - 1
            in train_day_set
        ]
        plug_masks[entity_id] = on_mask(
            train_spans, n_samples, allowed=allowed, gap_samples=gap_samples
        )
        finite_train = p[allowed] if allowed.any() else np.empty(0, dtype=np.float64)
        finite_train = finite_train[np.isfinite(finite_train)]
        train_mean[entity_id] = (
            round(float(np.mean(finite_train)), 1) if len(finite_train) else 0.0
        )
        n_train_sessions[entity_id] = len(train_spans)

    kind_of = {str(a.get("signature_id")): str(a.get("kind", "")) for a in library}
    scores = {
        (entity_id, sig_id): jaccard(plug_masks[entity_id], artefact_masks[sig_id])
        for entity_id in eligible
        for sig_id in artefact_masks
    }
    assigned, unlinked = assign_links(scores, eligible, min_overlap=LINK_MIN_OVERLAP)

    links = []
    for entry in assigned:
        sig_id = str(entry["signature_id"])
        entity_id = str(entry["entity_id"])
        links.append(
            {
                "entity_id": entity_id,
                "channel": channel_of[entity_id],
                "signature_id": sig_id,
                "kind": kind_of.get(sig_id, ""),
                "overlap": round(float(entry["overlap"]), 4),
            }
        )
    not_discovered = [
        {
            "entity_id": entity_id,
            "channel": channel_of[entity_id],
            "n_train_sessions": int(n_train_sessions[entity_id]),
        }
        for entity_id in unlinked
    ]

    last_train_val = max(train_days + val_days) if (train_days or val_days) else 0
    link_dict = {
        "links": links,
        "not_discovered": not_discovered,
        "plugs": {entity_id: {"train_mean_w": train_mean[entity_id]} for entity_id in eligible},
        "train_cutoff_ms": int(bounds[last_train_val + 1]),
    }
    link_path = stage_dir / "shadow_link.json"
    _write_atomic(link_path, canonical_json(link_dict))

    return {
        "stage_dir": str(stage_dir),
        "result": str(link_path),
        "n_links": len(links),
        "n_not_discovered": len(not_discovered),
    }


def run_stage(stage: str, job: Mapping) -> dict:
    """Dispatch stage execution."""
    if stage == "extract":
        return extract(job)
    if stage == "compile_score":
        return compile_score(job)
    if stage == "rescore":
        return rescore(job)
    if stage == "refit":
        return refit(job)
    if stage == "validate":
        return validate(job)
    if stage == "validate_sessions":
        return validate_sessions(job)
    if stage == "shadow_link":
        return shadow_link(job)
    raise ValueError(f"Unknown stage: {stage}")


def run_stage_from_file(stage: str, job_path: str | Path) -> dict:
    """Load job from JSON file and run stage."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    return run_stage(stage, job)


def stage_outputs(stage: str, result: Mapping) -> dict:
    """Return the JSON-safe outputs a stage reports to the supervisor in its `done` event.

    This is the single output contract between the worker and the supervisor: the
    supervisor reads exactly these keys from `_async_run_stage`, and the in-process test
    fixture goes through the same function.
    """
    stage_dir = Path(str(result.get("stage_dir", "")))
    if stage == "extract":
        return {
            "stage_dir": str(stage_dir),
            "digest": str(stage_dir / "digest.json"),
            "meta": str(stage_dir / "digest_meta.json"),
            "episodes": str(stage_dir / "episodes.npz"),
            "n_episodes": int(result.get("n_episodes", 0)),
            "n_clusters": int(result.get("n_clusters", 0)),
            "insufficient_data": bool(result.get("insufficient_data", False)),
        }
    if stage == "compile_score":
        return {
            "stage_dir": str(stage_dir),
            "library": str(result["library"]),
            "scorecard": str(result["scorecard"]),
            "val": str(result["val"]),
            "S": float(result["S"]),
            "conflict_share": float(result["conflict_share"]),
            "accepted": bool(result["accepted"]),
            "n_artefacts": int(result.get("n_artefacts", 0)),
            "unresolved": [str(r) for r in result.get("unresolved", [])],
        }
    if stage == "rescore":
        return {
            "S": float(result["S"]),
            "conflict_share": float(result["conflict_share"]),
            "per_signature": {
                str(sig): {
                    "share": float(v["share"]),
                    "n": int(v["n"]),
                    "validation_error_pct": (
                        None
                        if v.get("validation_error_pct") is None
                        else float(v["validation_error_pct"])
                    ),
                }
                for sig, v in dict(result["per_signature"]).items()
            },
            "days": int(result.get("days", 0)),
        }
    if stage == "refit":
        return {
            "stage_dir": str(stage_dir),
            "result": str(result["result"]),
            "accepted": bool(result["accepted"]),
            "current_S": float(result["current_S"]),
            "new_S": float(result["new_S"]),
        }
    if stage == "validate":
        return {
            "stage_dir": str(stage_dir),
            "report": str(stage_dir / "report.json"),
            "status": str(result.get("status", "")),
        }
    if stage == "validate_sessions":
        return {
            "stage_dir": str(stage_dir),
            "sessions": str(stage_dir / "sessions.json"),
            "n": int(result.get("n", 0)),
        }
    if stage == "shadow_link":
        return {
            "stage_dir": str(stage_dir),
            "result": str(result["result"]),
            "n_links": int(result["n_links"]),
            "n_not_discovered": int(result["n_not_discovered"]),
        }
    raise ValueError(f"Unknown stage: {stage}")


def _start_stdin_watchdog(
    stream: object,
    exit_fn: object,
    completed_event: threading.Event | None = None,
) -> threading.Thread:
    """Start daemon thread monitoring stdin EOF for orphan shutdown."""
    fd = None
    if hasattr(stream, "fileno"):
        try:
            fd = stream.fileno()
        except Exception:
            fd = None

    def _watch() -> None:
        try:
            if fd is not None:
                while os.read(fd, 1024):
                    pass
            else:
                read_fn = getattr(stream, "read", None)
                if read_fn is not None:
                    while read_fn(1024):
                        pass
        except Exception:
            pass
        if (completed_event is None or not completed_event.is_set()) and callable(exit_fn):
            exit_fn(3)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t


def _peak_rss(status_path: str = "/proc/self/status") -> tuple[int, str]:
    """Return this process's own peak resident memory in kB and where the figure came from.

    The source is "vmhwm", "ru_maxrss" or "unavailable" (figure 0), so consumers can say
    when the number is not VmHWM (IN-11).

    Reads VmHWM from /proc/self/status. On Linux, ru_maxrss keeps the spawning process's RSS
    high-water mark across fork/vfork and exec, so a worker spawned by Home Assistant would
    report at least Home Assistant's RSS (03-VERIFICATION reproduction: 326 MB reported vs
    11 MB real). VmHWM belongs to the process's own address space, which exec creates fresh.
    ru_maxrss is only the fallback where /proc is absent (macOS, the BSDs); on Linux that
    fallback figure includes the parent's high-water mark. macOS reports ru_maxrss in bytes,
    so it is converted to kB there. Without the resource module (Windows) it is unavailable.
    """
    try:
        with open(status_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]), "vmhwm"
    except (OSError, ValueError, IndexError):
        pass
    if resource is not None and hasattr(resource, "RUSAGE_SELF"):
        try:
            max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return 0, "unavailable"
        if sys.platform == "darwin":
            max_rss //= 1024
        return max_rss, "ru_maxrss"
    return 0, "unavailable"


def peak_memory_text(stats: Mapping[str, Any]) -> str:
    """Render a stage's peak memory for a log line, marking a figure that is not VmHWM (IN-11)."""
    source = stats.get("max_rss_source")
    if source == "unavailable":
        return "unavailable"
    text = f"{int(stats.get('max_rss_kb', 0)) // 1024} MB"
    if source == "ru_maxrss":
        text += " (ru_maxrss fallback, may include the parent's peak)"
    return text


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for worker script."""
    if argv is None:
        argv = sys.argv[1:]

    # Refuse execution if inside homeassistant process
    if "homeassistant" in sys.modules:
        print(
            json.dumps({"event": "error", "error": "homeassistant imported in worker process"}),
            flush=True,
        )
        return 4

    try:
        if hasattr(os, "nice"):
            os.nice(19)
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="PowerGuesser learning worker")
    parser.add_argument("stage", choices=STAGES, help="Stage to execute")
    parser.add_argument("--job", required=True, help="Path to job.json file")
    parser.add_argument("--watch-stdin", action="store_true", help="Exit on stdin EOF")
    args = parser.parse_args(argv)

    completed_event = threading.Event()
    if args.watch_stdin:
        _start_stdin_watchdog(sys.stdin.buffer, os._exit, completed_event=completed_event)

    nice_val = os.nice(0) if hasattr(os, "nice") else 0
    omp_val = os.environ.get("OMP_NUM_THREADS")
    print(
        json.dumps(
            {
                "event": "env",
                "nice": nice_val,
                "omp": omp_val,
                "ha_imported": False,
            }
        ),
        flush=True,
    )

    t0 = time.monotonic()
    print(json.dumps({"event": "progress", "stage": args.stage, "step": "start"}), flush=True)

    try:
        outputs = stage_outputs(args.stage, run_stage_from_file(args.stage, args.job))
        t1 = time.monotonic()
        print(json.dumps({"event": "progress", "stage": args.stage, "step": "complete"}), flush=True)

        max_rss, max_rss_source = _peak_rss()

        done_payload = {
            "event": "done",
            "outputs": outputs,
            "stats": {
                "wall_s": round(t1 - t0, 1),
                "max_rss_kb": int(max_rss),
                "max_rss_source": max_rss_source,
            },
        }
        completed_event.set()
        print(json.dumps(done_payload), flush=True)
        return 0
    except Exception as ex:
        completed_event.set()
        print(json.dumps({"event": "error", "error": f"{type(ex).__name__}: {ex}"}), flush=True)
        return 1
