"""Feature computation, DBSCAN pre-clustering, and canonical digest generation."""

from collections.abc import Mapping, Sequence
import hashlib
import json
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

RISE_SCALE = 0.08
DUR_SCALE = 0.35
EPS = 1.0
MIN_SAMPLES = 4
MAX_CLUSTERS = 40


def switched_features(table: Mapping[str, np.ndarray]) -> np.ndarray:
    """Extract 2D scaled log-space features for switched episodes."""
    n = len(table["rise_w"])
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    rise = np.maximum(table["rise_w"], 1.0)
    dur = np.maximum(table["dur_s"], 1.0)
    return np.column_stack([np.log(rise) / RISE_SCALE, np.log(dur) / DUR_SCALE])


def session_features(table: Mapping[str, np.ndarray]) -> np.ndarray:
    """Extract 4D scaled log-space features for sessions (plan 03-01: ln energy, ln duration, peak, n_segments).

    Scaled like the switched features so the same DBSCAN eps applies: the peak is a power
    level (RISE_SCALE, like the switched rise), the duration uses DUR_SCALE, and energy and
    segment count (both dominated by how long the program runs) use DUR_SCALE as well.
    """
    n = len(table["energy_wh"])
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    energy = np.maximum(table["energy_wh"], 1.0)
    dur = np.maximum(table["dur_s"], 1.0)
    peak = np.maximum(table["peak_w"], 1.0)
    n_seg = np.maximum(table["n_segments"].astype(np.float64), 1.0)
    return np.column_stack(
        [
            np.log(energy) / DUR_SCALE,
            np.log(dur) / DUR_SCALE,
            np.log(peak) / RISE_SCALE,
            np.log(n_seg) / DUR_SCALE,
        ]
    )


def dbscan(X: np.ndarray, eps: float = EPS, min_samples: int = MIN_SAMPLES) -> np.ndarray:
    """Deterministic spatial DBSCAN without pairwise distance matrices."""
    n = len(X)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    tree = scipy.spatial.cKDTree(X)
    nbrs = tree.query_ball_point(X, eps)
    core_mask = np.array([len(nbr_list) >= min_samples for nbr_list in nbrs], dtype=bool)
    core_indices = np.where(core_mask)[0]

    if len(core_indices) == 0:
        return np.full(n, -1, dtype=np.int32)

    # Build core-core sparse adjacency
    row_ind: list[int] = []
    col_ind: list[int] = []
    for u in core_indices:
        for v in nbrs[u]:
            if core_mask[v]:
                row_ind.append(u)
                col_ind.append(v)

    data = np.ones(len(row_ind), dtype=bool)
    adj = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    _, raw_labels = scipy.sparse.csgraph.connected_components(adj, directed=False)

    labels = np.full(n, -1, dtype=np.int32)
    for u in core_indices:
        labels[u] = raw_labels[u]

    # Assign non-core border points to nearest core neighbour
    for i in range(n):
        if not core_mask[i]:
            core_nbrs = [c for c in nbrs[i] if core_mask[c]]
            if core_nbrs:
                distances = np.linalg.norm(X[core_nbrs] - X[i], axis=1)
                nearest_core = core_nbrs[int(np.argmin(distances))]
                labels[i] = labels[nearest_core]

    # Relabel unique cluster IDs to 0, 1, 2, ...
    unique_clusters = sorted(set(labels) - {-1})
    remap = {c: idx for idx, c in enumerate(unique_clusters)}
    result = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if labels[i] in remap:
            result[i] = remap[labels[i]]

    return result


def format_rle_exemplar(rle: Sequence[Sequence[int]]) -> str:
    """Render RLE segment list as concise string: +180x2,+2050x20,..."""
    parts: list[str] = []
    for seg in rle:
        lvl = int(seg[0])
        cnt = int(seg[1])
        sign = "+" if lvl >= 0 else ""
        parts.append(f"{sign}{lvl}x{cnt}")
    return ",".join(parts)


def compute_co_occurrence(
    starts: np.ndarray,
    labels: np.ndarray,
    id_map: Mapping[int, str],
    window_samples: int = 2,
) -> dict[str, list[dict]]:
    """Compute co-occurring clusters whose episodes start within window_samples."""
    result: dict[str, list[dict]] = {}
    unique_labels = sorted(id_map.keys())

    for lab in unique_labels:
        cid = id_map[lab]
        idx_a = np.where(labels == lab)[0]
        n_a = len(idx_a)
        if n_a == 0:
            result[cid] = []
            continue

        starts_a = starts[idx_a]
        co_list: list[dict] = []

        for other_lab in unique_labels:
            if other_lab == lab:
                continue
            other_cid = id_map[other_lab]
            idx_b = np.where(labels == other_lab)[0]
            if len(idx_b) == 0:
                continue
            starts_b = starts[idx_b]

            # Count episodes in A that start within window_samples of any episode in B
            # Using searchsorted for fast range queries
            matched = 0
            for s in starts_a:
                pos = np.searchsorted(starts_b, s)
                near = False
                if pos < len(starts_b) and abs(starts_b[pos] - s) <= window_samples:
                    near = True
                elif pos > 0 and abs(starts_b[pos - 1] - s) <= window_samples:
                    near = True
                if near:
                    matched += 1

            share = matched / float(n_a)
            if round(share, 2) > 0.0:
                co_list.append({"id": other_cid, "share": round(share, 2)})

        co_list.sort(key=lambda item: (-item["share"], item["id"]))
        result[cid] = co_list[:3]

    return result


def canonical_json(obj: object) -> bytes:
    """Format object as deterministic canonical UTF-8 JSON bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def input_digest(digest: Mapping) -> str:
    """Compute sha256 hex digest of canonical digest JSON."""
    return hashlib.sha256(canonical_json(digest)).hexdigest()


def build_digest(
    table: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    grid_s: int,
    threshold_w: float,
    data_days: int,
    residual_kwh: float = 0.0,
    phases_mapped: bool = False,
    plugs_data: Mapping[int, Mapping[str, object]] | None = None,
    all_rle: Sequence[Sequence[Sequence[int]]] | None = None,
) -> tuple[dict, dict]:
    """Build canonical digest dictionary and companion local meta."""
    total_ep_energy_wh = float(np.sum(table["energy_wh"])) if len(table["energy_wh"]) > 0 else 0.0
    if residual_kwh <= 0.0:
        residual_kwh = round(total_ep_energy_wh / 1000.0, 1)

    unique_clusters = sorted(set(labels) - {-1})
    cluster_records: list[dict] = []

    has_phases = bool(
        phases_mapped
        and "l1_share" in table
        and not np.all(np.isnan(table["l1_share"]))
    )

    for l_id in unique_clusters:
        idx = np.where(labels == l_id)[0]
        n_members = len(idx)
        kind = int(table["kind"][idx[0]])
        rises = table["rise_w"][idx]
        falls = table["fall_w"][idx]
        durs_min = table["dur_s"][idx] / 60.0
        energies = table["energy_wh"][idx]
        cluster_energy_wh = float(np.sum(energies))

        rise_p10, rise_p50, rise_p90 = np.percentile(rises, [10, 50, 90])
        fall_p10, fall_p50, fall_p90 = np.percentile(falls, [10, 50, 90])
        dur_p10, dur_p50, dur_p90 = np.percentile(durs_min, [10, 50, 90])

        e_share = cluster_energy_wh / (residual_kwh * 1000.0) if residual_kwh > 0 else 0.0

        # Hour and dow histograms
        hour_counts = [0] * 24
        dow_counts = [0] * 7
        if "hour" in table:
            for h in table["hour"][idx]:
                if 0 <= h < 24:
                    hour_counts[int(h)] += 1
        if "dow" in table:
            for d in table["dow"][idx]:
                if 0 <= d < 7:
                    dow_counts[int(d)] += 1

        # Phase median shares
        phase_dict: dict[str, float] | None = None
        if has_phases:
            p1_finite = table["l1_share"][idx][np.isfinite(table["l1_share"][idx])]
            p2_finite = table["l2_share"][idx][np.isfinite(table["l2_share"][idx])]
            p3_finite = table["l3_share"][idx][np.isfinite(table["l3_share"][idx])]
            if len(p1_finite) > 0:
                phase_dict = {
                    "l1": round(float(np.median(p1_finite)), 2),
                    "l2": round(float(np.median(p2_finite)), 2),
                    "l3": round(float(np.median(p3_finite)), 2),
                }

        # Exemplars for session clusters
        exemplars: list[str] = []
        if kind == 1 and all_rle is not None:
            med_e = float(np.median(energies))
            dist_to_med = [(abs(energies[k] - med_e), table["start_i"][idx[k]], idx[k]) for k in range(n_members)]
            dist_to_med.sort(key=lambda item: (item[0], item[1]))
            for _, _, member_idx in dist_to_med[:3]:
                if member_idx < len(all_rle):
                    exemplars.append(format_rle_exemplar(all_rle[member_idx]))

        record = {
            "kind": "session" if kind == 1 else "switched",
            "n": n_members,
            "rise_w": {
                "p10": int(round(rise_p10)),
                "p50": int(round(rise_p50)),
                "p90": int(round(rise_p90)),
            },
            "fall_w": {
                "p10": int(round(fall_p10)),
                "p50": int(round(fall_p50)),
                "p90": int(round(fall_p90)),
            },
            "dur_min": {
                "p10": round(float(dur_p10), 1),
                "p50": round(float(dur_p50), 1),
                "p90": round(float(dur_p90), 1),
            },
            "energy_share": round(e_share, 3),
            "hour": hour_counts,
            "dow": dow_counts,
            "phase": phase_dict,
        }
        if kind == 1:
            record["exemplars"] = exemplars

        cluster_records.append(
            {
                "raw_label": l_id,
                "n": n_members,
                "explained_energy": cluster_energy_wh,
                "rise_p50": float(rise_p50),
                "record": record,
            }
        )

    # Sort clusters by explained energy descending, ties by rise p50 descending, then n descending
    cluster_records.sort(
        key=lambda cr: (cr["explained_energy"], cr["rise_p50"], cr["n"]),
        reverse=True,
    )

    # Map cluster index to "c01", "c02", ...
    id_map: dict[int, str] = {}
    for i, cr in enumerate(cluster_records[:MAX_CLUSTERS]):
        cid = f"c{i + 1:02d}"
        id_map[cr["raw_label"]] = cid

    # Compute co-occurrence
    starts_arr = table["start_i"] if "start_i" in table else np.empty(0, dtype=np.int64)
    conc_map = compute_co_occurrence(starts_arr, labels, id_map)

    digest_clusters: list[dict] = []
    meta_clusters: dict[str, int] = {}

    for i, cr in enumerate(cluster_records[:MAX_CLUSTERS]):
        cid = f"c{i + 1:02d}"
        rec = dict(cr["record"])
        rec["conc_with"] = conc_map.get(cid, [])
        ordered_rec = {"id": cid}
        ordered_rec.update(rec)
        digest_clusters.append(ordered_rec)
        meta_clusters[cid] = cr["n"]

    # "other" summary if > MAX_CLUSTERS
    other_dict: dict | None = None
    if len(cluster_records) > MAX_CLUSTERS:
        other_records = cluster_records[MAX_CLUSTERS:]
        other_n = sum(cr["n"] for cr in other_records)
        other_energy = sum(cr["explained_energy"] for cr in other_records)
        other_share = other_energy / (residual_kwh * 1000.0) if residual_kwh > 0 else 0.0
        other_dict = {
            "n": other_n,
            "energy_share": round(other_share, 3),
        }

    clustered_energy_wh = sum(cr["explained_energy"] for cr in cluster_records[:MAX_CLUSTERS])
    clustered_share = clustered_energy_wh / (residual_kwh * 1000.0) if residual_kwh > 0 else 0.0

    # Build plug reference lines
    plug_lines: list[dict] = []
    plug_meta: dict[str, int] = {}
    if plugs_data:
        sorted_plugs = []
        for ch_id, p_info in plugs_data.items():
            p_table = p_info["sessions"]
            tot_plug_energy = float(np.sum(p_table["energy_wh"])) if len(p_table["energy_wh"]) > 0 else 0.0
            sorted_plugs.append((ch_id, tot_plug_energy, p_table, p_info["rle"]))
        sorted_plugs.sort(key=lambda item: (-item[1], item[0]))

        for idx_p, (ch_id, tot_energy, p_table, p_rle) in enumerate(sorted_plugs):
            pid = f"p{idx_p + 1}"
            plug_meta[pid] = ch_id
            n_plug = len(p_table["dur_s"])
            if n_plug > 0:
                p_durs_min = p_table["dur_s"] / 60.0
                p_dur_p10, p_dur_p50, p_dur_p90 = np.percentile(p_durs_min, [10, 50, 90])
                p_peak = int(round(float(np.max(p_table["peak_w"]))))
                tot_kwh = tot_energy / 1000.0
                e_per_day = round(tot_kwh / max(1, data_days), 2)
            else:
                p_dur_p10 = p_dur_p50 = p_dur_p90 = 0.0
                p_peak = 0
                e_per_day = 0.0

            # Plug hour and dow histograms
            p_hour = [0] * 24
            p_dow = [0] * 7
            if "hour" in p_table:
                for h in p_table["hour"]:
                    if 0 <= h < 24:
                        p_hour[int(h)] += 1
            if "dow" in p_table:
                for d in p_table["dow"]:
                    if 0 <= d < 7:
                        p_dow[int(d)] += 1

            p_exemplars: list[str] = []
            if n_plug > 0 and p_rle:
                p_energies = p_table["energy_wh"]
                p_med_e = float(np.median(p_energies))
                p_dist = [(abs(p_energies[k] - p_med_e), p_table["start_i"][k], k) for k in range(n_plug)]
                p_dist.sort(key=lambda item: (item[0], item[1]))
                for _, _, mem_idx in p_dist[:3]:
                    if mem_idx < len(p_rle):
                        p_exemplars.append(format_rle_exemplar(p_rle[mem_idx]))

            plug_lines.append(
                {
                    "id": pid,
                    "n": n_plug,
                    "energy_kwh_per_day": e_per_day,
                    "dur_min": {
                        "p10": round(float(p_dur_p10), 1),
                        "p50": round(float(p_dur_p50), 1),
                        "p90": round(float(p_dur_p90), 1),
                    },
                    "peak_w": p_peak,
                    "hour": p_hour,
                    "dow": p_dow,
                    "exemplars": p_exemplars,
                }
            )

    header = {
        "grid_s": int(grid_s),
        "days": int(data_days),
        "residual_kwh": round(float(residual_kwh), 1),
        "clustered_share": round(float(clustered_share), 3),
        "edge_threshold_w": int(round(threshold_w)),
        "phases_mapped": bool(phases_mapped),
        "plugs": len(plug_lines),
    }

    digest: dict[str, object] = {
        "header": header,
        "clusters": digest_clusters,
    }
    if other_dict is not None:
        digest["other"] = other_dict
    if plug_lines:
        digest["plugs"] = plug_lines

    digest_hash = input_digest(digest)
    meta = {
        "input_digest": digest_hash,
        "clusters": meta_clusters,
        "cluster_map": {cid: int(raw_lab) for raw_lab, cid in id_map.items()},
        "plugs": plug_meta,
        "edge_threshold_w": int(round(threshold_w)),
        "grid_s": int(grid_s),
    }

    return digest, meta
