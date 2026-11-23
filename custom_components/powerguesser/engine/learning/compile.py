"""Compile validated LLM proposals into refitted SignatureArtefact v1 dicts."""

from collections.abc import Mapping, Sequence
import hashlib
import numpy as np

from .artefact import Box, SignatureArtefact, State
from .prompts import PROMPT_VERSION
from .proposal import sanitize_label
from .stages import expand_rle, fit_stages

MIN_TRAIN_EPISODES = 3
WIDEN_FRAC = 0.05


def fit_state_boxes(
    table: Mapping[str, np.ndarray],
    indices: np.ndarray,
    threshold_w: float,
    *,
    name: str,
) -> State:
    """Fit one state's boxes from its member episode rows (moved as-is from compile_proposal).

    The rise, fall and plateau boxes widen by max(threshold_w / 2, WIDEN_FRAC x |p|);
    the duration box widens by WIDEN_FRAC x p only (lo floored at 0) — the duration
    rule is deliberately NOT unified with the other boxes. Phase-share boxes are
    p5/p95 rounded to 2 decimals when the table carries phase shares.
    """
    st_idx_arr = np.asarray(indices, dtype=np.int64)

    # Rise W box
    r_p5, r_p50, r_p95 = np.percentile(table["rise_w"][st_idx_arr], [5, 50, 95])
    r_lo = r_p5 - max(threshold_w / 2.0, WIDEN_FRAC * abs(r_p5))
    r_hi = r_p95 + max(threshold_w / 2.0, WIDEN_FRAC * abs(r_p95))
    rise_box = Box(lo=round(float(r_lo), 1), p50=round(float(r_p50), 1), hi=round(float(r_hi), 1))

    # Fall W box
    f_p5, f_p50, f_p95 = np.percentile(table["fall_w"][st_idx_arr], [5, 50, 95])
    f_lo = f_p5 - max(threshold_w / 2.0, WIDEN_FRAC * abs(f_p5))
    f_hi = f_p95 + max(threshold_w / 2.0, WIDEN_FRAC * abs(f_p95))
    fall_box = Box(lo=round(float(f_lo), 1), p50=round(float(f_p50), 1), hi=round(float(f_hi), 1))

    # Duration s box
    d_p5, d_p50, d_p95 = np.percentile(table["dur_s"][st_idx_arr], [5, 50, 95])
    d_lo = max(0.0, d_p5 - WIDEN_FRAC * d_p5)
    d_hi = d_p95 + WIDEN_FRAC * d_p95
    dur_box = Box(lo=round(float(d_lo), 1), p50=round(float(d_p50), 1), hi=round(float(d_hi), 1))

    # Plateau W box
    pl_p5, pl_p50, pl_p95 = np.percentile(table["plateau_w"][st_idx_arr], [5, 50, 95])
    pl_lo = pl_p5 - max(threshold_w / 2.0, WIDEN_FRAC * abs(pl_p5))
    pl_hi = pl_p95 + max(threshold_w / 2.0, WIDEN_FRAC * abs(pl_p95))
    plateau_box = Box(lo=round(float(pl_lo), 1), p50=round(float(pl_p50), 1), hi=round(float(pl_hi), 1))

    # Phase share boxes
    phase_share: dict[str, tuple[float, float]] | None = None
    if "l1_share" in table:
        l1_sample = table["l1_share"][st_idx_arr]
        if np.any(np.isfinite(l1_sample)):
            phase_share = {}
            for ph in ("l1", "l2", "l3"):
                shares = table[f"{ph}_share"][st_idx_arr]
                finite_s = shares[np.isfinite(shares)]
                if len(finite_s) > 0:
                    p5, p95 = np.percentile(finite_s, [5, 95])
                    phase_share[ph] = (round(float(p5), 2), round(float(p95), 2))
                else:
                    phase_share[ph] = (0.0, 0.0)

    return State(
        name=name,
        rise_w=rise_box,
        fall_w=fall_box,
        dur_s=dur_box,
        plateau_w=plateau_box,
        phase_share=phase_share,
    )


def compile_proposal(
    proposal: Mapping,
    table: Mapping[str, np.ndarray],
    cluster_index: Mapping[str, Sequence[int] | np.ndarray],
    *,
    run_id: str,
    round_no: int,
    model: str,
    input_digest: str,
    extraction: Mapping[str, object],
    threshold_w: float,
) -> tuple[list[SignatureArtefact], list[str]]:
    """Compile proposal devices into SignatureArtefact v1 objects refitted from train episodes."""
    artefacts: list[SignatureArtefact] = []
    unresolved_refs: list[str] = []

    devices = proposal.get("devices", [])
    block_arr = table.get("block", np.zeros(len(table.get("start_i", [])), dtype=np.int32))

    for dev in devices:
        ref = dev.get("ref", "")
        member_clusters = dev.get("member_clusters", [])

        # Collect train episodes for member clusters
        train_indices_list: list[int] = []
        for c_ref in member_clusters:
            c_indices = cluster_index.get(c_ref, [])
            for idx in c_indices:
                if block_arr[idx] == 0:
                    train_indices_list.append(int(idx))

        # Deduplicate
        dev_train_indices = sorted(set(train_indices_list))

        if len(dev_train_indices) < MIN_TRAIN_EPISODES:
            unresolved_refs.append(ref)
            continue

        # Compile states
        states: list[State] = []
        dev_states = dev.get("states", [])

        for st in dev_states:
            st_cref = st.get("cluster_ref", "")
            raw_st_indices = cluster_index.get(st_cref, [])
            st_indices = [int(i) for i in raw_st_indices if block_arr[i] == 0]

            if len(st_indices) == 0:
                st_indices = dev_train_indices

            states.append(
                fit_state_boxes(
                    table,
                    np.array(st_indices, dtype=np.int64),
                    threshold_w,
                    name=str(st.get("name", "state")),
                )
            )

        # Time prior (Laplace +1 smoothed and normalised)
        dev_idx_arr = np.array(dev_train_indices, dtype=np.int64)
        n_dev_train = len(dev_train_indices)

        hour_counts = np.bincount(table["hour"][dev_idx_arr].astype(int), minlength=24)[:24]
        hour_smoothed = (hour_counts + 1.0) / (n_dev_train + 24.0)

        dow_counts = np.bincount(table["dow"][dev_idx_arr].astype(int), minlength=7)[:7]
        dow_smoothed = (dow_counts + 1.0) / (n_dev_train + 7.0)

        time_prior = {
            "hour": [round(float(x), 4) for x in hour_smoothed],
            "dow": [round(float(x), 4) for x in dow_smoothed],
        }

        # Session boxes for program appliances
        tracker_kind = dev.get("tracker_kind", "switched")
        session: dict[str, object] | None = None
        if tracker_kind == "program":
            e_p5, e_p50, e_p95 = np.percentile(table["energy_wh"][dev_idx_arr], [5, 50, 95])
            e_lo = max(0.0, e_p5 - max(threshold_w / 2.0, WIDEN_FRAC * abs(e_p5)))
            e_hi = e_p95 + max(threshold_w / 2.0, WIDEN_FRAC * abs(e_p95))
            e_box = Box(lo=round(float(e_lo), 1), p50=round(float(e_p50), 1), hi=round(float(e_hi), 1))

            sd_p5, sd_p50, sd_p95 = np.percentile(table["dur_s"][dev_idx_arr], [5, 50, 95])
            sd_lo = max(0.0, sd_p5 - WIDEN_FRAC * sd_p5)
            sd_hi = sd_p95 + WIDEN_FRAC * sd_p95
            sd_box = Box(lo=round(float(sd_lo), 1), p50=round(float(sd_p50), 1), hi=round(float(sd_hi), 1))

            # n_segments from rle_offset
            n_segments_list: list[int] = []
            if "rle_offset" in table:
                rle_off = table["rle_offset"]
                for i in dev_train_indices:
                    if i + 1 < len(rle_off):
                        n_segments_list.append(int(rle_off[i + 1] - rle_off[i]))

            if n_segments_list:
                ns_p5, ns_p50, ns_p95 = np.percentile(n_segments_list, [5, 50, 95])
            else:
                ns_p5 = ns_p50 = ns_p95 = 1.0

            ns_lo = max(1.0, ns_p5 - WIDEN_FRAC * ns_p5)
            ns_hi = ns_p95 + WIDEN_FRAC * ns_p95
            ns_box = Box(lo=round(float(ns_lo), 1), p50=round(float(ns_p50), 1), hi=round(float(ns_hi), 1))

            # Up to 3 member RLE exemplars nearest median energy
            exemplars: list[str] = []
            if "rle_offset" in table and "rle_level_w" in table and "rle_n" in table:
                med_e = float(np.median(table["energy_wh"][dev_idx_arr]))
                sorted_by_med = sorted(
                    dev_train_indices,
                    key=lambda idx: abs(table["energy_wh"][idx] - med_e),
                )
                rle_off = table["rle_offset"]
                rle_lvl = table["rle_level_w"]
                rle_cnt = table["rle_n"]

                for s_idx in sorted_by_med[:3]:
                    if s_idx + 1 < len(rle_off):
                        start_k = int(rle_off[s_idx])
                        end_k = int(rle_off[s_idx + 1])
                        parts = []
                        for k in range(start_k, end_k):
                            lvl = int(rle_lvl[k])
                            cnt = int(rle_cnt[k])
                            sign = "+" if lvl >= 0 else ""
                            parts.append(f"{sign}{lvl}x{cnt}")
                        if parts:
                            exemplars.append(",".join(parts))

            # Local stage template (D-27, LRN-03): expand every train member's
            # RLE and fit stages by banded DTW; the LLM never supplies stage
            # numbers.  Absent when too few members exist.
            members: list[np.ndarray] = []
            if "rle_offset" in table and "rle_level_w" in table and "rle_n" in table:
                mem_off = table["rle_offset"]
                mem_lvl = table["rle_level_w"]
                mem_cnt = table["rle_n"]
                for i in dev_train_indices:
                    if i + 1 < len(mem_off):
                        start_k = int(mem_off[i])
                        end_k = int(mem_off[i + 1])
                        members.append(
                            expand_rle(mem_lvl[start_k:end_k], mem_cnt[start_k:end_k])
                        )
            template = None
            if members:
                template = fit_stages(
                    members,
                    threshold_w,
                    float(extraction.get("grid_s", 30) or 30),
                )

            session = {
                "energy_wh": e_box.to_dict(),
                "dur_s": sd_box.to_dict(),
                "n_segments": ns_box.to_dict(),
                "exemplars": exemplars,
            }
            if template is not None:
                session["stages"] = template["stages"]
                session["terminal"] = template["terminal"]

        signature_id = hashlib.sha256(f"{run_id}:{round_no}:{ref}".encode("utf-8")).hexdigest()[:32]
        sanitized_label = sanitize_label(str(dev.get("label", "")))

        provenance = {
            "run_id": run_id,
            "round": round_no,
            "device_ref": ref,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "input_digest": input_digest,
            "member_clusters": list(member_clusters),
            "reference_plug": str(dev.get("reference_plug", "")),
        }

        scores = {
            "train": None,
            "val": None,
            "validated": False,
            "conflicts": 0,
        }

        extraction_dict = {
            "grid_s": extraction.get("grid_s", 30),
            "edge_threshold_w": extraction.get("edge_threshold_w", int(round(threshold_w))),
            "merge_samples": extraction.get("merge_samples", 2),
            "pair_tol": extraction.get("pair_tol", 0.25),
        }

        artefact = SignatureArtefact(
            schema_version=1,
            signature_id=signature_id,
            kind=tracker_kind,
            label=sanitized_label,
            category=str(dev.get("category", "unknown")),
            extraction=extraction_dict,
            states=tuple(states),
            session=session,
            time_prior=time_prior,
            constraints={"max_concurrent": 1},
            provenance=provenance,
            scores=scores,
        )
        artefacts.append(artefact)

    return artefacts, unresolved_refs
