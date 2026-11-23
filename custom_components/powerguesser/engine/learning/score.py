"""Local replay scoring, consistency proxy, monotone acceptance, and train scorecards.

Shares and S are fractions of the block's residual energy (positive residual over the
block's whole days, the digest's residual definition), passed in as residual_wh; without
it the episode energy sum is used, which double-counts overlapping switched episodes and
sessions and is not residual energy (WR-12). ACCEPT_MARGIN is therefore 0.5 percentage
points of residual energy.

Train replay of member episodes is near-tautological by design (the boxes are percentiles
of those same members, widened), so train scorecards describe coverage, not discrimination,
and must never be used to tune constants or decide acceptance; only the validation S decides.

D-20: each residual sample is attributed once across switched and session claims. A
switched episode claimed by one signature that overlaps a session episode claimed by a
different signature covers samples whose energy the session claim already includes; the
overlap is removed from the session claim (ReplayResult.overlap_wh) so explained shares
and S are not inflated.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import numpy as np

from .artefact import SignatureArtefact

ACCEPT_MARGIN = 0.005

# D-17/D-25: the proposal gate's basis — both versions replay on the learning
# run's own validation block (the chronologically last ~30 % of usable days),
# which is out-of-sample for artefacts fit from train-block members only.
GATE_BASIS = "learning_split_val"


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying signature artefacts across a block of episodes."""

    per_signature: dict[str, dict[str, object]]
    conflict_wh: float
    unique_wh: float
    denominator_wh: float
    claims: dict[str, set[int]]
    conflicts_with: dict[str, set[str]]
    conflict_episodes: set[int]
    overlap_wh: float = 0.0


def replay(
    artefacts: Sequence[SignatureArtefact],
    table: Mapping[str, np.ndarray],
    block: int,
    residual_wh: float | None = None,
) -> ReplayResult:
    """Replay artefacts against episodes in the specified block (0=train, 1=val).

    residual_wh is the block's residual energy and becomes the denominator of every share
    and of S; the worker always passes it.
    """
    block_arr = table.get("block", np.empty(0, dtype=np.int32))
    block_indices = np.where(block_arr == block)[0]

    n_episodes = len(block_indices)
    if n_episodes == 0:
        return ReplayResult(
            per_signature={
                art.signature_id: {"n": 0, "claimed_wh": 0.0, "unique_wh": 0.0}
                for art in artefacts
            },
            conflict_wh=0.0,
            unique_wh=0.0,
            denominator_wh=float(residual_wh) if residual_wh is not None else 0.0,
            claims={art.signature_id: set() for art in artefacts},
            conflicts_with={art.signature_id: set() for art in artefacts},
            conflict_episodes=set(),
            overlap_wh=0.0,
        )

    denominator_wh = (
        float(residual_wh)
        if residual_wh is not None
        else float(np.sum(table["energy_wh"][block_indices]))
    )

    claims: dict[str, set[int]] = {art.signature_id: set() for art in artefacts}
    episode_to_sigs: dict[int, list[str]] = {int(idx): [] for idx in block_indices}

    for art in artefacts:
        sig_id = art.signature_id

        if art.kind == "switched":
            for idx in block_indices:
                if int(table["kind"][idx]) != 0:
                    continue
                ep_r = float(table["rise_w"][idx])
                ep_f = float(table["fall_w"][idx])
                ep_d = float(table["dur_s"][idx])
                ep_pl = float(table["plateau_w"][idx])

                for st in art.states:
                    if not (st.rise_w.lo <= ep_r <= st.rise_w.hi):
                        continue

                    # Check fall: handle negative or positive bounds
                    if st.fall_w.hi <= 0:
                        fall_match = (st.fall_w.lo <= ep_f <= st.fall_w.hi) or (
                            st.fall_w.lo <= -abs(ep_f) <= st.fall_w.hi
                        )
                    else:
                        fall_match = (st.fall_w.lo <= abs(ep_f) <= st.fall_w.hi)
                    if not fall_match:
                        continue

                    if not (st.dur_s.lo <= ep_d <= st.dur_s.hi):
                        continue
                    if not (st.plateau_w.lo <= ep_pl <= st.plateau_w.hi):
                        continue

                    # Phase check if present
                    if st.phase_share is not None and "l1_share" in table:
                        ph_match = True
                        for ph in ("l1", "l2", "l3"):
                            if ph in st.phase_share:
                                sh = float(table[f"{ph}_share"][idx])
                                if np.isfinite(sh):
                                    plo, phi = st.phase_share[ph]
                                    if not (plo <= sh <= phi):
                                        ph_match = False
                                        break
                        if not ph_match:
                            continue

                    # Matched state
                    claims[sig_id].add(int(idx))
                    episode_to_sigs[int(idx)].append(sig_id)
                    break

        elif art.kind == "program" and art.session is not None:
            sess = art.session
            e_box = sess.get("energy_wh", {})
            d_box = sess.get("dur_s", {})
            ns_box = sess.get("n_segments", {})

            for idx in block_indices:
                if int(table["kind"][idx]) != 1:
                    continue
                ep_e = float(table["energy_wh"][idx])
                ep_d = float(table["dur_s"][idx])

                if not (float(e_box.get("lo", 0.0)) <= ep_e <= float(e_box.get("hi", 1e9))):
                    continue
                if not (float(d_box.get("lo", 0.0)) <= ep_d <= float(d_box.get("hi", 1e9))):
                    continue

                if "rle_offset" in table and idx + 1 < len(table["rle_offset"]):
                    n_seg = int(table["rle_offset"][idx + 1] - table["rle_offset"][idx])
                    if not (float(ns_box.get("lo", 1.0)) <= n_seg <= float(ns_box.get("hi", 1e9))):
                        continue

                claims[sig_id].add(int(idx))
                episode_to_sigs[int(idx)].append(sig_id)

    unique_wh = 0.0
    conflict_wh = 0.0
    conflict_episodes: set[int] = set()
    conflicts_with: dict[str, set[str]] = {art.signature_id: set() for art in artefacts}

    per_signature: dict[str, dict[str, object]] = {
        art.signature_id: {"n": len(claims[art.signature_id]), "claimed_wh": 0.0, "unique_wh": 0.0}
        for art in artefacts
    }

    for idx, sigs in episode_to_sigs.items():
        ep_energy = float(table["energy_wh"][idx])
        if len(sigs) == 1:
            s_id = sigs[0]
            unique_wh += ep_energy
            per_signature[s_id]["unique_wh"] = float(per_signature[s_id]["unique_wh"]) + ep_energy
            per_signature[s_id]["claimed_wh"] = float(per_signature[s_id]["claimed_wh"]) + ep_energy
        elif len(sigs) > 1:
            conflict_wh += ep_energy
            conflict_episodes.add(idx)
            for s_id in sigs:
                per_signature[s_id]["claimed_wh"] = float(per_signature[s_id]["claimed_wh"]) + ep_energy
                for other_s in sigs:
                    if other_s != s_id:
                        conflicts_with[s_id].add(other_s)

    # D-20: attribute each residual sample once. The switched claim takes precedence
    # over a session claim covering the same samples (the session's energy includes
    # the switched load's energy), so the overlapping energy is removed from the
    # session claim instead of being counted for both signatures. Only single-
    # claimant episodes take part: existing conflict handling stays authoritative for
    # multi-claimant switched episodes, and claims/conflicts_with/n are untouched.
    overlap_wh = 0.0
    single_sessions: list[tuple[int, int, str]] = []
    single_switched: list[tuple[int, int, str, float]] = []
    for idx, sigs in episode_to_sigs.items():
        if len(sigs) != 1:
            continue
        start_i = int(table["start_i"][idx])
        end_i = int(table["end_i"][idx])
        if int(table["kind"][idx]) == 1:
            single_sessions.append((start_i, end_i, sigs[0]))
        else:
            single_switched.append((start_i, end_i, sigs[0], float(table["energy_wh"][idx])))

    if single_sessions and single_switched:
        # Sort-and-sweep over start_i: O((n + m) log(n + m)). A switched episode that
        # ends at or before the session start cannot overlap this or any later session.
        single_sessions.sort()
        single_switched.sort()
        switched_pos = 0
        n_switched = len(single_switched)
        for s_start, s_end, p_sig in single_sessions:
            while switched_pos < n_switched and single_switched[switched_pos][0] < s_end:
                k_start, k_end, k_sig, k_energy = single_switched[switched_pos]
                switched_pos += 1
                if k_sig == p_sig or k_end <= s_start:
                    continue
                overlap_samples = min(s_end, k_end) - max(s_start, k_start)
                if overlap_samples <= 0:
                    continue
                removed = k_energy * overlap_samples / max(1, k_end - k_start)
                removed = min(removed, float(per_signature[p_sig]["unique_wh"]))
                if removed <= 0.0:
                    continue
                per_signature[p_sig]["unique_wh"] = float(per_signature[p_sig]["unique_wh"]) - removed
                per_signature[p_sig]["claimed_wh"] = max(
                    0.0, float(per_signature[p_sig]["claimed_wh"]) - removed
                )
                unique_wh -= removed
                overlap_wh += removed

    return ReplayResult(
        per_signature=per_signature,
        conflict_wh=conflict_wh,
        unique_wh=unique_wh,
        denominator_wh=denominator_wh,
        claims=claims,
        conflicts_with=conflicts_with,
        conflict_episodes=conflict_episodes,
        overlap_wh=overlap_wh,
    )


def block_score(result: ReplayResult) -> tuple[float, float]:
    """Compute validation score S and conflict_share from replay results."""
    denom = result.denominator_wh
    if denom <= 0.0:
        return 0.0, 0.0
    s_score = (result.unique_wh - result.conflict_wh) / denom
    conflict_share = result.conflict_wh / denom
    return float(s_score), float(conflict_share)


def gate_proposal(
    proposed: SignatureArtefact,
    active: SignatureArtefact,
    table: Mapping[str, np.ndarray],
    residual_wh: float,
) -> dict[str, object]:
    """D-17/D-25 no-regression gate for a re-learn proposal (zero tolerance).

    Both the proposed and the active version replay ALONE on the same run's
    validation block (``block=1``) against the block's residual energy, using
    exactly worker.refit's keep-or-reject rule (replay each artefact alone,
    ``block_score``, no ACCEPT_MARGIN). The active version claims under the
    proposed one's signature id so both sides score under one key. A proposal
    passes only when proposed_S >= active_S: the user is offered a parameter
    update only when it replays no worse than the frozen active version on the
    same recent window. Stored scores from different splits are never compared
    — both numbers come from this run's own validation block (RESEARCH
    Pitfall 9: a no-regression check, not proof of improvement).
    """
    proposed_S, _prop_conf = block_score(
        replay([proposed], table, block=1, residual_wh=residual_wh)
    )
    active_as_proposed = replace(active, signature_id=proposed.signature_id)
    active_S, _act_conf = block_score(
        replay([active_as_proposed], table, block=1, residual_wh=residual_wh)
    )
    return {
        "passed": bool(proposed_S >= active_S),
        "proposed_S": round(float(proposed_S), 5),
        "active_S": round(float(active_S), 5),
        "basis": GATE_BASIS,
    }


def accept(
    new_score: tuple[float, float] | Mapping[str, float],
    best_score: tuple[float, float] | Mapping[str, float] | None,
) -> bool:
    """Evaluate monotone acceptance of a new refinement candidate."""
    if best_score is None:
        return True

    if isinstance(new_score, tuple):
        new_s, new_conf = new_score
    else:
        new_s = float(new_score["S"])
        new_conf = float(new_score["conflict_share"])

    if isinstance(best_score, tuple):
        best_s, best_conf = best_score
    else:
        best_s = float(best_score["S"])
        best_conf = float(best_score["conflict_share"])

    return bool(new_s >= best_s + ACCEPT_MARGIN and new_conf <= best_conf)


def scorecard(
    result: ReplayResult,
    artefacts_by_ref: Mapping[str, SignatureArtefact],
    digest: Mapping[str, object],
    round_no: int = 1,
) -> dict[str, object]:
    """Generate train scorecard for refinement rounds; contains no validation block metrics."""
    denom = result.denominator_wh
    devices_card: list[dict[str, object]] = []

    sig_to_ref = {art.signature_id: ref for ref, art in artefacts_by_ref.items()}

    for ref, art in artefacts_by_ref.items():
        sig_id = art.signature_id
        stats = result.per_signature.get(sig_id, {})
        claimed_wh = float(stats.get("claimed_wh", 0.0))
        claimed_share = claimed_wh / denom if denom > 0 else 0.0
        conflicts = [
            sig_to_ref[other_sig]
            for other_sig in result.conflicts_with.get(sig_id, set())
            if other_sig in sig_to_ref
        ]

        devices_card.append(
            {
                "ref": ref,
                "matched": int(stats.get("n", 0)),
                "claimed_share": round(claimed_share, 3),
                "conflicts_with": sorted(conflicts),
            }
        )

    # Find unexplained clusters from digest
    claimed_clusters: set[str] = set()
    for art in artefacts_by_ref.values():
        for c in art.provenance.get("member_clusters", []):
            claimed_clusters.add(c)

    unexplained: list[dict[str, object]] = []
    for c in digest.get("clusters", []):
        cid = c.get("id", "")
        if cid and cid not in claimed_clusters:
            unexplained.append(
                {
                    "id": cid,
                    "energy_share": float(c.get("energy_share", 0.0)),
                }
            )
    unexplained.sort(key=lambda item: (-item["energy_share"], item["id"]))

    total_claimed_share = (
        sum(float(stats.get("claimed_wh", 0.0)) for stats in result.per_signature.values()) / denom
        if denom > 0
        else 0.0
    )
    conflict_share = result.conflict_wh / denom if denom > 0 else 0.0

    return {
        "round": int(round_no),
        "devices": devices_card,
        "unexplained_clusters": unexplained[:10],
        "total_claimed_share": round(total_claimed_share, 3),
        "conflict_share": round(conflict_share, 3),
    }
