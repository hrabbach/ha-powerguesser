"""Cross-run candidate identity matching by local signature overlap and decision preservation."""

from collections.abc import Collection, Mapping, Sequence
import numpy as np

from .artefact import Box, SignatureArtefact, State
from .score import replay

JACCARD_MIN = 0.5
BOX_IOU_MIN = 0.5
TWIN_OVERLAP_MIN = 0.5


def _box_iou_2d(
    dur1: tuple[float, float, float],
    axis1: tuple[float, float, float],
    dur2: tuple[float, float, float],
    axis2: tuple[float, float, float],
) -> float:
    """2D box IoU over (log duration x axis value), the Phase 4 state formula.

    ``axis`` is the rise box for a switched state and the stage level box for a
    program stage; both go through the identical expressions, so the switched
    IoU values are byte-identical to before.
    """
    dur1_lo = np.log(max(1.0, dur1[0]))
    dur1_hi = np.log(max(1.0, dur1[2]))
    axis1_lo = float(axis1[0])
    axis1_hi = float(axis1[2])

    dur2_lo = np.log(max(1.0, dur2[0]))
    dur2_hi = np.log(max(1.0, dur2[2]))
    axis2_lo = float(axis2[0])
    axis2_hi = float(axis2[2])

    inter_dur = max(0.0, min(dur1_hi, dur2_hi) - max(dur1_lo, dur2_lo))
    inter_axis = max(0.0, min(axis1_hi, axis2_hi) - max(axis1_lo, axis2_lo))
    inter_area = inter_dur * inter_axis

    area1 = max(0.0, dur1_hi - dur1_lo) * max(0.0, axis1_hi - axis1_lo)
    area2 = max(0.0, dur2_hi - dur2_lo) * max(0.0, axis2_hi - axis2_lo)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return (
            1.0
            if (
                area1 == 0.0
                and area2 == 0.0
                and dur1_lo == dur2_lo
                and axis1_lo == axis2_lo
            )
            else 0.0
        )
    return float(inter_area / union_area)


def _calc_state_iou(s1: State, s2: State) -> float:
    """Compute 2D box IoU over (log duration x rise) space."""
    return _box_iou_2d(
        (s1.dur_s.lo, s1.dur_s.p50, s1.dur_s.hi),
        (s1.rise_w.lo, s1.rise_w.p50, s1.rise_w.hi),
        (s2.dur_s.lo, s2.dur_s.p50, s2.dur_s.hi),
        (s2.rise_w.lo, s2.rise_w.p50, s2.rise_w.hi),
    )


def _box_tuple(raw: object) -> tuple[float, float, float] | None:
    """Read a lo/p50/hi mapping as a tuple, or None when it is unusable."""
    if not isinstance(raw, Mapping):
        return None
    try:
        return (float(raw["lo"]), float(raw["p50"]), float(raw["hi"]))
    except (KeyError, TypeError, ValueError):
        return None


def _artefact_stages(art: SignatureArtefact) -> list[Mapping[str, object]]:
    """Stage entries of a program artefact's template (empty when unusable)."""
    session = art.session
    if not isinstance(session, Mapping):
        return []
    stages = session.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return []
    return [stage for stage in stages if isinstance(stage, Mapping)]


def _calc_stage_iou(st1: Mapping[str, object], st2: Mapping[str, object]) -> float:
    """2D box IoU of two program stages over (log duration x stage level)."""
    dur1 = _box_tuple(st1.get("dur_s"))
    level1 = _box_tuple(st1.get("level_w"))
    dur2 = _box_tuple(st2.get("dur_s"))
    level2 = _box_tuple(st2.get("level_w"))
    if dur1 is None or level1 is None or dur2 is None or level2 is None:
        return 0.0
    return _box_iou_2d(dur1, level1, dur2, level2)


def _calc_artefact_box_iou(art1: SignatureArtefact, art2: SignatureArtefact) -> float:
    """Average box IoU between two artefacts (states, or stages for a program).

    A program pair is compared on its stage templates: the mean of the zipped
    stage IoUs in (log duration x level) space, scaled by min(n1, n2) / max(n1,
    n2) so a template with extra or missing stages can never look identical.
    Either side without stages reads 0.0 — never a match (RESEARCH Pitfall 1).
    """
    if art1.kind == "program" or art2.kind == "program":
        stages1 = _artefact_stages(art1)
        stages2 = _artefact_stages(art2)
        if not stages1 or not stages2:
            return 0.0
        ious = [_calc_stage_iou(a, b) for a, b in zip(stages1, stages2)]
        scale = min(len(stages1), len(stages2)) / max(len(stages1), len(stages2))
        return float(np.mean(ious)) * float(scale)
    if not art1.states or not art2.states:
        return 0.0
    ious = []
    for s1, s2 in zip(art1.states, art2.states):
        ious.append(_calc_state_iou(s1, s2))
    return float(np.mean(ious))


def _to_artefact(item: SignatureArtefact | Mapping[str, object], sig_id_override: str | None = None) -> SignatureArtefact:
    """Ensure item is a SignatureArtefact instance."""
    if isinstance(item, SignatureArtefact):
        return item
    d = dict(item)
    if sig_id_override is not None:
        d["signature_id"] = sig_id_override
    if "schema_version" not in d:
        # Partial reconstruction from fingerprint/decision
        fp = d.get("fingerprint", {})
        states_raw = fp.get("states", [])
        states = []
        for i, s_fp in enumerate(states_raw):
            states.append(
                State(
                    name=f"state_{i}",
                    rise_w=Box.from_dict(s_fp["rise_w"]),
                    fall_w=Box.from_dict(s_fp["fall_w"]),
                    dur_s=Box.from_dict(s_fp["dur_s"]),
                    plateau_w=Box.from_dict(s_fp["plateau_w"]),
                    phase_share=s_fp.get("phase_share"),
                )
            )
        # A program fingerprint carries its stage template under
        # "session_stages" (Plan 06-04); without it the rebuilt artefact has no
        # stages, so the stage IoU reads 0.0 rather than a false match.
        raw_stages = fp.get("session_stages")
        session = None
        if isinstance(raw_stages, Sequence) and not isinstance(
            raw_stages, (str, bytes)
        ):
            session = {
                "stages": [dict(s) for s in raw_stages if isinstance(s, Mapping)]
            }
        return SignatureArtefact(
            schema_version=1,
            signature_id=sig_id_override or str(d.get("signature_id", "prior")),
            kind=str(fp.get("kind", "switched")),
            label=str(d.get("name") or d.get("label") or "prior"),
            category="unknown",
            extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
            states=tuple(states),
            session=session,
            time_prior={"hour": [1/24]*24, "dow": [1/7]*7},
            constraints={"max_concurrent": 1},
            provenance={},
            scores={},
        )
    return SignatureArtefact.from_dict(d)


def similar_dismissed(
    artefact: SignatureArtefact | Mapping[str, object],
    decisions: Mapping[str, Mapping[str, object]],
) -> str | None:
    """The id of a dismissed candidate whose fingerprint the artefact matches (D-07).

    Among the decisions with ``status == "dismissed"`` and a non-empty
    fingerprint of the artefact's kind, the one with the highest box IoU
    (``_calc_artefact_box_iou``) at or above ``BOX_IOU_MIN`` wins; ties break by
    dismissed id ascending. Returns None when nothing matches. Pure: no new
    constant, no store write. Used by the HA half of D-07 to note a
    re-captured dismissed shape without suppressing the preview.
    """
    art = artefact if isinstance(artefact, SignatureArtefact) else _to_artefact(artefact)
    best_id: str | None = None
    best_iou = 0.0
    for dec_id in sorted(decisions):
        dec = decisions[dec_id]
        if dec.get("status") != "dismissed":
            continue
        fingerprint = dec.get("fingerprint")
        if not fingerprint:
            continue
        if str(fingerprint.get("kind") or "") != art.kind:
            continue
        other = _to_artefact(dec, sig_id_override=dec_id)
        iou = _calc_artefact_box_iou(art, other)
        if iou >= BOX_IOU_MIN and (best_id is None or iou > best_iou):
            best_id = dec_id
            best_iou = iou
    return best_id


def match_identities(
    new_artefacts: Sequence[SignatureArtefact],
    prior_artefacts: Sequence[SignatureArtefact | Mapping[str, object]],
    decisions: Mapping[str, Mapping[str, object]],
    table: Mapping[str, np.ndarray],
    *,
    owned: Collection[str] = (),
    anchors: Mapping[str, Collection[int]] | None = None,
    capture_prior_ids: Collection[str] = (),
) -> list[dict[str, object]]:
    """Match new candidate artefacts against prior artefacts and user review decisions.

    Matching is purely structural based on episode overlap (Jaccard >= 0.5) and box IoU,
    never comparing display labels.

    D-19: after the greedy one-to-one pass, a new artefact still unmatched whose train
    claims overlap a dismissed same-kind prior with |A and B| / min(|A|, |B|) >=
    TWIN_OVERLAP_MIN is a near twin of the dismissed candidate (the same load re-proposed
    under a different cluster set, too large to match one-to-one). It is suppressed and
    records the dismissed id in identity.twin_of for diagnosis; the review UI never lists
    it. The replay result is reused — no second replay.

    OUT-05 / D-11 / D-27: ``owned`` carries the signature ids (and aliases) of confirmed
    or retired devices. Task 1 only accepts it; Plan 04-09 Task 2 uses it to suppress
    near twins of a confirmed device and to mark non-twin overlaps.

    CAP-09 (Plan 08-08): ``capture_prior_ids`` are capture-origin priors. A candidate
    pair whose prior is one of them sorts first in the greedy pass, so a capture of a
    shape the user dismissed earlier reconciles to the capture beside a higher-scoring
    dismissed twin (D-07) instead of being suppressed. ``anchors`` maps a prior's
    signature id to the episode rows its captured run snapped to; when it is given, a
    still-unmatched new artefact and a still-unmatched prior of the same kind are matched
    when the artefact's own claims (blocks 0, 1 and 2) contain the prior's anchored rows —
    the containment-anchor pass that reconciles a validation-block capture whose preview
    boxes over-claim too many train rows to pass the Jaccard gate (RESEARCH Pattern 4).
    The solo replay of blocks 1 and 2 is exact, not an approximation: score.replay
    decides claims per artefact by box containment and never removes a claim for a
    competing artefact, so claiming against the artefact's own claims (the intended
    semantics: "does this artefact itself claim the user's captured run?", not "does it
    win that run against the priors?") is independent of the priors — pinned by
    test_anchor_claims_do_not_depend_on_priors. Absent both keywords, the output is
    byte-identical to before.
    """
    if not new_artefacts:
        return []

    # Assemble comprehensive prior pool: prior_artefacts + any decision fingerprints not already present
    priors_by_id: dict[str, SignatureArtefact] = {}
    for p in prior_artefacts:
        art = _to_artefact(p)
        priors_by_id[art.signature_id] = art

    for p_id, dec in decisions.items():
        if p_id not in priors_by_id and "fingerprint" in dec:
            priors_by_id[p_id] = _to_artefact(dec, sig_id_override=p_id)

    # If no priors exist, all new artefacts receive status "new"
    if not priors_by_id:
        result = []
        for new_art in new_artefacts:
            art_d = new_art.to_dict()
            art_d["identity"] = {
                "status": "new",
                "matched": None,
                "merged_into": None,
                "twin_of": None,
            }
            result.append(art_d)
        return result

    prior_list = list(priors_by_id.values())

    # Replay both on current train block (block == 0)
    all_artefacts = list(new_artefacts) + prior_list
    replay_res = replay(all_artefacts, table, block=0)

    # Evaluate pair scores for matching kinds
    candidate_pairs: list[tuple[float, str, str]] = []  # (score, new_id, prior_id)
    pair_details: dict[tuple[str, str], float] = {}

    for new_art in new_artefacts:
        new_id = new_art.signature_id
        new_claims = replay_res.claims.get(new_id, set())

        for prior_art in prior_list:
            if new_art.kind != prior_art.kind:
                continue
            prior_id = prior_art.signature_id
            prior_claims = replay_res.claims.get(prior_id, set())

            union_len = len(new_claims | prior_claims)
            if union_len > 0:
                jaccard = len(new_claims & prior_claims) / float(union_len)
                if jaccard >= JACCARD_MIN:
                    candidate_pairs.append((jaccard, new_id, prior_id))
                    pair_details[(new_id, prior_id)] = jaccard
            else:
                # Fallback: per-state box IoU over (log duration x rise)
                iou = _calc_artefact_box_iou(new_art, prior_art)
                if iou >= BOX_IOU_MIN:
                    candidate_pairs.append((iou, new_id, prior_id))
                    pair_details[(new_id, prior_id)] = iou

    # Greedy one-to-one matching: capture-origin priors first (CAP-09/D-07), then
    # score descending, then (new_id, prior_id) ascending. Without any capture
    # prior every key starts with 1, so the order equals today's.
    capture_prior_set = set(capture_prior_ids)
    candidate_pairs.sort(
        key=lambda item: (
            0 if item[2] in capture_prior_set else 1,
            -item[0],
            item[1],
            item[2],
        )
    )

    matched_new: dict[str, str] = {}  # new_id -> prior_id
    matched_prior: set[str] = set()

    for score, n_id, p_id in candidate_pairs:
        if n_id not in matched_new and p_id not in matched_prior:
            matched_new[n_id] = p_id
            matched_prior.add(p_id)

    # CAP-09 (Plan 08-08): the containment-anchor pass, after the greedy pass and
    # before the D-19 twin pass. A capture's snapped run is a block-1/2 episode,
    # but the one-to-one pass decides on block-0 (train) claims only, and a
    # preview's tolerance boxes over-claim many train rows (Jaccard < JACCARD_MIN).
    # So a still-unmatched new artefact and a captured prior are matched when the
    # artefact's OWN claims contain the prior's anchored rows.
    #
    # The claims are the union of the block-0 (train) claims from the shared
    # replay and a SOLO replay of blocks 1 and 2 (a block without rows yields
    # empty claims). The solo replay is exact, not an approximation: score.replay
    # decides claims per artefact by box containment and never removes a claim
    # for a competing artefact, so adding the priors would not change any new
    # artefact's claims — solo claims are the intended semantics (the anchor asks
    # whether the artefact itself claims the user's captured run, not whether it
    # wins that run against priors), pinned by
    # test_anchor_claims_do_not_depend_on_priors.
    if anchors:
        anchor_claims: dict[str, set[int]] = {
            art.signature_id: set(replay_res.claims.get(art.signature_id, set()))
            for art in new_artefacts
        }
        for block_no in (1, 2):
            block_res = replay(new_artefacts, table, block=block_no)
            for art in new_artefacts:
                anchor_claims[art.signature_id] |= block_res.claims.get(
                    art.signature_id, set()
                )

        anchor_pairs: list[tuple[int, str, str]] = []  # (count, new_id, prior_id)
        for new_art in new_artefacts:
            new_id = new_art.signature_id
            if new_id in matched_new:
                continue
            new_claims = anchor_claims.get(new_id, set())
            if not new_claims:
                continue
            for prior_art in prior_list:
                prior_id = prior_art.signature_id
                if prior_id in matched_prior or prior_art.kind != new_art.kind:
                    continue
                anchored_rows = anchors.get(prior_id)
                if not anchored_rows:
                    continue
                count = sum(1 for row in anchored_rows if int(row) in new_claims)
                if count > 0:
                    anchor_pairs.append((count, new_id, prior_id))
        anchor_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _count, n_id, p_id in anchor_pairs:
            if n_id not in matched_new and p_id not in matched_prior:
                matched_new[n_id] = p_id
                matched_prior.add(p_id)

    # D-19: near-twin suppression of dismissed candidates, reusing the replay claims.
    twin_of: dict[str, str] = {}
    twin_coeffs: dict[str, float] = {}
    for new_art in new_artefacts:
        new_id = new_art.signature_id
        if new_id in matched_new:
            continue
        new_claims = replay_res.claims.get(new_id, set())
        if not new_claims:
            continue
        best_coeff = 0.0
        best_prior: str | None = None
        for prior_art in prior_list:
            prior_id = prior_art.signature_id
            dec = decisions.get(prior_id)
            if not dec or dec.get("status") != "dismissed":
                continue
            if prior_art.kind != new_art.kind:
                continue
            prior_claims = replay_res.claims.get(prior_id, set())
            if not prior_claims:
                continue
            overlap = len(new_claims & prior_claims)
            min_len = min(len(new_claims), len(prior_claims))
            if min_len <= 0:
                continue
            coeff = overlap / float(min_len)
            if coeff < TWIN_OVERLAP_MIN:
                continue
            # Highest coefficient wins; ties break by dismissed id ascending.
            if (
                best_prior is None
                or coeff > best_coeff
                or (coeff == best_coeff and prior_id < best_prior)
            ):
                best_coeff = coeff
                best_prior = prior_id
        if best_prior is not None:
            twin_of[new_id] = best_prior
            twin_coeffs[new_id] = best_coeff

    # D-27 (OUT-05): owned near-twin suppression and the overlaps marker.
    # A new artefact still unmatched whose pair with an owned prior was scored
    # by the one-to-one pass (Jaccard >= JACCARD_MIN, or the box-IoU fallback
    # when neither side claims anything) but stayed unmatched only because
    # another artefact took that owned prior is the same load re-proposed: it
    # is suppressed exactly like a near twin of a dismissed candidate, reading
    # the retained pair_details and matched_prior — never recomputed. Any
    # other owned overlap at TWIN_OVERLAP_MIN (a superset, a fragment) keeps
    # the artefact listed: identity stays new and gains "overlaps" = the owned
    # id (highest coefficient, ties by id ascending) so the review UI can show
    # " — overlaps <device name>". Without owned, this pass is skipped and the
    # output equals the Plan 04-02 behaviour.
    overlaps: dict[str, str] = {}
    owned_set = set(owned)
    if owned_set:
        for new_art in new_artefacts:
            new_id = new_art.signature_id
            if new_id in matched_new:
                continue
            new_claims = replay_res.claims.get(new_id, set())
            twin_coeff = 0.0
            twin_prior: str | None = None
            overlap_coeff = 0.0
            overlap_prior: str | None = None
            for prior_art in prior_list:
                prior_id = prior_art.signature_id
                if prior_id not in owned_set or prior_art.kind != new_art.kind:
                    continue
                if not new_claims:
                    # Box-IoU fallback: neither side claims train episodes, so
                    # the retained pair score is the near-twin evidence.
                    if replay_res.claims.get(prior_id):
                        continue
                    score = pair_details.get((new_id, prior_id))
                    if score is None:
                        continue
                    coeff = score
                else:
                    prior_claims = replay_res.claims.get(prior_id, set())
                    if not prior_claims:
                        continue
                    min_len = min(len(new_claims), len(prior_claims))
                    if min_len <= 0:
                        continue
                    coeff = len(new_claims & prior_claims) / float(min_len)
                # Near twin: the one-to-one pass scored the pair as a match
                # candidate and the owned prior was taken by another artefact.
                if (new_id, prior_id) in pair_details and prior_id in matched_prior:
                    if (
                        twin_prior is None
                        or coeff > twin_coeff
                        or (coeff == twin_coeff and prior_id < twin_prior)
                    ):
                        twin_coeff = coeff
                        twin_prior = prior_id
                # Any owned overlap at TWIN_OVERLAP_MIN marks the candidate.
                if coeff >= TWIN_OVERLAP_MIN and (
                    overlap_prior is None
                    or coeff > overlap_coeff
                    or (coeff == overlap_coeff and prior_id < overlap_prior)
                ):
                    overlap_coeff = coeff
                    overlap_prior = prior_id
            if twin_prior is not None:
                # Suppression candidates combine D-19 dismissed twins and owned
                # near twins: the highest coefficient wins, ties by id ascending.
                existing = twin_of.get(new_id)
                if (
                    existing is None
                    or twin_coeff > twin_coeffs[new_id]
                    or (twin_coeff == twin_coeffs[new_id] and twin_prior < existing)
                ):
                    twin_of[new_id] = twin_prior
                    twin_coeffs[new_id] = twin_coeff
            elif overlap_prior is not None and new_id not in twin_of:
                overlaps[new_id] = overlap_prior

    # Build output artefact dicts
    result_dicts: list[dict[str, object]] = []
    for new_art in new_artefacts:
        n_id = new_art.signature_id
        art_d = new_art.to_dict()

        if n_id in matched_new:
            p_id = matched_new[n_id]
            # Inherit prior signature id
            art_d["signature_id"] = p_id

            # Apply review decision if present
            dec = decisions.get(p_id)
            if dec and dec.get("status") == "dismissed":
                art_d["identity"] = {
                    "status": "suppressed",
                    "matched": p_id,
                    "merged_into": None,
                    "twin_of": None,
                }
            elif dec and dec.get("status") == "merged":
                art_d["identity"] = {
                    "status": "merged",
                    "matched": p_id,
                    "merged_into": dec.get("merged_into"),
                    "twin_of": None,
                }
            else:
                art_d["identity"] = {
                    "status": "matched",
                    "matched": p_id,
                    "merged_into": None,
                    "twin_of": None,
                }
        elif n_id in twin_of:
            art_d["identity"] = {
                "status": "suppressed",
                "matched": None,
                "merged_into": None,
                "twin_of": twin_of[n_id],
            }
        else:
            ident_new: dict[str, object] = {
                "status": "new",
                "matched": None,
                "merged_into": None,
                "twin_of": None,
            }
            if n_id in overlaps:
                ident_new["overlaps"] = overlaps[n_id]
            art_d["identity"] = ident_new

        result_dicts.append(art_d)

    return result_dicts
