"""Tests for cross-run identity matching, suppression, merge mapping, and tie breaking."""

from typing import Any

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning.artefact import Box, SignatureArtefact, State
from custom_components.powerguesser.engine.learning.identity import (
    BOX_IOU_MIN,
    JACCARD_MIN,
    match_identities,
    similar_dismissed,
)


def _make_test_artefact(
    sig_id: str,
    label: str,
    rise_lo: float,
    rise_p50: float,
    rise_hi: float,
    dur_lo: float,
    dur_p50: float,
    dur_hi: float,
    kind: str = "switched",
) -> SignatureArtefact:
    """Helper to construct a simple test artefact."""
    st = State(
        name="on",
        rise_w=Box(lo=rise_lo, p50=rise_p50, hi=rise_hi),
        fall_w=Box(lo=-rise_hi, p50=-rise_p50, hi=-rise_lo),
        dur_s=Box(lo=dur_lo, p50=dur_p50, hi=dur_hi),
        plateau_w=Box(lo=rise_lo, p50=rise_p50, hi=rise_hi),
        phase_share=None,
    )
    return SignatureArtefact(
        schema_version=1,
        signature_id=sig_id,
        kind=kind,
        label=label,
        category="unknown",
        extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
        states=(st,),
        session=None,
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={"run_id": "test", "round": 1},
        scores={"train": None, "val": None, "validated": True, "conflicts": 0},
    )


def _make_test_table(episodes: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Helper to construct a minimal episodes table."""
    n = len(episodes)
    return {
        "start_i": np.arange(n, dtype=np.int64),
        "end_i": np.arange(n, dtype=np.int64) + 1,
        "rise_w": np.array([ep["rise_w"] for ep in episodes], dtype=np.float64),
        "fall_w": np.array([ep["fall_w"] for ep in episodes], dtype=np.float64),
        "dur_s": np.array([ep["dur_s"] for ep in episodes], dtype=np.float64),
        "plateau_w": np.array([ep.get("plateau_w", ep["rise_w"]) for ep in episodes], dtype=np.float64),
        "energy_wh": np.array([ep.get("energy_wh", 50.0) for ep in episodes], dtype=np.float64),
        "kind": np.array([ep.get("kind", 0) for ep in episodes], dtype=np.int32),
        "block": np.zeros(n, dtype=np.int32),  # all train
    }


def test_identity_ignores_llm_names():
    """Giving every new and prior artefact a different, random display name leaves all matches unchanged."""
    table = _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 120.0},
        {"rise_w": 1500.0, "fall_w": -1500.0, "dur_s": 600.0},
    ])

    new_art = _make_test_artefact("new1", "Random New Name 123", 1900, 2000, 2100, 90, 120, 150)
    prior_art = _make_test_artefact("prior1", "Completely Different Prior Name 789", 1900, 2000, 2100, 90, 120, 150)

    matched = match_identities([new_art], [prior_art], {}, table)
    assert len(matched) == 1
    assert matched[0]["signature_id"] == "prior1"
    assert matched[0]["identity"]["status"] == "matched"
    assert matched[0]["identity"]["matched"] == "prior1"


def test_identity_suppression_and_merge_mapping():
    """Prior dismissed decision produces suppressed status, merged decision produces merged status with target."""
    table = _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 120.0},
        {"rise_w": 1500.0, "fall_w": -1500.0, "dur_s": 600.0},
    ])

    new_art1 = _make_test_artefact("new1", "Appliance 1", 1900, 2000, 2100, 90, 120, 150)
    new_art2 = _make_test_artefact("new2", "Appliance 2", 1400, 1500, 1600, 500, 600, 700)

    prior_art1 = _make_test_artefact("prior1", "Prior 1", 1900, 2000, 2100, 90, 120, 150)
    prior_art2 = _make_test_artefact("prior2", "Prior 2", 1400, 1500, 1600, 500, 600, 700)

    decisions = {
        "prior1": {"status": "dismissed", "name": None, "merged_into": None},
        "prior2": {"status": "merged", "name": None, "merged_into": "target_sig_99"},
    }

    matched = match_identities([new_art1, new_art2], [prior_art1, prior_art2], decisions, table)
    res_by_id = {item["identity"]["matched"]: item for item in matched}

    assert res_by_id["prior1"]["identity"]["status"] == "suppressed"
    assert res_by_id["prior1"]["signature_id"] == "prior1"

    assert res_by_id["prior2"]["identity"]["status"] == "merged"
    assert res_by_id["prior2"]["signature_id"] == "prior2"
    assert res_by_id["prior2"]["identity"]["merged_into"] == "target_sig_99"


def test_jaccard_boundary_exactly_half():
    """Jaccard exactly 0.5 counts as a match, while 0.49 does not."""
    # 3 episodes: ep0, ep1, ep2
    table = _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 100.0},
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 200.0},
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 300.0},
    ])

    # new_art matches ep0 and ep1 (dur 50-250) -> claims {0, 1}
    new_art = _make_test_artefact("new1", "New", 1900, 2000, 2100, 50, 150, 250)

    # prior_art matches ep1 and ep2 (dur 150-350) -> claims {1, 2}
    # intersection={1} (len 1), union={0, 1, 2} (len 3) -> J = 1/3 = 0.33 -> unmatched
    prior_art_third = _make_test_artefact("prior_third", "Prior Third", 1900, 2000, 2100, 150, 250, 350)

    res_third = match_identities([new_art], [prior_art_third], {}, table)
    assert res_third[0]["identity"]["status"] == "new"

    # Exactly Jaccard 0.5:
    # 2 episodes: ep0 and ep1
    table_half = _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 100.0},
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 200.0},
    ])
    # art_a matches ep0 and ep1 (len 2)
    art_a = _make_test_artefact("new_half", "A", 1900, 2000, 2100, 50, 150, 250)
    # art_b matches only ep0 (len 1)
    # intersection={0} (len 1), union={0, 1} (len 2) -> J = 1/2 = 0.5 exactly!
    art_b = _make_test_artefact("prior_half", "B", 1900, 2000, 2100, 50, 100, 150)

    res_half = match_identities([art_a], [art_b], {}, table_half)
    assert res_half[0]["identity"]["status"] == "matched"
    assert res_half[0]["signature_id"] == "prior_half"


def test_iou_fallback_when_no_episodes_claimed():
    """When both artefacts claim zero episodes in the train block, box IoU >= 0.5 decides match."""
    # Table has no matching episodes (episodes at 500 W)
    table_empty = _make_test_table([
        {"rise_w": 500.0, "fall_w": -500.0, "dur_s": 30.0},
    ])

    # Both are around 2000 W and 100 s
    new_art = _make_test_artefact("new_iou", "New", 1900, 2000, 2100, 80, 100, 120)
    prior_art = _make_test_artefact("prior_iou", "Prior", 1920, 2000, 2080, 85, 100, 115)

    res = match_identities([new_art], [prior_art], {}, table_empty)
    assert res[0]["identity"]["status"] == "matched"
    assert res[0]["signature_id"] == "prior_iou"

    # Completely different box: 1000 W vs 2000 W -> IoU = 0.0 -> "new"
    prior_diff = _make_test_artefact("prior_diff", "Prior Diff", 900, 1000, 1100, 80, 100, 120)
    res_diff = match_identities([new_art], [prior_diff], {}, table_empty)
    assert res_diff[0]["identity"]["status"] == "new"


def test_tie_breaking_order():
    """Ties between candidate pairs are broken deterministically by (new_id, prior_id) ascending."""
    table = _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 120.0},
    ])

    new_a = _make_test_artefact("new_a", "New A", 1900, 2000, 2100, 90, 120, 150)
    new_b = _make_test_artefact("new_b", "New B", 1900, 2000, 2100, 90, 120, 150)

    prior_1 = _make_test_artefact("prior_1", "Prior 1", 1900, 2000, 2100, 90, 120, 150)

    # Both new_a and new_b match prior_1 with score 1.0
    # Tied! (new_a, prior_1) < (new_b, prior_1), so new_a matches prior_1 and new_b becomes new
    res = match_identities([new_a, new_b], [prior_1], {}, table)
    res_dict = {item["identity"]["matched"] or item["signature_id"]: item for item in res}

    assert res_dict["prior_1"]["identity"]["status"] == "matched"
    assert res_dict["new_b"]["identity"]["status"] == "new"


# --- D-19: near twins of dismissed candidates stay hidden ---


def _dur_table(n: int) -> dict[str, np.ndarray]:
    """Table of n switched episodes; episode i has dur_s = (i + 1) * 10."""
    return _make_test_table([
        {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": float((i + 1) * 10)}
        for i in range(n)
    ])


def _range_artefact(sig_id: str, a: int, b: int) -> SignatureArtefact:
    """Switched artefact claiming exactly episodes a..b (via its duration box)."""
    return _make_test_artefact(
        sig_id,
        sig_id,
        1900,
        2000,
        2100,
        (a + 1) * 10 - 5,
        ((a + b) // 2 + 1) * 10,
        (b + 1) * 10 + 5,
    )


def _decision_for_ranges(
    ranges: list[tuple[int, int]],
    *,
    status: str = "dismissed",
    kind: str = "switched",
) -> dict[str, Any]:
    """A review decision whose fingerprint states claim the given episode ranges."""
    states = []
    for a, b in ranges:
        states.append(
            {
                "rise_w": {"lo": 1500.0, "p50": 2000.0, "hi": 2500.0},
                "fall_w": {"lo": -2500.0, "p50": -2000.0, "hi": -1500.0},
                "dur_s": {
                    "lo": float((a + 1) * 10 - 5),
                    "p50": float(((a + b) // 2 + 1) * 10),
                    "hi": float((b + 1) * 10 + 5),
                },
                "plateau_w": {"lo": 1500.0, "p50": 2000.0, "hi": 2500.0},
                "phase_share": None,
            }
        )
    return {
        "status": status,
        "name": None,
        "merged_into": None,
        "fingerprint": {"kind": kind, "states": states},
    }


def test_near_twin_of_dismissed_is_suppressed():
    """D-19: an unmatched new artefact fully overlapping a dismissed prior is suppressed as a near twin."""
    table = _dur_table(50)
    new_art = _range_artefact("new1", 0, 9)  # claims {0..9}
    # Dismissed prior claims {0..9} and {20..49}: overlap coefficient 10/10 = 1.0,
    # but Jaccard 10/30 < 0.5, so the one-to-one pass leaves the new artefact unmatched.
    decisions = {"priorD": _decision_for_ranges([(0, 9), (20, 49)])}

    res = match_identities([new_art], [], decisions, table)
    ident = res[0]["identity"]

    assert ident["status"] == "suppressed"
    assert ident["matched"] is None
    assert ident["twin_of"] == "priorD"
    # The near twin keeps its own id (no identity is inherited from the dismissed prior).
    assert res[0]["signature_id"] == "new1"


def test_twin_threshold_inclusive():
    """D-19: overlap coefficient exactly 0.5 suppresses (inclusive); 0.45 stays new."""
    # Exactly 0.5: new claims {0..9} (10), dismissed claims {6..13} (8), overlap {6..9} (4).
    table = _dur_table(14)
    new_art = _range_artefact("new_half", 0, 9)
    decisions = {"priorD": _decision_for_ranges([(6, 13)])}
    res = match_identities([new_art], [], decisions, table)
    assert res[0]["identity"]["status"] == "suppressed"
    assert res[0]["identity"]["twin_of"] == "priorD"

    # 0.45: new claims {0..19} (20), dismissed claims {0..8, 20..39} (29), overlap 9; 9/20 = 0.45.
    table_low = _dur_table(40)
    new_low = _range_artefact("new_low", 0, 19)
    decisions_low = {"priorD": _decision_for_ranges([(0, 8), (20, 39)])}
    res_low = match_identities([new_low], [], decisions_low, table_low)
    assert res_low[0]["identity"]["status"] == "new"
    assert res_low[0]["identity"]["twin_of"] is None


def test_twin_requires_dismissed_same_kind():
    """D-19: only dismissed decisions of the same kind can twin-suppress a new artefact."""
    table = _dur_table(50)
    new_art = _range_artefact("new1", 0, 9)
    full_overlap = [(0, 9), (20, 49)]

    for status in ("named", "kept", "merged"):
        decisions = {"priorX": _decision_for_ranges(full_overlap, status=status)}
        res = match_identities([new_art], [], decisions, table)
        assert res[0]["identity"]["status"] == "new", status
        assert res[0]["identity"]["twin_of"] is None, status

    # Dismissed but a different kind: never a twin.
    decisions = {"priorP": _decision_for_ranges(full_overlap, kind="program")}
    res = match_identities([new_art], [], decisions, table)
    assert res[0]["identity"]["status"] == "new"
    assert res[0]["identity"]["twin_of"] is None


def test_matched_artefact_never_twin():
    """D-19: an artefact matched one-to-one is never re-classified as a near twin."""
    table = _dur_table(60)
    new_art = _range_artefact("new1", 0, 9)  # matches priorM with Jaccard 1.0
    new_other = _range_artefact("new_other", 50, 59)  # no overlap with any prior
    prior_m = _range_artefact("priorM", 0, 9)
    # A dismissed prior the new artefact fully overlaps: would be a twin if unmatched.
    decisions = {"priorD": _decision_for_ranges([(0, 9), (20, 49)])}

    res = match_identities([new_art, new_other], [prior_m], decisions, table)
    by_id = {a["signature_id"]: a for a in res}

    assert by_id["priorM"]["identity"]["status"] == "matched"
    assert by_id["priorM"]["identity"]["twin_of"] is None
    assert by_id["new_other"]["identity"]["status"] == "new"
    assert by_id["new_other"]["identity"]["twin_of"] is None


# --- D-27: owned signatures (confirmed devices) ---


def _ranges_artefact(
    sig_id: str,
    ranges: list[tuple[int, int]],
    *,
    kind: str = "switched",
) -> SignatureArtefact:
    """Artefact with one state per episode range (claims the union of the ranges)."""
    states = []
    for a, b in ranges:
        states.append(
            State(
                name=f"state_{a}_{b}",
                rise_w=Box(lo=1900.0, p50=2000.0, hi=2100.0),
                fall_w=Box(lo=-2100.0, p50=-2000.0, hi=-1900.0),
                dur_s=Box(
                    lo=(a + 1) * 10 - 5,
                    p50=((a + b) // 2 + 1) * 10,
                    hi=(b + 1) * 10 + 5,
                ),
                plateau_w=Box(lo=1900.0, p50=2000.0, hi=2100.0),
                phase_share=None,
            )
        )
    return SignatureArtefact(
        schema_version=1,
        signature_id=sig_id,
        kind=kind,
        label=sig_id,
        category="unknown",
        extraction={"grid_s": 30, "edge_threshold_w": 120, "merge_samples": 2, "pair_tol": 0.25},
        states=tuple(states),
        session=None,
        time_prior={"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        constraints={"max_concurrent": 1},
        provenance={"run_id": "test", "round": 1},
        scores={"train": None, "val": None, "validated": True, "conflicts": 0},
    )


def _ident_by_id(res: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index match_identities output by the output signature id."""
    return {a["signature_id"]: a["identity"] for a in res}


def test_unmatched_twin_of_owned_is_suppressed():
    """D-27: a new artefact that is a near twin of an owned prior (Jaccard >=
    JACCARD_MIN, scored by the one-to-one pass) left unmatched only because
    another artefact took the owned prior is suppressed with twin_of."""
    table = _dur_table(20)
    owned = _range_artefact("ownedO", 0, 19)  # claims {0..19}
    m = _range_artefact("newM", 0, 19)  # takes ownedO one-to-one (Jaccard 1.0)
    n = _range_artefact("newN", 0, 11)  # claims {0..11}: Jaccard 12/20 = 0.6

    res = match_identities([m, n], [owned], {}, table, owned={"ownedO"})
    by_id = _ident_by_id(res)

    assert by_id["ownedO"]["status"] == "matched"
    ident = by_id["newN"]
    assert ident["status"] == "suppressed"
    assert ident["matched"] is None
    assert ident["merged_into"] is None
    assert ident["twin_of"] == "ownedO"
    assert "overlaps" not in ident


def test_superset_of_owned_gets_overlaps():
    """D-27: a superset of an owned prior (Jaccard 0.4 < JACCARD_MIN, overlap
    coefficient 1.0) is NOT hidden: identity stays new and gains overlaps."""
    table = _dur_table(50)
    owned = _range_artefact("ownedO", 0, 19)  # claims {0..19}
    n = _range_artefact("newN", 0, 49)  # claims {0..49}: Jaccard 20/50 = 0.4

    res = match_identities([n], [owned], {}, table, owned={"ownedO"})
    ident = res[0]["identity"]

    assert ident["status"] == "new"
    assert ident["matched"] is None
    assert ident["merged_into"] is None
    assert ident["twin_of"] is None
    assert ident["overlaps"] == "ownedO"


def test_owned_twin_threshold_and_kind():
    """D-27: Jaccard exactly JACCARD_MIN suppresses; below it with full overlap
    gains overlaps; partial overlap below TWIN_OVERLAP_MIN stays plain new; a
    different kind never triggers either; a prior that is neither owned nor
    dismissed never does."""
    owned = _range_artefact("ownedO", 0, 19)  # claims {0..19}
    m = _range_artefact("newM", 0, 19)  # takes ownedO one-to-one

    # Jaccard exactly 0.5 -> suppressed as the owned near twin.
    table = _dur_table(20)
    n_half = _range_artefact("new_half", 0, 9)  # claims {0..9}: Jaccard 10/20
    res = match_identities([m, n_half], [owned], {}, table, owned={"ownedO"})
    ident = _ident_by_id(res)["new_half"]
    assert ident["status"] == "suppressed"
    assert ident["twin_of"] == "ownedO"
    assert "overlaps" not in ident

    # Jaccard 0.45, overlap coefficient 1.0 -> status new with overlaps.
    n_low = _range_artefact("new_low", 0, 8)  # claims {0..8}: Jaccard 9/20
    res = match_identities([m, n_low], [owned], {}, table, owned={"ownedO"})
    ident = _ident_by_id(res)["new_low"]
    assert ident["status"] == "new"
    assert ident["twin_of"] is None
    assert ident["overlaps"] == "ownedO"

    # Overlap coefficient 0.45 (partial overlap) -> plain new, no overlaps key.
    table_frag = _dur_table(31)
    n_frag = _ranges_artefact("new_frag", [(0, 8), (20, 30)])  # claims 20, overlap 9
    res = match_identities([m, n_frag], [owned], {}, table_frag, owned={"ownedO"})
    ident = _ident_by_id(res)["new_frag"]
    assert ident["status"] == "new"
    assert ident["twin_of"] is None
    assert "overlaps" not in ident

    # Owned prior of a different kind: never a twin, never overlaps.
    owned_prog = _ranges_artefact("ownedP", [(0, 19)], kind="program")
    res = match_identities([m, n_half], [owned_prog], {}, table, owned={"ownedP"})
    ident = _ident_by_id(res)["new_half"]
    assert ident["status"] == "new"
    assert ident["twin_of"] is None
    assert "overlaps" not in ident

    # Prior neither owned nor dismissed: no suppression, no overlaps.
    prior_p = _range_artefact("priorP", 0, 19)
    res = match_identities(
        [m, n_half], [prior_p], {"priorP": {"status": "kept"}}, table
    )
    ident = _ident_by_id(res)["new_half"]
    assert ident["status"] == "new"
    assert ident["twin_of"] is None
    assert "overlaps" not in ident


def test_owned_and_dismissed_twin_best_wins():
    """Suppression combines dismissed twins and owned near twins: twin_of names
    the highest overlap coefficient (ties by id ascending), and a dismissed twin
    suppresses even when the artefact is also an owned overlap (no overlaps key)."""
    # Tie at coefficient 1.0 between the owned near twin and the dismissed twin:
    # the id ascending tie-break picks the owned id.
    table = _dur_table(20)
    owned = _range_artefact("ownedO", 0, 19)
    m_owned = _range_artefact("newM", 0, 19)  # takes ownedO
    dismissed = _range_artefact("priorD", 0, 9)
    m_dis = _range_artefact("newM2", 0, 9)  # takes priorD one-to-one
    n = _range_artefact("newN", 0, 9)  # coeff 1.0 with both candidates
    decisions = {"priorD": {"status": "dismissed", "name": None, "merged_into": None}}
    res = match_identities(
        [m_owned, m_dis, n], [owned, dismissed], decisions, table, owned={"ownedO"}
    )
    ident = _ident_by_id(res)["newN"]
    assert ident["status"] == "suppressed"
    assert ident["twin_of"] == "ownedO"  # tie 1.0 vs 1.0, "ownedO" < "priorD"
    assert "overlaps" not in ident

    # A dismissed twin suppresses even when the artefact is also an owned
    # superset (which alone would only gain the overlaps marker).
    table_super = _dur_table(50)
    owned_s = _range_artefact("ownedO", 0, 19)
    m_s = _range_artefact("newM", 0, 19)  # takes ownedO
    dismissed_s = _range_artefact("priorD", 0, 19)  # identical claims as ownedO
    n_super = _range_artefact("newN", 0, 49)  # Jaccard 20/50 = 0.4 with both priors
    decisions_s = {
        "priorD": {"status": "dismissed", "name": None, "merged_into": None}
    }
    res = match_identities(
        [m_s, n_super], [owned_s, dismissed_s], decisions_s, table_super,
        owned={"ownedO"},
    )
    ident = _ident_by_id(res)["newN"]
    assert ident["status"] == "suppressed"
    assert ident["twin_of"] == "priorD"  # only suppression candidate
    assert "overlaps" not in ident


def test_owned_never_reclassifies_matched():
    """D-27: an artefact matched one-to-one to an owned prior (or any prior) is
    never re-classified: no twin suppression, no overlaps marker."""
    table = _dur_table(40)
    owned = _range_artefact("ownedO", 0, 9)  # claims {0..9}
    owned2 = _range_artefact("ownedP", 0, 19)  # fully overlaps the match
    n = _range_artefact("newN", 0, 9)  # matches ownedO with Jaccard 1.0

    res = match_identities([n], [owned, owned2], {}, table, owned={"ownedO", "ownedP"})
    ident = res[0]["identity"]

    assert res[0]["signature_id"] == "ownedO"
    assert ident["status"] == "matched"
    assert ident["matched"] == "ownedO"
    assert ident["twin_of"] is None
    assert "overlaps" not in ident


def test_without_owned_unchanged():
    """Without owned, D-27 inputs behave exactly as Plan 04-02: the near twin and
    the superset both stay new, and no identity dict carries an overlaps key."""
    table = _dur_table(50)
    owned = _range_artefact("ownedO", 0, 19)
    m = _range_artefact("newM", 0, 19)
    n_twin = _range_artefact("newN", 0, 11)  # Jaccard 0.6, but O is not owned here
    res = match_identities([m, n_twin], [owned], {}, table)
    by_id = _ident_by_id(res)
    assert by_id["newN"]["status"] == "new"
    assert by_id["newN"]["twin_of"] is None
    assert "overlaps" not in by_id["newN"]

    n_super = _range_artefact("newS", 0, 49)
    res2 = match_identities([n_super], [owned], {}, table)
    ident2 = res2[0]["identity"]
    assert ident2["status"] == "new"
    assert "overlaps" not in ident2


# --- Plan 06-04: program stage templates (D-40, RESEARCH Pitfall 1) ---------


def _program_art(sig_id: str, stages: list[tuple[float, int]]) -> SignatureArtefact:
    """Program artefact dict from the Plan 06-01 fixture, as a SignatureArtefact."""
    from tests.fixtures.programs import program_artefact

    return SignatureArtefact.from_dict(program_artefact(sig_id, stages))


def _states_json(art: SignatureArtefact) -> str:
    import json

    return json.dumps(art.to_dict()["states"], sort_keys=True)


def test_detection_differs_switched_pair_compares_states_only() -> None:
    """A switched pair keeps the Phase 4 rule: states decide, nothing else."""
    from custom_components.powerguesser.engine.learning.artefact import (
        detection_differs,
    )

    base = _make_test_artefact("a", "A", 1900, 2000, 2100, 90, 120, 150)
    same = _make_test_artefact("b", "B", 1900, 2000, 2100, 90, 120, 150)
    other = _make_test_artefact("c", "C", 1400, 1500, 1600, 90, 120, 150)

    assert detection_differs(base.to_dict(), same.to_dict()) is False
    assert detection_differs(base.to_dict(), other.to_dict()) is True
    # No active version: always differing (a device confirmed while the run
    # was in flight), exactly as _artefact_states_differ read it.
    assert detection_differs(None, base.to_dict()) is True
    assert detection_differs({}, base.to_dict()) is True


def test_detection_differs_program_pair_compares_the_stage_template() -> None:
    """Equal degenerate states still differ when a stage level or duration moved."""
    from custom_components.powerguesser.engine.learning.artefact import (
        detection_differs,
    )

    a = _program_art("p1", [(2000.0, 20), (150.0, 30)])
    same = _program_art("p2", [(2000.0, 20), (150.0, 30)])
    moved_level = _program_art("p3", [(2000.0, 20), (400.0, 30)])
    moved_dur = _program_art("p4", [(2000.0, 40), (150.0, 30)])

    # The degenerate session states are identical for every program fixture, so
    # only the stage template can tell them apart (RESEARCH Pitfall 1).
    assert _states_json(a) == _states_json(moved_level)
    assert _states_json(a) == _states_json(moved_dur)

    assert detection_differs(a.to_dict(), same.to_dict()) is False
    assert detection_differs(a.to_dict(), moved_level.to_dict()) is True
    assert detection_differs(a.to_dict(), moved_dur.to_dict()) is True


def test_fingerprint_switched_unchanged_and_program_adds_session_stages() -> None:
    """A switched fingerprint keeps its two keys; a program one adds the stages."""
    switched = _make_test_artefact("s1", "S", 1900, 2000, 2100, 90, 120, 150)
    fp = switched.fingerprint()
    assert set(fp) == {"kind", "states"}

    prog = _program_art("p1", [(2000.0, 20), (150.0, 30)])
    pfp = prog.fingerprint()
    assert set(pfp) == {"kind", "states", "session_stages"}
    stages = pfp["session_stages"]
    assert [s["level_w"]["p50"] for s in stages] == [2000.0, 150.0]
    assert [s["dur_s"]["p50"] for s in stages] == [600.0, 900.0]


def test_program_box_iou_uses_stage_levels_and_durations() -> None:
    """Two program templates with different stage levels never look alike."""
    import dataclasses

    from custom_components.powerguesser.engine.learning.identity import (
        _calc_artefact_box_iou,
    )

    a = _program_art("p1", [(2000.0, 20), (150.0, 30), (2000.0, 16), (200.0, 20)])
    same = _program_art("p2", [(2000.0, 20), (150.0, 30), (2000.0, 16), (200.0, 20)])
    other = _program_art("p3", [(900.0, 20), (300.0, 30)])
    no_stages = dataclasses.replace(a, session=None)

    assert _calc_artefact_box_iou(a, same) == pytest.approx(1.0)
    assert _calc_artefact_box_iou(a, other) < BOX_IOU_MIN
    assert _calc_artefact_box_iou(a, no_stages) == 0.0
    assert _calc_artefact_box_iou(no_stages, a) == 0.0

    # Switched values are unchanged: identical states still read IoU 1.0.
    s1 = _make_test_artefact("s1", "S", 1900, 2000, 2100, 90, 120, 150)
    s2 = _make_test_artefact("s2", "T", 1900, 2000, 2100, 90, 120, 150)
    assert _calc_artefact_box_iou(s1, s2) == pytest.approx(1.0)


def test_program_identity_falls_back_to_stage_iou() -> None:
    """The box-IoU fallback matches programs by stages: different ones stay new."""
    table = _dur_table(10)  # switched episodes only: neither programme claims any
    a = _program_art("p1", [(2000.0, 20), (150.0, 30), (2000.0, 16), (200.0, 20)])
    other = _program_art("p2", [(900.0, 20), (300.0, 30)])
    same = _program_art("p3", [(2000.0, 20), (150.0, 30), (2000.0, 16), (200.0, 20)])

    res = match_identities([a], [other], {}, table)
    assert res[0]["identity"]["status"] == "new"

    res_same = match_identities([a], [same], {}, table)
    assert res_same[0]["identity"]["status"] == "matched"
    assert res_same[0]["signature_id"] == "p3"


def test_to_artefact_restores_session_stages_from_a_fingerprint() -> None:
    """A decision fingerprint carries the stage template back into the artefact."""
    from custom_components.powerguesser.engine.learning.identity import (
        _calc_artefact_box_iou,
        _to_artefact,
    )

    prog = _program_art("p1", [(2000.0, 20), (150.0, 30)])
    rebuilt = _to_artefact(
        {"kind": "program", "fingerprint": prog.fingerprint()},
        sig_id_override="p1",
    )
    assert rebuilt.session is not None
    assert rebuilt.kind == "program"
    assert _calc_artefact_box_iou(prog, rebuilt) == pytest.approx(1.0)

    # A fingerprint without stages rebuilds with no template: IoU 0.0.
    switched_fp = {"kind": "program", "states": prog.fingerprint()["states"]}
    bare = _to_artefact({"kind": "program", "fingerprint": switched_fp})
    assert _calc_artefact_box_iou(prog, bare) == 0.0


# --- D-07 (HA half): similar_dismissed -------------------------------------


def test_similar_dismissed_picks_best_overlap() -> None:
    """The best overlapping dismissed fingerprint wins; named/low/other-kind/missing are ignored."""
    from custom_components.powerguesser.engine.learning.identity import (
        _calc_artefact_box_iou,
        _to_artefact,
    )

    base = _make_test_artefact("new", "New", 1900, 2000, 2100, 90, 120, 150)
    hi = _make_test_artefact("dhi", "Dhi", 1900, 2000, 2100, 90, 120, 160).fingerprint()
    lo = _make_test_artefact("dlo", "Dlo", 1900, 2000, 2100, 90, 120, 220).fingerprint()
    named = _make_test_artefact("dname", "Named", 1900, 2000, 2100, 90, 120, 150).fingerprint()
    decisions: dict[str, Any] = {
        "dhi": {"status": "dismissed", "fingerprint": hi},
        "dlo": {"status": "dismissed", "fingerprint": lo},
        "dname": {"status": "named", "fingerprint": named},
    }

    iou_hi = _calc_artefact_box_iou(base, _to_artefact(decisions["dhi"], "dhi"))
    iou_lo = _calc_artefact_box_iou(base, _to_artefact(decisions["dlo"], "dlo"))
    iou_named = _calc_artefact_box_iou(base, _to_artefact(decisions["dname"], "dname"))
    assert iou_named > iou_hi > iou_lo >= BOX_IOU_MIN

    # The named decision has the best overlap but is not dismissed: dhi wins.
    assert similar_dismissed(base.to_dict(), decisions) == "dhi"

    # A dismissed fingerprint below BOX_IOU_MIN is ignored.
    far = _make_test_artefact("dfar", "Dfar", 100, 200, 300, 10, 20, 30).fingerprint()
    assert similar_dismissed(base, {"dfar": {"status": "dismissed", "fingerprint": far}}) is None

    # Another kind never matches.
    prog = {
        "status": "dismissed",
        "fingerprint": {"kind": "program", "states": base.fingerprint()["states"]},
    }
    assert similar_dismissed(base, {"prog": prog}) is None

    # A decision without a fingerprint is ignored.
    assert similar_dismissed(base, {"nofp": {"status": "dismissed"}}) is None
    assert similar_dismissed(base, {}) is None


# --- CAP-09 (Plan 08-08): containment anchors and capture-origin precedence ---


def _table_with_blocks(rows: list[tuple[float, int]]) -> dict[str, np.ndarray]:
    """Switched episodes with the given (dur_s, block); rise/fall/plateau 2000 W."""
    n = len(rows)
    return {
        "start_i": np.arange(n, dtype=np.int64),
        "end_i": np.arange(n, dtype=np.int64) + 1,
        "rise_w": np.full(n, 2000.0),
        "fall_w": np.full(n, -2000.0),
        "dur_s": np.array([float(d) for d, _b in rows], dtype=np.float64),
        "plateau_w": np.full(n, 2000.0),
        "energy_wh": np.full(n, 50.0),
        "kind": np.zeros(n, dtype=np.int32),
        "block": np.array([b for _d, b in rows], dtype=np.int32),
    }


def _anchor_fixture() -> tuple[
    dict[str, np.ndarray], SignatureArtefact, SignatureArtefact, dict[str, set[int]]
]:
    """A val-block capture whose preview over-claims the train rows.

    Ten train episodes (dur 10..100, block 0) plus one validation episode
    (dur 150, block 1): the captured kettle run. The LLM artefact's narrow box
    claims only the train row at dur 100 and the val row; the preview's wide
    tolerance box claims all ten train rows, so Jaccard on train claims is
    1/10 < 0.5 and the one-to-one pass cannot match them. The anchor maps the
    preview id to the val row (index 10).
    """
    rows = [(float((i + 1) * 10), 0) for i in range(10)] + [(150.0, 1)]
    table = _table_with_blocks(rows)
    new_art = _make_test_artefact("llm", "Llm", 1900, 2000, 2100, 95, 120, 210)
    preview = _make_test_artefact("cap_preview", "Kettle", 1900, 2000, 2100, 5, 55, 355)
    return table, new_art, preview, {"cap_preview": {10}}


def test_anchor_matches_capture_in_val_block() -> None:
    """CAP-09: a val-block capture reconciles the learned artefact to the preview.

    The preview's wide boxes over-claim the train rows (Jaccard < 0.5), so the
    one-to-one pass gives up; the containment-anchor pass matches because the
    learned artefact itself claims the captured val episode. Without the anchor
    there are two identities.
    """
    table, new_art, preview, anchors = _anchor_fixture()

    res = match_identities([new_art], [preview], {}, table, anchors=anchors)
    assert res[0]["signature_id"] == "cap_preview"
    assert res[0]["identity"]["status"] == "matched"
    assert res[0]["identity"]["matched"] == "cap_preview"

    res_no = match_identities([new_art], [preview], {}, table)
    assert res_no[0]["signature_id"] == "llm"
    assert res_no[0]["identity"]["status"] == "new"


def test_anchor_ties_and_kind() -> None:
    """CAP-09: the artefact containing more anchors wins, ties by new id ascending;
    a program artefact never anchor-matches a switched preview."""
    rows = [(float((i + 1) * 10), 0) for i in range(4)] + [(150.0, 1), (160.0, 1)]
    table = _table_with_blocks(rows)
    preview = _make_test_artefact("cap_preview", "Kettle", 1900, 2000, 2100, 5, 55, 355)
    anchors = {"cap_preview": {4, 5}}

    # new_a claims both anchored val rows; new_b only one -> new_a wins.
    new_a = _make_test_artefact("new_a", "A", 1900, 2000, 2100, 95, 120, 210)
    new_b = _make_test_artefact("new_b", "B", 1900, 2000, 2100, 145, 150, 155)
    res = match_identities([new_a, new_b], [preview], {}, table, anchors=anchors)
    assert res[0]["signature_id"] == "cap_preview"
    assert res[0]["identity"]["status"] == "matched"
    assert res[1]["signature_id"] == "new_b"
    assert res[1]["identity"]["status"] == "new"

    # Tie at one anchored row: the lower new id wins (new_c < new_d).
    new_c = _make_test_artefact("new_c", "C", 1900, 2000, 2100, 145, 150, 155)
    new_d = _make_test_artefact("new_d", "D", 1900, 2000, 2100, 135, 150, 165)
    res_tie = match_identities(
        [new_c, new_d], [preview], {}, table, anchors={"cap_preview": {4}}
    )
    assert res_tie[0]["signature_id"] == "cap_preview"
    assert res_tie[1]["signature_id"] == "new_d"
    assert res_tie[1]["identity"]["status"] == "new"

    # A program artefact never anchor-matches a switched preview.
    prog = _make_test_artefact("new_prog", "P", 1900, 2000, 2100, 145, 150, 155, kind="program")
    res_prog = match_identities(
        [prog], [preview], {}, table, anchors={"cap_preview": {4}}
    )
    assert res_prog[0]["identity"]["status"] == "new"


def test_capture_prior_beats_dismissed_twin() -> None:
    """D-07: capture-origin precedence matches the capture prior before a higher-
    scoring dismissed twin, so the artefact reconciles instead of being suppressed."""
    table = _dur_table(20)
    new_art = _range_artefact("new1", 0, 9)  # claims {0..9}
    capture = _range_artefact("cap_prior", 0, 5)  # Jaccard 6/10 = 0.6
    decisions = {"priorD": _decision_for_ranges([(0, 7)])}  # Jaccard 8/10 = 0.8

    res = match_identities(
        [new_art], [capture], decisions, table, capture_prior_ids=["cap_prior"]
    )
    assert res[0]["signature_id"] == "cap_prior"
    assert res[0]["identity"]["status"] == "matched"
    assert res[0]["identity"]["twin_of"] is None


def test_llm_twin_of_dismissed_still_suppressed() -> None:
    """D-07 regression: without a capture prior an LLM near twin of a dismissed
    candidate stays suppressed, and the new keywords change nothing."""
    table = _dur_table(50)
    new_art = _range_artefact("new1", 0, 9)
    decisions = {"priorD": _decision_for_ranges([(0, 9), (20, 49)])}

    res = match_identities(
        [new_art], [], decisions, table, anchors=None, capture_prior_ids=()
    )
    res_plain = match_identities([new_art], [], decisions, table)
    assert res == res_plain
    ident = res[0]["identity"]
    assert ident["status"] == "suppressed"
    assert ident["matched"] is None
    assert ident["twin_of"] == "priorD"


def test_no_capture_keywords_is_identical() -> None:
    """Without the capture keywords, match_identities output is byte-identical
    to the pre-CAP-09 call for every existing fixture shape in this file."""
    scenarios: list[tuple] = []

    # identity_ignores_llm_names
    t1 = _make_test_table(
        [
            {"rise_w": 2000.0, "fall_w": -2000.0, "dur_s": 120.0},
            {"rise_w": 1500.0, "fall_w": -1500.0, "dur_s": 600.0},
        ]
    )
    scenarios.append(
        (
            [_make_test_artefact("new1", "N", 1900, 2000, 2100, 90, 120, 150)],
            [_make_test_artefact("prior1", "P", 1900, 2000, 2100, 90, 120, 150)],
            {},
            t1,
            (),
        )
    )

    # suppression + merge mapping
    n1 = _make_test_artefact("new1", "A1", 1900, 2000, 2100, 90, 120, 150)
    n2 = _make_test_artefact("new2", "A2", 1400, 1500, 1600, 500, 600, 700)
    p1 = _make_test_artefact("prior1", "P1", 1900, 2000, 2100, 90, 120, 150)
    p2 = _make_test_artefact("prior2", "P2", 1400, 1500, 1600, 500, 600, 700)
    scenarios.append(
        (
            [n1, n2],
            [p1, p2],
            {
                "prior1": {"status": "dismissed", "name": None, "merged_into": None},
                "prior2": {"status": "merged", "name": None, "merged_into": "t99"},
            },
            t1,
            (),
        )
    )

    # near twin of dismissed
    t2 = _dur_table(50)
    scenarios.append(
        (
            [_range_artefact("new1", 0, 9)],
            [],
            {"priorD": _decision_for_ranges([(0, 9), (20, 49)])},
            t2,
            (),
        )
    )

    # owned twin suppression
    t3 = _dur_table(20)
    scenarios.append(
        (
            [_range_artefact("newM", 0, 19), _range_artefact("newN", 0, 11)],
            [_range_artefact("ownedO", 0, 19)],
            {},
            t3,
            ("ownedO",),
        )
    )

    # box-IoU fallback
    t4 = _make_test_table([{"rise_w": 500.0, "fall_w": -500.0, "dur_s": 30.0}])
    scenarios.append(
        (
            [_make_test_artefact("new_iou", "N", 1900, 2000, 2100, 80, 100, 120)],
            [_make_test_artefact("prior_iou", "P", 1920, 2000, 2080, 85, 100, 115)],
            {},
            t4,
            (),
        )
    )

    for new, priors, decisions, table, owned in scenarios:
        base = match_identities(new, priors, decisions, table, owned=owned)
        same = match_identities(
            new,
            priors,
            decisions,
            table,
            owned=owned,
            anchors=None,
            capture_prior_ids=(),
        )
        assert same == base


def test_anchor_claims_do_not_depend_on_priors() -> None:
    """The anchor pass's solo replay of blocks 1 and 2 is exact, and a competing
    non-capture prior that also contains the anchored row does not block the match."""
    from custom_components.powerguesser.engine.learning.score import replay

    table, new_art, preview, anchors = _anchor_fixture()

    for block in (1, 2):
        solo = replay([new_art], table, block=block).claims["llm"]
        joint = replay([new_art, preview], table, block=block).claims["llm"]
        assert solo == joint

    # A second, non-capture prior Q whose boxes also contain the anchored row
    # (and whose train claims do not match the new artefact) is not the anchor
    # target: the capture prior still wins.
    q = _make_test_artefact("q_prior", "Q", 1900, 2000, 2100, 5, 55, 355)
    res = match_identities(
        [new_art],
        [q, preview],
        {},
        table,
        anchors=anchors,
        capture_prior_ids=["cap_preview"],
    )
    assert res[0]["signature_id"] == "cap_preview"
    assert res[0]["identity"]["status"] == "matched"


def test_anchor_pass_without_block_two() -> None:
    """A table with blocks 0 and 1 only still runs the anchor pass (the block-2
    replay contributes empty claims) and still anchor-matches the val capture."""
    from custom_components.powerguesser.engine.learning.score import replay

    table, new_art, preview, anchors = _anchor_fixture()
    assert sorted(set(table["block"].tolist())) == [0, 1]
    assert replay([new_art], table, block=2).claims["llm"] == set()

    res = match_identities([new_art], [preview], {}, table, anchors=anchors)
    assert res[0]["signature_id"] == "cap_preview"
    assert res[0]["identity"]["status"] == "matched"

