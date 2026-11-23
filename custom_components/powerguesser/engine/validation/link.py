"""Local shadow-artefact <-> plug linking (D-03, A12, Plan 05-03).

The link is a comparison step, not a plug-fitting path: each shadow artefact
is scored against each eligible plug by the Jaccard overlap of their ON masks
on the shadow run's TRAIN-block days only (holdout overlap never counts, so
the link cannot leak holdout information). Both masks merge gaps of at most
``PLUG_GAP_S`` so a replayed artefact's claimed episodes and a merged plug
session are comparable. Assignment is one-to-one, greedy by overlap
descending, then signature_id ascending, then entity_id ascending, at or
above ``LINK_MIN_OVERLAP``; every plug left over is reported not_discovered
and never silently dropped (D-03).

HA-free, numpy only (tests/engine/test_no_ha_imports.py).
"""

from collections.abc import Mapping, Sequence

import numpy as np


def on_mask(
    spans: Sequence[tuple[int, int]],
    n: int,
    allowed: np.ndarray | None = None,
    gap_samples: int = 0,
) -> np.ndarray:
    """Boolean ON mask of length ``n`` from inclusive [start, end] sample spans.

    Only samples where ``allowed`` is True are kept (the train-day restriction,
    D-03). Gaps of at most ``gap_samples`` unmarked samples between two marked
    samples are then closed — the same merge rule as ``plug_sessions`` so both
    sides of the Jaccard are comparable.
    """
    mask = np.zeros(int(n), dtype=bool)
    for start, end in spans:
        mask[int(start) : int(end) + 1] = True
    if allowed is not None:
        mask &= np.asarray(allowed, dtype=bool)
    if gap_samples > 0:
        true_idx = np.flatnonzero(mask)
        # Filling every gap of <= gap_samples between originally marked samples
        # is exactly the transitive island merge (each fill reconnects the two
        # islands the pair belongs to).
        for prev, cur in zip(true_idx[:-1], true_idx[1:]):
            if cur - prev - 1 <= gap_samples:
                mask[prev + 1 : cur] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|a and b| / |a or b| over boolean masks; 0.0 for an empty union."""
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a_bool | b_bool))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a_bool & b_bool)) / float(union)


def assign_links(
    scores: Mapping[tuple[str, str], float],
    entities: Sequence[str],
    *,
    min_overlap: float,
) -> tuple[list[dict[str, object]], list[str]]:
    """One-to-one greedy assignment of plugs to shadow artefacts (D-03).

    ``scores`` maps (entity_id, signature_id) to their overlap. Pairs are
    considered greedily by overlap descending, then signature_id ascending,
    then entity_id ascending; a pair below ``min_overlap`` or whose plug or
    artefact is already used is skipped. Returns (links, unlinked): links
    carry entity_id, signature_id and the overlap; unlinked lists the input
    entity ids that stayed without an artefact, in input order.
    """
    ranked = sorted(
        (
            (float(overlap), str(signature_id), str(entity_id))
            for (entity_id, signature_id), overlap in scores.items()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_entities: set[str] = set()
    used_signatures: set[str] = set()
    links: list[dict[str, object]] = []
    for overlap, signature_id, entity_id in ranked:
        if overlap < min_overlap:
            continue
        if entity_id in used_entities or signature_id in used_signatures:
            continue
        used_entities.add(entity_id)
        used_signatures.add(signature_id)
        links.append(
            {
                "entity_id": entity_id,
                "signature_id": signature_id,
                "overlap": overlap,
            }
        )
    unlinked = [str(entity_id) for entity_id in entities if str(entity_id) not in used_entities]
    return links, unlinked
