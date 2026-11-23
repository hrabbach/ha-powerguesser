"""Joint subset scoring for overlapping edge attribution (HA-free).

Implements D-06/D-07/D-36/D-41 (Phase 6 owner decisions): every edge is matched
against single devices AND small device subsets in ONE candidate list under ONE
runner-up margin, so a single device never wins just because it was checked
first, and anything ambiguous is refused instead of being absorbed.

The rules, in the order they apply (D-07 precedes D-41):

- Every plausible explanation of the edge is a ``(key, action, box)`` tuple whose
  box is ``(lo, p50, hi)`` in the edge delta's sign (rises positive, falls
  negative). ``action`` is the state index of a switched device or the target
  stage index of a program device; it is opaque to this module.
- ``prefilter`` drops boxes that cannot contribute to any sum reaching the
  delta, keeps one explanation per key, and bounds the search to the
  ``JOINT_PREFILTER`` best by ``(distance, key)`` (T-06-12: at most 35 subsets
  of size 2-3 per edge, and only on edges, never per tick).
- ``best_subsets`` enumerates every combination of 2..``JOINT_MAX_K`` prefiltered
  explanations, keeps the subsets whose summed ``[lo, hi]`` contains the delta
  inclusively, and scores them exactly like the direction's singleton score —
  the normalised side distance to the summed box (identical to
  ``online._norm_distance``, which delegates here) or, for the fall path, the
  relative ``|delta - sum p50| / |sum p50|`` form — plus an Occam penalty
  ``JOINT_PENALTY x (k - 1)`` so widening a box by adding devices cannot win.
- ``pick`` decides the single candidate list: fewer than two containing
  singletons and a unique best entry at least ``margin`` above every other
  entry (singleton or subset) attributes; anything else refuses and returns the
  competing containing entries, so the caller can count them
  (``overlap_refused``) and feed the drift guard.
- ``split_delta`` splits a uniquely explained edge across its winning subset in
  proportion to each member's learned p50 (D-36), each share clamped to its own
  box.

``JOINT_MAX_K``, ``JOINT_PREFILTER`` and ``JOINT_PENALTY`` are untuned initial
values; replay tuning on the reference install's plug data is post-ship (D-22).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
import math

Box = tuple[float, float, float]
Explanation = tuple[str, int, Box]

# D-22 initial values (RESEARCH A4/B1): k <= 3, at most 6 candidates, and a
# per-extra-device penalty that keeps a widened box from winning by width alone.
JOINT_MAX_K = 3
JOINT_PREFILTER = 6
JOINT_PENALTY = 0.15


def norm_distance(delta: float, box: Box) -> float:
    """|delta - p50| / max(side half-width, 1.0) for the side p50 lies on.

    The one definition of the online detector's normalised box distance
    (``online._norm_distance`` delegates here), so a satisfied explanation
    scores identically whether it is a singleton or a summed subset.
    """
    lo, p50, hi = box
    half = (hi - p50) if delta >= p50 else (p50 - lo)
    return abs(delta - p50) / max(half, 1.0)


def _relative_distance(delta: float, box: Box) -> float:
    """|delta - p50| / |p50|: the fall path's singleton score form."""
    p50 = box[1]
    return abs(delta - p50) / max(abs(p50), 1.0)


def _can_contribute(delta: float, box: Box) -> bool:
    """False when no sum including this box can reach the delta.

    A sum's range is ``[sum lo, sum hi]``; on the positive (rise) side every
    box is positive-shaped, so a box whose ``lo`` already exceeds the delta can
    never be part of a containing sum, and on the negative (fall) side the same
    holds for a box whose ``hi`` is below the delta.
    """
    lo, _p50, hi = box
    return lo <= delta if delta >= 0.0 else hi >= delta


def prefilter(
    delta: float, explanations: Sequence[Explanation], n: int
) -> list[Explanation]:
    """Bound the subset search: contributing explanations, one per key, best n.

    Keeps, per key, the explanation with the smallest normalised distance,
    drops explanations whose box cannot contribute to a containing sum, and
    returns at most ``n`` (by ``(distance, key)``) sorted by key so the
    enumeration order is deterministic for any input order (D-41).
    """
    best: dict[str, tuple[float, int, Explanation]] = {}
    for order, exp in enumerate(explanations):
        key = str(exp[0])
        if not _can_contribute(delta, exp[2]):
            continue
        dist = norm_distance(delta, exp[2])
        cur = best.get(key)
        if cur is None or (dist, order) < (cur[0], cur[1]):
            best[key] = (dist, order, exp)
    ranked = sorted(best.values(), key=lambda item: (item[0], item[2][0]))
    return [item[2] for item in ranked[: max(0, int(n))]]


def best_subsets(
    delta: float,
    explanations: Sequence[Explanation],
    *,
    min_k: int = 2,
    max_k: int = JOINT_MAX_K,
    penalty: float = JOINT_PENALTY,
    relative: bool = False,
) -> list[tuple[float, tuple[str, ...]]]:
    """Score every containing subset of the prefiltered explanations.

    Returns ``[(score, members)]`` sorted by ``(score, member keys)``, where
    ``members`` is the winning subset's keys sorted. ``relative=True`` scores in
    the fall path's relative form so singleton and subset scores are comparable
    there; the default scores the summed box with the side-distance rule.
    """
    ordered = sorted(explanations, key=lambda exp: str(exp[0]))
    out: list[tuple[float, tuple[str, ...]]] = []
    for k in range(max(1, int(min_k)), min(int(max_k), len(ordered)) + 1):
        for combo in combinations(ordered, k):
            lo = sum(exp[2][0] for exp in combo)
            p50 = sum(exp[2][1] for exp in combo)
            hi = sum(exp[2][2] for exp in combo)
            if not (lo <= delta <= hi):
                continue
            score = (
                _relative_distance(delta, (lo, p50, hi))
                if relative
                else norm_distance(delta, (lo, p50, hi))
            ) + penalty * (k - 1)
            out.append((score, tuple(sorted(str(exp[0]) for exp in combo))))
    return sorted(out, key=lambda item: (item[0], item[1]))


def _competing(
    containing: Sequence[tuple[float, tuple[str, ...], bool]],
    best_score: float,
    margin: float,
) -> tuple[tuple[str, ...], ...]:
    """Every containing singleton, plus every containing subset within the margin.

    The keys of these entries get the per-device refused-edge counter (P4's rule
    for containing singletons, extended to competing subsets) and, when two or
    more entries compete, the overlap-refusal counter.
    """
    return tuple(
        sorted(
            (
                entry[1]
                for entry in containing
                if len(entry[1]) == 1 or entry[0] - best_score < margin
            ),
            key=lambda keys: keys,
        )
    )


def pick(
    singletons: Sequence[tuple[float, str, bool]],
    subsets: Sequence[tuple[float, tuple[str, ...]]],
    margin: float,
    *,
    strict_singletons: bool = True,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    """Decide ONE candidate list of singletons and subsets (D-41).

    ``singletons`` is ``[(score, key, contains)]`` exactly as the caller's
    existing singleton loop computes it (k = 1, no penalty); ``subsets`` is
    ``best_subsets``' output (every entry contains). Returns:

    - ``("attribute", (keys,))`` for the lowest-scoring containing entry by
      ``(score, sorted keys)``, only when fewer than two singletons contain the
      delta (``strict_singletons``, the rise path's D-07 rule) and every other
      entry scores at least ``margin`` above it;
    - ``("refuse", competing)`` when some entry contains but that test fails,
      ``competing`` being one entry (its sorted keys) per competing containing
      entry, so the caller can deduplicate keys and count the entries;
    - ``("none", ())`` when no entry contains.

    ``strict_singletons=False`` is the fall path: its singleton candidates keep
    the Phase 3/P4 margin rule (two containing candidates are resolved by the
    margin, not refused outright), so an edge no subset ties is decided exactly
    as before (D-41 must-have).
    """
    entries: list[tuple[float, tuple[str, ...], bool]] = [
        (float(score), (str(key),), bool(contains)) for score, key, contains in singletons
    ]
    entries.extend((float(score), tuple(keys), True) for score, keys in subsets)
    containing = [entry for entry in entries if entry[2]]
    if not containing:
        return ("none", ())
    best = min(containing, key=lambda entry: (entry[0], entry[1]))
    if strict_singletons and sum(1 for entry in containing if len(entry[1]) == 1) >= 2:
        return ("refuse", _competing(containing, best[0], margin))
    runner = min(
        (entry[0] for entry in entries if entry is not best), default=math.inf
    )
    if runner - best[0] >= margin:
        return ("attribute", (best[1],))
    return ("refuse", _competing(containing, best[0], margin))


def split_delta(
    delta: float, members: Sequence[Explanation]
) -> list[tuple[str, float]]:
    """D-36: split an edge across its subset in proportion to each member's p50.

    Each member gets ``p50 + (delta - sum p50) x p50 / sum p50`` — that is
    ``delta x p50 / sum p50`` — clamped to its own box, so no member claims
    power outside its learned bounds. Called only for a uniquely explained edge
    (a subset pick); a refused edge is never split (D-07).
    """
    total = sum(float(exp[2][1]) for exp in members)
    out: list[tuple[str, float]] = []
    for key, _action, box in members:
        lo, p50, hi = box
        share = (delta * p50 / total) if total != 0.0 else p50
        out.append((str(key), float(min(max(share, lo), hi))))
    return out
