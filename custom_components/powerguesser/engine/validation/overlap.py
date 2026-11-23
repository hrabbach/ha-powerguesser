"""Overlap metrics for the holdout validation report (Plan 06-08; D-08, D-28).

The judge is the plug ground truth, never the detector. An ON tick is a holdout
sample where a linked plug reads above ``PLUG_ON_W`` (the same threshold
``plug_sessions`` uses); an overlap tick is a holdout sample where two or more
linked plugs are ON at once, and a solo tick is a holdout sample where exactly
one is. The edge counters are the replay's cumulative ``joint_attributed`` /
``overlap_refused`` totals read over exactly those samples.

Every figure travels with its definition text, so a reader of the report never
has to reconstruct what it measured. Following the validation contract's
refusal-without-figures rule (05-LEARNINGS.md, metrics.py), a figure computed
from fewer than ``OVERLAP_MIN_TICKS`` ticks is refused (``None``), never 0:
that is the per-device solo/overlap energy error, the energy left in the
remainder and a pair row. Plain tick counts keep their numeric value (0 is a
fact there, not a refused measurement), and a refused report carries no overlap
block at all.

Pure numpy + stdlib, HA-free (tests/engine/test_no_ha_imports.py).
"""

from collections.abc import Mapping

import numpy as np

from ..learning.episodes import PLUG_ON_W
from .constants import OVERLAP_MIN_TICKS
from .metrics import energy_error_pct

OVERLAP_DEFINITION = (
    "an overlap tick is a holdout sample where two or more linked plugs read "
    f"above {PLUG_ON_W:g} W (plug ground truth); a solo tick is a holdout "
    "sample where exactly one linked plug is above it"
)

# The review concern L207@8b3be9c: edges and ticks are different units. The
# definition travels in the block as edges_definition so a reader cannot
# subtract one from the other by mistake.
EDGES_DEFINITION = (
    "edges counts the detector's edge DECISIONS on overlap samples, not ticks: "
    "the total increase of the cumulative attributed-jointly and refused "
    "counters over the samples where any overlap tick is set. Two loads "
    "switching in the same sample are one joint decision, and an edge one "
    "sample before a plug crosses the plug ON threshold counts as non-overlap, "
    "so edges and n_overlap_ticks are different units and must not be "
    "subtracted from each other"
)

ENERGY_LEFT_DEFINITION = (
    "the energy on overlap ticks the estimate did not cover: the sum over "
    "measured plug devices of max(0, plug reading - estimated power) x the grid "
    "step, in Wh; it includes, but is not limited to, refused edges"
)


def on_mask(y: object, on_w: float = PLUG_ON_W) -> np.ndarray:
    """Boolean ON mask of one plug series; a NaN (missing) reading counts as off."""
    arr = np.asarray(y, dtype=np.float64)
    return np.isfinite(arr) & (arr > float(on_w))


def shifted_copy(series: object, shift: int) -> np.ndarray:
    """A copy of ``series`` moved by ``shift`` samples (D-28's time shift).

    ``shift`` > 0 moves the series to later samples, ``shift`` < 0 to earlier
    ones, and the samples left empty are 0 W. A NaN (missing) reading counts as
    0 W, so the copy can be added to a residual and used as ground truth without
    dragging NaN in.
    """
    arr = np.nan_to_num(
        np.asarray(series, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    out = np.zeros(len(arr), dtype=np.float64)
    n = len(arr)
    if shift > 0:
        out[shift:] = arr[: n - shift]
    elif shift < 0:
        out[:shift] = arr[-shift:]
    else:
        out[:] = arr
    return out


def tick_masks(
    masks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Split every plug's ON ticks into solo and overlap ticks.

    ``masks`` maps an entity id to its boolean ON mask (all the same length).
    Returns ``(solo_by_entity, overlap_by_entity, n_on)``: for each entity its
    ON ticks where no other plug is ON (solo) and where at least one other plug
    is ON (overlap); ``n_on`` is the per-sample integer count of ON plugs, from
    which the report's ``n_overlap_ticks`` (n_on >= 2) and ``n_solo_ticks``
    (n_on == 1) are read.
    """
    arrays = {str(key): np.asarray(m, dtype=bool) for key, m in masks.items()}
    n = len(next(iter(arrays.values()))) if arrays else 0
    n_on = np.zeros(n, dtype=np.int64)
    for m in arrays.values():
        n_on = n_on + m.astype(np.int64)
    solo_by_entity = {key: m & (n_on == 1) for key, m in arrays.items()}
    overlap_by_entity = {key: m & (n_on >= 2) for key, m in arrays.items()}
    return solo_by_entity, overlap_by_entity, n_on


def _sum_wh(values: object, mask: np.ndarray, step_s: float) -> float:
    """Energy (Wh) of one series over the given ticks; missing values are left out."""
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    sel = arr[np.asarray(mask, dtype=bool)]
    sel = sel[np.isfinite(sel)]
    return float(np.sum(sel)) * float(step_s) / 3600.0


def split_errors(
    y: object,
    est: object,
    solo: np.ndarray,
    overlap: np.ndarray,
    step_s: float,
) -> dict[str, dict[str, float | int | None]]:
    """One device's energy error (%) over its solo and its overlap ticks.

    Returns ``{"solo": {"energy_error_pct", "n_ticks"}, "overlap": {...}}``.
    Each figure is null when its own tick count is below ``OVERLAP_MIN_TICKS``
    (refused, never 0); the tick counts stay numeric.
    """
    out: dict[str, dict[str, float | int | None]] = {}
    for name, raw in (("solo", solo), ("overlap", overlap)):
        mask = np.asarray(raw, dtype=bool)
        n_ticks = int(np.count_nonzero(mask))
        figure: float | None = None
        if n_ticks >= OVERLAP_MIN_TICKS:
            figure = energy_error_pct(
                _sum_wh(y, mask, step_s), _sum_wh(est, mask, step_s)
            )
        out[name] = {"energy_error_pct": figure, "n_ticks": n_ticks}
    return out


def _increase(cum: object, sel: np.ndarray) -> int:
    """Total increase of a cumulative counter over the selected samples."""
    if cum is None:
        return 0
    arr = np.asarray(cum, dtype=np.int64)
    if arr.size == 0 or not bool(np.any(sel)):
        return 0
    rates = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), arr)))
    return int(np.sum(rates[np.asarray(sel, dtype=bool)]))


def edge_counts(
    joint_cum: object, refused_cum: object, overlap_any: np.ndarray
) -> dict[str, int]:
    """Overlap edges the detector attributed jointly and the ones it refused.

    The unit is an edge DECISION, not a tick (review concern L207@8b3be9c): the
    total increase of each cumulative counter over the samples where any overlap
    tick is set, so two loads switching in one sample count 1 and an edge one
    sample before a plug crosses the ON threshold counts nothing.
    """
    sel = np.asarray(overlap_any, dtype=bool)
    return {
        "attributed_jointly": _increase(joint_cum, sel),
        "refused": _increase(refused_cum, sel),
    }


def remainder_energy_wh(
    ys: Mapping[str, object],
    ests: Mapping[str, object],
    overlaps: Mapping[str, np.ndarray],
    step_s: float,
) -> float:
    """Plug energy on overlap ticks the estimate did not cover, in Wh (D-08 c).

    The sum over measured plug devices of ``max(0, plug - estimate)`` over that
    device's overlap ticks. NaN readings contribute nothing.
    """
    total = 0.0
    for key, raw in overlaps.items():
        mask = np.asarray(raw, dtype=bool)
        if not bool(np.any(mask)):
            continue
        y = np.asarray(ys.get(key), dtype=np.float64) if key in ys else None
        est = np.asarray(ests.get(key), dtype=np.float64) if key in ests else None
        if y is None:
            continue
        diff = y[mask]
        if est is not None and est.size == y.size:
            diff = diff - est[mask]
        diff = diff[np.isfinite(diff)]
        total += float(np.sum(np.maximum(diff, 0.0))) * float(step_s) / 3600.0
    return total


def pair_table(
    ys: Mapping[str, object],
    ests: Mapping[str, object],
    masks: Mapping[str, np.ndarray],
    step_s: float,
) -> list[dict[str, object]]:
    """One row per device pair sharing at least ``OVERLAP_MIN_TICKS`` ON ticks.

    Rows carry ``{"a", "b", "n_ticks", "a_error_pct", "b_error_pct"}`` sorted by
    ``(a, b)``, the two entity ids, with both devices' energy error over the
    shared ticks, so the report shows that neither absorbed the other.

    Aggregation rule (review concern L208@8b3be9c): ``n_ticks`` and both errors
    aggregate every shared ON tick of the pair across the WHOLE holdout window,
    all overlap episodes summed, and ``OVERLAP_MIN_TICKS`` applies to that
    total, never per episode — a single kettle-during-wash overlap of 6 ticks is
    below it, seven of them are not.
    """
    keys = sorted(str(key) for key in masks)
    rows: list[dict[str, object]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = np.asarray(masks[a], dtype=bool) & np.asarray(masks[b], dtype=bool)
            n_ticks = int(np.count_nonzero(shared))
            if n_ticks < OVERLAP_MIN_TICKS:
                continue
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "n_ticks": n_ticks,
                    "a_error_pct": energy_error_pct(
                        _sum_wh(ys.get(a), shared, step_s),
                        _sum_wh(ests.get(a), shared, step_s),
                    ),
                    "b_error_pct": energy_error_pct(
                        _sum_wh(ys.get(b), shared, step_s),
                        _sum_wh(ests.get(b), shared, step_s),
                    ),
                }
            )
    return rows
