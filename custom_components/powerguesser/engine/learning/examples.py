"""HA-free selection, trimming and rendering of capture examples (CAP-10, D-08).

Captures the household owner snapped on purpose become labelled examples in the
next LLM learning run. This module owns the whole engine half of that: the byte
and count budget, the deterministic selection order, the opaque ``eN`` example
ids and ``tN`` target-group tokens, the rendered ``<examples>`` block and the
``<example_names>`` block under the names flag.

Rows are plain dicts produced by the HA side (09-04). Every row carries:
``capture_id``, ``resolved_at`` (used only for ordering, never sent), ``verdict``
(``clean`` / ``contaminated`` / ``ambiguous``), ``group`` (an opaque target-group
key), ``label`` and ``target_name`` (user-supplied text or ``None``), ``rise_w``,
``fall_w``, ``dur_s``, ``energy_wh``, ``hour``, ``dow`` and the optional phase
shares ``l1_share`` / ``l2_share`` / ``l3_share``.

Selection is deterministic (D-08): pass 1 takes one example per target group
(newest within the group, clean before contaminated/ambiguous), groups ordered by
their pick newest first; pass 2 fills newest first; every tie breaks on
``capture_id`` ascending. Ids ``e1..eN`` follow the selection order with no gaps.
The rendered block never carries a capture id, an entity id or an absolute
timestamp. numpy-free, no Home Assistant import.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any

from .budget import BYTES_PER_TOKEN
from .proposal import sanitize_label

# INITIAL VALUES (D-08): the fixed byte ceiling and count cap, and the share of
# the run input cap the example block may take. The byte ceiling wins for the
# default 300k run cap; a small user cap shrinks the budget.
EXAMPLE_MAX_BYTES = 12_288
EXAMPLE_MAX_COUNT = 40
EXAMPLE_BUDGET_SHARE = 0.02

# Quality order: clean runs are the strongest evidence, so pass 1 prefers them.
_VERDICT_RANK = {"clean": 0, "contaminated": 1, "ambiguous": 2}


def example_budget_bytes(run_input_tokens: int) -> int:
    """Byte budget for the example block: min(fixed ceiling, share of the run cap)."""
    shared = int(EXAMPLE_BUDGET_SHARE * run_input_tokens * BYTES_PER_TOKEN)
    return min(EXAMPLE_MAX_BYTES, shared)


@dataclass(frozen=True)
class ExampleSelection:
    """A rendered example block plus the bookkeeping the HA side keeps (D-08).

    ``text`` is the block appended inside ``<data>`` ("" when nothing is
    selected). ``ids`` lists the example ids in selection order; ``id_map`` maps
    each ``eN`` to the capture id it stands for; ``left_out`` is the number of
    eligible rows the budget left out.
    """

    text: str = ""
    ids: tuple[str, ...] = ()
    id_map: Mapping[str, str] = field(default_factory=dict)
    left_out: int = 0


EMPTY_SELECTION = ExampleSelection()


def _escaped_bytes(text: str) -> int:
    """UTF-8 bytes the block occupies once JSON-escaped (newlines as ``\\n``)."""
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _rank(verdict: Any) -> int:
    return _VERDICT_RANK.get(str(verdict), 2)


def _resolved_at_key(row: Mapping[str, Any]) -> str:
    """Total sort key for ``resolved_at`` that a missing value cannot break (WR-02).

    ``CaptureManager._valid_captures`` does not require ``resolved_at`` on a
    ``done`` record, so a hand-edited or partially written record can lack it.
    Sorting on the raw value would compare ``None`` against a ``str`` and raise
    ``TypeError``, failing the whole learning run. Coercing to a string treats a
    missing value as the oldest (``""`` sorts before any ISO timestamp) and keeps
    the ordering total, matching the guard the rest of the phase applies to this
    field (``capture.py`` ``_last_result`` / ``list_view``).
    """
    return str(row.get("resolved_at") or "")


def _group_key(row: Mapping[str, Any]) -> str:
    group = row.get("group")
    return str(group) if group is not None else str(row.get("capture_id", ""))


def _is_renderable(row: Mapping[str, Any]) -> bool:
    """True when the row carries a snapped pair this block can render."""
    if not isinstance(row, Mapping) or row.get("capture_id") is None:
        return False
    for key in ("rise_w", "fall_w", "dur_s", "energy_wh"):
        value = row.get(key)
        if value is None:
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def _label_for(row: Mapping[str, Any]) -> str | None:
    """The row's label, falling back to its target name, or None when both empty (D-05)."""
    for key in ("label", "target_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _cluster_for(row: Mapping[str, Any], digest: Mapping[str, Any]) -> str:
    """First switched digest cluster in id order that contains the example, else 'none' (D-08)."""
    try:
        rise = float(row.get("rise_w"))
        dur_min = float(row.get("dur_s")) / 60.0
    except (TypeError, ValueError):
        return "none"
    clusters = digest.get("clusters") or []
    for c in sorted(
        (c for c in clusters if isinstance(c, Mapping) and c.get("kind") == "switched"),
        key=lambda c: str(c.get("id", "")),
    ):
        rise_box = c.get("rise_w") or {}
        dur_box = c.get("dur_min") or {}
        r_lo, r_hi = rise_box.get("p10"), rise_box.get("p90")
        d_lo, d_hi = dur_box.get("p10"), dur_box.get("p90")
        if None in (r_lo, r_hi, d_lo, d_hi):
            continue
        if r_lo <= rise <= r_hi and d_lo <= dur_min <= d_hi:
            return str(c.get("id", "none"))
    return "none"


def _int_or_zero(value: Any) -> int:
    """Coerce a possibly-missing numeric field to an int, defaulting to 0 (WR-02 class).

    ``CaptureManager.example_rows`` builds ``hour``/``dow`` with
    ``features.get("hour")`` / ``features.get("dow")`` and a ``{}`` fallback, so a
    hand-edited or partially written record can carry them as ``None`` or omit the
    keys entirely. ``int(None)`` raises ``TypeError`` and would fail the whole
    learning run, so a missing or unparseable value renders as 0.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _share_or_none(value: Any) -> float | None:
    """Coerce a possibly-missing phase share to a float, else None (WR-02 class).

    ``CaptureManager.example_rows`` passes ``features.get("l1_share")`` and friends
    through raw, and ``_is_renderable`` validates only rise/fall/dur/energy, so a
    hand-edited or partially written record can carry a non-numeric share. Calling
    ``float()`` on it would raise ``ValueError`` mid-selection and fail the whole
    learning run (and its preview payload), so an unparseable share renders no
    phase suffix — the same guard the rest of this module applies to numeric fields.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_line(eid: str, token: str, row: Mapping[str, Any], cluster: str) -> str:
    """One numeric example line: opaque ids, hour/dow only, no timestamp."""
    phase = ""
    l1 = _share_or_none(row.get("l1_share"))
    l2 = _share_or_none(row.get("l2_share"))
    l3 = _share_or_none(row.get("l3_share"))
    if l1 is not None and l2 is not None and l3 is not None:
        phase = f" phase=l1:{l1:.2f},l2:{l2:.2f},l3:{l3:.2f}"
    return (
        f"{eid} t={token} q={row.get('verdict', 'clean')}"
        f" rise={round(float(row['rise_w']))} fall={round(float(row['fall_w']))}"
        f" dur_min={float(row['dur_s']) / 60.0:.1f} wh={float(row['energy_wh']):.1f}"
        f" hour={_int_or_zero(row.get('hour'))} dow={_int_or_zero(row.get('dow'))}"
        f"{phase} cluster={cluster}"
    )


def _order(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deterministic selection order: pass 1 one per group, then pass 2 newest-first (D-08)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    picks: list[Mapping[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda r: str(r.get("capture_id", "")))
        ordered = sorted(
            ordered, key=_resolved_at_key, reverse=True
        )
        ordered = sorted(ordered, key=lambda r: _rank(r.get("verdict")))
        picks.append(ordered[0])

    # Groups ordered by their pick, newest first; ties on capture_id ascending.
    picks = sorted(picks, key=lambda r: str(r.get("capture_id", "")))
    picks = sorted(picks, key=_resolved_at_key, reverse=True)

    chosen_ids = {id(r) for r in picks}
    remaining = [r for r in rows if id(r) not in chosen_ids]
    remaining = sorted(remaining, key=lambda r: str(r.get("capture_id", "")))
    remaining = sorted(remaining, key=_resolved_at_key, reverse=True)

    return [*picks, *remaining]


def _render_block(
    selected: Sequence[Mapping[str, Any]],
    digest: Mapping[str, Any],
    names_flag: bool,
    left_out: int,
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    """Render the ``<examples>`` / ``<example_names>`` block for a selected run."""
    tokens: dict[str, str] = {}
    lines: list[str] = []
    names_lines: list[str] = []
    ids: list[str] = []
    id_map: dict[str, str] = {}
    for index, row in enumerate(selected, start=1):
        eid = f"e{index}"
        ids.append(eid)
        id_map[eid] = str(row.get("capture_id", ""))
        group = _group_key(row)
        if group not in tokens:
            tokens[group] = f"t{len(tokens) + 1}"
        lines.append(_format_line(eid, tokens[group], row, _cluster_for(row, digest)))
        if names_flag:
            label = _label_for(row)
            if label is not None:
                names_lines.append(f"{eid}: {sanitize_label(label)}")

    parts = ["<examples>", *lines]
    if left_out > 0:
        parts.append(f"examples_left_out n={left_out}")
    parts.append("</examples>")
    if names_lines:
        parts.append("<example_names>")
        parts.extend(names_lines)
        parts.append("</example_names>")
    return "\n".join(parts), tuple(ids), id_map


def select_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    digest: Mapping[str, Any],
    names_flag: bool,
    max_bytes: int,
    max_count: int,
) -> ExampleSelection:
    """Select, trim and render capture examples within the byte and count budget.

    Rows are the plain dicts described in the module docstring. The block is grown
    in selection order and stops at the first row whose addition would push the
    JSON-escaped UTF-8 size past ``max_bytes`` or the count past ``max_count``. A
    row that is not renderable (no snapped pair) is skipped, and ``left_out``
    counts every eligible row that was not selected.
    """
    eligible = [r for r in rows if _is_renderable(r)]
    total = len(eligible)
    if total == 0:
        return EMPTY_SELECTION

    ordered = _order(eligible)
    selected: list[Mapping[str, Any]] = []
    block = ""
    ids: tuple[str, ...] = ()
    id_map: dict[str, str] = {}
    for row in ordered:
        if len(selected) >= max_count:
            break
        trial = [*selected, row]
        trial_block, trial_ids, trial_map = _render_block(
            trial, digest, names_flag, total - len(trial)
        )
        if _escaped_bytes(trial_block) > max_bytes:
            break
        selected = trial
        block, ids, id_map = trial_block, trial_ids, trial_map

    if not selected:
        return ExampleSelection(left_out=total)
    return ExampleSelection(text=block, ids=ids, id_map=id_map, left_out=total - len(selected))
