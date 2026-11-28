"""Tests for capture-example selection, trimming, ids and rendering (CAP-10, D-08)."""

import json

from custom_components.powerguesser.engine.learning.examples import (
    EXAMPLE_MAX_BYTES,
    EXAMPLE_MAX_COUNT,
    ExampleSelection,
    example_budget_bytes,
    select_examples,
)


DIGEST = {
    "header": {"grid_s": 30},
    "clusters": [
        {
            "id": "c03",
            "kind": "switched",
            "rise_w": {"p10": 1950, "p50": 2000, "p90": 2050},
            "dur_min": {"p10": 1.5, "p50": 2.5, "p90": 3.5},
        }
    ],
    "plugs": [],
}


def _escaped_bytes(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _row(
    capture_id,
    *,
    resolved_at=0,
    verdict="clean",
    group=None,
    label=None,
    target_name=None,
    rise_w=2010.0,
    fall_w=-2000.0,
    dur_s=180.0,
    energy_wh=100.0,
    hour=7,
    dow=2,
    phase=None,
):
    row = {
        "capture_id": capture_id,
        "resolved_at": resolved_at,
        "verdict": verdict,
        "group": group if group is not None else capture_id,
        "label": label,
        "target_name": target_name,
        "rise_w": rise_w,
        "fall_w": fall_w,
        "dur_s": dur_s,
        "energy_wh": energy_wh,
        "hour": hour,
        "dow": dow,
    }
    if phase is not None:
        row["l1_share"], row["l2_share"], row["l3_share"] = phase
    return row


def _select(rows, *, names_flag=False, max_bytes=EXAMPLE_MAX_BYTES, max_count=EXAMPLE_MAX_COUNT):
    return select_examples(
        rows, digest=DIGEST, names_flag=names_flag, max_bytes=max_bytes, max_count=max_count
    )


def test_selection_order_round_robin_then_newest():
    """Pass 1 picks each group's newest clean row, groups newest first; pass 2 fills newest first."""
    rows = [
        _row("g1-a", resolved_at=100, verdict="clean", group="g1"),
        _row("g1-b", resolved_at=300, verdict="clean", group="g1"),
        _row("g1-c", resolved_at=900, verdict="contaminated", group="g1"),
        _row("g2-a", resolved_at=200, verdict="clean", group="g2"),
        _row("g3-a", resolved_at=400, verdict="contaminated", group="g3"),
        _row("g3-b", resolved_at=800, verdict="contaminated", group="g3"),
    ]

    sel = _select(rows)

    assert sel.ids == ("e1", "e2", "e3", "e4", "e5", "e6")
    assert sel.left_out == 0
    # pass 1: g3 newest pick (800), g1 newest clean (300), g2 (200)
    # pass 2: the leftovers newest first
    assert [sel.id_map[i] for i in sel.ids] == [
        "g3-b", "g1-b", "g2-a", "g1-c", "g3-a", "g1-a",
    ]
    # tokens follow first appearance: g3 -> t1, g1 -> t2, g2 -> t3
    assert "e1 t=t1 q=contaminated" in sel.text
    assert "e2 t=t2 q=clean" in sel.text
    assert "e3 t=t3 q=clean" in sel.text
    assert "e4 t=t2" in sel.text
    assert "e5 t=t1" in sel.text
    assert "e6 t=t2" in sel.text


def test_selection_ties_break_on_capture_id():
    """Equal resolved_at rows order by capture_id ascending and share one target token."""
    rows = [
        _row("b", resolved_at=100, verdict="clean", group="g"),
        _row("a", resolved_at=100, verdict="clean", group="g"),
    ]

    sel = _select(rows)

    assert [sel.id_map[i] for i in sel.ids] == ["a", "b"]
    assert "e1 t=t1" in sel.text
    assert "e2 t=t1" in sel.text


def test_selection_respects_byte_and_count_budget():
    """The block stays within max_bytes/max_count; left_out counts eligible minus selected."""
    rows = [_row(f"cap{i:02d}", resolved_at=i) for i in range(60)]

    sel = _select(rows, max_count=40)
    assert len(sel.ids) == 40
    assert sel.left_out == 20
    assert "examples_left_out n=20" in sel.text
    assert _escaped_bytes(sel.text) <= EXAMPLE_MAX_BYTES

    # A small byte budget trims further and keeps the whole block under it.
    small = _select(rows, max_bytes=400)
    assert 0 < len(small.ids) < 40
    assert _escaped_bytes(small.text) <= 400
    assert small.left_out == 60 - len(small.ids)

    # When everything fits there is no left-out line.
    all_fit = _select(rows[:3])
    assert len(all_fit.ids) == 3
    assert all_fit.left_out == 0
    assert "examples_left_out" not in all_fit.text


def test_example_line_has_no_ids_or_timestamps():
    """A line carries the numeric fields but never a capture id, ISO date or 12-digit number."""
    rows = [
        _row(
            "cap_deadbeef",
            resolved_at=1_700_000_000_000,
            verdict="clean",
            group="grp",
            phase=(0.9, 0.05, 0.05),
        )
    ]

    sel = _select(rows)
    text = sel.text

    assert "e1 t=t1 q=clean" in text
    assert "rise=2010" in text
    assert "fall=-2000" in text
    assert "dur_min=3.0" in text
    assert "wh=100.0" in text
    assert "hour=7" in text
    assert "dow=2" in text
    assert "phase=l1:0.90,l2:0.05,l3:0.05" in text
    assert "cluster=c03" in text

    assert "cap_deadbeef" not in text
    assert "1700000000000" not in text
    import re

    assert not re.search(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text)
    assert not re.search(r"\b\d{12,}\b", text)


def test_cluster_tag_matches_digest_ranges():
    """An example inside a switched cluster's percentile box gets that id, else none."""
    inside = _select([_row("cap1", rise_w=2010.0, dur_s=180.0)])
    assert "cluster=c03" in inside.text

    outside = _select([_row("cap2", rise_w=5000.0, dur_s=180.0)])
    assert "cluster=none" in outside.text


def test_names_block_only_with_flag():
    """No labels without the flag; with it the label (falling back to target name) is sanitised."""
    row = _row("cap1", label="Kettle", target_name="Kitchen Kettle")

    off = _select([row], names_flag=False)
    assert "<example_names>" not in off.text
    assert "Kettle" not in off.text

    on = _select([row], names_flag=True)
    assert "<example_names>\ne1: Kettle\n</example_names>" in on.text

    # Empty label falls back to the target name; a non-string never forces a name.
    fallback = _select([_row("cap2", label="", target_name="Washer")], names_flag=True)
    assert "e1: Washer" in fallback.text

    # Neither label nor target name -> no names block at all.
    blank = _select([_row("cap3")], names_flag=True)
    assert "<example_names>" not in blank.text
    assert "e1: unknown" not in blank.text

    # A 60-character multi-byte label is cut to 40 code points and costs its real bytes.
    long_label = "\u00e9" * 60  # 60 x U+00E9, two UTF-8 bytes each
    trimmed = _select([_row("cap4", label=long_label)], names_flag=True)
    name_line = next(ln for ln in trimmed.text.splitlines() if ln.startswith("e1: "))
    assert name_line == "e1: " + "\u00e9" * 40  # 40 code points

    ascii_label = _select([_row("cap5", label="a" * 40)], names_flag=True)
    assert _escaped_bytes(trimmed.text) > _escaped_bytes(ascii_label.text)


def test_example_budget_bytes_matches_contract():
    """example_budget_bytes is min(fixed ceiling, share x run cap x bytes per token)."""
    assert example_budget_bytes(300_000) == EXAMPLE_MAX_BYTES
    assert example_budget_bytes(10_000) == int(0.02 * 10_000 * 3.0)


def test_select_examples_returns_empty_selection_when_no_rows():
    """No eligible rows give the shared EMPTY_SELECTION with no block."""
    sel = _select([])
    assert sel == ExampleSelection()
    assert sel.text == ""
    assert sel.left_out == 0


def test_missing_resolved_at_does_not_break_ordering():
    """A record without ``resolved_at`` orders oldest and never raises (WR-02).

    ``_valid_captures`` never requires ``resolved_at`` on a ``done`` record, so a
    hand-edited or partially written row can lack it (or carry ``None``). Sorting on
    the raw value would compare it against a ``str``/``int`` and raise ``TypeError``,
    failing the whole learning run; the ordering must stay total instead.
    """
    rows = [
        _row("g1-new", resolved_at=500, group="g1"),
        _row("g1-old", resolved_at=100, group="g1"),
        _row("g1-none", group="g1"),  # key removed below
        _row("g2-none", group="g2"),  # explicit None below
        _row("g3-x", resolved_at=200, group="g3"),
    ]
    del rows[2]["resolved_at"]
    rows[3]["resolved_at"] = None

    sel = _select(rows)  # must not raise

    assert sel.ids == ("e1", "e2", "e3", "e4", "e5")
    # Dated rows sort newest first; rows without a resolved_at are treated as
    # oldest and keep their capture_id-ascending tie order.
    assert [sel.id_map[i] for i in sel.ids] == [
        "g1-new",
        "g3-x",
        "g2-none",
        "g1-old",
        "g1-none",
    ]


def test_missing_or_none_hour_and_dow_render_as_zero():
    """A row whose hour/dow are absent or None renders hour=0 dow=0, never raises (WR-02 class).

    ``CaptureManager.example_rows`` builds these with ``features.get("hour")`` and a
    ``{}`` fallback, so a hand-edited or partially written record can carry them as
    ``None`` or omit the keys. ``int(None)`` raises ``TypeError`` and would fail the
    whole learning run.
    """
    rows = [
        _row("cap-none", hour=None, dow=None),
        _row("cap-missing"),
    ]
    del rows[1]["hour"]
    del rows[1]["dow"]

    sel = _select(rows)

    assert len(sel.ids) == 2
    assert "hour=0" in sel.text
    assert "dow=0" in sel.text


def test_non_numeric_phase_share_skips_the_phase_suffix():
    """A row whose phase shares are non-numeric renders without the phase suffix (WR-02 class).

    ``_is_renderable`` checks only rise/fall/dur/energy, and ``example_rows`` passes
    ``features.get("l1_share")`` and friends through raw, so a hand-edited or
    partially written share can be a non-numeric string. Calling ``float()`` on it
    raised ``ValueError`` mid-selection and failed the whole learning run; the phase
    suffix must be skipped instead, leaving the numeric line intact.
    """
    rows = [_row("cap-bad", phase=("abc", 0.1, 0.2))]

    sel = _select(rows)  # must not raise

    assert len(sel.ids) == 1
    assert "phase=" not in sel.text
    # The rest of the line still renders.
    assert "e1 t=t1 q=clean" in sel.text
    assert "rise=2010" in sel.text

