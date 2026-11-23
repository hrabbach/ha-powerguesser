"""Device class derivation and the class pass bar (Plan 05-02; D-06, D-08, D-21).

The class is derived; the user sets nothing (D-06): kind + band + level count.
A switched device takes the band of the largest state rise_w.p50 (A1); a
program device's states are degenerate rise boxes around 0 W (RESEARCH
Pitfall 1), so it takes the band of the largest stage level_w.p50 of its
session template instead, and has no class at all without a usable template
(P5 D-08, D-26 groundwork). A 15 % hysteresis at the band edges keeps the class
stored at the previous validation while the measured p50 stays near the shared
edge, so a device does not flip class on noise (D-06, T-05-24).

Pure stdlib; HA-free (tests/engine/test_no_ha_imports.py).
"""

from collections.abc import Mapping, Sequence
import math
import statistics

from .constants import (
    BAND_EDGES_W,
    BAND_HYSTERESIS,
    CLASS_MAX_MEDIAN_PCT,
    CLASS_MAX_PEER_PCT,
    CLASS_MIN_PASSING,
)

# D-06 band vocabulary (costly: user-visible through Plan 05-04's
# validation_class; the format is the flagged discretionary choice A/D-09).
BANDS = ("under_0.5kw", "0.5_1.5kw", "1.5_3kw", "over_3kw")


def _band_of(p50: float) -> str:
    """Band of a rise p50 in watts against the D-06 edges."""
    if p50 < BAND_EDGES_W[0]:
        return BANDS[0]
    if p50 < BAND_EDGES_W[1]:
        return BANDS[1]
    if p50 < BAND_EDGES_W[2]:
        return BANDS[2]
    return BANDS[3]


def _parse_class(previous: str) -> tuple[str, str, str] | None:
    """Split a stored class into (kind, band, levels); None when unparseable."""
    parts = previous.split("_")
    # f"{kind}_{band}_{levels}": kind and the level count are single tokens,
    # the band carries its own underscores (e.g. 1.5_3kw).
    if len(parts) < 4:
        return None
    kind = parts[0]
    levels = "_".join(parts[-2:])
    band = "_".join(parts[1:-2])
    if band not in BANDS or levels not in ("1_level", "multi_level"):
        return None
    return kind, band, levels


def _apply_hysteresis(p50: float, kind: str, levels: str, previous: str) -> str:
    """D-06: keep the previous band while p50 stays within the hysteresis window.

    Only a previous class of the same kind and level count naming a band
    adjacent to the new one is honoured, and only while p50 stays within
    BAND_HYSTERESIS x the shared edge on either side of it (T-05-24: a corrupt
    stored entry cannot pin an arbitrary band).
    """
    band = _band_of(p50)
    parsed = _parse_class(previous)
    if parsed is None:
        return band
    prev_kind, prev_band, prev_levels = parsed
    if prev_kind != kind or prev_levels != levels or prev_band == band:
        return band
    prev_idx = BANDS.index(prev_band)
    new_idx = BANDS.index(band)
    if abs(prev_idx - new_idx) != 1:
        return band
    edge = BAND_EDGES_W[min(prev_idx, new_idx)]
    if edge * (1.0 - BAND_HYSTERESIS) <= p50 <= edge * (1.0 + BAND_HYSTERESIS):
        return prev_band
    return band


def _stage_level_p50s(artefact: Mapping) -> list[float]:
    """Numeric level_w.p50 of every stage in the artefact's session template.

    A program artefact's own states are degenerate rise boxes around 0 W
    (RESEARCH Pitfall 1), so the class band comes from its stage template. A
    program without a session, without stages, or whose stages carry no numeric
    level p50 has no usable template, so ``derive_class`` returns None for it
    and it can never be counted under an unrelated passing class (P5 D-08).
    Non-mapping stages and non-finite level p50 values (NaN, Infinity) are
    skipped, the same semantics as the detector-side ``_check_finite``
    rejection, so a malformed stored template never reaches ``_band_of``
    (T-06-29).
    """
    session = artefact.get("session")
    raw_stages = session.get("stages") if isinstance(session, Mapping) else None
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        return []
    levels: list[float] = []
    for stage in raw_stages:
        if not isinstance(stage, Mapping):
            continue
        level = stage.get("level_w")
        if not isinstance(level, Mapping) or level.get("p50") is None:
            continue
        try:
            value = float(level["p50"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        levels.append(value)
    return levels


def derive_class(artefact: Mapping, previous: str | None = None) -> str | None:
    """Derive the device class from a linked shadow artefact (D-06, A1, D-26).

    Switched devices: p50 is the largest rise_w.p50 across the artefact's
    states; levels are "1_level" for one state else "multi_level".

    Program devices (kind "program"): the artefact's states are degenerate rise
    boxes around 0 W (RESEARCH Pitfall 1), so p50 is the largest level_w.p50
    over the artefact's session stages and levels are "1_level" for one stage
    else "multi_level"; a program without a usable stage template has no class
    (None), so it can never be counted under an unrelated passing class
    (P5 D-08).

    With previous set (the class stored for this plug at the last ok
    validation), band edges apply 15 % hysteresis so the class does not flip
    on noise.
    """
    kind = str(artefact.get("kind", "switched"))
    if kind == "program":
        p50s = _stage_level_p50s(artefact)
        if not p50s:
            return None
        p50 = max(p50s)
        levels = "1_level" if len(p50s) == 1 else "multi_level"
    else:
        states = [s for s in (artefact.get("states") or []) if isinstance(s, Mapping)]
        if not states:
            return None
        rise_p50s = [
            float(s["rise_w"]["p50"])
            for s in states
            if isinstance(s.get("rise_w"), Mapping) and s["rise_w"].get("p50") is not None
        ]
        if not rise_p50s:
            return None
        p50 = max(rise_p50s)
        levels = "1_level" if len(states) == 1 else "multi_level"
    band = _band_of(p50)
    if previous:
        band = _apply_hysteresis(p50, kind, levels, str(previous))
    return f"{kind}_{band}_{levels}"


def class_passes(peer_errors: Sequence[float | None], n_passing: int) -> bool:
    """The D-08 class bar over every peer with a figure (A2).

    peer_errors holds the energy error of every plug device of the class that
    earned a figure, including ones that failed the device bar; None entries
    (a zero reference energy refuses the figure) cannot satisfy the caps and
    are excluded from the median/max, but the peer still counts as not passing.
    """
    if n_passing < CLASS_MIN_PASSING:
        return False
    errors = [float(e) for e in peer_errors if e is not None]
    if not errors:
        return False
    return (
        statistics.median(errors) <= CLASS_MAX_MEDIAN_PCT
        and max(errors) <= CLASS_MAX_PEER_PCT
    )
