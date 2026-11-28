"""HA-free D-11 similarity and near-twin for a snapped capture (CAP-12, D-11).

``similar`` scores every visible candidate and confirmed device against a
capture's snapped pair and returns the result the record stores under
``result.similar``: a top-k display list, a ``near_twin`` flag and the ``twin``
it names. The near-twin rule is the D-11 rule the owner confirmed: an artefact
is a near twin when the capture is claimed by the artefact's own state boxes
(the learner's ``score.replay`` predicate) or by the capture's own D-02
tolerance box (the artefact's p50 rise and p50 duration inside it), or when the
digest-space distance ``d`` from the capture to the artefact's nearest state is
at most ``EPS``. ``near_twin`` and ``twin`` are read from the FULL scored pool,
never from the displayed top list, so a twin ranked below the top list is still
named.

Box IoU is deliberately not used: measured against the real buffer, a capture's
IoU with its own learned candidate has a median of 0.45 and never reaches
``identity.BOX_IOU_MIN``, so it would report no twins at all (RESEARCH
Pitfall 1). Only ``switched`` artefacts take part. numpy only, no Home Assistant
import.
"""

from collections.abc import Mapping
from math import log, sqrt
from typing import Any

import numpy as np

from ..learning import digest as digest_mod
from ..learning import score
from ..learning.artefact import SignatureArtefact
from . import preview as preview_mod
from .config import DEFAULT_CONFIG, CaptureConfig

_PARSE_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)


def _capture_table(capture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """One-row episode table for the capture, plus the columns replay needs."""
    table = preview_mod._build_table([capture])
    table["kind"] = np.zeros(1, dtype=np.int32)
    table["block"] = np.zeros(1, dtype=np.int32)
    table["start_i"] = np.zeros(1, dtype=np.int64)
    table["end_i"] = np.ones(1, dtype=np.int64)
    table["energy_wh"] = np.asarray(
        [preview_mod._features_value(capture, "energy_wh", "energy_wh")],
        dtype=np.float64,
    )
    return table


def _capture_box(
    capture: Mapping[str, Any], config: CaptureConfig
) -> dict[str, Any] | None:
    """The capture's own D-02 tolerance box (via fit_preview), or None.

    The capture is marked clean so ``fit_preview`` always builds the one-run
    tolerance box, whichever verdict the capture actually carries.
    """
    marked = {
        **dict(capture),
        "result": {**dict(capture.get("result") or {}), "verdict": "clean"},
    }
    artefact = preview_mod.fit_preview(
        "cap_similar", "similar", [marked], config=config
    )
    if not artefact or not artefact.get("states"):
        return None
    return artefact["states"][0]


def _state_in_capture_box(state: Mapping[str, Any], capture_box: Mapping[str, Any]) -> bool:
    """True when the state's p50 rise and duration lie inside the capture box."""
    rise = state.get("rise_w") or {}
    dur = state.get("dur_s") or {}
    r = rise.get("p50")
    d = dur.get("p50")
    c_rise = capture_box.get("rise_w") or {}
    c_dur = capture_box.get("dur_s") or {}
    if r is None or d is None:
        return False
    if c_rise.get("lo") is None or c_dur.get("lo") is None:
        return False
    return bool(c_rise["lo"] <= r <= c_rise["hi"] and c_dur["lo"] <= d <= c_dur["hi"])


def _nearest(
    capture: Mapping[str, Any], artefact: SignatureArtefact
) -> tuple[float | None, Any]:
    """Digest-space distance to the nearest artefact state and that state, or (None, None)."""
    result = capture.get("result") or {}
    step_w = result.get("step_w")
    dur_s = result.get("dur_s")
    if step_w is None or dur_s is None:
        return None, None
    try:
        step_w = float(step_w)
        dur_s = float(dur_s)
    except (TypeError, ValueError):
        return None, None
    if step_w <= 0 or dur_s <= 0:
        return None, None
    best_d: float | None = None
    best_state: Any = None
    for state in artefact.states:
        r = state.rise_w.p50
        d = state.dur_s.p50
        if r is None or d is None or r <= 0 or d <= 0:
            continue
        dist = sqrt(
            ((log(step_w) - log(r)) / digest_mod.RISE_SCALE) ** 2
            + ((log(dur_s) - log(d)) / digest_mod.DUR_SCALE) ** 2
        )
        if best_d is None or dist < best_d:
            best_d = dist
            best_state = state
    return best_d, best_state


def _score_entry(
    entry: Mapping[str, Any],
    artefact: SignatureArtefact,
    capture: Mapping[str, Any],
    table: Mapping[str, np.ndarray],
    capture_box: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Score one pool entry into the internal full-pool record."""
    claims = bool(
        score.replay([artefact], table, block=0).claims.get(artefact.signature_id)
    )
    if not claims and capture_box is not None:
        for state in artefact.states:
            if _state_in_capture_box(state.to_dict(), capture_box):
                claims = True
                break
    d, best_state = _nearest(capture, artefact)
    score_value = round(1.0 / (1.0 + d), 3) if d is not None else 0.0
    step_p50 = best_state.rise_w.p50 if best_state is not None else None
    dur_p50 = best_state.dur_s.p50 if best_state is not None else None
    return {
        "id": str(entry.get("id", "")),
        "kind": str(entry.get("kind", "")),
        "name": entry.get("name"),
        "score": score_value,
        "step_w": step_p50,
        "dur_s": dur_p50,
        "claims": claims,
        "d": d,
    }


def similar(
    capture: Mapping[str, Any],
    pool: Any,
    *,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return ``{top, near_twin, twin}`` for a capture over the whole pool.

    ``capture`` is a capture record (its ``result`` carries the snapped pair);
    ``pool`` is a list of ``{id, kind, name, artefact}`` plain-data entries. A
    non-switched or unparseable artefact is skipped. Every remaining entry is
    scored (the full scored pool); ``near_twin`` and ``twin`` are decided over
    that full pool, while ``top`` (score desc, then name, then id) keeps at most
    ``config.similar_top_k`` entries as the display list only.
    """
    table = _capture_table(capture)
    capture_box = _capture_box(capture, config)
    scored: list[dict[str, Any]] = []
    for entry in pool or []:
        if not isinstance(entry, Mapping):
            continue
        artefact_dict = entry.get("artefact")
        if not isinstance(artefact_dict, Mapping) or artefact_dict.get("kind") != "switched":
            continue
        try:
            artefact = SignatureArtefact.from_dict(artefact_dict)
        except _PARSE_ERRORS:
            continue
        scored.append(_score_entry(entry, artefact, capture, table, capture_box))

    order = sorted(scored, key=lambda e: (-e["score"], e["name"] or "", e["id"]))
    near_twin = False
    twin: dict[str, Any] | None = None
    for entry in order:
        d = entry["d"]
        if entry["claims"] or (d is not None and d <= digest_mod.EPS):
            near_twin = True
            twin = {"id": entry["id"], "kind": entry["kind"], "name": entry["name"]}
            break

    top = [
        {
            "id": e["id"],
            "kind": e["kind"],
            "name": e["name"],
            "score": e["score"],
            "step_w": e["step_w"],
            "dur_s": e["dur_s"],
            "claims": e["claims"],
        }
        for e in order[: config.similar_top_k]
    ]
    return {"top": top, "near_twin": near_twin, "twin": twin}
