"""Shared review backend for the PowerGuesser options flow (D-09, D-14).

Plan 07-01 lands the read-only view builder here: the pure formatter that
turns a candidate's stored ``evidence`` dict into the markdown block the
options-flow Review shows. Plan 07-05 adds the confirm / rename / dismiss /
merge actions (the behaviour-preserving extraction of D-14) to this same
module, which is the single backend the later websocket panel will share.

The view builder is pure: no I/O, no recompute, no job start. Review is
read-only (D-12), and the block carries only numbers, dates and fixed words,
so it is NOT passed through ``_escape_markdown`` (that would mangle the ``-``
bullets, the ``**`` bold markers, the dates and ``+2150 W``).

The actions mutate the review state through the supervisor and the device
manager exactly as the options-flow steps did before the move: every action
keeps the 0.2.0 lookup-then-act order and raises the same errors (typed as
``InvalidName`` / ``InvalidSource`` / the re-exported ``CandidateGone``) the
flow maps back to ``name_invalid`` / ``candidate_gone``.
"""

from collections.abc import Mapping, Sequence
import copy
import logging
from typing import Any

from homeassistant.util import dt as dt_util

from .const import EVIDENCE_MAX_FAILURES, SUBENTRY_TYPE_DEVICE
from .engine.capture.constants import CONFIRM_MIN_RUNS
from .engine.learning.evidence import (
    EVIDENCE_VERSION,
    MIN_RHYTHM_RUNS,
    TOD_MIN_CONCENTRATION,
)
from .learning import CandidateGone
from .review_store import normalise_name

_LOGGER = logging.getLogger(__name__)

# How many occurrences the Review block lists before the "…and N more" line.
SHOWN_OCCURRENCES = 5

# D-12, verbatim: shown while the evidence stage is queued/running or a
# candidate has no evidence yet, and after the failure cap is hit.
EVIDENCE_PENDING_TEXT = (
    "Evidence is being computed in the background. Reopen Review in a few minutes."
)
EVIDENCE_FAILED_TEXT = "Evidence unavailable, will retry at the next learning run"


class InvalidName(ValueError):
    """A display name rejected by ``review_store.normalise_name``."""


class InvalidSource(Exception):
    """A merge source value that resolves to no known candidate or device."""


def format_energy_wh(wh: float | None) -> str:
    """Format energy as whole Wh, or kWh with two decimals from 1000 Wh up."""
    if wh is None:
        return "n/a"
    rounded = round(float(wh))
    if rounded >= 1000:
        return f"{rounded / 1000:.2f} kWh"
    return f"{rounded} Wh"


def format_duration(dur_s: float | None) -> str:
    """Format a duration in whole minutes; "<1 min" below a minute."""
    if dur_s is None:
        return "n/a"
    if dur_s < 60:
        return "<1 min"
    minutes = round(float(dur_s) / 60.0)
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    return f"{hours} h {mins} min"


def _local_date(ms: int | None) -> str:
    """A UTC-millisecond timestamp as a local YYYY-MM-DD date, or n/a."""
    if ms is None:
        return "n/a"
    return dt_util.as_local(dt_util.utc_from_timestamp(ms / 1000)).strftime("%Y-%m-%d")


def _format_occurrence(occurrence: Mapping[str, Any]) -> str:
    """One bullet: local weekday/date/time, duration, step W and energy.

    EVID-05: a user-captured occurrence ends with a fixed-word `` (captured)``
    suffix; an appended captured run the detector did not claim ends with
    `` (captured, not matched by the detector)``. Only fixed words are added, so
    the block stays outside ``_escape_markdown``.
    """
    start_ms = occurrence.get("start_ms")
    when = (
        dt_util.as_local(dt_util.utc_from_timestamp(start_ms / 1000)).strftime(
            "%a %Y-%m-%d %H:%M"
        )
        if start_ms is not None
        else "n/a"
    )
    step_w = occurrence.get("step_w")
    # A missing step reads a bare "n/a" (like duration/energy/profile), never
    # the odd "step +n/a W" (MR !17 note 4918).
    step = "n/a" if step_w is None else f"+{round(float(step_w))} W"
    suffix = ""
    if occurrence.get("captured"):
        suffix = (
            " (captured, not matched by the detector)"
            if occurrence.get("detected") is False
            else " (captured)"
        )
    return (
        f"- {when}, {format_duration(occurrence.get('dur_s'))}, "
        f"step {step}, {format_energy_wh(occurrence.get('energy_wh'))}{suffix}"
    )


_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def evidence_state(evidence: Any, failures: int) -> str:
    """The D-12 review state for an evidence value: ready / pending / failed.

    Takes the evidence value rather than the candidate, so the device path
    reuses the same rule. ``ready`` needs a current-version dict; otherwise the
    failure count against ``EVIDENCE_MAX_FAILURES`` decides pending vs failed.
    """
    if isinstance(evidence, Mapping) and evidence.get("v") == EVIDENCE_VERSION:
        return "ready"
    if int(failures or 0) >= EVIDENCE_MAX_FAILURES:
        return "failed"
    return "pending"


def format_gap(seconds: float | None) -> str:
    """A typical gap as minutes, hours or days+hours; ``n/a`` for an unknown."""
    if seconds is None:
        return "n/a"
    minutes = round(float(seconds) / 60.0)
    if minutes < 60:
        return f"{minutes} min"
    hours = round(float(seconds) / 3600.0)
    if hours < 48:
        return f"{hours} h"
    days, rem_hours = divmod(hours, 24)
    return f"{days} d {rem_hours} h"


def _clock_time(minute_of_day: Any) -> str:
    """A minute-of-day as HH:MM."""
    minute = int(minute_of_day or 0) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _weekday_phrase(dow_counts: Sequence[Any]) -> str:
    """``most often`` plus up to three dominant weekday abbreviations, or a flat week."""
    counts = [int(count or 0) for count in dow_counts]
    if not counts:
        return "spread over the week"
    threshold = 1.5 * (sum(counts) / len(counts))
    picked = [i for i, count in enumerate(counts) if count > 0 and count >= threshold]
    if not picked:
        return "spread over the week"
    # Count order, ties in weekday order; at most three names.
    picked.sort(key=lambda i: (-counts[i], i))
    picked = picked[:3]
    return "most often " + ", ".join(_WEEKDAY_ABBR[i] for i in picked)


def _rhythm_line(
    rhythm: Any, n_runs: int, observed_days: int
) -> str:
    """The EVID-02 rhythm line, or the D-02 short form below MIN_RHYTHM_RUNS."""
    if not isinstance(rhythm, Mapping):
        return f"Rhythm: too few runs yet ({n_runs} of {MIN_RHYTHM_RUNS} needed)"
    runs_per_day = float(rhythm.get("runs_per_day") or 0.0)
    runs_per_week = float(rhythm.get("runs_per_week") or 0.0)
    tod = rhythm.get("tod") or {}
    concentration = float(tod.get("concentration") or 0.0)
    if concentration >= TOD_MIN_CONCENTRATION:
        time_of_day = (
            f"mostly {_clock_time(tod.get('lo_min'))}-{_clock_time(tod.get('hi_min'))}"
        )
    else:
        time_of_day = "no fixed time of day"
    weekdays = _weekday_phrase(rhythm.get("dow_counts") or [])
    gap = format_gap(rhythm.get("gap_median_s"))
    return (
        f"Rhythm: about {runs_per_day:.1f} runs per day ({runs_per_week:.1f} per week), "
        f"{time_of_day}, {weekdays}, typical gap {gap} "
        f"(from {n_runs} runs over {observed_days} observed days)"
    )


def _profile_line(profile: Any) -> str:
    """The EVID-03 profile line; every missing value prints ``n/a``, never 0 (D-04)."""
    data: Mapping[str, Any] = profile if isinstance(profile, Mapping) else {}
    step = data.get("step_w")
    duration = data.get("dur_s")
    step_p50 = step.get("p50") if isinstance(step, Mapping) else None
    dur_p50 = duration.get("p50") if isinstance(duration, Mapping) else None
    wh_per_run = data.get("wh_per_run")
    wh_per_day = data.get("wh_per_day")
    phase = data.get("phase")
    step_text = f"~{round(float(step_p50))} W" if step_p50 is not None else "n/a"
    dur_text = f"~{format_duration(dur_p50)}" if dur_p50 is not None else "n/a"
    run_text = f"~{format_energy_wh(wh_per_run)}" if wh_per_run is not None else "n/a"
    day_text = f"~{format_energy_wh(wh_per_day)}" if wh_per_day is not None else "n/a"
    phase_text = f"L{str(phase)[-1].upper()}" if phase else "n/a"
    return (
        f"Profile: step {step_text}, {dur_text}, {run_text} per run, "
        f"{day_text} per day, phase {phase_text}"
    )


def _hints_line(hints: Any, n_runs: int, observed_days: int) -> str:
    """The D-05 heuristic label and the phrases for the engine's hint codes."""
    phrases: list[str] = []
    for code in hints or []:
        if code == "few_runs":
            phrases.append(f"few runs ({n_runs})")
        elif code == "wide_step_spread":
            phrases.append("step size varies widely")
        elif code == "short_coverage":
            phrases.append(f"short coverage ({observed_days} observed days)")
        else:
            phrases.append(str(code))
    if not phrases:
        return "Hints (heuristic): none triggered"
    return "Hints (heuristic): " + "; ".join(phrases)


def evidence_text(evidence: Any, state: str) -> str:
    """Render the evidence block for the Review menu description.

    ``state`` is ``"ready"`` (a stored evidence dict), ``"pending"`` (missing,
    queued or running), ``"failed"`` (the failure cap was hit) or ``"none"`` (a
    device with no active artefact). Each section of the ready block (header,
    summary, recent runs, Rhythm, Profile, Hints) is its own markdown paragraph,
    separated from the next by exactly one blank line, because CommonMark joins
    two adjacent non-blank lines into one paragraph: the layout must not depend
    on how the frontend renders a single newline. The only adjacent non-blank
    lines allowed are the ``- `` recent-run bullets.
    """
    if state == "pending":
        return EVIDENCE_PENDING_TEXT
    if state == "failed":
        return EVIDENCE_FAILED_TEXT
    if state == "none":
        # A device with no active artefact (nothing to replay, D-07a): every
        # value n/a, never 0, never the pending text.
        return "\n".join(
            [
                "**Evidence** (heuristics, not accuracy)",
                "",
                "Runs: n/a",
                "",
                "Recent runs: n/a",
                "",
                "Rhythm: n/a",
                "",
                "Profile: step n/a, n/a, n/a per run, n/a per day, phase n/a",
                "",
                "Hints (heuristic): n/a",
            ]
        )
    if not isinstance(evidence, Mapping):
        return EVIDENCE_PENDING_TEXT

    n_runs = int(evidence.get("n_runs") or 0)
    window = evidence.get("window") or {}
    observed = int(window.get("observed_days") or 0)
    n_days = int(window.get("n_days") or 0)
    tz_name = str(dt_util.get_default_time_zone())

    lines = ["**Evidence** (heuristics, not accuracy)", ""]
    n_captured = int(evidence.get("n_captured") or 0)
    captured_part = f", {n_captured} captured" if n_captured > 0 else ""
    if n_runs > 0:
        lines.append(
            f"{n_runs} runs{captured_part}, first {_local_date(evidence.get('first_ms'))}, "
            f"last {_local_date(evidence.get('last_ms'))}, "
            f"{observed} of {n_days} days observed, times in {tz_name}"
        )
    else:
        lines.append(
            f"0 runs{captured_part} in {observed} of {n_days} days observed, "
            f"times in {tz_name}"
        )
    lines.append("")

    occurrences = list(evidence.get("occurrences") or [])
    # The stored occurrences list is capped (MAX_OCCURRENCES), so the tail count
    # is the true total: detected runs plus captures that matched no detected run
    # (a captured occurrence whose rise matched a run is already counted in
    # n_runs). Key the empty branch on the overlaid data instead of n_runs: a
    # capture-only group (zero detected runs, read-time overlay appending the
    # captures) must list them, not render "Recent runs: none" (EVID-05).
    n_total = n_runs + int(evidence.get("n_captured_unmatched") or 0)
    if not occurrences:
        lines.append("Recent runs: none")
    else:
        lines.append("Recent runs, newest first:")
        lines.append("")
        for occurrence in occurrences[:SHOWN_OCCURRENCES]:
            lines.append(_format_occurrence(occurrence))
        # EVID-05: the tail count covers every occurrence (detected runs and
        # appended captured-only runs together) beyond the shown five.
        if n_total > SHOWN_OCCURRENCES:
            lines.append(f"- …and {n_total - SHOWN_OCCURRENCES} more")

    # One blank line between each remaining section, so Rhythm, Profile and
    # Hints each render on their own line whatever the frontend does with a
    # single newline (Plan 07-01 layout rule).
    lines.append("")
    lines.append(_rhythm_line(evidence.get("rhythm"), n_runs, observed))
    lines.append("")
    lines.append(_profile_line(evidence.get("profile")))
    lines.append("")
    lines.append(_hints_line(evidence.get("hints"), n_runs, observed))

    return "\n".join(lines)


def _resolve_unconfirmed_uuid(hub: Any, cand: Mapping[str, Any]) -> str | None:
    """The device uuid an unconfirmed-device review row was built from (D-07a).

    ``devices.unconfirmed_candidates()`` builds each row from its record's
    active artefact, so the row's ``states`` name the record it came from:
    prefer the evidence source whose artefact states equal the row's states,
    then the lowest uuid (``evidence_sources()`` is sorted by uuid). This is
    deterministic; an arbitrary first match would show the wrong device's
    evidence when two records share a signature.
    """
    sig_id = str(cand.get("signature_id") or "")
    row_states = cand.get("states")
    matches = [
        source
        for source in hub.devices.evidence_sources()
        if str(source.get("status")) == "unconfirmed"
        and str(source.get("signature_id") or "") == sig_id
    ]
    if not matches:
        return None
    for source in matches:
        artefact = source.get("artefact") or {}
        if artefact.get("states") == row_states:
            return str(source.get("uuid"))
    return str(matches[0].get("uuid"))


def overlay_captured(
    evidence: Mapping[str, Any], captured_runs: Sequence[Mapping[str, Any]], step_ms: int
) -> dict[str, Any]:
    """Overlay user-captured runs on a stored evidence dict (EVID-05, A9).

    Read-time and pure: a deep copy is returned, so the stored evidence is never
    mutated. A detected occurrence whose ``start_ms`` is within ``step_ms`` of a
    captured ``rise_ms`` (inclusive) gains ``captured: True``; several captures
    of one occurrence mark it once. A captured run matched by no occurrence is
    appended as ``{start_ms, dur_s, step_w, energy_wh, captured: True,
    detected: False}``; unmatched captures whose rises lie within ``step_ms`` of
    an already-appended one are dropped, processing them in the fixed order
    ``(verdict != "clean", rise_ms, capture_id)`` so the clean capture wins
    whatever the Store record order. The listed occurrences are re-sorted
    newest first and ``n_captured``/``n_captured_unmatched`` are set. With an
    empty ``captured_runs`` the evidence is returned unchanged (no new keys).
    """
    if not captured_runs:
        return evidence
    out = copy.deepcopy(dict(evidence))
    occurrences = [dict(occ) for occ in (out.get("occurrences") or [])]
    step = int(step_ms)

    unmatched: list[Mapping[str, Any]] = []
    for run in captured_runs:
        rise = run.get("rise_ms")
        matched = False
        if rise is not None:
            rise_int = int(rise)
            for occurrence in occurrences:
                start = occurrence.get("start_ms")
                if start is not None and abs(int(start) - rise_int) <= step:
                    occurrence["captured"] = True
                    matched = True
                    break
        if not matched:
            unmatched.append(run)

    unmatched.sort(
        key=lambda run: (
            run.get("verdict") != "clean",
            int(run.get("rise_ms") or 0),
            str(run.get("capture_id") or ""),
        )
    )
    appended: list[dict[str, Any]] = []
    for run in unmatched:
        rise_int = int(run.get("rise_ms") or 0)
        if any(abs(rise_int - int(a["start_ms"])) <= step for a in appended):
            continue
        appended.append(
            {
                "start_ms": rise_int,
                "dur_s": run.get("dur_s"),
                "step_w": run.get("step_w"),
                "energy_wh": run.get("energy_wh"),
                "captured": True,
                "detected": False,
            }
        )

    occurrences.extend(appended)
    occurrences.sort(key=lambda occ: occ.get("start_ms") or 0, reverse=True)
    out["occurrences"] = occurrences
    out["n_captured"] = sum(1 for occ in occurrences if occ.get("captured"))
    out["n_captured_unmatched"] = len(appended)
    return out


def _capture_step_ms(captured_runs: Sequence[Mapping[str, Any]]) -> int:
    """The overlay step for a run list: the first run's grid as milliseconds."""
    for run in captured_runs:
        grid_s = run.get("grid_s")
        if grid_s:
            return int(round(float(grid_s) * 1000))
    return 30_000


def evidence_view(
    hub: Any, *, cand: Any = None, dev_uuid: str | None = None
) -> tuple[Any, str]:
    """The evidence value and D-12 state for a Review row (D-07, D-07a, D-12).

    A ``dev_uuid`` without an active artefact (nothing to replay) is the
    ``none`` state; otherwise the supervisor's stored device evidence (or its
    read-time fallback) decides ready/pending/failed. A ``cand`` that is one of
    the supervisor's own candidates uses its stored evidence; a capture-origin
    preview uses its own ``capture`` evidence; an unconfirmed-device row is
    resolved to its record and read as a device, so it never shows the pending
    text forever. A ready learned candidate's or device's evidence is overlaid
    with the captures that resolve to it (EVID-05). Read-only: it starts no job
    and writes no Store (D-12).
    """
    learning = hub.learning
    data = getattr(getattr(learning, "coordinator", None), "data", None) or {}
    failures = int(data.get("evidence_failures", 0) or 0)
    captures = getattr(hub, "captures", None)
    if dev_uuid is not None:
        # Comment: the artefact decides "none"; the evidence value never does.
        artefact, _active_v = hub.devices.active_artefact_and_version(dev_uuid)
        if artefact is None:
            return None, "none"
        device_evidence = learning.device_evidence(dev_uuid)
        state = evidence_state(device_evidence, failures)
        if state == "ready" and captures is not None and isinstance(device_evidence, Mapping):
            captured = captures.captured_runs("device", str(dev_uuid))
            if captured:
                device_evidence = overlay_captured(
                    device_evidence, captured, _capture_step_ms(captured)
                )
        return device_evidence, state
    if cand is not None:
        sig_id = str(cand.get("signature_id") or "")
        if str(cand.get("origin") or "") == "capture":
            # A capture preview's evidence is built from its own captured runs.
            preview_evidence = cand.get("evidence")
            return preview_evidence, evidence_state(preview_evidence, failures)
        own_sigs = {str(c.get("signature_id") or "") for c in learning.candidates}
        if sig_id in own_sigs:
            candidate_evidence = cand.get("evidence")
            state = evidence_state(candidate_evidence, failures)
            if state == "ready" and captures is not None and isinstance(candidate_evidence, Mapping):
                captured = captures.captured_runs("candidate", sig_id)
                if captured:
                    candidate_evidence = overlay_captured(
                        candidate_evidence, captured, _capture_step_ms(captured)
                    )
            return candidate_evidence, state
        # The row came from devices.unconfirmed_candidates(): it is a device view.
        resolved = _resolve_unconfirmed_uuid(hub, cand)
        if resolved is None:
            return None, "none"
        return evidence_view(hub, dev_uuid=resolved)
    return None, "none"


# ---------------------------------------------------------------------------
# Review actions (D-14): the behaviour-preserving extraction of the confirm,
# rename, dismiss and merge logic for candidates and confirmed devices.
# ---------------------------------------------------------------------------


def find_candidate(hub: Any, cand_id: str | None) -> dict[str, Any] | None:
    """The visible candidate with that signature id, or None (D-14)."""
    return next(
        (
            c
            for c in hub.learning.visible_candidates()
            if c.get("signature_id") == cand_id
        ),
        None,
    )


def candidate_is_preview(hub: Any, cand: Mapping[str, Any]) -> bool:
    """True when the candidate is a preview: today's confirm rule (D-14).

    ``scores.validated`` wins; when it is absent the stored candidates
    record's ``validated`` flag decides, exactly as ``async_step_confirm`` did.
    """
    scores = cand.get("scores") or {}
    validated = scores.get("validated")
    if validated is None:
        validated = hub.learning.candidates_validated
    return not bool(validated)


def captured_runs_phrase(n: int) -> str:
    """Render a captured-run count as ``1 captured run`` / ``N captured runs`` (D-03).

    The single place this wording is built: the confirm form warning, the
    override WARNING log line and the ``PreviewNeedsRuns`` message all call it,
    so the gate cannot show two different counts for the same preview.
    """
    count = int(n)
    if count == 1:
        return "1 captured run"
    return f"{count} captured runs"


class PreviewNeedsRuns(Exception):
    """A capture preview backed by fewer than ``CONFIRM_MIN_RUNS`` clean runs (D-03).

    Raised by ``async_confirm_candidate`` unless the caller passes
    ``override=True``. The options flow maps it to the ``preview_needs_runs``
    form error and offers the ``confirm_anyway`` override checkbox.
    """

    def __init__(self, n: int, required: int) -> None:
        self.n = int(n)
        self.required = int(required)
        super().__init__(
            f"This preview is based on {captured_runs_phrase(self.n)}, "
            f"fewer than the {self.required} needed to confirm without an "
            "explicit override"
        )


def capture_gate(cand: Mapping[str, Any]) -> tuple[int, int] | None:
    """The D-03 confirm gate for a candidate: ``(n, required)`` or None.

    A capture-origin preview backed by fewer clean captured runs than
    ``CONFIRM_MIN_RUNS`` is gated; every other candidate (a mature preview with
    enough runs, any learned candidate) returns None. This is the one gate rule:
    ``async_confirm_candidate`` refuses on it and ``config_flow.async_step_confirm``
    shows the ``confirm_anyway`` override exactly when it is not None (D-03). The
    gate lives only on confirm: preview creation, identity priors and
    reconciliation never consult it.
    """
    if str(cand.get("origin") or "") != "capture":
        return None
    scores = cand.get("scores") or {}
    n = int(scores.get("n_examples") or 0)
    if n < CONFIRM_MIN_RUNS:
        return (n, CONFIRM_MIN_RUNS)
    return None


async def async_confirm_candidate(
    hub: Any, cand_id: str | None, raw_name: str, *, override: bool = False
) -> None:
    """Confirm a candidate as a device (D-14, D-03 override).

    Raises ``CandidateGone`` when the candidate is no longer visible,
    ``InvalidName`` when the name is rejected and ``PreviewNeedsRuns`` when a
    capture preview is backed by fewer than ``CONFIRM_MIN_RUNS`` clean runs and
    ``override`` is not set.

    On an override the refusal is logged at WARNING and recorded on the review
    decision through ``ReviewStore.async_record_confirm_override`` *before* the
    device is created, so the record means "override attempted" (a confirm that
    then fails keeps the audit entry). A capture preview passes its group's
    ``created_at`` as ``trained_at``; every other candidate keeps the last
    learning run's ``candidates_created_at``.
    """
    cand = find_candidate(hub, cand_id)
    if cand is None:
        raise CandidateGone(f"Candidate {cand_id} is no longer visible")
    try:
        norm_name = normalise_name(raw_name)
    except ValueError as err:
        raise InvalidName(str(err)) from err

    gate = capture_gate(cand)
    trained_at = hub.learning.candidates_created_at
    if gate is not None:
        n, required = gate
        if not override:
            raise PreviewNeedsRuns(n, required)
        _LOGGER.warning(
            "Confirming capture preview %s with %s (%d of %d required) via override",
            cand_id,
            captured_runs_phrase(n),
            n,
            required,
        )
        await hub.learning.review.async_record_confirm_override(
            cand_id, n_runs=n, required=required
        )

    if str(cand.get("origin") or "") == "capture":
        created_at = (cand.get("provenance") or {}).get("created_at")
        if created_at:
            trained_at = str(created_at)

    await hub.devices.async_confirm(
        cand,
        norm_name,
        preview=candidate_is_preview(hub, cand),
        trained_at=trained_at,
    )


async def async_rename_candidate(
    hub: Any, cand_id: str | None, raw_name: str
) -> None:
    """Rename a visible candidate (D-14); CandidateGone propagates from the decision."""
    if find_candidate(hub, cand_id) is None:
        raise CandidateGone(f"Candidate {cand_id} is no longer visible")
    try:
        norm_name = normalise_name(raw_name)
    except ValueError as err:
        raise InvalidName(str(err)) from err
    await hub.learning.async_record_decision("rename", cand_id, name=norm_name)


async def async_rename_device(
    hass: Any, entry: Any, hub: Any, dev_uuid: str, raw_name: str
) -> None:
    """Rename a confirmed device through its config subentry (D-14, OUT-05)."""
    try:
        norm_name = normalise_name(raw_name)
    except ValueError as err:
        raise InvalidName(str(err)) from err
    rec = hub.devices.record(dev_uuid)
    if rec is None:
        raise CandidateGone(f"Device {dev_uuid} is no longer available")
    sub = next(
        (
            s
            for s in entry.get_subentries_of_type(SUBENTRY_TYPE_DEVICE)
            if s.subentry_id == rec.get("subentry_id")
        ),
        None,
    )
    if sub is None:
        raise CandidateGone(f"Device {dev_uuid} has no config subentry")
    hass.config_entries.async_update_subentry(
        entry,
        sub,
        title=norm_name,
        data={**sub.data, "name": norm_name},
    )


async def async_dismiss_candidate(hub: Any, cand_id: str) -> None:
    """Dismiss a candidate (D-14); CandidateGone propagates from the decision."""
    await hub.learning.async_record_decision("dismiss", cand_id)


async def async_merge_candidate(hub: Any, target_id: str, sources: Any) -> None:
    """Merge candidates into a candidate target (D-14).

    ``sources`` is normalised exactly as the flow did (a bare string becomes a
    one-element list). CandidateGone and ValueError propagate unchanged: the
    candidate path keeps its 0.2.0 behaviour, where a ValueError from the
    decision is not caught (a latent crash, kept on purpose).
    """
    if isinstance(sources, str):
        sources = [sources]
    elif not isinstance(sources, list):
        sources = list(sources or [])
    await hub.learning.async_record_decision("merge", target_id, sources=sources)


async def async_merge_device(hub: Any, dev_uuid: str, option_values: Any) -> None:
    """Merge devices or candidates into a confirmed device (D-14, D-12/D-13/D-23).

    Each pick is re-resolved server-side: ``device_<uuid>`` maps to that
    record's signature id, any other value to a visible switched candidate.
    An unresolvable value raises ``InvalidSource`` (the flow's
    ``candidate_gone``).
    """
    rec = hub.devices.record(dev_uuid)
    target_sig = str((rec or {}).get("signature_id") or "")
    raw = option_values
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, list):
        raw = list(raw or [])
    source_sigs: list[str] = []
    for value in raw:
        value = str(value)
        if value.startswith("device_"):
            src_uuid = value[len("device_") :]
            src_rec = hub.devices.record(src_uuid) if src_uuid else None
            if src_rec is None or not src_rec.get("signature_id"):
                raise InvalidSource(f"Unknown merge source device {value}")
            source_sigs.append(str(src_rec["signature_id"]))
        else:
            # An unconfirmed source: a visible switched candidate only (D-13);
            # signature ids never start with "device_".
            cand = next(
                (
                    c
                    for c in hub.learning.visible_candidates()
                    if c.get("signature_id") == value
                ),
                None,
            )
            if cand is None or cand.get("kind") != "switched":
                raise InvalidSource(f"Unknown merge source {value}")
            source_sigs.append(value)
    await hub.learning.async_record_decision("merge", target_sig, sources=source_sigs)
