"""HA-side capture manager: captures Store, retroactive capture, list, previews.

Owns its own ``powerguesser.captures`` Store (never a sibling key of the review
Store: RESEARCH Pitfall 7) and turns an admin ``capture_window`` call into a
snapped capture record plus a read-time preview candidate for Review.
"""

from collections.abc import Callable, Mapping
import functools
import logging
from typing import Any
import unicodedata
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, issue_registry as ir
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    CANDIDATE_DECORATION_KEYS,
    DOMAIN,
    ISSUE_CAPTURE_ORPHANS,
)
from .engine.capture import snap, window
from .engine.capture.config import DEFAULT_CONFIG
from .engine.capture.constants import (
    CAPTURE_STATES,
    MAX_CAPTURE_RECORDS,
    OPEN_STATES,
    PREVIEW_ID_PREFIX,
)
from .engine.capture.preview import capture_evidence, fit_preview
from .engine.capture.replay_check import replay_check
from .engine.capture.window import CaptureRefused
from .engine.learning.artefact import SignatureArtefact
from .engine.learning.identity import similar_dismissed
from .review_store import normalise_name, resolve_session_target

_LOGGER = logging.getLogger(__name__)

# A live resolution that found a pair with one of these verdicts is `done`.
VERDICTS_DONE = ("clean", "contaminated", "ambiguous")
# The record cap may only ever prune these states (never a `done` teaching example).
_DROPPABLE_STATES = ("failed", "cancelled", "interrupted")
# How many candidate/device names a target error lists in its choices placeholder.
_TARGET_CHOICES = 10


def _match_key(value: Any) -> str:
    """NFC normalise, collapse whitespace and casefold a display name for matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


class CaptureManager:
    """Manages capture records and the preview candidates derived from them."""

    def __init__(self, hass: HomeAssistant, entry: Any, hub: Any) -> None:
        """Initialize the capture manager."""
        self.hass = hass
        self.entry = entry
        self.hub = hub
        self._store = Store[dict[str, Any]](
            hass, CAPTURES_STORE_VERSION, CAPTURES_STORE_KEY
        )
        self._records: list[dict[str, Any]] = []
        # Live-capture timers, owned by this manager and cancelled on unload
        # (RESEARCH Pitfall 7 / T-08-09).
        self._unsub_expiry: Callable[[], None] | None = None
        self._unsub_settle: Callable[[], None] | None = None
        self._unsub_coordinator: Callable[[], None] | None = None
        self._extended: set[str] = set()

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a copy of the stored capture records."""
        return [dict(rec) for rec in self._records]

    async def async_load(self) -> None:
        """Load the stored capture records and close any open capture (D-08).

        On HA restart or entry reload every ``recording`` or ``resolving`` record
        ends ``interrupted`` with reason ``restart``, a gap flag and a holdout
        window; no ``recording`` state survives and no timer is armed.
        """
        data = await self._store.async_load()
        if isinstance(data, dict) and isinstance(data.get("captures"), list):
            raw = data["captures"]
        else:
            raw = []
        self._records = self._valid_captures(raw)
        if self._async_sweep_open_captures():
            await self._store.async_save({"captures": self._records})
        # Orphan captures (a target that no longer resolves, D-06): surface the
        # Repairs issue at load and keep it fresh on every learning-coordinator
        # update (dismiss/merge/confirm all push one).
        self._refresh_orphan_issue()
        self._async_register_orphan_listener()

    def _async_sweep_open_captures(self) -> bool:
        """End every open record as interrupted/restart; return True if any changed."""
        changed = False
        now_iso = dt_util.utcnow().isoformat()
        for index, rec in enumerate(self._records):
            if rec.get("state") not in OPEN_STATES:
                continue
            press_start_ms = _iso_to_ms(rec.get("press_start_ts")) or 0
            press_stop_ms = _iso_to_ms(rec.get("press_stop_ts"))
            if press_stop_ms is None:
                press_stop_ms = press_start_ms + DEFAULT_CONFIG.live_max_open_s * 1000
            self._records[index] = {
                **rec,
                "state": "interrupted",
                "reason_code": "restart",
                "gap_flagged": True,
                "resolved_at": now_iso,
                "holdout_window_ms": [press_start_ms, press_stop_ms],
            }
            changed = True
        return changed

    @staticmethod
    def _valid_captures(raw: Any) -> list[dict[str, Any]]:
        """Keep only well-formed capture records (T-08-06).

        A malformed record (not a dict, unknown state, a missing target, an
        unparseable created_at) is dropped at load with a debug log, so a corrupt
        captures Store can never block setup.
        """
        if not isinstance(raw, list):
            return []
        records: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                _LOGGER.debug("Dropping a non-dict capture record")
                continue
            capture_id = item.get("capture_id")
            target_id = item.get("target_id")
            created_at = item.get("created_at")
            if (
                not isinstance(capture_id, str)
                or not capture_id
                or item.get("mode") not in ("live", "window")
                or item.get("state") not in CAPTURE_STATES
                or item.get("target_kind") not in ("candidate", "device")
                or not isinstance(target_id, str)
                or not target_id
                or not isinstance(created_at, str)
                or dt_util.parse_datetime(created_at) is None
            ):
                _LOGGER.debug("Dropping a malformed capture record")
                continue
            records.append(dict(item))
        return records

    def _device_lookup(self, device_uuid: str) -> Mapping[str, Any] | None:
        devices = getattr(self.hub, "devices", None)
        return devices.record(device_uuid) if devices is not None else None

    # --- target resolution (CAP-03) -------------------------------------

    @staticmethod
    def _join_choices(names: list[str]) -> str:
        return ", ".join(names[:_TARGET_CHOICES])

    def _target_unknown(
        self, target: str, choices: list[str]
    ) -> ServiceValidationError:
        return ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="capture_target_unknown",
            translation_placeholders={
                "target": target,
                "choices": self._join_choices(choices),
            },
        )

    def _target_ambiguous(
        self, target: str, choices: list[str]
    ) -> ServiceValidationError:
        return ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="capture_target_ambiguous",
            translation_placeholders={
                "target": target,
                "choices": self._join_choices(choices),
            },
        )

    def _resolve_target(
        self, target_type: str, target: str, device_id: str | None
    ) -> tuple[str, str, str | None]:
        """Resolve a service target once (CAP-03) to ``(target_kind, target_id, name)``.

        ``new_device`` allocates a fresh preview id and keeps the user's name (a
        raised ValueError for an invalid name is the caller's capture_invalid_name).
        ``candidate`` matches a visible candidate by exact signature id first, then
        by NFC + whitespace-collapsed + casefolded display name. ``device`` with a
        ``device_id`` reads the device registry entry whose identifiers carry
        ``(DOMAIN, uuid)`` and requires a confirmed PowerGuesser record; without one
        it matches the confirmed records by exact uuid then by casefolded name, so a
        confirmed device whose registry entry is gone stays targetable.
        """
        if target_type == "new_device":
            name = normalise_name(target)
            return ("candidate", PREVIEW_ID_PREFIX + uuid4().hex[:28], name)

        if target_type == "candidate":
            visible = self.hub.learning.visible_candidates()
            for cand in visible:
                if cand.get("signature_id") == target:
                    return ("candidate", target, None)
            key = _match_key(target)
            matches = [
                cand
                for cand in visible
                if _match_key(self.hub.learning._display_name(cand)) == key
            ]
            if not matches:
                names = [
                    self.hub.learning._display_name(c)
                    for c in visible
                ]
                raise self._target_unknown(target, names)
            if len(matches) > 1:
                names = [self.hub.learning._display_name(c) for c in matches]
                raise self._target_ambiguous(target, names)
            return ("candidate", str(matches[0]["signature_id"]), None)

        if target_type == "device":
            if device_id is not None:
                return self._resolve_device_id(target, device_id)
            return self._resolve_device_record(target)

        raise self._target_unknown(target, [])

    def _confirmed_names(self) -> list[str]:
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return []
        return [str(e.get("label")) for e in devices.confirmed_entries()]

    def _resolve_device_id(self, target: str, device_id: str) -> tuple[str, str, None]:
        entry = dr.async_get(self.hass).async_get(device_id)
        uuid: str | None = None
        if entry is not None:
            for domain, ident in entry.identifiers:
                if domain == DOMAIN:
                    uuid = ident
                    break
        devices = getattr(self.hub, "devices", None)
        rec = devices.record(uuid) if (devices is not None and uuid) else None
        if rec is None or rec.get("status") != "confirmed":
            raise self._target_unknown(target, self._confirmed_names())
        return ("device", str(uuid), None)

    def _resolve_device_record(self, target: str) -> tuple[str, str, None]:
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            raise self._target_unknown(target, [])
        rec = devices.record(target)
        if rec is not None and rec.get("status") == "confirmed":
            return ("device", target, None)
        key = _match_key(target)
        entries = devices.confirmed_entries()
        matches = [e for e in entries if _match_key(e.get("label")) == key]
        if not matches:
            raise self._target_unknown(target, self._confirmed_names())
        if len(matches) > 1:
            raise self._target_ambiguous(
                target, [str(e.get("label")) for e in matches]
            )
        return ("device", str(matches[0]["device_uuid"]), None)

    @staticmethod
    def _replay_for(
        context: Mapping[str, Any],
        is_device: bool,
        device_artefact: Mapping[str, Any] | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Run the read-only replay for a device target that snapped a pair."""
        if not is_device or device_artefact is None:
            return None
        if result.get("rise_ms") is None:
            return None
        return replay_check(
            context,
            device_artefact,
            rise_ms=int(result["rise_ms"]),
            fall_ms=int(result["fall_ms"]),
        )

    # --- live capture (start / stop / settle / cancel / unload) ----------

    def _effective_grid_s(self) -> int:
        """Return the current effective grid in whole seconds (default 30)."""
        coordinator = getattr(self.hub, "profile_coordinator", None)
        data = getattr(coordinator, "data", None)
        if data is not None:
            return int(round(float(getattr(data, "effective_s", 30.0) or 30.0)))
        return 30

    def _find(self, capture_id: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("capture_id") == capture_id:
                return rec
        return None

    def _index_of(self, capture_id: str) -> int | None:
        for index, rec in enumerate(self._records):
            if rec.get("capture_id") == capture_id:
                return index
        return None

    def _open_record(self) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("state") in OPEN_STATES:
                return rec
        return None

    def _not_found(self, capture_id: str) -> ServiceValidationError:
        return ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="capture_not_found",
            translation_placeholders={"capture_id": capture_id},
        )

    def _cancel_expiry(self) -> None:
        if self._unsub_expiry is not None:
            self._unsub_expiry()
            self._unsub_expiry = None

    def _cancel_settle(self) -> None:
        if self._unsub_settle is not None:
            self._unsub_settle()
            self._unsub_settle = None

    def _cancel_timers(self) -> None:
        self._cancel_expiry()
        self._cancel_settle()

    async def async_start(
        self,
        *,
        label: str,
        target_type: str,
        target: str,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a live capture ("on now") and arm its auto-expiry timer.

        Refuses ``capture_already_open`` while any capture is open (single open
        capture, RESEARCH Pitfall 7). Returns ``{capture_id, state, started_at}``.
        """
        open_rec = self._open_record()
        if open_rec is not None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_already_open",
                translation_placeholders={
                    "capture_id": str(open_rec.get("capture_id"))
                },
            )
        try:
            norm_label = normalise_name(label)
            target_kind, target_id, new_name = self._resolve_target(
                target_type, target, device_id
            )
        except ValueError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_invalid_name",
                translation_placeholders={"reason": str(err)},
            ) from err

        capture_id = uuid4().hex
        created_at = dt_util.utcnow().isoformat()
        record: dict[str, Any] = {
            "capture_id": capture_id,
            "mode": "live",
            "state": "recording",
            "target_kind": target_kind,
            "target_id": target_id,
            "new_device_name": new_name,
            "label": norm_label,
            "press_start_ts": created_at,
            "press_stop_ts": None,
            "window_start_ms": None,
            "window_end_ms": None,
            "created_at": created_at,
            "resolved_at": None,
            "reassigned_at": None,
            "reason_code": None,
            "gap_flagged": False,
            "result": None,
            "features": None,
            "trace": None,
            "holdout_excluded": True,
            "holdout_window_ms": None,
            "replay": None,
        }
        # Same cap as the retroactive path (T-08-18): after target resolution and
        # immediately before the append, so a refusal leaves the store untouched.
        self._enforce_record_cap()
        self._records.append(record)
        await self._store.async_save({"captures": self._records})
        self._arm_expiry(capture_id)
        return {"capture_id": capture_id, "state": "recording", "started_at": created_at}

    def _arm_expiry(self, capture_id: str) -> None:
        self._cancel_expiry()
        self._unsub_expiry = async_call_later(
            self.hass,
            DEFAULT_CONFIG.live_max_open_s,
            functools.partial(self._async_expire, capture_id),
        )

    async def _async_expire(self, capture_id: str, _now: Any) -> None:
        """Auto-expire an open capture as interrupted/timeout (Pitfall 7)."""
        self._unsub_expiry = None
        rec = self._find(capture_id)
        if rec is None or rec.get("state") not in OPEN_STATES:
            return
        index = self._index_of(capture_id)
        if index is None:
            return
        press_start_ms = _iso_to_ms(rec.get("press_start_ts")) or 0
        self._cancel_settle()
        self._records[index] = {
            **rec,
            "state": "interrupted",
            "reason_code": "timeout",
            "resolved_at": dt_util.utcnow().isoformat(),
            "holdout_window_ms": [
                press_start_ms,
                press_start_ms + DEFAULT_CONFIG.live_max_open_s * 1000,
            ],
        }
        await self._store.async_save({"captures": self._records})
        self._refresh_orphan_issue()

    async def async_stop(self, capture_id: str | None = None) -> dict[str, Any]:
        """Stop a live capture; returns ``resolving`` at once (owner Q2).

        A recording capture moves to ``resolving`` and arms the settle timer. A
        capture already resolving, or a terminal one named by id, returns its
        current state unchanged without saving or arming a second timer (Edge
        CAP-02 idempotency: a double stop press runs exactly one resolution).
        """
        if capture_id is not None:
            rec = self._find(capture_id)
            if rec is None:
                raise self._not_found(capture_id)
            if rec.get("state") not in OPEN_STATES:
                return {"capture_id": capture_id, "state": rec.get("state")}
            if rec.get("state") == "resolving":
                return {"capture_id": capture_id, "state": "resolving"}
            return await self._begin_resolve(rec)
        open_rec = self._open_record()
        if open_rec is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="capture_not_open"
            )
        if open_rec.get("state") == "resolving":
            return {"capture_id": open_rec["capture_id"], "state": "resolving"}
        return await self._begin_resolve(open_rec)

    async def _begin_resolve(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        capture_id = str(rec["capture_id"])
        index = self._index_of(capture_id)
        if index is None:
            raise self._not_found(capture_id)
        self._records[index] = {
            **rec,
            "state": "resolving",
            "press_stop_ts": dt_util.utcnow().isoformat(),
        }
        self._cancel_expiry()
        await self._store.async_save({"captures": self._records})
        grid_s = self._effective_grid_s()
        delay = max(
            DEFAULT_CONFIG.live_stop_settle_min_s,
            min(
                DEFAULT_CONFIG.live_stop_settle_max_s,
                DEFAULT_CONFIG.live_stop_settle_samples * grid_s,
            ),
        )
        self._arm_settle(capture_id, float(delay))
        return {"capture_id": capture_id, "state": "resolving"}

    def _arm_settle(self, capture_id: str, delay_s: float) -> None:
        self._cancel_settle()
        self._unsub_settle = async_call_later(
            self.hass, delay_s, functools.partial(self._async_settle, capture_id)
        )

    async def _async_settle(self, capture_id: str, _now: Any) -> None:
        """Flush, snap the live run and finish (or wait once more on still-on)."""
        self._unsub_settle = None
        rec = self._find(capture_id)
        if rec is None or rec.get("state") != "resolving":
            return
        press_start_ms = _iso_to_ms(rec.get("press_start_ts")) or 0
        press_stop_ms = _iso_to_ms(rec.get("press_stop_ts")) or press_start_ms
        grid_s = self._effective_grid_s()
        now_ms = int(round(dt_util.utcnow().timestamp() * 1000))
        rise_range, fall_range = snap.live_ranges(
            press_start_ms, press_stop_ms, now_ms, DEFAULT_CONFIG
        )

        writer = getattr(self.hub, "writer", None)
        if writer is not None:
            await writer.async_flush()

        device_artefact: Mapping[str, Any] | None = None
        is_device = rec.get("target_kind") == "device"
        if is_device:
            devices = getattr(self.hub, "devices", None)
            if devices is not None:
                device_artefact, _v = devices.active_artefact_and_version(
                    str(rec.get("target_id"))
                )

        def _job() -> dict[str, Any]:
            context = self.hub.learning.build_context_job()
            context["tz_name"] = str(dt_util.get_default_time_zone())
            snapped = snap.snap_run(
                context,
                rise_range_ms=rise_range,
                fall_range_ms=fall_range,
                ref_start_ms=press_start_ms,
                ref_end_ms=press_stop_ms,
            )
            replay = self._replay_for(
                context, is_device, device_artefact, snapped["result"]
            )
            return {"snapped": snapped, "replay": replay}

        try:
            out = await self.hass.async_add_executor_job(_job)

            # The record may have been cancelled or deleted while the job ran: write
            # nothing unless it is still resolving (T-08-10).
            rec = self._find(capture_id)
            if rec is None or rec.get("state") != "resolving":
                return

            snapped = out["snapped"]
            replay = out["replay"]
            result = snapped["result"]
            verdict = result["verdict"]
            still_on = "still_on" in (result.get("quality_flags") or [])
            if (
                verdict == "none"
                and result.get("reason_code") == "no_off_edge"
                and still_on
                and capture_id not in self._extended
            ):
                self._extended.add(capture_id)
                extend_to = press_stop_ms + DEFAULT_CONFIG.live_stop_max_wait_s * 1000
                self._arm_settle(capture_id, max(1.0, (extend_to - now_ms) / 1000.0))
                return

            if result["rise_ms"] is not None:
                result["press_to_rise_s"] = (result["rise_ms"] - press_start_ms) / 1000.0
                result["press_to_fall_s"] = (result["fall_ms"] - press_stop_ms) / 1000.0
                holdout_window_ms = [
                    int(result["rise_ms"]),
                    int(result["fall_ms"]) + grid_s * 1000,
                ]
                state = "done" if verdict in VERDICTS_DONE else "failed"
            else:
                holdout_window_ms = [
                    press_start_ms - DEFAULT_CONFIG.live_rise_before_s * 1000,
                    press_stop_ms,
                ]
                state = "failed"

            updated = {
                **rec,
                "state": state,
                "resolved_at": dt_util.utcnow().isoformat(),
                "reason_code": result.get("reason_code"),
                "result": result,
                "features": snapped["features"],
                "trace": snapped["trace"],
                "holdout_window_ms": holdout_window_ms,
                "replay": replay,
            }
            # Refit/name the preview (and record the D-07 similar note) before the
            # record is saved, so the mutation reaches the Store.
            if verdict == "clean":
                await self._register_clean_preview(updated)
            # The preview refit above can await (a new-device target writes the review
            # Store), so re-resolve the record afterwards rather than reusing the
            # pre-await state: a concurrent delete/cancel can remove or reorder
            # ``_records``, which would make a stale index overwrite the wrong record
            # or raise IndexError. Write nothing if it is gone or no longer resolving
            # (T-08-10, WR-03).
            rec = self._find(capture_id)
            if rec is None or rec.get("state") != "resolving":
                return
            index = self._index_of(capture_id)
            if index is None:
                return
            self._records[index] = updated
            self._extended.discard(capture_id)
            await self._store.async_save({"captures": self._records})
        except Exception:  # noqa: BLE001 - a settle failure must not strand the record
            _LOGGER.exception(
                "Live capture %s failed to settle; marking it interrupted", capture_id
            )
            await self._interrupt_settle(capture_id, "settle_error")
            return
        self._refresh_orphan_issue()

    async def _interrupt_settle(self, capture_id: str, reason_code: str) -> None:
        """Mark a still-resolving capture interrupted after a settle failure.

        A raised exception in the settle job must not strand the record in
        ``resolving`` with no save and no re-armed timer; it ends ``interrupted``
        with the given reason so the record is terminal and persisted.
        """
        rec = self._find(capture_id)
        if rec is None or rec.get("state") != "resolving":
            return
        index = self._index_of(capture_id)
        if index is None:
            return
        self._extended.discard(capture_id)
        self._records[index] = {
            **rec,
            "state": "interrupted",
            "reason_code": reason_code,
            "resolved_at": dt_util.utcnow().isoformat(),
        }
        await self._store.async_save({"captures": self._records})
        self._refresh_orphan_issue()

    async def async_cancel(self, capture_id: str | None = None) -> dict[str, Any]:
        """Cancel an open capture; idempotent on a terminal record."""
        if capture_id is None:
            rec = self._open_record()
            if rec is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN, translation_key="capture_not_open"
                )
        else:
            found = self._find(capture_id)
            if found is None:
                raise self._not_found(capture_id)
            if found.get("state") not in OPEN_STATES:
                return {"capture_id": capture_id, "state": found.get("state")}
            rec = found

        capture_id = str(rec["capture_id"])
        index = self._index_of(capture_id)
        if index is None:
            raise self._not_found(capture_id)
        self._cancel_timers()
        self._extended.discard(capture_id)
        press_start_ms = _iso_to_ms(rec.get("press_start_ts")) or 0
        press_stop_ms = _iso_to_ms(rec.get("press_stop_ts"))
        if press_stop_ms is None:
            press_stop_ms = int(round(dt_util.utcnow().timestamp() * 1000))
        self._records[index] = {
            **rec,
            "state": "cancelled",
            "resolved_at": dt_util.utcnow().isoformat(),
            "holdout_window_ms": [press_start_ms, press_stop_ms],
        }
        await self._store.async_save({"captures": self._records})
        self._refresh_orphan_issue()
        return {"capture_id": capture_id, "state": "cancelled"}

    async def async_unload(self) -> None:
        """Cancel every capture timer (T-08-09: no timer survives an unload)."""
        self._cancel_timers()
        if self._unsub_coordinator is not None:
            self._unsub_coordinator()
            self._unsub_coordinator = None

    def _preview_group(self, target_id: str) -> list[dict[str, Any]]:
        """Every record that resolves to the given preview candidate id."""
        decisions = self.hub.learning.review.decisions
        group: list[dict[str, Any]] = []
        for rec in self._records:
            resolved = resolve_session_target(rec, decisions, self._device_lookup)
            if (
                resolved is not None
                and resolved[0] == "candidate"
                and str(resolved[1]) == target_id
            ):
                group.append(rec)
        return group

    async def _register_clean_preview(self, record: dict[str, Any]) -> None:
        """Refit the preview for a clean capture and note a dismissed match (SC5, D-07).

        Shared by the retroactive and live paths. Only a preview target (a
        ``cap_`` id) has a preview: a new-device target names its decision with the
        refitted fingerprint, an existing-preview target keeps its name. D-07 (HA
        half): when the refitted preview's shape matches a dismissed candidate's
        fingerprint, the capture result names it in ``similar_dismissed``; the
        label note is added by ``candidate_label``.
        """
        target_id = record.get("target_id")
        if not target_id or not str(target_id).startswith(PREVIEW_ID_PREFIX):
            return
        tid = str(target_id)
        name = record.get("new_device_name")
        if not name:
            dec = self.hub.learning.review.decisions.get(tid) or {}
            name = dec.get("name") or tid
        group = self._preview_group(tid)
        if not any(rec is record for rec in group):
            group.append(record)
        artefact = fit_preview(tid, str(name), group)
        if artefact is None:
            return
        if record.get("new_device_name"):
            fingerprint = SignatureArtefact.from_dict(artefact).fingerprint()
            await self.hub.learning.review.async_set_name(
                tid, str(name), fingerprint, label=str(name)
            )
        result = record.get("result")
        if isinstance(result, dict):
            result["similar_dismissed"] = similar_dismissed(
                artefact, self.hub.learning.review.decisions
            )

    # --- capture_window -------------------------------------------------

    async def async_capture_window(
        self,
        *,
        start: Any,
        end: Any,
        label: str,
        target_type: str,
        target: str,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Snap a retroactive window to its real rise/fall pair and store a record.

        Returns the new record's ``list_view`` entry: every stored record key
        (replay included) plus the read-time target resolution and window echo.
        """
        try:
            norm_label = normalise_name(label)
            target_kind, target_id, new_device_name = self._resolve_target(
                target_type, target, device_id
            )
        except ValueError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_invalid_name",
                translation_placeholders={"reason": str(err)},
            ) from err

        device_artefact: Mapping[str, Any] | None = None
        is_device = target_kind == "device"
        if is_device:
            devices = getattr(self.hub, "devices", None)
            if devices is not None:
                device_artefact, _v = devices.active_artefact_and_version(target_id)

        # Make sure the off-edge is flushed to SQLite before the read-only snap
        # connection (RESEARCH anti-pattern: the writer batches 60 s).
        writer = getattr(self.hub, "writer", None)
        if writer is not None:
            await writer.async_flush()

        def _job() -> dict[str, Any]:
            context = self.hub.learning.build_context_job()
            context["tz_name"] = str(dt_util.get_default_time_zone())
            # resolve_window and snap_run (through _load_job_residual) each open
            # their own read-only SQLite connection on purpose: both run inside
            # this one executor job, the reads are bounded (a window of at most
            # 6 h plus 25 h of context), and no connection object is shared
            # across helpers or across the executor boundary.
            w_start_ms, w_end_ms = window.resolve_window(
                start,
                end,
                tz_name=context["tz_name"],
                now_ms=int(round(dt_util.utcnow().timestamp() * 1000)),
                grid_s=int(context.get("grid_s", 30)),
                enabled_event_types=context.get("enabled_event_types", []),
                buffer_path=context["buffer_path"],
                config=DEFAULT_CONFIG,
            )
            w_rise_range, w_fall_range = snap.retro_ranges(w_start_ms, w_end_ms)
            snapped = snap.snap_run(
                context,
                rise_range_ms=w_rise_range,
                fall_range_ms=w_fall_range,
                ref_start_ms=w_start_ms,
                ref_end_ms=w_end_ms,
            )
            replay = self._replay_for(
                context, is_device, device_artefact, snapped["result"]
            )
            return {
                "snapped": snapped,
                "start_ms": w_start_ms,
                "end_ms": w_end_ms,
                "replay": replay,
            }

        try:
            job_out = await self.hass.async_add_executor_job(_job)
        except CaptureRefused as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_" + err.code,
                translation_placeholders=err.details,
            ) from err
        snapped = job_out["snapped"]
        start_ms = job_out["start_ms"]
        end_ms = job_out["end_ms"]
        replay = job_out["replay"]
        result = snapped["result"]
        verdict = result["verdict"]

        created_at = dt_util.utcnow().isoformat()
        grid_s = int(result.get("grid_s") or 30)

        if result.get("rise_ms") is not None:
            holdout_window_ms = [
                int(result["rise_ms"]),
                int(result["fall_ms"]) + grid_s * 1000,
            ]
        else:
            holdout_window_ms = [int(start_ms), int(end_ms)]

        record: dict[str, Any] = {
            "capture_id": uuid4().hex,
            "mode": "window",
            "state": "done" if verdict in VERDICTS_DONE else "failed",
            "target_kind": target_kind,
            "target_id": target_id,
            "new_device_name": new_device_name,
            "label": norm_label,
            "press_start_ts": None,
            "press_stop_ts": None,
            "window_start_ms": int(start_ms),
            "window_end_ms": int(end_ms),
            "created_at": created_at,
            "resolved_at": dt_util.utcnow().isoformat(),
            "reassigned_at": None,
            "reason_code": result.get("reason_code"),
            "gap_flagged": False,
            "result": result,
            "features": snapped["features"],
            "trace": snapped["trace"],
            "holdout_excluded": True,
            "holdout_window_ms": holdout_window_ms,
            "replay": replay,
        }

        # The user's name wins over a later learned label (SC5, Pattern 3). The
        # decision carries the fitted preview's fingerprint so a later run's
        # identity match can reconcile with it.
        if verdict == "clean":
            await self._register_clean_preview(record)

        # Prune only once the capture is known to be storable AND the preview
        # registration above has not raised: the prune and the append must be
        # atomic with respect to a failure in preview registration, or the
        # in-memory records would be pruned while the Store is never saved (the
        # WR-02 memory/Store divergence). A refusal above
        # (capture_invalid_name / capture_target_* / CaptureRefused) never
        # reaches here either, so it cannot prune without saving (T-08-18,
        # WR-02).
        self._enforce_record_cap()
        self._records.append(record)
        await self._store.async_save({"captures": self._records})
        self._refresh_orphan_issue()
        return self._list_view_entry(record)

    def _enforce_record_cap(self) -> None:
        """Keep the Store bounded, pruning only non-teaching records (T-08-18).

        Drops the oldest ``failed``/``cancelled``/``interrupted`` records until a
        new record fits; when only ``done`` (teaching) records remain at the cap
        the new capture is refused with ``capture_store_full`` and nothing is
        removed (never drop a taught example, PITFALLS 21).
        """
        if len(self._records) < MAX_CAPTURE_RECORDS:
            return
        work = list(self._records)
        while len(work) >= MAX_CAPTURE_RECORDS:
            drop_index = next(
                (
                    i
                    for i, rec in enumerate(work)
                    if rec.get("state") in _DROPPABLE_STATES
                ),
                None,
            )
            if drop_index is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="capture_store_full",
                    translation_placeholders={"limit": str(MAX_CAPTURE_RECORDS)},
                )
            del work[drop_index]
        self._records = work

    # --- delete / reassign / orphan issue (CAP-05, D-06) ----------------

    async def async_delete(self, capture_id: str) -> dict[str, Any]:
        """Remove a capture record; idempotent on an unknown id (Edge CAP-05).

        An open record has its timers cancelled first; a resolution that finishes
        for a deleted record writes nothing (the post-await state re-check from
        08-03 covers it, the test pins it). Returns ``{deleted, preview_removed}``
        where ``preview_removed`` is True when a preview card for the record's
        group existed before and not after.
        """
        rec = self._find(capture_id)
        if rec is None:
            return {"deleted": False}
        if rec.get("state") in OPEN_STATES:
            self._cancel_timers()
        self._extended.discard(capture_id)
        before = self._preview_group_ids()
        index = self._index_of(capture_id)
        if index is not None:
            del self._records[index]
        await self._store.async_save({"captures": self._records})
        after = self._preview_group_ids()
        self._refresh_orphan_issue()
        return {"deleted": True, "preview_removed": bool(before - after)}

    async def async_reassign(
        self,
        capture_id: str,
        *,
        target_type: str,
        target: str,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        """Re-target a capture (CAP-05, D-06).

        The only action that rewrites ``target_kind``/``target_id``/
        ``new_device_name`` and stamps ``reassigned_at``. Refuses an open capture
        with ``capture_busy``; re-targeting to the target the record already
        resolves to returns the unchanged view without writing; a device target
        recomputes the replay block.
        """
        rec = self._find(capture_id)
        if rec is None:
            raise self._not_found(capture_id)
        if rec.get("state") in OPEN_STATES:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_busy",
                translation_placeholders={"capture_id": capture_id},
            )
        try:
            kind, ident, new_name = self._resolve_target(
                target_type, target, device_id
            )
        except ValueError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_invalid_name",
                translation_placeholders={"reason": str(err)},
            ) from err

        resolved_now = resolve_session_target(
            rec, self.hub.learning.review.decisions, self._device_lookup
        )
        if resolved_now is not None and resolved_now == (kind, ident):
            return self._list_view_entry(rec)

        replay = await self._reassigned_replay(kind, ident, rec)

        index = self._index_of(capture_id)
        if index is None:
            raise self._not_found(capture_id)
        updated = {
            **rec,
            "target_kind": kind,
            "target_id": ident,
            "new_device_name": new_name,
            "reassigned_at": dt_util.utcnow().isoformat(),
            "replay": replay,
        }
        self._records[index] = updated
        await self._store.async_save({"captures": self._records})
        self._refresh_orphan_issue()
        return self._list_view_entry(updated)

    async def _reassigned_replay(
        self, kind: str, ident: str, rec: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Recompute the read-only replay for a device target that has a pair."""
        result = rec.get("result") or {}
        if kind != "device" or result.get("rise_ms") is None:
            return None
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return None
        artefact, _v = devices.active_artefact_and_version(ident)
        if artefact is None:
            return None
        rise_ms = int(result["rise_ms"])
        fall_ms = int(result["fall_ms"])

        def _job() -> dict[str, Any]:
            context = self.hub.learning.build_context_job()
            context["tz_name"] = str(dt_util.get_default_time_zone())
            return replay_check(context, artefact, rise_ms=rise_ms, fall_ms=fall_ms)

        return await self.hass.async_add_executor_job(_job)

    def orphans(self) -> list[dict[str, Any]]:
        """Return records whose target no longer resolves (D-06)."""
        decisions = self.hub.learning.review.decisions
        return [
            dict(rec)
            for rec in self._records
            if resolve_session_target(rec, decisions, self._device_lookup) is None
        ]

    def captured_runs(self, target_kind: str, target_id: str) -> list[dict[str, Any]]:
        """Return the snapped pair of every record resolving to a target (used by 08-07)."""
        decisions = self.hub.learning.review.decisions
        out: list[dict[str, Any]] = []
        for rec in self._records:
            resolved = resolve_session_target(rec, decisions, self._device_lookup)
            if resolved is None or resolved != (target_kind, target_id):
                continue
            result = rec.get("result") or {}
            if result.get("rise_ms") is None:
                continue
            out.append(
                {
                    "capture_id": rec.get("capture_id"),
                    "rise_ms": result.get("rise_ms"),
                    "fall_ms": result.get("fall_ms"),
                    "step_w": result.get("step_w"),
                    "dur_s": result.get("dur_s"),
                    "energy_wh": result.get("energy_wh"),
                    "grid_s": result.get("grid_s"),
                    "verdict": result.get("verdict"),
                }
            )
        return out

    def identity_priors(self) -> list[dict[str, Any]]:
        """Prior preview artefacts for the next run's identity matching (CAP-09).

        One clean prior per capture group whose capture-origin id resolves to an
        unmerged, un-dismissed, unowned candidate: the decorated preview dict with
        CANDIDATE_DECORATION_KEYS removed, so only the fit_preview artefact enters
        the prior pool. A group that resolves to a device, to a merged id or to
        nothing contributes nothing (read-time resolution, D-06).
        """
        devices = getattr(self.hub, "devices", None)
        owned = devices.owned_signature_ids() if devices is not None else set()
        priors: list[dict[str, Any]] = []
        for cand in self.preview_candidates():
            sig_id = str(cand.get("signature_id", ""))
            if not sig_id or sig_id in owned:
                continue
            priors.append(
                {k: v for k, v in cand.items() if k not in CANDIDATE_DECORATION_KEYS}
            )
        return priors

    def capture_prior_ids(self) -> list[str]:
        """The signature ids of the capture-origin priors (CAP-09/D-07)."""
        return [
            str(p["signature_id"])
            for p in self.identity_priors()
            if p.get("signature_id")
        ]

    def capture_anchors(self) -> list[dict[str, Any]]:
        """Anchors for every clean capture whose target resolves to a candidate.

        An anchor on a learned candidate can re-bind a new LLM artefact of a later
        run to that existing learned id; that is the intended "same load, same identity"
        behaviour (the user said this run belongs to that candidate), not a hijack,
        because the artefact must itself claim the captured episode (T-08-24).
        Preview groups and attached learned candidates alike contribute; a capture
        resolving to a device or to nothing does not.
        """
        decisions = self.hub.learning.review.decisions
        anchors: list[dict[str, Any]] = []
        for rec in self._records:
            result = rec.get("result")
            if not isinstance(result, Mapping):
                continue
            if result.get("verdict") != "clean" or result.get("rise_ms") is None:
                continue
            resolved = resolve_session_target(rec, decisions, self._device_lookup)
            if resolved is None or resolved[0] != "candidate":
                continue
            anchors.append(
                {
                    "signature_id": str(resolved[1]),
                    "rise_ms": result.get("rise_ms"),
                    "fall_ms": result.get("fall_ms"),
                }
            )
        return anchors

    def _preview_group_ids(self) -> set[str]:
        """Return the preview-candidate signature ids the records resolve to."""
        decisions = self.hub.learning.review.decisions
        ids: set[str] = set()
        for rec in self._records:
            resolved = resolve_session_target(rec, decisions, self._device_lookup)
            if (
                resolved is not None
                and resolved[0] == "candidate"
                and str(resolved[1]).startswith(PREVIEW_ID_PREFIX)
            ):
                ids.add(str(resolved[1]))
        return ids

    def _async_register_orphan_listener(self) -> None:
        coordinator = getattr(self.hub.learning, "coordinator", None)
        if coordinator is None:
            return
        self._unsub_coordinator = coordinator.async_add_listener(
            self._refresh_orphan_issue
        )

    def _refresh_orphan_issue(self) -> None:
        """Create or clear the non-fixable capture_orphans Repairs issue (D-06)."""
        count = len(self.orphans())
        if count:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                ISSUE_CAPTURE_ORPHANS,
                is_fixable=False,
                is_persistent=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="capture_orphans",
                translation_placeholders={"count": str(count)},
                config_entry_id=self.entry.entry_id,
            )
        else:
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_CAPTURE_ORPHANS)

    # --- list / previews ------------------------------------------------

    def list_view(self, capture_id: str | None = None) -> list[dict[str, Any]]:
        """Return the list_view entries, in created_at order (or the named one)."""
        records = sorted(self._records, key=lambda r: str(r.get("created_at") or ""))
        if capture_id is not None:
            records = [r for r in records if r.get("capture_id") == capture_id]
        return [self._list_view_entry(rec) for rec in records]

    def _list_view_entry(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Build one record's list_view entry (never rewrites the record)."""
        resolved = resolve_session_target(
            record, self.hub.learning.review.decisions, self._device_lookup
        )
        if resolved is None:
            target = None
            target_state = "target_gone"
            attached_as = "target_gone"
        else:
            target = self._target_view(resolved[0], resolved[1])
            target_state = "ok"
            verdict = (record.get("result") or {}).get("verdict")
            if verdict != "clean":
                attached_as = "not_teaching"
            elif resolved[0] == "device":
                attached_as = "device_example"
            elif str(resolved[1]).startswith(PREVIEW_ID_PREFIX):
                attached_as = "preview"
            else:
                attached_as = "candidate_example"
        return {
            **dict(record),
            "target": target,
            "target_state": target_state,
            "attached_as": attached_as,
            "window": self._window_echo(record),
        }

    def _target_view(self, kind: str, ident: str) -> dict[str, Any] | None:
        if kind == "candidate":
            dec = self.hub.learning.review.decisions.get(ident) or {}
            name = dec.get("name")
            if not name:
                for rec in self._records:
                    if rec.get("target_id") == ident and rec.get("new_device_name"):
                        name = rec["new_device_name"]
                        break
            return {"kind": "candidate", "id": ident, "name": name or ident}
        if kind == "device":
            devices = getattr(self.hub, "devices", None)
            rec = devices.record(ident) if devices is not None else None
            return {"kind": "device", "id": ident, "name": (rec or {}).get("name") or ident}
        return None

    def _window_echo(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Echo the interpreted window as local and UTC time.

        A still-``recording`` live capture has no stop time yet: the end keys are
        emitted as ``None`` rather than a 1970 timestamp (the 0 sentinel below is
        only a fallback for a genuinely missing start).
        """
        tz = dt_util.get_default_time_zone()
        start_ms = record.get("window_start_ms")
        end_ms = record.get("window_end_ms")
        if start_ms is None:
            start_ms = _iso_to_ms(record.get("press_start_ts"))
        if end_ms is None:
            end_ms = _iso_to_ms(record.get("press_stop_ts"))
        start_utc = dt_util.utc_from_timestamp((start_ms or 0) / 1000.0)
        if end_ms is None:
            end_local: str | None = None
            end_utc: str | None = None
        else:
            end_dt = dt_util.utc_from_timestamp(end_ms / 1000.0)
            end_local = end_dt.astimezone(tz).isoformat()
            end_utc = end_dt.isoformat()
        return {
            "start_local": start_utc.astimezone(tz).isoformat(),
            "end_local": end_local,
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc,
            "tz": str(tz),
        }

    def preview_candidates(self) -> list[dict[str, Any]]:
        """Derive the preview candidates from the capture records at read time.

        Groups every record whose target resolves to a ``cap_`` candidate, fits
        one preview per group from its clean examples, and decorates the artefact
        dict for Review. Never stored, so delete/reassign/merge/dismiss can never
        leave a stale preview (D-06).
        """
        decisions = self.hub.learning.review.decisions
        groups: dict[str, list[dict[str, Any]]] = {}
        for rec in self._records:
            resolved = resolve_session_target(rec, decisions, self._device_lookup)
            if resolved is None or resolved[0] != "candidate":
                continue
            ident = str(resolved[1])
            if not ident.startswith(PREVIEW_ID_PREFIX):
                continue
            groups.setdefault(ident, []).append(rec)

        previews: list[dict[str, Any]] = []
        for ident, records in groups.items():
            dec = decisions.get(ident) or {}
            name = dec.get("name")
            if not name:
                ordered = sorted(records, key=lambda r: str(r.get("created_at") or ""))
                name = ordered[0].get("new_device_name") or ident
            artefact = fit_preview(ident, str(name), records)
            if artefact is None:
                continue
            # EVID-05: the preview's own evidence block, built from its captured
            # runs (source "capture", v1 schema, every run marked captured).
            artefact["evidence"] = capture_evidence(
                records,
                tz_name=str(dt_util.get_default_time_zone()),
                now_iso=dt_util.utcnow().isoformat(),
            )
            artefact["origin"] = "capture"
            artefact["identity"] = {
                "status": "new",
                "matched": None,
                "merged_into": None,
                "twin_of": None,
            }
            artefact["fingerprint"] = SignatureArtefact.from_dict(artefact).fingerprint()
            artefact["first_label"] = str(name)
            # D-03: the group's creation time is its earliest clean capture's
            # resolved_at; a confirmed preview trains from this, not the last
            # learning run's created_at.
            clean_resolved = [
                str(rec["resolved_at"])
                for rec in records
                if isinstance(rec.get("result"), Mapping)
                and rec["result"].get("verdict") == "clean"
                and rec.get("resolved_at")
            ]
            if clean_resolved and isinstance(artefact.get("provenance"), dict):
                artefact["provenance"]["created_at"] = min(clean_resolved)
            previews.append(artefact)
        return previews

    def holdout_windows(self) -> list[list[int]]:
        """Return each record's holdout window as ``[start_ms, end_ms]``."""
        return [
            list(rec["holdout_window_ms"])
            for rec in self._records
            if rec.get("holdout_window_ms")
        ]


def _iso_to_ms(value: Any) -> int | None:
    """Parse an ISO timestamp to a millisecond epoch, or None."""
    if not isinstance(value, str) or not value:
        return None
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        return None
    return int(round(parsed.timestamp() * 1000))


async def async_remove_capture_data(hass: HomeAssistant) -> None:
    """Remove the captures Store (entry removal)."""
    await Store(hass, CAPTURES_STORE_VERSION, CAPTURES_STORE_KEY).async_remove()
