"""Review store and normalisation for PowerGuesser candidates."""

from collections.abc import Callable, Mapping
from datetime import datetime
import logging
from typing import Any
import unicodedata
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import (
    LEARNING_STORES_VERSION,
    REVIEW_STORE_KEY,
    SESSIONS_STORE_KEY,
    SESSIONS_STORE_VERSION,
)

_LOGGER = logging.getLogger(__name__)

NAME_MAX = 64

# The session record's end_reason vocabulary (D-09, Plan 06-06 artifacts).
SESSION_END_USER = "user"
SESSION_END_DATE = "end_date"
SESSION_END_PLUG_METERED = "plug_metered"

# Hop limit for the target-resolution chains (candidate merged_into, device
# retired_into); a longer or cyclic chain resolves to None (T-06-14).
_TARGET_MAX_HOPS = 10


def resolve_session_target(
    record: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    device_lookup: Callable[[str], Mapping[str, Any] | None],
) -> tuple[str, str] | None:
    """Resolve a session record's target to its current stable identity.

    Returns ``("candidate", signature_id)`` or ``("device", uuid)``, or None
    when the target no longer resolves (dismissed candidate, a broken or
    cyclic merge/retire chain, a missing device). The record itself is never
    rewritten: rename, merge, confirm and retire all move the *resolution*,
    so a label shown for an earlier session is derived from its record and the
    current state, never from where the plug is now (D-09, SC3).
    """
    target_kind = record.get("target_kind")
    target_id = str(record.get("target_id") or "")

    if target_kind == "candidate":
        sig_id = target_id
        seen: set[str] = set()
        for _ in range(_TARGET_MAX_HOPS):
            if not sig_id or sig_id in seen:
                return None
            seen.add(sig_id)
            dec = decisions.get(sig_id) or {}
            if dec.get("status") == "dismissed":
                return None
            merged = dec.get("merged_into")
            if merged:
                sig_id = str(merged)
                continue
            device_uuid = dec.get("device_uuid")
            if device_uuid:
                linked = device_lookup(str(device_uuid))
                if linked is not None and linked.get("status") in (
                    "confirmed",
                    "retired",
                ):
                    return _resolve_device(str(device_uuid), device_lookup)
            return ("candidate", sig_id)
        return None

    if target_kind == "device":
        return _resolve_device(target_id, device_lookup)

    return None


def _resolve_device(
    device_uuid: str,
    device_lookup: Callable[[str], Mapping[str, Any] | None],
) -> tuple[str, str] | None:
    """Follow a device's retired_into chain to the surviving device uuid."""
    seen: set[str] = set()
    for _ in range(_TARGET_MAX_HOPS):
        if not device_uuid or device_uuid in seen:
            return None
        seen.add(device_uuid)
        rec = device_lookup(device_uuid)
        if rec is None:
            return None
        retired_into = rec.get("retired_into")
        if retired_into:
            device_uuid = str(retired_into)
            continue
        return ("device", device_uuid)
    return None


def _as_iso_utc(value: str | datetime | None) -> str | None:
    """Normalise a planned/actual end time to an ISO 8601 UTC string, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = dt_util.parse_datetime(str(value))
    if parsed is None:
        return None
    return dt_util.as_utc(parsed).isoformat()


def normalise_name(raw: str) -> str:
    """Normalise candidate name: NFC, collapse whitespace, strip, 1..64 code points, no control chars."""
    if not isinstance(raw, str):
        raise ValueError("Name must be a string")
    nfc = unicodedata.normalize("NFC", raw)
    collapsed = " ".join(nfc.split())
    if not collapsed:
        raise ValueError("Name cannot be empty")
    if len(collapsed) > NAME_MAX:
        raise ValueError(f"Name cannot exceed {NAME_MAX} characters")
    for ch in collapsed:
        if unicodedata.category(ch).startswith("C"):
            raise ValueError("Name cannot contain control characters")
    return collapsed


class ReviewStore:
    """Persistent storage for candidate review decisions (names, merges, dismissals)."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the review store."""
        self._store = Store[dict[str, Any]](
            hass, LEARNING_STORES_VERSION, REVIEW_STORE_KEY
        )
        # Plug sessions live in their own Store (D-09, RESEARCH Pitfall 7): the
        # seven decision writers above save {"decisions": ...} only, so a sibling
        # key in the same file would be erased on the next dismiss/rename/merge.
        self._sessions_store = Store[dict[str, Any]](
            hass, SESSIONS_STORE_VERSION, SESSIONS_STORE_KEY
        )
        self._decisions: dict[str, dict[str, Any]] = {}
        self._sessions: list[dict[str, Any]] = []

    @property
    def decisions(self) -> dict[str, dict[str, Any]]:
        """Return the in-memory mapping of review decisions."""
        return self._decisions

    @property
    def sessions(self) -> list[dict[str, Any]]:
        """Return a copy of the stored plug sessions, oldest first."""
        return [dict(rec) for rec in self._sessions]

    def active_sessions(self, now: datetime) -> list[dict[str, Any]]:
        """Return the sessions that are still open at time ``now`` (D-11).

        A session is active while it has no end_ts and its optional planned end
        date has not passed yet.
        """
        active: list[dict[str, Any]] = []
        for rec in self._sessions:
            if rec.get("end_ts") is not None:
                continue
            planned = rec.get("planned_end_ts")
            if isinstance(planned, str):
                parsed = dt_util.parse_datetime(planned)
                if parsed is not None and parsed <= now:
                    continue
            active.append(dict(rec))
        return active

    async def async_start_session(
        self,
        plug_entity_id: str,
        target_kind: str,
        target_id: str,
        planned_end_ts: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Append a new open session record and save immediately (D-09)."""
        record = {
            "session_id": uuid4().hex,
            "plug_entity_id": str(plug_entity_id),
            "target_kind": str(target_kind),
            "target_id": str(target_id),
            "start_ts": dt_util.utcnow().isoformat(),
            "end_ts": None,
            "planned_end_ts": _as_iso_utc(planned_end_ts),
            "end_reason": None,
        }
        self._sessions.append(record)
        await self._sessions_store.async_save({"sessions": self._sessions})
        return dict(record)

    async def async_end_session(
        self,
        session_id: str,
        reason: str,
        end_ts: str | datetime | None = None,
    ) -> None:
        """Close one open session with a reason, writing only that record (D-09).

        Key-preserving ({**rec}): every other field of the record — including
        the plug, the target and the start time — stays byte-identical, so
        starting a later session for another target never rewrites this one
        (SC3). No-op for an unknown or already ended session.
        """
        ended_at = _as_iso_utc(end_ts) or dt_util.utcnow().isoformat()
        for index, rec in enumerate(self._sessions):
            if rec.get("session_id") != session_id or rec.get("end_ts") is not None:
                continue
            self._sessions[index] = {
                **rec,
                "end_ts": ended_at,
                "end_reason": str(reason),
            }
            await self._sessions_store.async_save({"sessions": self._sessions})
            return

    async def async_load(self) -> dict[str, dict[str, Any]]:
        """Load decisions and sessions from disk into memory."""
        data = await self._store.async_load()
        if data and isinstance(data, dict) and "decisions" in data:
            raw = data.get("decisions")
            if isinstance(raw, dict):
                self._decisions = {str(k): dict(v) for k, v in raw.items()}
            else:
                self._decisions = {}
        else:
            self._decisions = {}

        sessions_data = await self._sessions_store.async_load()
        self._sessions = self._valid_sessions(sessions_data)
        return self._decisions

    @staticmethod
    def _valid_sessions(data: Any) -> list[dict[str, Any]]:
        """Keep only well-typed session records with a parseable start_ts (T-06-14)."""
        if not isinstance(data, dict):
            return []
        raw = data.get("sessions")
        if not isinstance(raw, list):
            return []
        sessions: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                _LOGGER.debug("Dropping a non-dict plug session record")
                continue
            session_id = item.get("session_id")
            plug = item.get("plug_entity_id")
            target_id = item.get("target_id")
            start_ts = item.get("start_ts")
            if (
                not isinstance(session_id, str)
                or not session_id
                or not isinstance(plug, str)
                or not plug
                or item.get("target_kind") not in ("candidate", "device")
                or not isinstance(target_id, str)
                or not target_id
                or not isinstance(start_ts, str)
                or dt_util.parse_datetime(start_ts) is None
            ):
                _LOGGER.debug("Dropping a malformed plug session record")
                continue
            sessions.append(
                {
                    "session_id": session_id,
                    "plug_entity_id": plug,
                    "target_kind": str(item["target_kind"]),
                    "target_id": target_id,
                    "start_ts": start_ts,
                    "end_ts": (
                        item["end_ts"] if isinstance(item.get("end_ts"), str) else None
                    ),
                    "planned_end_ts": (
                        item["planned_end_ts"]
                        if isinstance(item.get("planned_end_ts"), str)
                        else None
                    ),
                    "end_reason": (
                        item["end_reason"]
                        if isinstance(item.get("end_reason"), str)
                        else None
                    ),
                }
            )
        return sessions

    async def async_dismiss(
        self,
        sig_id: str,
        fingerprint: Mapping[str, Any] | None,
        *,
        label: str | None = None,
    ) -> None:
        """Mark a candidate as dismissed and save immediately.

        Key-preserving ({**rec}): unknown keys (e.g. a future device link) survive
        the rewrite. When label is given it is stored as the display label shown
        at decision time (D-18).
        """
        rec = self._decisions.get(sig_id, {})
        self._decisions[sig_id] = {
            **rec,
            "status": "dismissed",
            "name": rec.get("name"),
            "merged_into": rec.get("merged_into"),
            "aliases": list(rec.get("aliases") or []),
            "label": label if label is not None else rec.get("label"),
            "decided_at": dt_util.utcnow().isoformat(),
            "fingerprint": dict(fingerprint or {}),
        }
        await self._store.async_save({"decisions": self._decisions})

    async def async_set_name(
        self,
        sig_id: str,
        name: str,
        fingerprint: Mapping[str, Any] | None,
        *,
        label: str | None = None,
    ) -> None:
        """Mark a candidate as named and save immediately (key-preserving, D-18 label)."""
        rec = self._decisions.get(sig_id, {})
        self._decisions[sig_id] = {
            **rec,
            "status": "named",
            "name": name,
            "merged_into": rec.get("merged_into"),
            "aliases": list(rec.get("aliases") or []),
            "label": label if label is not None else rec.get("label"),
            "decided_at": dt_util.utcnow().isoformat(),
            "fingerprint": dict(fingerprint or {}),
        }
        await self._store.async_save({"decisions": self._decisions})

    async def async_set_device_link(
        self,
        sig_id: str,
        *,
        name: str,
        fingerprint: Mapping[str, Any] | None,
        label: str | None,
        device_uuid: str,
        subentry_id: str | None,
    ) -> None:
        """Link a review record to its confirmed device and save immediately.

        Key-preserving ({**rec}): unknown keys survive the rewrite. The status
        enum stays kept|named|dismissed|merged (D-01) — a confirmed candidate's
        record becomes "named", carrying the device link fields. The link is
        permanent: once subentry_id is set, a repeated call keeps the stored
        pre_confirm_status (Plan 04-05 restores it on un-confirm, D-05).
        """
        rec = self._decisions.get(sig_id, {})
        if rec.get("subentry_id"):
            # Already linked: never overwrite the captured pre-confirm status.
            pre_confirm_status = rec.get("pre_confirm_status", "named")
        else:
            pre_confirm_status = "kept" if rec.get("status") == "kept" else "named"
        self._decisions[sig_id] = {
            **rec,
            "status": "named",
            "name": name,
            "merged_into": rec.get("merged_into"),
            "aliases": list(rec.get("aliases") or []),
            "label": label if label is not None else rec.get("label"),
            "fingerprint": dict(fingerprint or {}),
            "device_uuid": device_uuid,
            "subentry_id": subentry_id,
            "pre_confirm_status": pre_confirm_status,
            "decided_at": dt_util.utcnow().isoformat(),
        }
        await self._store.async_save({"decisions": self._decisions})

    async def async_clear_device_link(self, sig_id: str) -> None:
        """Clear the device link on un-confirm (D-05) and save immediately.

        Key-preserving ({**rec}): the review record keeps its name and
        device_uuid; status returns to the pre-confirm status stored at link
        time (a candidate that was kept returns as kept, a named one as
        named) and subentry_id is cleared. Status is left as is when the
        record has no pre_confirm_status. No-op without a record.
        """
        rec = self._decisions.get(sig_id)
        if rec is None:
            return
        status = rec.get("pre_confirm_status") or rec.get("status")
        self._decisions[sig_id] = {
            **rec,
            "status": status,
            "subentry_id": None,
        }
        await self._store.async_save({"decisions": self._decisions})

    async def async_rename_linked(self, sig_id: str, name: str) -> None:
        """Rename a device-linked review record, keeping its status (D-05)."""
        rec = self._decisions.get(sig_id)
        if rec is None:
            return
        self._decisions[sig_id] = {**rec, "name": name}
        await self._store.async_save({"decisions": self._decisions})

    async def async_update_fingerprint(
        self, sig_id: str, fingerprint: Mapping[str, Any] | None
    ) -> None:
        """Update the stored fingerprint of a device-linked record (D-11, D-25).

        Key-preserving ({**rec}): version activation (accept or rollback)
        rewrites the review record's fingerprint to the activated artefact's
        fingerprint so a later run's identity match compares against the
        active version. No-op without a record; saves immediately.
        """
        rec = self._decisions.get(sig_id)
        if rec is None:
            return
        self._decisions[sig_id] = {**rec, "fingerprint": dict(fingerprint or {})}
        await self._store.async_save({"decisions": self._decisions})

    async def async_merge(
        self,
        target_id: str,
        target_fingerprint: Mapping[str, Any] | None,
        sources: list[tuple[str, Mapping[str, Any] | None]],
        *,
        label: str | None = None,
    ) -> None:
        """Merge source candidates into a target candidate and save immediately.

        Key-preserving ({**rec}): unknown keys survive on both target and source
        records. When label is given it is stored on the target record (D-18).
        """
        now = dt_util.utcnow().isoformat()
        t_rec = self._decisions.get(target_id, {})
        aliases = list(t_rec.get("aliases") or [])
        for s_id, _ in sources:
            if s_id not in aliases:
                aliases.append(s_id)
        t_status = "named" if t_rec.get("status") == "named" else "kept"
        self._decisions[target_id] = {
            **t_rec,
            "status": t_status,
            "name": t_rec.get("name"),
            "merged_into": None,
            "aliases": aliases,
            "label": label if label is not None else t_rec.get("label"),
            "decided_at": now,
            "fingerprint": dict(target_fingerprint or {}),
        }
        for s_id, s_fp in sources:
            s_rec = self._decisions.get(s_id, {})
            self._decisions[s_id] = {
                **s_rec,
                "status": "merged",
                "name": s_rec.get("name"),
                "merged_into": target_id,
                "aliases": list(s_rec.get("aliases") or []),
                "decided_at": now,
                "fingerprint": dict(s_fp or {}),
            }
        await self._store.async_save({"decisions": self._decisions})
