"""Holdout validation manager (Plan 05-01): the shadow and validation Stores and run_validation.

The manager owns two Stores: powerguesser.shadow (the stored shadow set, D-01
contract; Plan 05-03 writes it) and powerguesser.validation ({"report":
latest report or None, "plug_classes": {entity_id: class}}).

Cleanup contract: the validate stage directory <learning root>/validation is
transient scratch space. run_validation recreates it (parents, exist_ok) on
every run and overwrites its job and report files; the next learning run's
sibling cleanup (learning.py) deletes it on purpose, which loses nothing
because the latest report lives in the powerguesser.validation Store; the
single job slot keeps a learning run from deleting it while a validation is
using it; entry removal deletes it with the learning root, and nothing in it
is read after the run returns.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import (
    BUFFER_DIR,
    DOMAIN,
    LEARNING_DIR,
    SHADOW_STORE_KEY,
    VALIDATION_DIR,
    VALIDATION_STORE_KEY,
    VALIDATION_STORES_VERSION,
)
from .engine.buffer import KIND_METERED, KIND_SESSION
from .engine.validation.constants import (
    HISTORY_MAX,
    REPORT_SCHEMA_VERSION,
    SHADOW_MAX_AGE_DAYS,
    SHADOW_SCHEMA_VERSION,
)
from .engine.validation.drift import (
    clean_holdout_key_history,
    evaluate_holdout_drift,
    holdout_state_to_dict,
)
from .learning import LearningStageError
from .review_store import resolve_session_target


class ValidationManager:
    """Owns the shadow and validation Stores and drives the validate stage."""

    def __init__(self, hass: HomeAssistant, entry: Any, hub: Any) -> None:
        """Initialize the manager; async_load reads both Stores defensively."""
        self.hass = hass
        self.entry = entry
        self.hub = hub
        self._shadow_store = Store[dict[str, Any]](
            hass, VALIDATION_STORES_VERSION, SHADOW_STORE_KEY
        )
        self._validation_store = Store[dict[str, Any]](
            hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
        )
        self._shadow: dict[str, Any] | None = None
        self._report: dict[str, Any] | None = None
        self._plug_classes: dict[str, str] = {}
        # Plan 05-06: per-plug and per-class ("class:<class>") holdout history
        # and the D-15 holdout-drift states, both keyed alike. History entries
        # {shadow_run_id, holdout_end, energy_error_pct} are compared only
        # within one shadow set (D-17).
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._holdout_drift: dict[str, dict[str, Any]] = {}

    @property
    def shadow_set(self) -> dict[str, Any] | None:
        """Return the stored shadow set, or None when none is stored."""
        return self._shadow

    @property
    def latest_report(self) -> dict[str, Any] | None:
        """Return the latest validation report kept in the validation Store."""
        return self._report

    async def async_load(self) -> None:
        """Load both Stores defensively (T-05-02).

        A non-dict loads as nothing, and a shadow set whose schema_version
        differs from SHADOW_SCHEMA_VERSION loads as absent, so an unreadable
        or future set is refused (no_shadow_set), never guessed.
        """
        shadow = await self._shadow_store.async_load()
        if isinstance(shadow, dict) and shadow.get("schema_version") == SHADOW_SCHEMA_VERSION:
            self._shadow = dict(shadow)
        else:
            self._shadow = None

        data = await self._validation_store.async_load()
        if isinstance(data, dict):
            report = data.get("report")
            self._report = dict(report) if isinstance(report, dict) else None
            classes = data.get("plug_classes")
            self._plug_classes = (
                {str(k): str(v) for k, v in classes.items()}
                if isinstance(classes, dict)
                else {}
            )
            # Plan 05-06: defensive load (T-05-21). Malformed history entries
            # (missing shadow_run_id, holdout_end or a numeric error) are
            # dropped, per-key lists are trimmed to HISTORY_MAX, malformed
            # states and non-string keys never load — for plug keys and
            # "class:" keys alike.
            raw_history = data.get("history")
            self._history = {}
            if isinstance(raw_history, dict):
                for key, entries in raw_history.items():
                    if not isinstance(key, str) or not key:
                        continue
                    clean = clean_holdout_key_history(entries)
                    if clean:
                        self._history[key] = clean
            raw_drift = data.get("holdout_drift")
            self._holdout_drift = {}
            if isinstance(raw_drift, dict):
                for key, state in raw_drift.items():
                    if not isinstance(key, str) or not isinstance(state, dict):
                        continue
                    shadow_run_id = state.get("shadow_run_id")
                    if not isinstance(shadow_run_id, str) or not shadow_run_id:
                        continue
                    self._holdout_drift[key] = dict(state)
        else:
            self._report = None
            self._plug_classes = {}
            self._history = {}
            self._holdout_drift = {}
        # Plan 05-04: the latest report (ok or refused) shapes every device's
        # class-holdout view; devices loaded first, so this is a plain call.
        self._apply_to_devices(self.latest_report)

    async def async_store_shadow(self, shadow: dict[str, Any]) -> None:
        """Save a shadow set and keep it in memory (Plan 05-03's writer)."""
        self._shadow = dict(shadow)
        await self._shadow_store.async_save(self._shadow)

    def shadow_due(
        self,
        eligible: Sequence[str],
        grid_s: int,
        enabled_event_types: Sequence[str],
        now: datetime,
    ) -> str | None:
        """Return why a shadow pass is due, or None when it is not (D-04).

        "first_run" without a stored set; "stale" when the stored set is
        older than SHADOW_MAX_AGE_DAYS (strictly, matching the validate
        stage's shadow_stale flag); "eligible_changed" when the sorted
        eligible plug set differs from the stored one; "grid_changed" when
        grid_s differs; "event_types_changed" when the sorted enabled event
        types differ (order-insensitive, D-26 (1)). The grid and event-types
        checks keep validation from refusing grid_mismatch or
        event_types_mismatch for up to 30 days after such a change: a
        refused set is refreshed by the next learning run.
        """
        stored = self._shadow
        if stored is None:
            return "first_run"
        created_raw = stored.get("created")
        try:
            created = datetime.fromisoformat(str(created_raw))
        except (TypeError, ValueError):
            created = None
        if created is not None:
            if created.tzinfo is None:
                created = created.replace(tzinfo=dt_util.UTC)
            age_days = (now - created).total_seconds() / 86_400.0
            if age_days > SHADOW_MAX_AGE_DAYS:
                return "stale"
        if sorted(str(e) for e in eligible) != sorted(
            str(e) for e in stored.get("eligible", [])
        ):
            return "eligible_changed"
        if int(grid_s) != int(stored.get("grid_s", grid_s)):
            return "grid_changed"
        if sorted(str(t) for t in enabled_event_types) != sorted(
            str(t) for t in stored.get("enabled_event_types", [])
        ):
            return "event_types_changed"
        return None

    async def async_run_validation(
        self, trigger: str = "service", *, synthetic_overlap: bool = False
    ) -> dict[str, Any]:
        """Replay the stored shadow set over the holdout block and return the report.

        With no stored shadow set (or one with a foreign schema_version, which
        async_load already treats as absent) the report is {status refused,
        reason no_shadow_set}, carries no number, and is stored as the latest
        report (D-12, A15). The stage runs through the shared single job slot
        (async_run_exclusive_stage), which raises already_running while another
        PowerGuesser job holds it; a worker failure raises HomeAssistantError
        validation_failed and keeps the previous report.

        ``synthetic_overlap`` (D-28, Plan 06-08) rides in the job and makes the
        validate stage add its labelled, purely synthetic ``overlap_synthetic``
        block; it never touches the real figures.

        Plan 06-09 (REV-04, D-13): when device-target plug sessions exist, a
        validate_sessions stage runs after the validate stage (or alone when the
        shadow part is refused) and its verdicts are attached as the report's
        separate ``sessions`` list. With no device-target session the
        no_shadow_set branch is byte-for-byte as before: no worker runs and the
        report carries no sessions key.
        """
        stage_dir = Path(self.hass.config.path(BUFFER_DIR, LEARNING_DIR, VALIDATION_DIR))

        if self._shadow is None:
            report = {"status": "refused", "reason": "no_shadow_set"}
            self._stamp_report(report, trigger)
            # Only when device-target sessions exist does this branch run the
            # validate_sessions stage alone (no shadow, no validate) and attach
            # its sessions to the still-refused report.
            await self._maybe_validate_sessions(stage_dir, report)
            self._report = report
            # A refusal, including this one, clears every class result: a
            # no-shadow-set report never validates a class (T-05-11). The
            # coverage list still names every confirmed device, unvalidated.
            self._apply_to_devices(report)
            report["confirmed_devices"] = self._coverage_list()
            await self._save_validation_store()
            return report

        validation_time_ms = int(round(dt_util.utcnow().timestamp() * 1000))

        def _prepare_job() -> Path:
            stage_dir.mkdir(parents=True, exist_ok=True)
            job = dict(self.hub.learning.build_window_job(stage_dir))
            job["stage_dir"] = str(stage_dir)
            job["shadow"] = self._shadow
            job["plug_classes"] = dict(self._plug_classes)
            job["validation_time_ms"] = validation_time_ms
            job["synthetic_overlap"] = bool(synthetic_overlap)
            job_path = stage_dir / "job_validate.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            return job_path

        job_path = await self.hass.async_add_executor_job(_prepare_job)
        try:
            await self.hub.learning.async_run_exclusive_stage("validate", job_path)
        except LearningStageError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="validation_failed",
                translation_placeholders={"error": f"{err}"[:200]},
            ) from err

        def _read_report() -> dict[str, Any]:
            with open(stage_dir / "report.json", "r", encoding="utf-8") as f:
                return json.load(f)

        report = await self.hass.async_add_executor_job(_read_report)
        self._stamp_report(report, trigger)
        # Plan 06-09 (REV-04, D-13): run the session validation stage and attach
        # its separate sessions list, on an ok and a refused report alike. No
        # device-target session leaves the report byte-identical.
        await self._maybe_validate_sessions(stage_dir, report)
        # Plan 05-06 (D-15, D-23): after an ok report, upsert every figure-
        # bearing plug error and every numeric class median into the holdout
        # history under the run's shadow set, evaluate the D-15 rule and
        # write drift, drift_reason, drift_since and best_error_pct into the
        # report's devices/classes entries — leaving every other field
        # (including unvalidated, a class's passes, peers and median)
        # unchanged. Refused reports touch neither history nor states. This
        # runs before the report is stored and before apply_validation and
        # the confirmed_devices coverage list read it, so devices see the
        # enriched class entries.
        if report.get("status") == "ok":
            shadow_run_id = self._shadow.get("run_id") if self._shadow else None
            if isinstance(shadow_run_id, str) and shadow_run_id:
                self._enrich_report_drift(report, shadow_run_id)
        # D-06 hysteresis memory: only an ok report's plug_classes merge into
        # the stored map; a refused report, whatever keys it carries, and the
        # no_shadow_set refusal never change it. Keep-forever semantics: keys
        # are never pruned — a plug that leaves the eligible set keeps its
        # class so a transient gap (briefly unavailable plug, an
        # eligible_changed shadow refresh) cannot reset its D-06 band
        # hysteresis; the map is small (one short class name per plug ever
        # validated), so unbounded growth is not a concern.
        if report.get("status") == "ok" and isinstance(
            report.get("plug_classes"), Mapping
        ):
            self._plug_classes.update(
                {str(k): str(v) for k, v in report["plug_classes"].items()}
            )
        self._report = report
        # Plan 05-04: refused reports included, the new latest report shapes
        # every device's class-holdout view and republishes the snapshot, so
        # the device sensors show the new values without an aligner publish.
        self._apply_to_devices(report)
        # VAL-03 coverage: the report names every confirmed device with its
        # class, basis and unvalidated flag, sorted by name.
        report["confirmed_devices"] = self._coverage_list()
        await self._save_validation_store()
        return report

    # -- Device-target plug sessions (Plan 06-09; REV-04, D-13) ------------

    def _device_target_sessions(self) -> list[dict[str, Any]]:
        """Build the device-target session list for the validate_sessions stage.

        Runs in the executor (reads the buffer channels table). A session is a
        device target only when its resolved target is a confirmed device with a
        usable active artefact; the job carries the device's frozen active
        artefact and version, its session plug's buffer channel and the session
        window (end_ts, or now for an open session). Candidate targets and
        unresolvable sessions are skipped: a candidate session shapes learning
        (Task 2), it never validates a device.
        """
        review = getattr(self.hub.learning, "review", None)
        devices = getattr(self.hub, "devices", None)
        if review is None or devices is None:
            return []
        decisions = review.decisions

        channel_rows = self.hub.writer.db.channel_rows()
        session_ch: dict[str, int] = {}
        for ch, entity_id, _role, kind in channel_rows:
            if kind == KIND_SESSION:
                session_ch[str(entity_id)] = int(ch)

        now_ms = int(round(dt_util.utcnow().timestamp() * 1000))
        out: list[dict[str, Any]] = []
        for record in review.sessions:
            resolved = resolve_session_target(record, decisions, devices.record)
            if resolved is None or resolved[0] != "device":
                continue
            device_uuid = str(resolved[1])
            dev = devices.record(device_uuid)
            if dev is None or dev.get("status") != "confirmed":
                continue
            artefact, active_v = devices.active_artefact_and_version(device_uuid)
            if artefact is None:
                continue
            plug_channel = session_ch.get(str(record.get("plug_entity_id")))
            if plug_channel is None:
                continue
            start_ms = _parse_iso_ms(record.get("start_ts"))
            end_ms = _parse_iso_ms(record.get("end_ts")) or now_ms
            if start_ms is None or end_ms <= start_ms:
                continue
            out.append(
                {
                    "session_id": str(record.get("session_id") or ""),
                    "device_uuid": device_uuid,
                    "artefact": dict(artefact),
                    "artefact_v": int(active_v),
                    "plug_channel": int(plug_channel),
                    "window_start_ms": int(start_ms),
                    "window_end_ms": int(end_ms),
                }
            )
        return out

    async def _maybe_validate_sessions(
        self, stage_dir: Path, report: dict[str, Any]
    ) -> None:
        """Run the validate_sessions stage when device-target sessions exist.

        With none, nothing runs and the report is untouched (no sessions key).
        With at least one, the stage runs through the shared single job slot and
        its verdicts are Read, stamped with the device name and ISO window
        times, and attached as ``report["sessions"]`` on an ok or a refused
        report alike (D-13).
        """
        def _prepare_sessions_job() -> Path | None:
            sessions = self._device_target_sessions()
            if not sessions:
                return None
            stage_dir.mkdir(parents=True, exist_ok=True)
            window = dict(self.hub.learning.build_window_job(stage_dir))
            channel_rows = self.hub.writer.db.channel_rows()
            metered_entity_ids = getattr(self.hub.model, "metered_entity_ids", set())
            metered_channels = [
                int(ch)
                for ch, entity_id, _role, kind in channel_rows
                if kind == KIND_METERED and entity_id in metered_entity_ids
            ]
            others = [
                dict(artefact) for artefact in self.hub.devices.gate_actives().values()
            ]
            job = {
                "buffer_path": window["buffer_path"],
                "stage_dir": str(stage_dir),
                "grid_s": window["grid_s"],
                "day_bounds_ms": window["day_bounds_ms"],
                "enabled_event_types": window["enabled_event_types"],
                "phase_channels": window["phase_channels"],
                "metered_channels": metered_channels,
                "others": others,
                "sessions": sessions,
            }
            job_path = stage_dir / "job_validate_sessions.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            return job_path

        job_path = await self.hass.async_add_executor_job(_prepare_sessions_job)
        if job_path is None:
            return
        try:
            await self.hub.learning.async_run_exclusive_stage(
                "validate_sessions", job_path
            )
        except LearningStageError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="validation_failed",
                translation_placeholders={"error": f"{err}"[:200]},
            ) from err

        def _read_sessions() -> list[dict[str, Any]]:
            with open(stage_dir / "sessions.json", "r", encoding="utf-8") as f:
                entries = json.load(f)
            return entries if isinstance(entries, list) else []

        entries = await self.hass.async_add_executor_job(_read_sessions)
        self._stamp_sessions(entries)
        report["sessions"] = entries

    def _stamp_sessions(self, entries: list[dict[str, Any]]) -> None:
        """Add the device name to each session entry (ISO times are in the entry)."""
        devices = getattr(self.hub, "devices", None)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            device_uuid = str(entry.get("device_uuid") or "")
            record = devices.record(device_uuid) if devices is not None else None
            entry["name"] = str(
                (record or {}).get("name") or device_uuid
            )

    def _coverage_list(self) -> list[dict[str, Any]]:
        """Return the confirmed-devices coverage list from the device runtime."""
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return []
        return devices.coverage_summary()

    # -- Holdout-basis drift enrichment (Plan 05-06; D-15, D-23) -----------

    def _update_holdout_drift(
        self,
        key: str,
        shadow_run_id: str,
        holdout_end: str,
        error_pct: float,
    ) -> dict[str, Any]:
        """Upsert one history entry, evaluate the D-15 rule and store the state.

        One shared helper serves plug keys (entity ids) and "class:<class>"
        # keys (D-23), so the two kinds cannot diverge. The upsert key is
        (shadow_run_id, holdout_end): a re-run on the same window of the same
        shadow set replaces the entry instead of counting as a new evaluation
        (RESEARCH Pitfall 6). The list is trimmed to HISTORY_MAX by dropping
        the oldest holdout_end across all shadow sets (earlier sets' entries
        stay within the bound for display only, D-17).
        """
        entries = self._history.setdefault(key, [])
        entries[:] = [
            entry
            for entry in entries
            if not (
                entry.get("shadow_run_id") == shadow_run_id
                and entry.get("holdout_end") == holdout_end
            )
        ]
        entries.append(
            {
                "shadow_run_id": shadow_run_id,
                "holdout_end": holdout_end,
                "energy_error_pct": float(error_pct),
            }
        )
        entries.sort(key=lambda e: e["holdout_end"])
        if len(entries) > HISTORY_MAX:
            del entries[: len(entries) - HISTORY_MAX]
        state = evaluate_holdout_drift(
            entries, self._holdout_drift.get(key), shadow_run_id
        )
        self._holdout_drift[key] = holdout_state_to_dict(state)
        return state

    def _enrich_report_drift(self, report: dict[str, Any], shadow_run_id: str) -> None:
        """Write the holdout-drift fields into an ok report's entries.

        For every devices entry with a numeric energy_error_pct the error is
        tracked under the plug's entity id; for every classes entry with a
        numeric median_error_pct the median is tracked under "class:<class>"
        (D-23, tracked whether or not the class passes so its best recorded
        value exists when it starts passing). Only drift, drift_reason,
        drift_since and best_error_pct are written; every other entry field
        is returned and stored unchanged (D-16 ties unvalidated to the
        D-07/D-08 pass, drift only sets its own flags).
        """
        holdout = report.get("holdout")
        end_ms = holdout.get("end_ms") if isinstance(holdout, Mapping) else None
        if not isinstance(end_ms, (int, float)) or isinstance(end_ms, bool):
            return
        holdout_end = (
            dt_util.as_local(datetime.fromtimestamp(end_ms / 1000.0, tz=dt_util.UTC))
            .date()
            .isoformat()
        )

        def _numeric(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        for entry in report.get("devices") or []:
            if not isinstance(entry, dict):
                continue
            error = _numeric(entry.get("energy_error_pct"))
            key = str(entry.get("entity_id") or "")
            if error is None or not key:
                continue
            state = self._update_holdout_drift(key, shadow_run_id, holdout_end, error)
            entry["drift"] = bool(state["drift"])
            entry["drift_reason"] = state["reason"]
            entry["drift_since"] = state["since"]
            entry["best_error_pct"] = state["best_error_pct"]
        for entry in report.get("classes") or []:
            if not isinstance(entry, dict):
                continue
            error = _numeric(entry.get("median_error_pct"))
            klass = entry.get("class")
            if error is None or not klass:
                continue
            state = self._update_holdout_drift(
                f"class:{klass}", shadow_run_id, holdout_end, error
            )
            entry["drift"] = bool(state["drift"])
            entry["drift_reason"] = state["reason"]
            entry["drift_since"] = state["since"]
            entry["best_error_pct"] = state["best_error_pct"]

    def _apply_to_devices(self, report: Mapping[str, Any] | None) -> None:
        """Push the latest report into the device runtime (no-op before devices load)."""
        devices = getattr(self.hub, "devices", None)
        if devices is not None:
            devices.apply_validation(report)

    def _stamp_report(self, report: dict[str, Any], trigger: str) -> None:
        """Stamp schema_version, generated_at, trigger, ISO cutoffs and names."""
        report["schema_version"] = REPORT_SCHEMA_VERSION
        report["generated_at"] = dt_util.utcnow().isoformat()
        report["trigger"] = trigger
        train_cutoff_ms = report.get("train_cutoff_ms")
        if isinstance(train_cutoff_ms, int | float):
            report["train_cutoff"] = datetime.fromtimestamp(
                train_cutoff_ms / 1000.0, tz=dt_util.UTC
            ).isoformat()
        holdout = report.get("holdout")
        if isinstance(holdout, dict):
            start_ms = holdout.get("start_ms")
            end_ms = holdout.get("end_ms")
            if isinstance(start_ms, int | float):
                holdout["start"] = datetime.fromtimestamp(
                    start_ms / 1000.0, tz=dt_util.UTC
                ).isoformat()
            if isinstance(end_ms, int | float):
                holdout["end"] = datetime.fromtimestamp(
                    end_ms / 1000.0, tz=dt_util.UTC
                ).isoformat()
            days = holdout.get("days")
            if isinstance(days, list):
                for day in days:
                    if not isinstance(day, dict):
                        continue
                    day_start_ms = day.get("day_start_ms")
                    if isinstance(day_start_ms, int | float):
                        day["date"] = (
                            dt_util.as_local(
                                datetime.fromtimestamp(
                                    day_start_ms / 1000.0, tz=dt_util.UTC
                                )
                            )
                            .date()
                            .isoformat()
                        )
        for collection in ("devices", "refusals", "not_discovered"):
            entries = report.get(collection)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "name" not in entry:
                    entity_id = str(entry.get("entity_id", ""))
                    state = self.hass.states.get(entity_id) if entity_id else None
                    entry["name"] = state.name if state is not None else entity_id

    async def _save_validation_store(self) -> None:
        """Persist the latest report next to the stored plug_classes map."""
        await self._validation_store.async_save(
            {
                "report": self._report,
                "plug_classes": self._plug_classes,
                "history": self._history,
                "holdout_drift": self._holdout_drift,
            }
        )


async def async_remove_validation_data(hass: HomeAssistant) -> None:
    """Remove the powerguesser.shadow and powerguesser.validation Stores."""
    await Store(hass, VALIDATION_STORES_VERSION, SHADOW_STORE_KEY).async_remove()
    await Store(hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY).async_remove()


def _parse_iso_ms(value: Any) -> int | None:
    """Parse an ISO 8601 timestamp to a millisecond epoch, or None when unusable."""
    if not isinstance(value, str) or not value:
        return None
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        return None
    return int(round(parsed.timestamp() * 1000))
