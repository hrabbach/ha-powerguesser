"""Learning supervisor and worker subprocess orchestration for PowerGuesser."""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import copy
import dataclasses
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid

import asyncio

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import (
    async_call_later,
    async_track_time_change,
    async_track_time_interval,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
import homeassistant.util.dt as dt_util

from .consent import ConsentStore, consent_valid, format_origin
from .review_store import ReviewStore, normalise_name, resolve_session_target
from .const import (
    BUFFER_DIR,
    BUFFER_FILE,
    CANDIDATES_STORE_KEY,
    CANDIDATE_DECORATION_KEYS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROBE,
    CONF_MAX_TOKENS,
    CONF_MONTH_COST_LIMIT,
    CONF_MONTH_INPUT_TOKENS,
    CONF_MONTH_OUTPUT_TOKENS,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    CONF_PRICE_INPUT,
    CONF_PRICE_OUTPUT,
    CONF_RUN_INPUT_TOKENS,
    CONF_RUN_OUTPUT_TOKENS,
    CONF_SCHEDULE,
    CONF_SCHEDULE_TIME,
    CONF_SCHEDULE_WEEKDAY,
    CONF_SHARE_NAMES,
    CONF_TRUNCATION_GUARD,
    CONF_VALIDATE_AFTER_LEARNING,
    CONSENT_STORE_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MONTH_INPUT_TOKENS,
    DEFAULT_MONTH_OUTPUT_TOKENS,
    DEFAULT_RUN_INPUT_TOKENS,
    DEFAULT_RUN_OUTPUT_TOKENS,
    DEFAULT_SCHEDULE,
    DEFAULT_SCHEDULE_TIME,
    DEFAULT_SCHEDULE_WEEKDAY,
    DEFAULT_VALIDATE_AFTER_LEARNING,
    DOMAIN,
    EVIDENCE_MAX_FAILURES,
    EVIDENCE_REFRESH_AGE,
    ISSUE_LEARNING_CAPPED,
    ISSUE_LEARNING_CONSENT,
    ISSUE_LLM_AUTH,
    LEARNING_DIR,
    LEARNING_STORE_KEY,
    LEARNING_STORES_VERSION,
    MAX_REFINE_ROUNDS,
    MAX_REPAIR_TURNS,
    MIN_LEARNING_DAYS,
    PREPARE_DEBOUNCE_S,
    PREPARE_DIR,
    PREPARE_STARTUP_DELAY_S,
    REPAIR_OUTPUT_MAX_CHARS,
    REVIEW_STORE_KEY,
    SESSIONS_STORE_KEY,
    SESSIONS_STORE_VERSION,
    STAGE_TIMEOUT_S,
    USABLE_DAYS_REFRESH,
    VALIDATED_DAYS,
    WEEKDAYS,
)
from .engine.buffer import KIND_METERED, KIND_PHASE, KIND_SESSION, RETENTION_DAYS
from .engine.coverage import MS_PER_DAY
from .engine.intervals import gate
from .engine.learning import prompts, proposal
from .engine.learning.artefact import SignatureArtefact
from .engine.learning.budget import (
    BYTES_PER_TOKEN,
    CAP_ORDER,
    Caps,
    Usage,
    check_request,
    estimate_tokens,
    month_key,
    output_estimate,
    request_cost,
    truncation_suspected,
    update_ratio,
)
from .engine.learning.evidence import EVIDENCE_VERSION
from .engine.learning.prompts import payload_bytes
from .engine.learning.worker import SESSION_KEY_BASE, peak_memory_text
from .engine.llm_wire import extract_json, model_matches
from .engine.validation.constants import (
    MIN_USABLE_DAYS,
    SHADOW_HOLDOUT_DAYS,
    SHADOW_SCHEMA_VERSION,
)
from .llm_client import (
    LlmAuthError,
    LlmBadResponse,
    LlmBudgetError,
    LlmClient,
    LlmConnectError,
    LlmHttpError,
    LlmTimeoutError,
    endpoint_location,
    endpoint_origin,
    endpoint_secrets,
    redact_secrets,
    secret_text_forms,
)

_LOGGER = logging.getLogger(__name__)

WORKER_ENTRY = Path(__file__).parent / "engine" / "worker_entry.py"
IONICE = shutil.which("ionice")


class LearningStageError(Exception):
    """Raised when a learning worker stage fails or times out."""


class LearningRunError(Exception):
    """Raised when a learning run fails with a known error code."""

    def __init__(self, code: str) -> None:
        """Initialize learning run error."""
        super().__init__(code)
        self.code = code


class LearningCapped(Exception):
    """Raised when a request would exceed a token cap or cost limit."""

    def __init__(self, cap: str, limit: Any) -> None:
        """Initialize learning capped exception."""
        super().__init__(f"cap_reached: {cap}")
        self.cap = cap
        self.limit = limit


class CandidateGone(Exception):
    """Raised when a candidate or merge source is no longer visible."""


# Keys each stage must report in its worker `done` event (worker.stage_outputs).
_REQUIRED_STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "compile_score": ("S", "conflict_share", "accepted"),
    "rescore": ("S", "conflict_share", "per_signature"),
    "refit": ("result", "accepted", "current_S", "new_S"),
    "evidence": ("result",),
}


def _require_outputs(stage: str, outputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise LearningStageError when a stage result lacks a key the supervisor relies on."""
    missing = [k for k in _REQUIRED_STAGE_OUTPUTS.get(stage, ()) if k not in outputs]
    if missing:
        raise LearningStageError(f"{stage}: worker result lacks {', '.join(missing)}")
    return outputs


def _parse_iso_ms(value: Any) -> int | None:
    """Parse an ISO 8601 timestamp to a millisecond epoch, or None when unusable."""
    if not isinstance(value, str) or not value:
        return None
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        return None
    return int(round(parsed.timestamp() * 1000))


def _evidence_is_aged(evidence: Mapping[str, Any]) -> bool:
    """True when an evidence dict's computed_at is missing, unparsable or stale.

    D-11a: only the 6-hourly refresh checks age; a missing or unparsable
    computed_at counts as aged so the stage recomputes it.
    """
    computed = evidence.get("computed_at")
    if not isinstance(computed, str):
        return True
    parsed = dt_util.parse_datetime(computed)
    if parsed is None:
        return True
    return parsed < dt_util.utcnow() - EVIDENCE_REFRESH_AGE


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Immutable snapshot of configuration taken at learning run start."""

    base_url: str
    api_key: str
    model: str
    profile: dict[str, Any]
    names_flag: bool
    max_tokens: int
    validated: bool
    caps: Caps = dataclasses.field(default_factory=Caps)
    truncation_guard: bool = True


@dataclasses.dataclass
class _RoundsState:
    """Mutable holder for the shared LLM rounds (production and shadow, Plan 05-03).

    A cap hit mid-loop raises LearningCapped and must leave the paid-for
    library readable by the caller: the production LearningCapped handler
    keeps saving best_library, and the shadow pass turns the cap into a skip
    that never costs the production result (D-04, D-21 shadow cap).
    """

    best_library: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    incumbent_round: int = 1
    rounds_run: int = 1
    refine_error: str | None = None
    proposal_empty: bool = False


class LearningSupervisor:
    """Manages background learning runs, worker subprocesses, and candidate storage."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, hub: Any) -> None:
        """Initialize the learning supervisor."""
        self.hass = hass
        self.entry = entry
        self.hub = hub
        self.coordinator = DataUpdateCoordinator[dict[str, Any]](
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN}_learning",
            update_interval=None,
        )
        self.coordinator.data = {
            "status": "idle",
            "running_job": None,
            "rescore_failures": 0,
            "evidence_failures": 0,
            "blocked_reason": None,
            "last_run": None,
            "last_error": None,
            "last_attempt": None,
            "pending_candidates": 0,
            "tokens_run": {"input": 0, "output": 0},
            "model_mismatch_run": None,
            "trimmed_run": 0,
            "month": {"key": None, "input": 0, "output": 0, "cost": None},
            "usable_days": {"value": 0.0, "computed_at": None},
        }
        self._candidates_store = Store[dict[str, Any]](
            hass, LEARNING_STORES_VERSION, CANDIDATES_STORE_KEY
        )
        self._learning_store = Store[dict[str, Any]](
            hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY
        )
        self.consent_store = ConsentStore(hass)
        self.consent = self.consent_store
        self.review = ReviewStore(hass)
        self.candidates: list[dict[str, Any]] = []
        self._candidates_record: dict[str, Any] | None = None
        self._rescore_failures: int = 0
        self._evidence_failures: int = 0
        # D-07a/D-13: device evidence lives here, keyed by device uuid, never in
        # a device record (whose schema is pinned and whose versions D-13 keeps
        # evidence-free).
        self._device_evidence: dict[str, dict[str, Any]] = {}
        self._calibration: dict[str, dict[str, Any]] = {}
        self.run_task: asyncio.Task[Any] | None = None
        # Set synchronously while a run or rescore start is between its slot check and
        # the task assignment, which spans awaits (WR-09)
        self._start_pending = False
        # Set by async_stop(): follow-up job starts (refit/rescore) must not begin a new
        # background task while teardown is in flight (WR-01).
        self._stopped = False
        self._proc: asyncio.subprocess.Process | None = None
        self._usable_days_cache: dict[str, Any] | None = None
        self._channel_rows_cache: list[tuple[int, str, str, str]] | None = None
        self._unsub_prepare_timer: Callable[[], None] | None = None
        self._unsub_at_started: Callable[[], None] | None = None
        self._prepare_task: asyncio.Task[None] | None = None
        # Serialises prepares (consent job, preview action) on the shared prepare dir (WR-08)
        self._prepare_lock = asyncio.Lock()
        self._unsub_refresh: Callable[[], None] | None = None
        self._unsub_schedule: Callable[[], None] | None = None

    def _build_learning_store_data(self) -> dict[str, Any]:
        """Build dictionary for persisting learning supervisor state."""
        data = self.coordinator.data or {}
        return {
            "status": data.get("status", "idle"),
            "blocked_reason": data.get("blocked_reason"),
            "last_run": data.get("last_run"),
            "last_error": data.get("last_error"),
            "last_attempt": data.get("last_attempt"),
            "tokens_run": data.get("tokens_run", {"input": 0, "output": 0}),
            "model_mismatch_run": data.get("model_mismatch_run"),
            "trimmed_run": data.get("trimmed_run", 0),
            "month": data.get(
                "month",
                {"key": month_key(dt_util.now()), "input": 0, "output": 0, "cost": None},
            ),
            "usable_days": data.get("usable_days", {"value": 0.0, "computed_at": None}),
            "rescore_failures": getattr(self, "_rescore_failures", 0),
            "evidence_failures": getattr(self, "_evidence_failures", 0),
            "device_evidence": getattr(self, "_device_evidence", {}),
            "calibration": getattr(self, "_calibration", {}),
        }

    @staticmethod
    def _attempt_record(
        started: str | None, outcome: str, ended: str | None
    ) -> dict[str, Any]:
        """Build the persisted last_attempt record (CLEAN-02, D-15).

        outcome is one of running, success, failed, capped, stopped, interrupted.
        """
        return {"started": started, "ended": ended, "outcome": outcome}

    async def async_start(self) -> None:
        """Load candidate and consent stores, start interval timer, and check gates."""
        await self.consent_store.async_load()
        await self.review.async_load()

        stored_learning = await self._learning_store.async_load()
        if stored_learning and isinstance(stored_learning, dict):
            c_data = dict(self.coordinator.data)
            if stored_learning.get("status") == "running":
                stored_learning["status"] = "failed"
                stored_learning["last_error"] = "interrupted"
                await self._learning_store.async_save(stored_learning)

            # D-15a: a lingering running attempt (the restart killed the run) becomes
            # interrupted with its start time kept, so the sensor still shows when it
            # happened. Independent of the status fix-up above.
            stored_attempt = stored_learning.get("last_attempt")
            if (
                isinstance(stored_attempt, dict)
                and stored_attempt.get("outcome") == "running"
            ):
                stored_learning["last_attempt"] = self._attempt_record(
                    stored_attempt.get("started"), "interrupted", None
                )
                await self._learning_store.async_save(stored_learning)

            # A last_error stored before 03-10 may carry the key, plain or JSON-escaped:
            # scrub it before it reaches the sensor attribute again (IN-10).
            stored_error = stored_learning.get("last_error")
            if isinstance(stored_error, str):
                secrets = secret_text_forms(
                    endpoint_secrets(
                        str(self.entry.data.get(CONF_LLM_API_KEY) or ""),
                        str(self.entry.data.get(CONF_LLM_BASE_URL) or ""),
                    )
                )
                scrubbed = redact_secrets(stored_error, secrets)
                if scrubbed != stored_error:
                    stored_learning["last_error"] = scrubbed
                    await self._learning_store.async_save(stored_learning)

            for k in (
                "status",
                "blocked_reason",
                "last_run",
                "last_error",
                "last_attempt",
                "tokens_run",
                "model_mismatch_run",
                "trimmed_run",
                "month",
                "usable_days",
                "rescore_failures",
                "evidence_failures",
                "device_evidence",
            ):
                if k in stored_learning:
                    c_data[k] = stored_learning[k]

            self._rescore_failures = int(c_data.get("rescore_failures", 0))
            self._evidence_failures = int(c_data.get("evidence_failures", 0))
            stored_device_evidence = c_data.get("device_evidence")
            if isinstance(stored_device_evidence, dict):
                self._device_evidence = {
                    str(uuid): dict(entry)
                    for uuid, entry in stored_device_evidence.items()
                    if isinstance(entry, dict)
                }
            if "calibration" in stored_learning and isinstance(
                stored_learning["calibration"], dict
            ):
                self._calibration = dict(stored_learning["calibration"])

            curr_month = month_key(dt_util.now())
            month_dict = c_data.get("month") or {}
            if month_dict.get("key") != curr_month:
                c_data["month"] = {"key": curr_month, "input": 0, "output": 0, "cost": None}
                ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CAPPED)
                if c_data.get("status") == "capped":
                    c_data["status"] = "idle"
                    c_data["last_error"] = None

            self.coordinator.data = c_data

        data = await self._candidates_store.async_load()
        if data and isinstance(data, dict):
            self._candidates_record = data
            self.candidates = data.get("candidates", [])
            data_dict = dict(self.coordinator.data)
            data_dict["pending_candidates"] = len(self.visible_candidates())
            self.coordinator.data = data_dict

        self._unsub_refresh = async_track_time_interval(
            self.hass, self._async_periodic_refresh, USABLE_DAYS_REFRESH
        )
        if self.coordinator.data.get("status") not in ("failed", "capped"):
            await self._async_reevaluate_gates()

        await self._async_check_consent(is_startup=True)
        await self._async_maybe_rescore()
        # A retrain queued while a job was running (or before a restart)
        # resumes here (D-13); single-flight keeps it behind any job above.
        await self._async_maybe_refit()
        # D-10: the evidence refresh runs last, so a due rescore at startup keeps
        # the slot and evidence follows from that rescore's own finally.
        await self._async_maybe_evidence()
        self._async_arm_schedule()

    async def async_stop(self) -> None:
        """Cancel background run and clean up resources."""
        # Latch FIRST, unconditionally: even with no active job, a periodic-refresh
        # callback already in flight could otherwise spawn a stray rescore task
        # during/after teardown (WR-01 re-review).
        self._stopped = True
        self._cancel_prepare_timers()
        if self._prepare_task is not None and not self._prepare_task.done():
            self._prepare_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._prepare_task
            self._prepare_task = None
        if self._unsub_refresh is not None:
            self._unsub_refresh()
            self._unsub_refresh = None
        if self._unsub_schedule is not None:
            self._unsub_schedule()
            self._unsub_schedule = None
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.run_task
            # The cancelled job's finally may have released the slot and started a
            # follow-up job while we were awaiting it: cancel and await that too
            # rather than dropping the reference untracked (WR-01).
            if self.run_task is not None and not self.run_task.done():
                self.run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.run_task
            self.run_task = None

    def _is_run_capped(self) -> bool:
        """Return True when the status is capped by a per-run cap (run_input/run_output).

        Per-run caps reset with every run, so unlike month caps they must not keep
        the schedule or the gate re-evaluation suppressed until the next month (WR-01).
        """
        data = self.coordinator.data or {}
        return data.get("status") == "capped" and str(data.get("last_error") or "") in (
            "cap_reached: run_input",
            "cap_reached: run_output",
        )

    async def async_settings_changed(self) -> None:
        """Handle entry settings or options change."""
        if self._is_run_capped():
            # The user may have raised the per-run cap; a run cap does not outlive the run.
            c_data = dict(self.coordinator.data)
            c_data["status"] = "idle"
            c_data["last_error"] = None
            self.coordinator.data = c_data
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CAPPED)
            await self._learning_store.async_save(self._build_learning_store_data())
        await self._async_reevaluate_gates()
        await self._async_check_consent()
        self._async_arm_schedule()
        # The status sensor shows the stored probe result from entry.data
        self.coordinator.async_update_listeners()

    def _async_arm_schedule(self) -> None:
        """Arm or re-arm weekly learning schedule timer."""
        if self._unsub_schedule is not None:
            self._unsub_schedule()
            self._unsub_schedule = None

        if self.entry.options.get(CONF_SCHEDULE, DEFAULT_SCHEDULE) != "weekly":
            return

        time_str = str(
            self.entry.options.get(CONF_SCHEDULE_TIME, DEFAULT_SCHEDULE_TIME)
        )
        parts = [int(p) for p in time_str.split(":") if p.strip().isdigit()]
        hour = parts[0] if len(parts) > 0 else 3
        minute = parts[1] if len(parts) > 1 else 0
        second = parts[2] if len(parts) > 2 else 0

        self._unsub_schedule = async_track_time_change(
            self.hass,
            self._async_on_schedule,
            hour=hour,
            minute=minute,
            second=second,
        )

    async def _async_on_schedule(self, now: datetime) -> None:
        """Handle weekly scheduled time event."""
        if self.entry.options.get(CONF_SCHEDULE, DEFAULT_SCHEDULE) != "weekly":
            return

        wday_opt = self.entry.options.get(
            CONF_SCHEDULE_WEEKDAY, DEFAULT_SCHEDULE_WEEKDAY
        )
        target_weekday = WEEKDAYS.index(wday_opt) if wday_opt in WEEKDAYS else 0
        if now.weekday() != target_weekday:
            return

        if self._job_active():
            return

        status = self.coordinator.data.get("status")
        if status == "blocked" or (status == "capped" and not self._is_run_capped()):
            return

        try:
            await self.async_request_run("schedule")
        except ServiceValidationError as err:
            _LOGGER.info("Scheduled learning run skipped by gate: %s", err)

    def invalidate_usable_days_cache(self) -> None:
        """Clear usable days cache."""
        self._usable_days_cache = None

    async def _async_compute_usable_days(self) -> float:
        """Compute usable buffer days from live SQLite coverage."""
        def _calc() -> float:
            gross_layout_raw = self.hub.writer.db.get_meta("gross_layout")
            gross_roles: dict[int, str] = {}
            if gross_layout_raw:
                try:
                    layout_data = json.loads(gross_layout_raw)
                    for item in layout_data:
                        r = item["role"]
                        for c in item["channels"]:
                            gross_roles[c] = r
                except Exception:
                    pass
            now_ms = int(round(dt_util.utcnow().timestamp() * 1000))
            active_rows = self.hub.writer.db.channel_rows()
            windows = {
                ch: (now_ms - RETENTION_DAYS * MS_PER_DAY, now_ms)
                for ch, _, _, _ in active_rows
            }
            val, _ = self.hub.writer.db.coverage_for(windows, gross_roles)
            return float(val)

        try:
            val = await self.hass.async_add_executor_job(_calc)
        except AssertionError:
            # The buffer can close while this executor job is running:
            # async_stop unsubscribes future refresh fires but cannot recall an
            # already-dispatched callback, and BufferDB raises AssertionError
            # ("BufferDB not open") once closed (CI teardown race, WR-01 family).
            # During teardown a skipped refresh is harmless; while running the
            # error must propagate.
            if not self._stopped:
                raise
            if self._usable_days_cache is not None:
                return float(self._usable_days_cache["value"])
            return 0.0
        now_dt = dt_util.utcnow()
        self._usable_days_cache = {
            "value": val,
            "computed_at": now_dt,
        }
        c_data = dict(self.coordinator.data)
        c_data["usable_days"] = {
            "value": round(val, 2),
            "computed_at": now_dt.isoformat(),
        }
        self.coordinator.data = c_data
        return val

    async def _async_get_usable_days(self) -> float:
        """Return cached usable days if younger than 6 hours, else recompute."""
        if self._usable_days_cache is not None:
            age = dt_util.utcnow() - self._usable_days_cache["computed_at"]
            if age < USABLE_DAYS_REFRESH:
                return float(self._usable_days_cache["value"])
        return await self._async_compute_usable_days()

    def _evaluate_endpoint_and_consent(self) -> str | None:
        """Evaluate endpoint and consent gates. Return reason or None."""
        base_url = self.entry.data.get(CONF_LLM_BASE_URL)
        probe = self.entry.data.get(CONF_LLM_PROBE)
        if not base_url or not probe:
            return "no_endpoint"

        names_flag = bool(self.entry.options.get(CONF_SHARE_NAMES, False))
        if not consent_valid(self.consent_store.record, str(base_url), names_flag):
            return "no_consent"

        return None

    def _evaluate_gates(self, usable_days: float) -> str | None:
        """Evaluate learning gates in strict order. Return first failing reason or None."""
        reason = self._evaluate_endpoint_and_consent()
        if reason is not None:
            return reason

        if usable_days < MIN_LEARNING_DAYS:
            return "insufficient_data"

        return None

    async def _async_handle_month_rollover_if_needed(self) -> bool:
        """Check if month has rolled over, reset counters, clear cap issue, and re-evaluate gates if capped."""
        curr_month = month_key(dt_util.now())
        c_data = dict(self.coordinator.data)
        month_dict = c_data.get("month") or {}
        if month_dict.get("key") != curr_month:
            c_data["month"] = {"key": curr_month, "input": 0, "output": 0, "cost": None}
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CAPPED)
            was_capped = c_data.get("status") == "capped"
            if was_capped:
                c_data["status"] = "idle"
                c_data["last_error"] = None
            self.coordinator.data = c_data
            await self._learning_store.async_save(self._build_learning_store_data())
            if was_capped:
                await self._async_reevaluate_gates()
            return True
        return False

    def _get_current_usage(self) -> Usage:
        """Return current Usage snapshot, checking for month rollover."""
        curr_month = month_key(dt_util.now())
        c_data = dict(self.coordinator.data)
        month_dict = c_data.get("month") or {}
        if month_dict.get("key") != curr_month:
            month_dict = {"key": curr_month, "input": 0, "output": 0, "cost": None}
            c_data["month"] = month_dict
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CAPPED)
            self.coordinator.data = c_data

        tokens_run = c_data.get("tokens_run") or {}
        return Usage(
            run_input=int(tokens_run.get("input", 0)),
            run_output=int(tokens_run.get("output", 0)),
            month_input=int(month_dict.get("input", 0)),
            month_output=int(month_dict.get("output", 0)),
            month_cost=float(month_dict["cost"]) if month_dict.get("cost") is not None else None,
        )

    async def _async_reevaluate_gates(self) -> None:
        """Re-evaluate gates passively when idle or blocked."""
        await self._async_handle_month_rollover_if_needed()

        if self.coordinator.data.get("status") not in ("idle", "blocked"):
            return
        reason = self._evaluate_endpoint_and_consent()
        if reason is None:
            usable_days = await self._async_get_usable_days()
            if usable_days < MIN_LEARNING_DAYS:
                reason = "insufficient_data"

        c_data = dict(self.coordinator.data)
        if reason is not None:
            c_data["status"] = "blocked"
            c_data["blocked_reason"] = reason
        else:
            c_data["status"] = "idle"
            c_data["blocked_reason"] = None
        self.coordinator.async_set_updated_data(c_data)

    async def _async_periodic_refresh(self, _now: Any) -> None:
        """Refresh usable days and re-evaluate gates every 6 hours."""
        # An already-dispatched callback can outlive async_stop's unsubscription;
        # nothing below is safe or useful during teardown (WR-01 family).
        if self._stopped:
            return
        await self._async_handle_month_rollover_if_needed()
        await self._async_compute_usable_days()
        await self._async_reevaluate_gates()
        await self._async_maybe_rescore()
        # D-11a: the existing 6-hourly refresh also recomputes aged candidate and
        # device evidence; all other hook sites leave age unchecked.
        await self._async_maybe_evidence(include_aged=True)

    def _job_active(self) -> bool:
        """Return True while a learning run or rescore is running or being started."""
        return self._start_pending or (self.run_task is not None and not self.run_task.done())

    async def async_request_run(self, trigger: str) -> None:
        """Request a learning run to start in the background (single-flight)."""
        if self._job_active():
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="already_running",
            )
        # Claim the slot before the first await, so a concurrent request or rescore
        # cannot pass the same check (WR-09)
        self._start_pending = True
        try:
            await self._async_request_run_claimed(trigger)
        finally:
            self._start_pending = False

    async def _async_request_run_claimed(self, trigger: str) -> None:
        """Check the gates and start the run; the caller holds the start slot."""
        reason = self._evaluate_endpoint_and_consent()
        if reason is None:
            usable_days = await self._async_get_usable_days()
            if usable_days < MIN_LEARNING_DAYS:
                reason = "insufficient_data"
                days_rem = str(math.ceil(MIN_LEARNING_DAYS - usable_days))
                c_data = dict(self.coordinator.data)
                c_data["status"] = "blocked"
                c_data["blocked_reason"] = reason
                self.coordinator.async_set_updated_data(c_data)
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key=reason,
                    translation_placeholders={"days_remaining": days_rem},
                )

        if reason is not None:
            if reason == "no_consent":
                issue_reg = ir.async_get(self.hass)
                if not issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT):
                    await self._async_check_consent(immediate=True)

            c_data = dict(self.coordinator.data)
            c_data["status"] = "blocked"
            c_data["blocked_reason"] = reason
            self.coordinator.async_set_updated_data(c_data)
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key=reason,
            )

        snapshot = self._build_snapshot()

        run_id = uuid.uuid4().hex
        self.run_task = self.entry.async_create_background_task(
            self.hass,
            self._async_run(run_id, trigger, snapshot),
            "powerguesser_learning",
        )

    def _build_snapshot(self) -> RunSnapshot:
        """Build the run snapshot from the entry; shared by the run, the preview and consent prepare.

        One builder keeps the previewed body and the consent estimate identical to what
        the run sends (WR-07).
        """
        caps = Caps(
            run_input=int(self.entry.options.get(CONF_RUN_INPUT_TOKENS, DEFAULT_RUN_INPUT_TOKENS)),
            run_output=int(self.entry.options.get(CONF_RUN_OUTPUT_TOKENS, DEFAULT_RUN_OUTPUT_TOKENS)),
            month_input=int(self.entry.options.get(CONF_MONTH_INPUT_TOKENS, DEFAULT_MONTH_INPUT_TOKENS)),
            month_output=int(self.entry.options.get(CONF_MONTH_OUTPUT_TOKENS, DEFAULT_MONTH_OUTPUT_TOKENS)),
            month_cost=float(self.entry.options[CONF_MONTH_COST_LIMIT]) if self.entry.options.get(CONF_MONTH_COST_LIMIT) is not None else None,
            price_input_per_1m=float(self.entry.options[CONF_PRICE_INPUT]) if self.entry.options.get(CONF_PRICE_INPUT) is not None else None,
            price_output_per_1m=float(self.entry.options[CONF_PRICE_OUTPUT]) if self.entry.options.get(CONF_PRICE_OUTPUT) is not None else None,
        )

        return RunSnapshot(
            base_url=str(self.entry.data.get(CONF_LLM_BASE_URL, "")),
            api_key=str(self.entry.data.get(CONF_LLM_API_KEY, "")),
            model=str(self.entry.data.get(CONF_LLM_MODEL, "")),
            profile=self.entry.data.get(CONF_LLM_PROBE) or {},
            names_flag=bool(self.entry.options.get(CONF_SHARE_NAMES, False)),
            max_tokens=int(self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
            validated=False,
            caps=caps,
            truncation_guard=bool(self.entry.options.get(CONF_TRUNCATION_GUARD, True)),
        )

    def _metered_channel_rows(
        self, channel_rows: list[tuple[int, str, str, str]]
    ) -> list[tuple[int, str]]:
        """(channel, entity_id) pairs for metered, model-known channels.

        The single copy of the metered-predicate shared by the extract job
        (which needs both the channel list and the masked dicts) and the
        shadow pass (IN-04).
        """
        metered_entity_ids = getattr(self.hub.model, "metered_entity_ids", set())
        return [
            (ch, entity_id)
            for ch, entity_id, _role, kind in channel_rows
            if kind == KIND_METERED and entity_id in metered_entity_ids
        ]

    def _build_extract_job_data(
        self, stage_dir: Path, *, shadow: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        """Build job_extract dict, prepare cache key, and window_end_ms.

        With shadow=True (D-01/D-05) the job learns the household with every
        eligible plug added back and no plug reference lines: both the
        prepare cache key and the job carry holdout_calendar_days =
        SHADOW_HOLDOUT_DAYS and empty metered_channels, and the job
        additionally names the masked channels (channel + entity_id, sorted
        by entity_id) for the shadow_link stage and the D-04 eligibility
        fingerprint. Production callers pass nothing: holdout_days 0,
        holdout_calendar_days 0, metered plugs subtracted, holdout empty.
        """
        channel_rows = self.hub.writer.db.channel_rows()
        self._channel_rows_cache = channel_rows
        # One shared projection of the metered/model-known predicate (IN-04),
        # shaped two ways below.
        metered_rows = self._metered_channel_rows(channel_rows)
        metered_channels = [ch for ch, _entity_id in metered_rows]
        masked_channels = sorted(
            (
                {"channel": ch, "entity_id": entity_id}
                for ch, entity_id in metered_rows
            ),
            key=lambda m: str(m["entity_id"]),
        )
        if shadow:
            holdout_calendar_days = SHADOW_HOLDOUT_DAYS
            job_metered_channels: list[int] = []
        else:
            holdout_calendar_days = 0
            job_metered_channels = metered_channels

        phase_map = {
            CONF_PHASE_L1: "l1",
            CONF_PHASE_L2: "l2",
            CONF_PHASE_L3: "l3",
        }
        phase_opts = {
            role: self.entry.options.get(conf)
            for conf, role in phase_map.items()
        }
        phase_channels: dict[str, int] = {}
        for ch, entity_id, role, kind in channel_rows:
            if kind == KIND_PHASE:
                for r_name, opt_eid in phase_opts.items():
                    if opt_eid and opt_eid == entity_id:
                        phase_channels[r_name] = ch

        now = dt_util.now()
        today_midnight = dt_util.start_of_local_day(now)
        day_bounds_ms: list[int] = []
        for i in range(RETENTION_DAYS, -1, -1):
            target_date = today_midnight.date() - timedelta(days=i)
            target_dt = datetime.combine(
                target_date, time.min, tzinfo=today_midnight.tzinfo
            )
            local_midnight = dt_util.start_of_local_day(target_dt)
            day_bounds_ms.append(int(round(local_midnight.timestamp() * 1000)))

        first_date = today_midnight.date() - timedelta(days=RETENTION_DAYS)
        first_weekday = first_date.weekday()

        profile_data = getattr(self.hub, "profile_coordinator", None)
        snap_data = profile_data.data if profile_data else None
        if snap_data is not None and getattr(snap_data, "effective_s", None) is not None:
            grid_s = int(round(snap_data.effective_s))
            enabled_types = [t.value for t in snap_data.enabled]
        else:
            grid_s = 30
            enabled_types = [t.value for t in gate(30.0, True)[0]]

        # The prepare cache key is (last day bound, grid_s, enabled types, metered channels, phase channels, holdout_days)
        # and deliberately excludes the names flag and the model, which are applied at request-build time
        # (review 03-06 C4). Do not add names_flag or model here.
        # D-05: the shadow key differs from the production key by
        # holdout_calendar_days and the emptied metered_channels, so a stale
        # 0-holdout split is never served to a shadow extract.
        # Plan 06-09: the production key also carries the session windows, so a
        # new or ended session invalidates a cached extract.
        session_windows = (
            []
            if shadow
            else self._session_windows_for_extract(channel_rows, day_bounds_ms[-1])
        )
        cache_key = {
            "last_day_bound": day_bounds_ms[-1],
            "grid_s": grid_s,
            "enabled_types": enabled_types,
            "metered_channels": job_metered_channels,
            "phase_channels": phase_channels,
            "holdout_days": 0,
            "holdout_calendar_days": holdout_calendar_days,
            "session_windows": session_windows,
        }

        job_extract = {
            "buffer_path": self.hass.config.path(BUFFER_DIR, BUFFER_FILE),
            "stage_dir": str(stage_dir),
            "grid_s": grid_s,
            "day_bounds_ms": day_bounds_ms,
            "first_weekday": first_weekday,
            "holdout_days": 0,
            "holdout_calendar_days": holdout_calendar_days,
            "train_fraction": 0.7,
            "enabled_event_types": enabled_types,
            "metered_channels": job_metered_channels,
            "phase_channels": phase_channels,
            "masked_channels": masked_channels if shadow else [],
            "session_windows": session_windows,
        }
        return job_extract, cache_key, day_bounds_ms[-1]

    def _session_windows_for_extract(
        self, channel_rows: Sequence[tuple[int, str, str, str]], window_end_ms: int
    ) -> list[dict[str, Any]]:
        """Build the extract job's session_windows (Plan 06-09; REV-04, D-25/D-12).

        One entry per session whose plug has a session channel and whose target
        still resolves, so every session plug is subtracted only inside its own
        window and every candidate-target window becomes a labelled plug
        reference line (D-12). The window end is the record's end_ts or the
        job's window end; the key starts at worker.SESSION_KEY_BASE so it can
        never collide with a buffer channel id; the label is the resolved
        target's display name, derived from the record, never from the plug
        entity (D-09, SC3). Runs in the executor (reads the channels table
        cached in the caller).
        """
        review = self.review
        sessions = review.sessions
        if not sessions:
            return []
        session_channels = {
            str(entity_id): int(ch)
            for ch, entity_id, _role, kind in channel_rows
            if kind == KIND_SESSION
        }
        if not session_channels:
            return []
        decisions = review.decisions
        devices = getattr(self.hub, "devices", None)

        def _lookup(device_uuid: str) -> Any:
            return devices.record(device_uuid) if devices is not None else None

        windows: list[dict[str, Any]] = []
        for index, record in enumerate(sessions):
            plug = str(record.get("plug_entity_id") or "")
            channel = session_channels.get(plug)
            if channel is None:
                continue
            resolved = resolve_session_target(record, decisions, _lookup)
            if resolved is None:
                continue
            target_kind, ident = resolved
            start_ms = _parse_iso_ms(record.get("start_ts"))
            if start_ms is None:
                continue
            end_ts = record.get("end_ts")
            end_ms = _parse_iso_ms(end_ts) if end_ts else window_end_ms
            if end_ms is None or end_ms <= start_ms:
                continue
            windows.append(
                {
                    "channel": channel,
                    "key": SESSION_KEY_BASE + index,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "target_kind": str(target_kind),
                    "label": self._session_window_label(target_kind, ident),
                }
            )
        return windows

    def _session_window_label(self, target_kind: str, ident: str) -> str:
        """Return a session window's label: the resolved target's display name.

        Derived from the resolved identity (the final candidate after any
        merged_into chain, or the surviving device), so an earlier session's
        line keeps its target's name after the plug moves on (D-09, SC3).
        """
        if target_kind == "device":
            devices = getattr(self.hub, "devices", None)
            record = devices.record(ident) if devices is not None else None
            return str((record or {}).get("name") or ident)
        for candidate in self.candidates:
            if candidate.get("signature_id") == ident:
                return self._display_name(candidate)
        decision = self.review.decisions.get(ident) or {}
        return str(decision.get("name") or ident)

    def _plug_session_labels(self) -> dict[int, str]:
        """Map each session window key to its label for _plug_names (Plan 06-09).

        Recomputed from the same ReviewStore sessions the extract job used, so
        the key (SESSION_KEY_BASE + index) matches the job's entry. Only used
        when the share-names flag is on, like the metered plug names.
        """
        sessions = self.review.sessions
        if not sessions:
            return {}
        channel_rows = self._channel_rows_cache
        if channel_rows is None:
            try:
                channel_rows = self.hub.writer.db.channel_rows()
            except RuntimeError:
                channel_rows = []
        session_channels = {
            str(entity_id)
            for _ch, entity_id, _role, kind in channel_rows
            if kind == KIND_SESSION
        }
        devices = getattr(self.hub, "devices", None)

        def _lookup(device_uuid: str) -> Any:
            return devices.record(device_uuid) if devices is not None else None

        labels: dict[int, str] = {}
        for index, record in enumerate(sessions):
            plug = str(record.get("plug_entity_id") or "")
            if plug not in session_channels:
                continue
            resolved = resolve_session_target(record, self.review.decisions, _lookup)
            if resolved is None:
                continue
            labels[SESSION_KEY_BASE + index] = self._session_window_label(
                resolved[0], resolved[1]
            )
        return labels

    def build_window_job(self, stage_dir: Path) -> dict[str, Any]:
        """Build a window-only job (buffer_path, grid_s, day_bounds_ms, enabled_event_types, phase_channels).

        Reads the buffer DB through _build_extract_job_data and therefore must
        run in the executor. Written-down side effect: _build_extract_job_data
        overwrites self._channel_rows_cache with the current channel rows; that
        is harmless because every learning run and rescore refreshes the cache
        before reading it.
        """
        job_extract, _cache_key, _window_end_ms = self._build_extract_job_data(stage_dir)
        return {
            "buffer_path": job_extract["buffer_path"],
            "grid_s": job_extract["grid_s"],
            "day_bounds_ms": job_extract["day_bounds_ms"],
            "enabled_event_types": job_extract["enabled_event_types"],
            "phase_channels": job_extract["phase_channels"],
        }

    async def async_run_exclusive_stage(
        self, stage: str, job_path: Path | str
    ) -> dict[str, Any]:
        """Run one worker stage through the shared single-flight job slot.

        Raises ServiceValidationError already_running while a learning run,
        rescore, refit or another stage holds the slot. The slot is released
        before queueing a follow-up refit (release-before-follow-up, P4-12).
        """
        if self._job_active() or self._stopped:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="already_running",
            )
        task = self.entry.async_create_background_task(
            self.hass,
            self._async_run_stage(stage, job_path),
            f"powerguesser_{stage}",
        )
        self.run_task = task
        try:
            return await task
        finally:
            if self.run_task is task:
                self.run_task = None
            await self._async_maybe_refit()
            await self._async_maybe_evidence()

    def _plug_names(
        self,
        meta: Mapping[str, Any],
        channel_rows: Sequence[tuple[int, str, str, str]] | None = None,
    ) -> dict[str, str]:
        """Map each pN plug id to its entity display name.

        Plan 06-09: a synthetic session-window key at or above
        worker.SESSION_KEY_BASE maps to that window's resolved target label
        instead of a plug entity name, so a candidate session's reference line
        is named after the target from the record, never after the plug (D-09).
        """
        if channel_rows is None:
            if self._channel_rows_cache is not None:
                channel_rows = self._channel_rows_cache
            else:
                try:
                    channel_rows = self.hub.writer.db.channel_rows()
                except RuntimeError:
                    channel_rows = []
        ch_to_entity = {ch: entity_id for ch, entity_id, role, kind in channel_rows}
        window_labels = self._plug_session_labels()
        names: dict[str, str] = {}
        for pid, ch_id in meta.get("plugs", {}).items():
            try:
                ch_int = int(ch_id)
            except (TypeError, ValueError):
                continue
            if ch_int >= SESSION_KEY_BASE:
                label = window_labels.get(ch_int)
                if label:
                    names[pid] = label
                continue
            entity_id = ch_to_entity.get(ch_int)
            if entity_id:
                state = self.hass.states.get(entity_id)
                if state is not None and state.name:
                    names[pid] = state.name
        return names

    async def _async_prepare(
        self, snapshot: RunSnapshot | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run or reuse extract stage in prepare directory and return (digest, meta).

        Concurrent callers are serialised, so the second one reuses the first one's
        cached extract instead of racing it on the same files.
        """
        async with self._prepare_lock:
            return await self._async_prepare_unlocked()

    async def _async_prepare_unlocked(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run or reuse the extract stage in the prepare directory (caller holds the lock)."""
        prepare_dir = Path(
            self.hass.config.path(BUFFER_DIR, LEARNING_DIR, PREPARE_DIR)
        )

        def _check_cached_or_prepare_job() -> tuple[
            dict[str, Any] | None,
            dict[str, Any] | None,
            Path | None,
            dict[str, Any],
        ]:
            job_extract, cache_key, _ = self._build_extract_job_data(prepare_dir)
            key_file = prepare_dir / "key.json"
            digest_file = prepare_dir / "digest.json"
            meta_file = prepare_dir / "digest_meta.json"

            if key_file.exists() and digest_file.exists() and meta_file.exists():
                try:
                    with open(key_file, "r", encoding="utf-8") as f:
                        cached_key = json.load(f)
                    if cached_key == cache_key:
                        with open(digest_file, "r", encoding="utf-8") as f:
                            digest = json.load(f)
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        return digest, meta, None, cache_key
                except Exception:
                    pass

            prepare_dir.mkdir(parents=True, exist_ok=True)
            job_path = prepare_dir / "job_extract.json"
            job_path.write_text(json.dumps(job_extract, indent=2), encoding="utf-8")
            return None, None, job_path, cache_key

        digest, meta, job_path, cache_key = await self.hass.async_add_executor_job(
            _check_cached_or_prepare_job
        )

        if digest is not None and meta is not None:
            return digest, meta

        assert job_path is not None
        await self._async_run_stage("extract", job_path)

        def _save_key_and_read_results() -> tuple[dict[str, Any], dict[str, Any]]:
            digest_file = prepare_dir / "digest.json"
            meta_file = prepare_dir / "digest_meta.json"
            key_file = prepare_dir / "key.json"

            with open(digest_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            with open(meta_file, "r", encoding="utf-8") as f:
                m = json.load(f)

            key_file.write_text(json.dumps(cache_key, indent=2), encoding="utf-8")
            return d, m

        return await self.hass.async_add_executor_job(_save_key_and_read_results)

    async def async_preview_payload(self) -> dict[str, Any]:
        """Return the exact request body the next learning run will send."""
        base_url = self.entry.data.get(CONF_LLM_BASE_URL)
        probe = self.entry.data.get(CONF_LLM_PROBE)
        if not base_url or not probe:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="no_endpoint",
            )

        snapshot = self._build_snapshot()

        try:
            digest, meta = await self._async_prepare(snapshot)
        except Exception as err:
            _LOGGER.warning("Prepare stage failed for preview: %s", err)
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="preview_unavailable",
                translation_placeholders={"reason": str(err)},
            ) from err

        names: dict[str, str] | None = None
        if snapshot.names_flag:
            channel_rows = await self.hass.async_add_executor_job(
                self.hub.writer.db.channel_rows
            )
            self._channel_rows_cache = channel_rows
            names = self._plug_names(meta, channel_rows)

        body = prompts.build_request(
            digest,
            model=snapshot.model,
            profile=snapshot.profile,
            max_tokens=snapshot.max_tokens,
            names=names,
        )

        n_bytes = payload_bytes(body)
        return {
            "url": f"{snapshot.base_url}/chat/completions",
            "body": body,
            "bytes": n_bytes,
            "estimated_input_tokens": estimate_tokens(n_bytes),
            "max_tokens": snapshot.max_tokens,
            "entity_names_included": bool(names),
        }

    def visible_candidates(self) -> list[dict[str, Any]]:
        """Return visible candidates excluding dismissed, suppressed, merged or owned candidates.

        Owned signatures (confirmed or retired devices, incl. aliases) are never
        offered as candidates: Review contract C1, asserted by
        tests/ha/test_device_confirm.py::test_confirmed_candidate_leaves_review_list.

        Unconfirmed device records (deleted devices, D-05) re-appear as ordinary
        candidates keyed by their signature id, built from the device record's
        active artefact, when the current run's candidates do not already list
        them — so a deleted device can always be re-confirmed.
        """
        decisions = self.review.decisions
        devices = getattr(self.hub, "devices", None)
        owned = devices.owned_signature_ids() if devices is not None else set()
        stored = list(self.candidates)
        if devices is not None:
            stored_sigs = {c.get("signature_id") for c in stored}
            for cand in devices.unconfirmed_candidates():
                sig_id = cand.get("signature_id", "")
                if sig_id and sig_id not in stored_sigs and sig_id not in owned:
                    stored.append(cand)
        visible: list[dict[str, Any]] = []
        for c in stored:
            ident = c.get("identity") or {}
            if ident.get("status") in ("suppressed", "merged"):
                continue
            sig_id = c.get("signature_id", "")
            if sig_id in owned:
                continue
            dec = decisions.get(sig_id)
            if dec and dec.get("status") in ("dismissed", "merged"):
                continue
            visible.append(c)

        def _sort_key(cand: dict[str, Any]) -> tuple[float, str]:
            scores = cand.get("scores") or {}
            train = scores.get("train") or {}
            share = float(train.get("share") or 0.0)
            return (-share, str(cand.get("signature_id", "")))

        visible.sort(key=_sort_key)
        return visible

    def review_entries(self) -> list[dict[str, Any]]:
        """Review list: visible candidates followed by confirmed device entries (C2).

        Candidates keep their signature id as option value; device entries
        carry option_value ``device_<uuid>`` (Review contract C2). Devices are
        listed after the candidates, sorted by name then signature_id.
        """
        entries = self.visible_candidates()
        devices = getattr(self.hub, "devices", None)
        if devices is not None:
            entries = entries + devices.confirmed_entries()
        return entries

    @property
    def candidates_created_at(self) -> str | None:
        """Creation time of the stored candidates record (read-only, D-18)."""
        rec = self._candidates_record
        created = rec.get("created") if rec else None
        return str(created) if created else None

    @property
    def candidates_validated(self) -> bool | None:
        """Validated flag of the stored candidates record (read-only, D-18)."""
        rec = self._candidates_record
        if rec is None:
            return None
        validated = rec.get("validated")
        return bool(validated) if validated is not None else None

    def _display_name(self, candidate: dict[str, Any]) -> str:
        """Display name without merge info: decision name, else label.

        The review list, the Confirm pre-fill and the rename pre-fill all follow
        the learning run's latest label: a re-run has more data behind its guess,
        so that is what the user should see and be offered. A user-given decision
        name still wins. The first-ever label stays stored on the candidate as
        first_label (carry-over in _apply_first_labels is unchanged) and is
        surfaced as history in the rename dialog, not as the display name —
        this supersedes the D-18 display precedence.
        """
        sig_id = candidate.get("signature_id", "")
        dec = self.review.decisions.get(sig_id)
        return (
            (dec.get("name") if dec else None)
            or candidate.get("label")
            or "Unknown"
        )

    def _apply_first_labels(self, library: list[dict[str, Any]]) -> None:
        """Set first_label on every artefact dict of a freshly compiled library (D-18).

        A candidate matched across learning runs keeps the label it first had as
        first_label: provenance shown as history in the rename dialog ("First seen
        as: …"), never the display name — display follows the latest label
        (decision name > label; supersedes the D-18 display precedence).
        Precedence: the prior stored candidate's first_label, else its label,
        else the label stored on the review decision record at decision time,
        else the artefact's own label. New identities get first_label = their
        own label. Read-only over the review Store: this helper must never
        write it (REV-01).
        """
        priors = {c.get("signature_id"): c for c in self.candidates}
        decisions = self.review.decisions
        for art in library:
            sig_id = art.get("signature_id")
            first_label: Any = None
            prior = priors.get(sig_id)
            if prior is not None:
                first_label = prior.get("first_label")
                if first_label is None:
                    first_label = prior.get("label")
            if first_label is None:
                dec = decisions.get(sig_id)
                if dec is not None:
                    first_label = dec.get("label")
            if first_label is None:
                first_label = art.get("label")
            if first_label is not None:
                art["first_label"] = first_label

    async def _route_device_matches(
        self, library: list[dict[str, Any]], run_id: str, created: str
    ) -> None:
        """Route a run's matched artefacts into device update proposals (D-11).

        Runs right after _apply_first_labels at every library save: a run
        artefact whose signature id belongs to a confirmed device and whose
        states differ from the device's active version becomes the device's
        pending_proposal only when its D-17 proposal_gate passed (the worker
        replayed it no worse than the frozen active version on the run's
        validation block). A differing artefact with a failing gate, or with
        no gate (a device confirmed while the run was in flight), is not
        offered: learning records validation.last_rejected_proposal with both
        scores, the basis and the reason for diagnosis only. Identical states
        store nothing. Retired and unconfirmed devices never get proposals
        (device_for_signature only resolves confirmed records), and a matched
        artefact never changes the device's parameters — only the user's
        explicit Accept does (T-04-20). The artefact stays in the stored
        library: visible_candidates already hides owned signatures from
        Review.
        """
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return
        now = dt_util.utcnow().isoformat()
        for art in library:
            sig_id = art.get("signature_id")
            if not sig_id:
                continue
            dev_uuid = devices.device_for_signature(str(sig_id))
            if dev_uuid is None:
                continue
            if not devices.active_version_differs(dev_uuid, art):
                continue
            gate = art.get("proposal_gate")
            if isinstance(gate, Mapping) and gate.get("passed") is True:
                await devices.async_set_proposal(
                    dev_uuid, art, run_id=run_id, proposed_at=now, trained_at=created
                )
                continue
            if isinstance(gate, Mapping):
                info = {
                    "run_id": run_id,
                    "rejected_at": now,
                    "proposed_S": gate.get("proposed_S"),
                    "active_S": gate.get("active_S"),
                    "basis": gate.get("basis"),
                    "reason": "worse_than_active",
                }
            else:
                info = {
                    "run_id": run_id,
                    "rejected_at": now,
                    "proposed_S": None,
                    "active_S": None,
                    "basis": None,
                    "reason": "not_gated",
                }
            await devices.async_record_rejected_proposal(dev_uuid, info)

    def candidate_label(self, candidate: dict[str, Any], *, include_overlaps: bool = True) -> str:
        """Format candidate label for selector.

        D-27: a candidate whose identity carries "overlaps" (a confirmed or
        retired device's signature it overlaps without being its near twin)
        shows " — overlaps <device name>" at the end of the label, so the
        review option, the candidate menu description and the confirm dialog's
        {candidate} placeholder explain the overlap before the user decides.
        The note rides inside the whole label, so it reaches every description
        placeholder only through the existing _escape_markdown calls in
        config_flow.py. Review-record labels pass include_overlaps=False: the
        stored label stays the plain recognition label.
        """
        sig_id = candidate.get("signature_id", "")
        if candidate.get("confirmed"):
            # Review contract C2 device entry: the plain device name, the
            # confirmed-device marker, the update note while a proposal is
            # pending (D-11 minimal surface) and the D-16 drift note while
            # live drift is set (from confirmed_entries()["drift"]).
            label = f"{self._display_name(candidate)} — confirmed device"
            if candidate.get("update_available"):
                label += " — update available"
            if candidate.get("drift"):
                label += " — drift detected"
            if self._session_target_active(
                "device", str(candidate.get("device_uuid") or "")
            ):
                label += " — plug session active"
            return label
        dec = self.review.decisions.get(sig_id)
        name = self._display_name(candidate)
        aliases = (dec.get("aliases") if dec else None) or []
        if aliases:
            name = f"{name} (+{len(aliases)} merged)"
        scores = candidate.get("scores") or {}
        train = scores.get("train") or {}
        train_share = float(train.get("share") or 0.0)
        share_pct = f"{train_share * 100:.1f}"

        # D-07: preview candidates show no train/validation figures beyond share
        is_validated = scores.get("validated")
        if is_validated is None:
            is_validated = self.candidates_validated
        if bool(is_validated):
            val = scores.get("val") or {}
            val_share = float(val.get("share") or 0.0)
            val_pct = f"{val_share * 100:.1f}"
            label = f"{name} — {share_pct} % of residual energy — explained: train {share_pct} % / validation {val_pct} %"
        else:
            label = f"{name} — {share_pct} % of residual energy — preview, not yet validated"

        if include_overlaps:
            ident = candidate.get("identity") or {}
            overlaps = ident.get("overlaps")
            if overlaps:
                devices = getattr(self.hub, "devices", None)
                owner = (
                    devices.owner_name(str(overlaps)) if devices is not None else None
                )
                label += " — overlaps " + (owner or "a confirmed device")
        if self._session_target_active("candidate", str(sig_id)):
            label += " — plug session active"
        return label

    def _session_target_active(self, kind: str, ident: str) -> bool:
        """True while an active plug session resolves to this candidate or device.

        The marker is derived at read time from the session records and the
        current identities, so it never rewrites an earlier session (D-09, D-11).
        """
        review = getattr(self, "review", None)
        if review is None or not ident:
            return False
        sessions = review.active_sessions(dt_util.utcnow())
        if not sessions:
            return False
        devices = getattr(self.hub, "devices", None)

        def _lookup(device_uuid: str) -> Any:
            return devices.record(device_uuid) if devices is not None else None

        decisions = review.decisions
        for session in sessions:
            if resolve_session_target(session, decisions, _lookup) == (kind, ident):
                return True
        return False

    async def async_record_decision(
        self,
        action: str,
        sig_id: str,
        *,
        name: str | None = None,
        sources: list[str] | None = None,
    ) -> None:
        """Record user review decision (dismiss, rename, merge) for a candidate."""
        devices = getattr(self.hub, "devices", None)
        if action == "merge" and devices is not None:
            # Device branch (D-12, D-23): the target signature belongs to a
            # confirmed device, so it is not a visible candidate (owned
            # signatures never are, C1) — handle it before the
            # visible-candidate lookup below.
            target_uuid = devices.device_for_signature(sig_id)
            target = devices.record(target_uuid) if target_uuid else None
            if target is not None and target.get("status") == "confirmed":
                await self._async_merge_confirmed_devices(
                    str(target_uuid), sig_id, sources
                )
                new_visible = self.visible_candidates()
                c_data = dict(self.coordinator.data or {})
                c_data["pending_candidates"] = len(new_visible)
                self.coordinator.async_set_updated_data(c_data)
                return

        visible = self.visible_candidates()
        cand = next((c for c in visible if c.get("signature_id") == sig_id), None)
        if cand is None:
            raise CandidateGone(f"Candidate {sig_id} is no longer visible")

        fp = cand.get("fingerprint")
        if action == "dismiss":
            await self.review.async_dismiss(sig_id, fp, label=self._display_name(cand))
        elif action == "rename":
            if name is None:
                raise ValueError("Name required for rename")
            norm_name = normalise_name(name)
            await self.review.async_set_name(sig_id, norm_name, fp, label=norm_name)
        elif action == "merge":
            if not sources:
                raise ValueError("Sources required for merge")
            source_pairs: list[tuple[str, Any]] = []
            for s_id in sources:
                s_cand = next((c for c in visible if c.get("signature_id") == s_id), None)
                if s_cand is None or s_id == sig_id:
                    raise CandidateGone(f"Source candidate {s_id} is no longer visible")
                source_pairs.append((s_id, s_cand.get("fingerprint")))
            await self.review.async_merge(
                sig_id, fp, source_pairs, label=self._display_name(cand)
            )
        else:
            raise ValueError(f"Unknown review action: {action}")

        new_visible = self.visible_candidates()
        c_data = dict(self.coordinator.data or {})
        c_data["pending_candidates"] = len(new_visible)
        self.coordinator.async_set_updated_data(c_data)

    @staticmethod
    def _device_fingerprint(rec: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Fingerprint of a device record's active version's artefact.

        The same value version activation stores on the review record, so a
        merged device's fingerprint stays comparable with the ACTIVE version.
        """
        for entry in rec.get("versions") or []:
            if not isinstance(entry, Mapping) or entry.get("v") != rec.get("active_v"):
                continue
            artefact = entry.get("artefact")
            if not isinstance(artefact, Mapping):
                return None
            try:
                return SignatureArtefact.from_dict(artefact).fingerprint()
            except (KeyError, TypeError, ValueError):
                return None
        return None

    async def _async_merge_confirmed_devices(
        self,
        target_uuid: str,
        target_sig: str,
        sources: list[str] | None,
    ) -> None:
        """Merge other confirmed devices or unconfirmed candidates into a device.

        A confirmed device's merge sources are the other confirmed devices
        (each retired in place through DeviceRuntime.async_retire, D-12/D-23)
        and the visible unconfirmed switched candidates (D-13): those add their
        signature as an alias with their artefact and start a versioned union
        retrain. Every merge is recorded through ReviewStore.async_merge
        ({**rec} writers) so later learning runs map the absorbed signature to
        identity.merged (hidden, no proposal) through the Plan 04-09 identity
        pipeline.
        """
        devices = self.hub.devices
        target = devices.record(target_uuid)
        if target is None or target.get("status") != "confirmed":
            raise ValueError(f"Unknown merge target device {target_uuid}")
        if not sources:
            raise ValueError("Sources required for merge")
        target_fp = self._device_fingerprint(target)
        target_name = str(target.get("name") or target_sig)
        for source_sig in sources:
            source_sig = str(source_sig)
            source_uuid = devices.device_for_signature(source_sig)
            if source_uuid is not None:
                # Confirmed device source: retire in place (D-12, D-23).
                source = devices.record(source_uuid)
                if (
                    source is None
                    or source_uuid == target_uuid
                    or source.get("status") != "confirmed"
                ):
                    raise ValueError(f"Invalid merge source {source_sig!r}")
                source_fp = self._device_fingerprint(source)
                await devices.async_retire(str(source_uuid), into=target_uuid)
                await self.review.async_merge(
                    target_sig,
                    target_fp,
                    [(source_sig, source_fp)],
                    label=target_name,
                )
                continue

            # Unconfirmed visible switched candidate source (D-13): alias +
            # union retrain. Program candidates merge only into unconfirmed
            # candidates, and a confirmed program device is never a merge
            # source or target in v0.x (D-40, RESEARCH Pitfall 9): a union
            # retrain has no meaning for a stage template.
            cand = next(
                (
                    c
                    for c in self.visible_candidates()
                    if c.get("signature_id") == source_sig
                ),
                None,
            )
            if (
                cand is None
                or source_sig == target_sig
                or cand.get("kind") != "switched"
            ):
                raise ValueError(f"Invalid merge source {source_sig!r}")
            artefact = {
                k: v
                for k, v in cand.items()
                if k not in CANDIDATE_DECORATION_KEYS
            }
            await self.review.async_merge(
                target_sig,
                target_fp,
                [(source_sig, cand.get("fingerprint"))],
                label=target_name,
            )
            await devices.async_add_aliases(
                target_uuid, [(source_sig, artefact)]
            )
            await self.async_request_refit(target_uuid)

    async def async_after_confirm(self) -> None:
        """Recompute and push pending_candidates after a device confirm (C1)."""
        new_visible = self.visible_candidates()
        c_data = dict(self.coordinator.data or {})
        c_data["pending_candidates"] = len(new_visible)
        self.coordinator.async_set_updated_data(c_data)

    def _cancel_prepare_timers(self) -> None:
        """Cancel any scheduled prepare timers or startup listeners."""
        if self._unsub_prepare_timer is not None:
            self._unsub_prepare_timer()
            self._unsub_prepare_timer = None
        if self._unsub_at_started is not None:
            self._unsub_at_started()
            self._unsub_at_started = None

    @callback
    def _async_scheduled_prepare_callback(self, _now: Any) -> None:
        """Callback from timer to start background prepare job."""
        self._unsub_prepare_timer = None
        self._start_prepare_job()

    def _start_prepare_job(self) -> None:
        """Start the background prepare task."""
        if self._prepare_task is not None and not self._prepare_task.done():
            return
        self._prepare_task = self.entry.async_create_background_task(
            self.hass,
            self._async_run_prepare_job(),
            "powerguesser_learning_prepare",
        )

    async def _async_check_consent(
        self, *, immediate: bool = False, is_startup: bool = False
    ) -> None:
        """Check consent and schedule or run prepare to create/update consent issue."""
        base_url = self.entry.data.get(CONF_LLM_BASE_URL)
        probe = self.entry.data.get(CONF_LLM_PROBE)
        if not base_url or not probe:
            self._cancel_prepare_timers()
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        names_flag = bool(self.entry.options.get(CONF_SHARE_NAMES, False))
        if consent_valid(self.consent.record, str(base_url), names_flag):
            self._cancel_prepare_timers()
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        self._cancel_prepare_timers()

        if immediate:
            self._start_prepare_job()
            return

        if is_startup:
            @callback
            def _on_started(_hass: HomeAssistant) -> None:
                self._unsub_at_started = None
                self._unsub_prepare_timer = async_call_later(
                    self.hass,
                    PREPARE_STARTUP_DELAY_S,
                    self._async_scheduled_prepare_callback,
                )

            self._unsub_at_started = async_at_started(self.hass, _on_started)
        else:
            self._unsub_prepare_timer = async_call_later(
                self.hass,
                PREPARE_DEBOUNCE_S,
                self._async_scheduled_prepare_callback,
            )

    async def _async_run_prepare_job(self) -> None:
        """Execute prepare stage and create or remove consent issue."""
        base_url = self.entry.data.get(CONF_LLM_BASE_URL)
        probe = self.entry.data.get(CONF_LLM_PROBE)
        if not base_url or not probe:
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        names_flag = bool(self.entry.options.get(CONF_SHARE_NAMES, False))
        if consent_valid(self.consent.record, str(base_url), names_flag):
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        snapshot = self._build_snapshot()

        tokens_str = "unknown"
        bytes_str = "unknown"
        try:
            digest, meta = await self._async_prepare(snapshot)
            names: dict[str, str] | None = None
            if snapshot.names_flag:
                channel_rows = await self.hass.async_add_executor_job(
                    self.hub.writer.db.channel_rows
                )
                self._channel_rows_cache = channel_rows
                names = self._plug_names(meta, channel_rows)

            body = prompts.build_request(
                digest,
                model=snapshot.model,
                profile=snapshot.profile,
                max_tokens=snapshot.max_tokens,
                names=names,
            )
            n_bytes = payload_bytes(body)
            tokens_str = str(estimate_tokens(n_bytes))
            bytes_str = str(n_bytes)
        except Exception as err:
            _LOGGER.debug("Consent prepare failed: %s", err)
            tokens_str = "unknown"
            bytes_str = "unknown"

        # Re-check consent before creating issue (user may have consented meanwhile)
        base_url = self.entry.data.get(CONF_LLM_BASE_URL)
        if not base_url:
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        loc = endpoint_location(base_url)
        names_flag = bool(self.entry.options.get(CONF_SHARE_NAMES, False))
        if consent_valid(self.consent.record, str(base_url), names_flag):
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CONSENT)
            return

        location_str = "local network" if loc.local else "remote"
        transport_str = "https" if loc.transport == "https" else "http, unencrypted"
        model_str = str(self.entry.data.get(CONF_LLM_MODEL, ""))
        names_str = "included" if names_flag else "not included"

        issue_data = {
            "entry_id": self.entry.entry_id,
            # The notice names the full origin consent is given for (WR-11)
            "endpoint": format_origin(*endpoint_origin(str(base_url))),
            "location": location_str,
            "transport": transport_str,
            "model": model_str,
            "tokens": tokens_str,
            "bytes": bytes_str,
            "names": names_str,
        }
        placeholders = {k: v for k, v in issue_data.items() if k != "entry_id"}

        ir.async_create_issue(
            self.hass,
            DOMAIN,
            ISSUE_LEARNING_CONSENT,
            is_fixable=True,
            is_persistent=False,
            config_entry_id=self.entry.entry_id,
            severity=ir.IssueSeverity.WARNING,
            translation_key=ISSUE_LEARNING_CONSENT,
            data=issue_data,
            translation_placeholders=placeholders,
        )

    async def _async_run(
        self, run_id: str, trigger: str, snapshot: RunSnapshot
    ) -> None:
        """Execute the multi-stage learning run in the background."""
        started = dt_util.utcnow().isoformat()
        coord_data = dict(self.coordinator.data)
        coord_data["status"] = "running"
        coord_data["running_job"] = "learning"
        coord_data["tokens_run"] = {"input": 0, "output": 0}
        coord_data["model_mismatch_run"] = None
        coord_data["trimmed_run"] = 0
        coord_data["blocked_reason"] = None
        coord_data["last_error"] = None
        # D-15a: persist the attempt start before any work, so one save records both
        coord_data["last_attempt"] = self._attempt_record(started, "running", None)
        self.coordinator.async_set_updated_data(coord_data)
        await self._learning_store.async_save(self._build_learning_store_data())

        best_library: list[dict[str, Any]] = []
        window_end_ms: int = 0
        state = _RoundsState()
        self._reached_idle = False

        try:
            fresh_days = await self._async_compute_usable_days()
            validated = fresh_days >= VALIDATED_DAYS
            snapshot = dataclasses.replace(snapshot, validated=validated)

            # 1. Setup stage directory and clean previous runs
            learning_root = Path(self.hass.config.path(BUFFER_DIR, LEARNING_DIR))
            stage_dir = learning_root / run_id

            def _prepare_fs_and_job() -> tuple[Path, int]:
                stage_dir.mkdir(parents=True, exist_ok=True)
                if learning_root.exists():
                    for item in learning_root.iterdir():
                        if item.is_dir() and item.name not in (run_id, PREPARE_DIR):
                            shutil.rmtree(item, ignore_errors=True)

                job_extract, _, window_end_ms = self._build_extract_job_data(stage_dir)
                job_extract_path = stage_dir / "job_extract.json"
                job_extract_path.write_text(json.dumps(job_extract, indent=2), encoding="utf-8")
                return job_extract_path, window_end_ms

            job_extract_path, window_end_ms = await self.hass.async_add_executor_job(
                _prepare_fs_and_job
            )

            # 2. Run extract stage in subprocess
            await self._async_run_stage("extract", job_extract_path)

            def _read_extract_results() -> tuple[dict[str, Any], dict[str, Any]]:
                with open(stage_dir / "digest.json", "r", encoding="utf-8") as f:
                    digest = json.load(f)
                with open(stage_dir / "digest_meta.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return digest, meta

            digest, meta = await self.hass.async_add_executor_job(_read_extract_results)

            if not digest.get("clusters") and not digest.get("plugs"):
                last_run = {
                    "run_id": run_id,
                    "trigger": trigger,
                    "started": started,
                    "ended": "",
                    "stage": "extract",
                    "candidates": 0,
                    "train_share": 0.0,
                    "val_share": 0.0,
                    "preview": not snapshot.validated,
                    "rounds": 1,
                    "model_mismatch": self.coordinator.data.get("model_mismatch_run"),
                    "trimmed_fields": self.coordinator.data.get("trimmed_run", 0),
                    "days": {
                        "train": len(meta.get("days", {}).get("train", [])),
                        "val": len(meta.get("days", {}).get("val", [])),
                        "dropped": len(meta.get("days", {}).get("dropped", [])),
                    },
                }
                await self._async_publish_idle(
                    run_id=run_id,
                    snapshot=snapshot,
                    fresh_days=fresh_days,
                    stage_dir=stage_dir,
                    last_run=last_run,
                    pending_candidates=0,
                    last_error=None,
                )
                return

            # 3-6. LLM rounds through the shared method (production identity
            # context; the shadow pass reuses the same code path with an
            # empty context and its own run_id, D-01)
            names: dict[str, str] | None = None
            if snapshot.names_flag:
                channel_rows = await self.hass.async_add_executor_job(
                    self.hub.writer.db.channel_rows
                )
                self._channel_rows_cache = channel_rows
                names = self._plug_names(meta, channel_rows)

            devices = getattr(self.hub, "devices", None)
            await self._async_llm_rounds(
                stage_dir=stage_dir,
                run_id=run_id,
                snapshot=snapshot,
                digest=digest,
                names=names,
                prior=list(self.candidates)
                + (devices.identity_priors() if devices is not None else []),
                owned=(
                    sorted(devices.owned_signature_ids())
                    if devices is not None
                    else []
                ),
                decisions=dict(self.review.decisions),
                state=state,
                # D-17: the confirmed devices' frozen active versions gate
                # re-learn proposals in compile_score ({} without devices).
                actives=devices.gate_actives() if devices is not None else {},
                # Plan 07-01 (D-11): the production run computes and attaches
                # per-candidate evidence; the shadow pass below writes False.
                with_evidence=True,
            )
            best_library = state.best_library

            if state.proposal_empty:
                ended = dt_util.utcnow().isoformat()
                last_run = {
                    "run_id": run_id,
                    "trigger": trigger,
                    "started": started,
                    "ended": ended,
                    "stage": "proposal",
                    "candidates": 0,
                    "train_share": 0.0,
                    "val_share": 0.0,
                    "preview": not snapshot.validated,
                    "rounds": 1,
                    "model_mismatch": self.coordinator.data.get("model_mismatch_run"),
                    "trimmed_fields": self.coordinator.data.get("trimmed_run", 0),
                    "days": {
                        "train": len(meta.get("days", {}).get("train", [])),
                        "val": len(meta.get("days", {}).get("val", [])),
                        "dropped": len(meta.get("days", {}).get("dropped", [])),
                    },
                }
                candidates_record = {
                    "run_id": run_id,
                    "created": ended,
                    "window_end_ms": window_end_ms,
                    "validated": snapshot.validated,
                    "candidates": [],
                }
                self._apply_first_labels(candidates_record["candidates"])
                await self._route_device_matches(
                    candidates_record["candidates"], run_id, ended
                )
                await self._candidates_store.async_save(candidates_record)
                self.candidates = []
                # Keep the in-memory record in sync with the store: the
                # candidates_created_at/candidates_validated properties and the
                # rescore gate read self._candidates_record, not the store, so
                # leaving the stale previous record makes memory and store
                # disagree until a reload (WR-05).
                self._candidates_record = candidates_record

                await self._async_publish_idle(
                    run_id=run_id,
                    snapshot=snapshot,
                    fresh_days=fresh_days,
                    stage_dir=stage_dir,
                    last_run=last_run,
                    pending_candidates=0,
                    last_error=None,
                )
                return

            # 7. Save candidates Store
            candidates_record = {
                "run_id": run_id,
                "created": dt_util.utcnow().isoformat(),
                "window_end_ms": window_end_ms,
                "validated": snapshot.validated,
                "candidates": best_library,
            }
            self._apply_first_labels(candidates_record["candidates"])
            await self._route_device_matches(
                candidates_record["candidates"], run_id, candidates_record["created"]
            )
            await self._candidates_store.async_save(candidates_record)
            self._candidates_record = candidates_record
            self.candidates = best_library
            # A run's candidates arrive with evidence from compile_score, so the
            # refresh counter starts over (D-12a).
            self._reset_evidence_failures()

            # 8. Complete run: the shadow pass (D-01) runs after the production
            # candidates are saved and before the idle publish.
            train_share = sum(
                c.get("scores", {}).get("train", {}).get("share", 0.0) for c in best_library
            )
            val_share = sum(
                (c.get("scores", {}).get("val") or {}).get("share", 0.0) for c in best_library
            )

            last_run = {
                "run_id": run_id,
                "trigger": trigger,
                "started": started,
                "ended": "",
                "stage": "compile_score",
                "candidates": len(best_library),
                "train_share": round(train_share, 3),
                "val_share": round(val_share, 3),
                "preview": not snapshot.validated,
                "rounds": state.rounds_run,
                "accepted_round": state.incumbent_round,
                "model_mismatch": self.coordinator.data.get("model_mismatch_run"),
                "trimmed_fields": self.coordinator.data.get("trimmed_run", 0),
                "days": {
                    "train": len(meta.get("days", {}).get("train", [])),
                    "val": len(meta.get("days", {}).get("val", [])),
                    "dropped": len(meta.get("days", {}).get("dropped", [])),
                },
            }

            await self._async_publish_idle(
                run_id=run_id,
                snapshot=snapshot,
                fresh_days=fresh_days,
                stage_dir=stage_dir,
                last_run=last_run,
                pending_candidates=len(self.visible_candidates()),
                last_error=(
                    f"refinement stopped: {state.refine_error}"[:500]
                    if state.refine_error is not None
                    else None
                ),
            )

        except LearningCapped as err:
            cap = err.cap
            limit = err.limit
            c_data = dict(self.coordinator.data)
            c_data["status"] = "capped"
            c_data["last_error"] = f"cap_reached: {cap}"
            c_data["last_attempt"] = self._attempt_record(
                started, "capped", dt_util.utcnow().isoformat()
            )
            if state.best_library:
                candidates_record = {
                    "run_id": run_id,
                    "created": dt_util.utcnow().isoformat(),
                    "window_end_ms": window_end_ms,
                    "validated": snapshot.validated,
                    "candidates": state.best_library,
                }
                self._apply_first_labels(candidates_record["candidates"])
                await self._route_device_matches(
                    candidates_record["candidates"],
                    run_id,
                    candidates_record["created"],
                )
                await self._candidates_store.async_save(candidates_record)
                self._candidates_record = candidates_record
                self.candidates = state.best_library
                c_data["pending_candidates"] = len(self.visible_candidates())
                self._reset_evidence_failures()

            ir.async_create_issue(
                self.hass,
                DOMAIN,
                ISSUE_LEARNING_CAPPED,
                is_fixable=False,
                is_persistent=False,
                config_entry_id=self.entry.entry_id,
                severity=ir.IssueSeverity.WARNING,
                translation_key=ISSUE_LEARNING_CAPPED,
                translation_placeholders={"cap": cap, "limit": str(limit)},
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmAuthError as err:
            _LOGGER.warning("Learning run %s auth failed: %s", run_id, err)
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                ISSUE_LLM_AUTH,
                is_fixable=True,
                is_persistent=False,
                config_entry_id=self.entry.entry_id,
                data={"entry_id": self.entry.entry_id},
                severity=ir.IssueSeverity.ERROR,
                translation_key=ISSUE_LLM_AUTH,
                translation_placeholders={"host": endpoint_location(snapshot.base_url).host},
            )
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = f"auth_failed (HTTP {err.status})"[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmBudgetError:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = "provider_budget_exceeded"
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmTimeoutError:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = "timeout"
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmConnectError:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = "cannot_connect"
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmHttpError as err:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = f"http_{err.status}: {err.excerpt}"[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LlmBadResponse as err:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = str(err.reason)[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LearningStageError as err:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = f"worker: {err}"[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except LearningRunError as err:
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = str(err.code)[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        except asyncio.CancelledError:
            # Keep the tokens this run already spent in the persisted month counter (WR-04).
            # A cancelled (user-stopped) run records its attempt as stopped (D-15).
            c_data = dict(self.coordinator.data)
            c_data["last_attempt"] = self._attempt_record(
                started, "stopped", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            with contextlib.suppress(Exception):
                await self._learning_store.async_save(self._build_learning_store_data())
            raise
        except Exception as err:
            _LOGGER.exception("Learning run %s failed: %s", run_id, err)
            c_data = dict(self.coordinator.data)
            c_data["status"] = "failed"
            c_data["last_error"] = f"{type(err).__name__}: {err}"[:500]
            c_data["last_attempt"] = self._attempt_record(
                started, "failed", dt_util.utcnow().isoformat()
            )
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
        finally:
            c_data = dict(self.coordinator.data)
            c_data["running_job"] = None
            self.coordinator.async_set_updated_data(c_data)
            # Release the slot before queueing follow-up jobs: while this
            # coroutine runs, run_task is still this (not yet done) task, so
            # _job_active() would otherwise see the slot as occupied.
            self.run_task = None
            # D-13: with the opt-in toggle, a run that reached the idle
            # publish chains exactly one validation (trigger after_learning)
            # after releasing the job slot and before queued refits start.
            # Failed, capped, cancelled or blocked runs chain nothing. The
            # chain logs and swallows failures so the refit queue still starts.
            if (
                self._reached_idle
                and not self._stopped
                and bool(
                    self.entry.options.get(
                        CONF_VALIDATE_AFTER_LEARNING,
                        DEFAULT_VALIDATE_AFTER_LEARNING,
                    )
                )
            ):
                try:
                    await self.hub.validation.async_run_validation(
                        trigger="after_learning"
                    )
                except (ServiceValidationError, HomeAssistantError) as err:
                    _LOGGER.warning("After-learning validation failed: %s", err)
                except Exception as err:  # noqa: BLE001 - the chain must never starve the refit queue
                    _LOGGER.warning(
                        "After-learning validation error: %s", err
                    )
            # The slot is free again: start a queued union retrain, if any
            # (D-13, T-04-25).
            await self._async_maybe_refit()
            await self._async_maybe_evidence()

    async def _async_publish_idle(
        self,
        *,
        run_id: str,
        snapshot: RunSnapshot,
        fresh_days: float,
        stage_dir: Path,
        last_run: dict[str, Any],
        pending_candidates: int,
        last_error: str | None,
    ) -> None:
        """Publish the idle state for a successful run after its shadow pass.

        The shadow pass (D-01) runs after the production candidates are saved
        and before the idle publish; its outcome lands in last_run["shadow"]
        and never touches status, last_error, Repairs issues, the candidates
        Store, the review Store or device records (_async_shadow_pass
        contract). A failed, capped or cancelled production run never reaches
        this tail, so it never runs a shadow pass (D-04). Sets
        self._reached_idle so the run's finally can chain the opt-in
        after-learning validation (D-13).
        """
        self._reached_idle = True
        last_run["shadow"] = await self._async_shadow_pass(
            run_id, snapshot, fresh_days, stage_dir
        )
        last_run["ended"] = dt_util.utcnow().isoformat()
        c_data = dict(self.coordinator.data)
        c_data["status"] = "idle"
        c_data["last_run"] = last_run
        # A successful run completes the attempt with last_run's own timestamps (the
        # adjacency truth of must_haves: last_attempt_* == last_run_* on success)
        c_data["last_attempt"] = self._attempt_record(
            last_run.get("started"), "success", last_run["ended"]
        )
        c_data["pending_candidates"] = pending_candidates
        c_data["last_error"] = last_error
        ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LEARNING_CAPPED)
        self.coordinator.async_set_updated_data(c_data)
        await self._learning_store.async_save(self._build_learning_store_data())

    async def _async_llm_rounds(
        self,
        *,
        stage_dir: Path,
        run_id: str,
        snapshot: RunSnapshot,
        digest: dict[str, Any],
        names: dict[str, str] | None,
        prior: list[dict[str, Any]],
        owned: list[str],
        decisions: dict[str, Any],
        state: _RoundsState,
        actives: dict[str, Any],
        with_evidence: bool,
    ) -> None:
        """Run the shared R1 + refinement LLM rounds and compile scores.

        Plan 05-03: the production run and the shadow pass use exactly this
        code path; the caller supplies the identity context (prior/owned/
        decisions), the run id and the names decision. The compile job
        writers read the passed context instead of the coordinator Stores, so
        the shadow pass (empty context, run_id suffixed "-shadow") can never
        surface hidden artefacts as review candidates or device proposals
        (D-01, RESEARCH Pitfalls 1-2). Plan 05-06 (D-17): the production run
        passes the confirmed devices' frozen active artefacts as ``actives``
        so compile_score can gate each differing proposal against the active
        version on the run's validation block; the shadow pass passes {} and
        never produces a gate entry, a proposal or a rejection record. When
        R1 proposes no devices, state.proposal_empty is set and the caller
        handles the empty exit; a cap raised mid-loop propagates with
        state.best_library holding the last accepted, already-paid-for
        library. ``with_evidence`` is forwarded to the compile job verbatim:
        the production run passes True and the shadow pass False, and the
        worker attaches per-candidate evidence only when the job sets that key
        to exactly True (Plan 07-01, D-11).
        """
        cluster_ids = [c["id"] for c in digest.get("clusters", [])]
        plug_ids = [p["id"] for p in digest.get("plugs", [])]
        cluster_kinds = {
            c["id"]: str(c.get("kind", "")) for c in digest.get("clusters", [])
        }

        body = prompts.build_request(
            digest,
            model=snapshot.model,
            profile=snapshot.profile,
            max_tokens=snapshot.max_tokens,
            names=names,
        )

        parsed_proposal = await self._async_ask(
            body,
            round_no=1,
            digest_bytes=prompts.digest_payload_bytes(digest, names),
            cluster_ids=cluster_ids,
            plug_ids=plug_ids,
            cluster_kinds=cluster_kinds,
            snapshot=snapshot,
            stage_dir=stage_dir,
        )

        if not parsed_proposal.get("devices"):
            state.proposal_empty = True
            return

        def _write_job(round_num: int, incumbent: dict[str, Any] | None) -> Path:
            job_compile = {
                "stage_dir": str(stage_dir),
                "episodes_path": str(stage_dir / "episodes.npz"),
                "proposal_path": str(stage_dir / f"proposal_{round_num}.json"),
                "run_id": run_id,
                "round": round_num,
                "model": getattr(self, "_last_llm_model", None) or snapshot.model,
                "incumbent": incumbent,
                "validated": snapshot.validated,
                # D-11/OUT-05: the device's frozen active version anchors
                # the prior pool — appended after the stored candidates,
                # it replaces any stored library artefact with the same
                # signature id. "owned" drives the D-27 twin pass.
                "prior": prior,
                "owned": owned,
                "decisions": decisions,
                # D-17 (Plan 05-06): the frozen active artefact of every
                # confirmed device, keyed by signature id; the worker gates
                # each differing matched artefact against it on this run's
                # validation block. The shadow pass passes {} (no actives),
                # so shadow artefacts never produce a gate entry.
                "actives": actives,
                # Plan 07-01 (D-11): explicit evidence opt-in for the worker;
                # the production caller writes True, the shadow caller False.
                # now_iso stamps each candidate's evidence computed_at, so the
                # HA-free engine reads no wall-clock time itself.
                "with_evidence": with_evidence,
                "now_iso": dt_util.utcnow().isoformat(),
            }
            job_compile_path = stage_dir / f"job_compile_{round_num}.json"
            job_compile_path.write_text(
                json.dumps(job_compile, indent=2), encoding="utf-8"
            )
            return job_compile_path

        job_compile_path = await self.hass.async_add_executor_job(_write_job, 1, None)

        compile_out = _require_outputs(
            "compile_score",
            await self._async_run_stage("compile_score", job_compile_path),
        )

        def _read_library(round_num: int) -> list[dict[str, Any]]:
            lib_path = stage_dir / f"library_{round_num}.json"
            with open(lib_path, "r", encoding="utf-8") as f:
                return json.load(f)

        library_1 = await self.hass.async_add_executor_job(_read_library, 1)

        incumbent = {
            "round": 1,
            "S": float(compile_out["S"]),
            "conflict_share": float(compile_out["conflict_share"]),
        }
        incumbent_proposal = parsed_proposal
        state.best_library = library_1

        for k in range(2, 2 + MAX_REFINE_ROUNDS):
            if not state.best_library:
                break

            def _read_scorecard(r: int) -> dict[str, Any]:
                with open(stage_dir / f"scorecard_{r}.json", "r", encoding="utf-8") as f:
                    return json.load(f)

            try:
                scorecard_data = await self.hass.async_add_executor_job(
                    _read_scorecard, incumbent["round"]
                )
            except Exception as err:
                state.refine_error = f"{type(err).__name__}: {err}"
                break

            body_k = prompts.build_refine_request(
                model=snapshot.model,
                profile=snapshot.profile,
                max_tokens=snapshot.max_tokens,
                digest=digest,
                previous=incumbent_proposal,
                scorecard=scorecard_data,
                names=names,
            )

            state.rounds_run = k
            try:
                proposal_k = await self._async_ask(
                    body_k,
                    round_no=k,
                    digest_bytes=prompts.digest_payload_bytes(digest, names),
                    cluster_ids=cluster_ids,
                    plug_ids=plug_ids,
                    cluster_kinds=cluster_kinds,
                    snapshot=snapshot,
                    stage_dir=stage_dir,
                )
            except LearningCapped:
                raise
            except LlmAuthError as err:
                ir.async_create_issue(
                    self.hass,
                    DOMAIN,
                    ISSUE_LLM_AUTH,
                    is_fixable=True,
                    is_persistent=False,
                    config_entry_id=self.entry.entry_id,
                    data={"entry_id": self.entry.entry_id},
                    severity=ir.IssueSeverity.ERROR,
                    translation_key=ISSUE_LLM_AUTH,
                    translation_placeholders={"host": endpoint_location(snapshot.base_url).host},
                )
                state.refine_error = f"auth_failed (HTTP {err.status})"
                break
            except LlmBudgetError:
                state.refine_error = "provider_budget_exceeded"
                break
            except LlmTimeoutError:
                state.refine_error = "timeout"
                break
            except LlmConnectError:
                state.refine_error = "cannot_connect"
                break
            except LlmHttpError as err:
                state.refine_error = f"http_{err.status}: {err.excerpt}"
                break
            except LlmBadResponse as err:
                state.refine_error = str(err.reason)
                break
            except LearningRunError as err:
                state.refine_error = str(err.code)
                break
            except Exception as err:
                state.refine_error = f"{type(err).__name__}: {err}"
                break

            if not proposal_k.get("devices"):
                continue

            # A worker or file failure in a refinement round stops refining but keeps
            # the incumbent library the run already paid for (WR-03).
            try:
                job_k_path = await self.hass.async_add_executor_job(_write_job, k, incumbent)
                compile_out_k = _require_outputs(
                    "compile_score",
                    await self._async_run_stage("compile_score", job_k_path),
                )
                library_k: list[dict[str, Any]] | None = None
                if compile_out_k["accepted"] is True:
                    library_k = await self.hass.async_add_executor_job(_read_library, k)
            except LearningStageError as err:
                state.refine_error = f"worker: {err}"
                break
            except Exception as err:
                state.refine_error = f"{type(err).__name__}: {err}"
                break

            if library_k is not None:
                incumbent = {
                    "round": k,
                    "S": float(compile_out_k["S"]),
                    "conflict_share": float(compile_out_k["conflict_share"]),
                }
                state.incumbent_round = k
                incumbent_proposal = proposal_k
                state.best_library = library_k

    @staticmethod
    def _shadow_result(
        status: str,
        reason: str | None,
        *,
        run_id: str | None = None,
        rounds: int = 0,
        artefacts: int = 0,
        links: int = 0,
        not_discovered: int = 0,
    ) -> dict[str, Any]:
        """One last_run.shadow-shaped dict (the Plan 05-01 contract)."""
        return {
            "status": status,
            "reason": reason,
            "run_id": run_id,
            "rounds": rounds,
            "artefacts": artefacts,
            "links": links,
            "not_discovered": not_discovered,
        }

    async def _async_shadow_pass(
        self,
        run_id: str,
        snapshot: RunSnapshot,
        fresh_days: float,
        stage_dir: Path,
    ) -> dict[str, Any]:
        """Run the hidden masked shadow pass and store its set (D-01..D-05).

        Only called after a successful production run, with the production
        candidates already saved. The pass re-learns the household on the
        gross residual (every eligible plug added back, no plug reference
        lines in the digest, names=None regardless of the share-names option,
        D-02) on the training block only (the extract carries
        holdout_calendar_days = SHADOW_HOLDOUT_DAYS, D-18), links what it
        found to the plugs locally in the shadow_link stage (D-03) and stores
        the set under run_id f"{run_id}-shadow" — the identity Plan 05-06
        tags plug validation history with, so holdout drift never compares
        errors across shadow sets (D-17).

        The pass never writes status, last_error, Repairs issues, the
        candidates Store, the review Store or device records, and never calls
        _apply_first_labels/_route_device_matches (RESEARCH Pitfalls 1-2). A
        cap hit is a skip (reason cap_reached) that keeps the previous shadow
        set (D-21); the worker counting fewer than MIN_USABLE_DAYS valid days
        is a skip (reason insufficient_data), never a failure (Pitfall 5); a
        cancellation propagates CancelledError; any other failure records
        {status: failed, reason} (<= 200 chars) and leaves the production
        result untouched.
        """
        def _masked_channels() -> list[dict[str, Any]]:
            channel_rows = self.hub.writer.db.channel_rows()
            return sorted(
                (
                    {"channel": ch, "entity_id": entity_id}
                    for ch, entity_id in self._metered_channel_rows(channel_rows)
                ),
                key=lambda m: str(m["entity_id"]),
            )

        masked = await self.hass.async_add_executor_job(_masked_channels)
        if not masked:
            return self._shadow_result("skipped", "no_plugs")
        if fresh_days < MIN_USABLE_DAYS:
            return self._shadow_result("skipped", "insufficient_data")

        try:
            shadow_dir = stage_dir / "shadow"

            def _prepare_shadow_job() -> Path:
                shadow_dir.mkdir(parents=True, exist_ok=True)
                job, _cache_key, _window_end = self._build_extract_job_data(
                    shadow_dir, shadow=True
                )
                job_path = shadow_dir / "job_shadow_extract.json"
                job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
                return job_path

            job_path = await self.hass.async_add_executor_job(_prepare_shadow_job)

            def _read_job_meta() -> tuple[list[str], int, list[str]]:
                with open(job_path, "r", encoding="utf-8") as f:
                    job = json.load(f)
                return (
                    sorted(str(m["entity_id"]) for m in job.get("masked_channels", [])),
                    int(job["grid_s"]),
                    [str(t) for t in job.get("enabled_event_types", [])],
                )

            eligible, grid_s, enabled_types = await self.hass.async_add_executor_job(
                _read_job_meta
            )
            # D-04: the pass costs tokens only when the stored set is missing
            # or stale; not_due returns before any extract or LLM request.
            due = self.hub.validation.shadow_due(
                eligible, grid_s, enabled_types, dt_util.utcnow()
            )
            if due is None:
                return self._shadow_result("skipped", "not_due")

            extract_outputs = await self._async_run_stage("extract", job_path)
            if extract_outputs.get("insufficient_data"):
                # The worker counted the valid days itself (Pitfall 5): a
                # skip, never a failure.
                return self._shadow_result("skipped", "insufficient_data")

            def _read_shadow_digest() -> dict[str, Any]:
                with open(shadow_dir / "digest.json", "r", encoding="utf-8") as f:
                    return json.load(f)

            digest = await self.hass.async_add_executor_job(_read_shadow_digest)

            state = _RoundsState()
            if digest.get("clusters"):
                await self._async_llm_rounds(
                    stage_dir=shadow_dir,
                    run_id=f"{run_id}-shadow",
                    snapshot=snapshot,
                    digest=digest,
                    names=None,
                    prior=[],
                    owned=[],
                    decisions={},
                    state=state,
                    # D-01/D-17: the shadow pass sends an empty actives map,
                    # so shadow artefacts never produce a proposal gate, a
                    # device proposal or a rejection record.
                    actives={},
                    # Plan 07-01: the shadow set stays lean — no evidence is
                    # computed or stored for shadow artefacts.
                    with_evidence=False,
                )

            def _write_shadow_library() -> str:
                if state.best_library:
                    path = shadow_dir / f"library_{state.incumbent_round}.json"
                    payload: Any = state.best_library
                else:
                    # Empty proposal: the set still stores every eligible plug
                    # as not_discovered, with no artefacts.
                    path = shadow_dir / "library_0.json"
                    payload = []
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return str(path)

            library_path = await self.hass.async_add_executor_job(
                _write_shadow_library
            )

            def _read_shadow_job() -> dict[str, Any]:
                with open(job_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            shadow_job = await self.hass.async_add_executor_job(_read_shadow_job)

            link_job = {
                "buffer_path": shadow_job["buffer_path"],
                "stage_dir": str(shadow_dir),
                "library_path": library_path,
                "masked_channels": shadow_job["masked_channels"],
                "grid_s": shadow_job["grid_s"],
                "day_bounds_ms": shadow_job["day_bounds_ms"],
            }

            def _write_link_job() -> Path:
                link_job_path = shadow_dir / "job_shadow_link.json"
                link_job_path.write_text(
                    json.dumps(link_job, indent=2), encoding="utf-8"
                )
                return link_job_path

            link_job_path = await self.hass.async_add_executor_job(_write_link_job)
            await self._async_run_stage("shadow_link", link_job_path)

            def _read_link_result() -> dict[str, Any]:
                with open(shadow_dir / "shadow_link.json", "r", encoding="utf-8") as f:
                    return json.load(f)

            link = await self.hass.async_add_executor_job(_read_link_result)

            shadow_set = {
                "schema_version": SHADOW_SCHEMA_VERSION,
                "created": dt_util.utcnow().isoformat(),
                "run_id": f"{run_id}-shadow",
                "grid_s": shadow_job["grid_s"],
                "enabled_event_types": list(shadow_job["enabled_event_types"]),
                "eligible": sorted(str(m["entity_id"]) for m in masked),
                "train_cutoff_ms": int(link["train_cutoff_ms"]),
                "holdout_days": SHADOW_HOLDOUT_DAYS,
                "artefacts": state.best_library,
                "links": link["links"],
                "not_discovered": link["not_discovered"],
                "plugs": link["plugs"],
            }
            await self.hub.validation.async_store_shadow(shadow_set)

            return self._shadow_result(
                "ran",
                due,
                run_id=f"{run_id}-shadow",
                rounds=state.rounds_run,
                artefacts=len(state.best_library),
                links=len(link["links"]),
                not_discovered=len(link["not_discovered"]),
            )
        except LearningCapped:
            # D-21 shadow cap: keep the production result, the previous
            # shadow set and the idle status; no capped status, no Repairs
            # issue. The shared per-run/month counters already recorded the
            # tokens both passes spent.
            return self._shadow_result("skipped", "cap_reached")
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - any shadow failure is recorded, never raised
            _LOGGER.warning("Shadow pass of run %s failed: %s", run_id, err)
            return self._shadow_result(
                "failed", f"{type(err).__name__}: {err}"[:200]
            )

    async def _async_maybe_rescore(self) -> None:
        """Trigger background candidate re-score once 21 days of data exist."""
        if self._stopped or self._job_active():
            return
        self._start_pending = True
        try:
            await self._async_maybe_rescore_claimed()
        finally:
            self._start_pending = False

    async def _async_maybe_rescore_claimed(self) -> None:
        """Start the rescore when due; the caller holds the start slot."""
        rescore_bounds_ms = await self._async_rescore_due_bounds()
        if rescore_bounds_ms is None:
            return

        # Re-check at the point of task creation: the flag may have been latched
        # by async_stop while this coroutine was suspended in the awaits above
        # (WR-01 re-review).
        if self._stopped:
            return

        self.run_task = self.entry.async_create_background_task(
            self.hass,
            self._async_run_rescore(rescore_bounds_ms),
            "powerguesser_rescore",
        )

    async def _async_rescore_due_bounds(self) -> list[int] | None:
        """Return the rescore's daily bounds when a rescore is due, else None (D-10).

        The exact predicate _async_maybe_rescore uses: a preview candidates
        record (present, candidates, not validated), at least VALIDATED_DAYS
        usable days, a window_end_ms and at least 2 local-midnight bounds from
        that window end to today. Extracted unchanged so the evidence hook can
        evaluate the same rule without duplicating it (a due rescore keeps
        priority over evidence at every hook site).
        """
        if self._candidates_record is None:
            data = await self._candidates_store.async_load()
            if data and isinstance(data, dict):
                self._candidates_record = data
                self.candidates = data.get("candidates", [])

        if not self._candidates_record or not self.candidates:
            return None

        if self._candidates_record.get("validated") is True:
            return None

        usable_days = await self._async_get_usable_days()
        if usable_days < VALIDATED_DAYS:
            return None

        window_end_ms = self._candidates_record.get("window_end_ms")
        if not window_end_ms:
            return None

        now = dt_util.now()
        today_midnight = dt_util.start_of_local_day(now)
        cur_dt = dt_util.start_of_local_day(
            datetime.fromtimestamp(window_end_ms / 1000.0, tz=today_midnight.tzinfo)
        )
        cur_date = cur_dt.date()
        rescore_bounds_ms: list[int] = []
        while cur_date <= today_midnight.date():
            target_dt = datetime.combine(cur_date, time.min, tzinfo=today_midnight.tzinfo)
            local_mid = dt_util.start_of_local_day(target_dt)
            rescore_bounds_ms.append(int(round(local_mid.timestamp() * 1000)))
            cur_date += timedelta(days=1)

        if len(rescore_bounds_ms) < 2:
            return None

        return rescore_bounds_ms

    async def _async_run_rescore(self, rescore_bounds_ms: list[int]) -> None:
        """Execute local replay rescore of preview candidates without LLM."""
        prev_status = self.coordinator.data.get("status", "idle")
        c_data = dict(self.coordinator.data)
        c_data["status"] = "running"
        c_data["running_job"] = "rescore"
        self.coordinator.async_set_updated_data(c_data)

        stage_dir = Path(tempfile.mkdtemp(prefix="powerguesser-rescore-"))
        try:
            def _prepare_rescore() -> Path:
                stage_dir.mkdir(parents=True, exist_ok=True)
                channel_rows = self.hub.writer.db.channel_rows()
                metered_entity_ids = getattr(self.hub.model, "metered_entity_ids", set())
                metered_channels = [
                    ch
                    for ch, entity_id, role, kind in channel_rows
                    if kind == KIND_METERED and entity_id in metered_entity_ids
                ]

                phase_map = {
                    CONF_PHASE_L1: "l1",
                    CONF_PHASE_L2: "l2",
                    CONF_PHASE_L3: "l3",
                }
                phase_opts = {
                    role: self.entry.options.get(conf)
                    for conf, role in phase_map.items()
                }
                phase_channels: dict[str, int] = {}
                for ch, entity_id, role, kind in channel_rows:
                    if kind == KIND_PHASE:
                        for r_name, opt_eid in phase_opts.items():
                            if opt_eid and opt_eid == entity_id:
                                phase_channels[r_name] = ch

                today_tz = dt_util.now().tzinfo
                first_dt = datetime.fromtimestamp(rescore_bounds_ms[0] / 1000.0, tz=today_tz)
                first_weekday = first_dt.weekday()

                profile_data = getattr(self.hub, "profile_coordinator", None)
                snap_data = profile_data.data if profile_data else None
                if snap_data is not None and getattr(snap_data, "effective_s", None) is not None:
                    grid_s = int(round(snap_data.effective_s))
                    enabled_types = [t.value for t in snap_data.enabled]
                else:
                    grid_s = 30
                    enabled_types = [t.value for t in gate(30.0, True)[0]]

                job_rescore = {
                    "buffer_path": self.hass.config.path(BUFFER_DIR, BUFFER_FILE),
                    "stage_dir": str(stage_dir),
                    "grid_s": grid_s,
                    "day_bounds_ms": rescore_bounds_ms,
                    "first_weekday": first_weekday,
                    "enabled_event_types": enabled_types,
                    "metered_channels": metered_channels,
                    "phase_channels": phase_channels,
                    "candidates": self.candidates,
                    # D-11: explicit evidence opt-in (the worker attaches
                    # per_signature evidence only when this key is exactly True)
                    # and the computed_at stamp the HA-free engine reads.
                    "with_evidence": True,
                    "now_iso": dt_util.utcnow().isoformat(),
                }
                job_path = stage_dir / "job_rescore.json"
                job_path.write_text(json.dumps(job_rescore, indent=2), encoding="utf-8")
                return job_path

            job_path = await self.hass.async_add_executor_job(_prepare_rescore)
            res = _require_outputs(
                "rescore", await self._async_run_stage("rescore", job_path)
            )

            per_sig = res["per_signature"]
            for cand in self.candidates:
                sig_id = cand["signature_id"]
                stats = per_sig.get(sig_id, {})
                cand_scores = dict(cand.get("scores", {}))
                cand_scores["val"] = {
                    "share": float(stats.get("share", 0.0)),
                    "n": int(stats.get("n", 0)),
                }
                cand_scores["validated"] = True
                # D-16 merge (Plan 04-07): the rescore entry's key is
                # "validation_error_pct" (rescore never emits the compile-side
                # "claimed_share"); a missing key means None for both value
                # and basis.
                vep = stats.get("validation_error_pct")
                cand_scores["validation_error_pct"] = (
                    None if vep is None else float(vep)
                )
                cand_scores["validation_basis"] = (
                    "learning_split" if vep is not None else None
                )
                cand["scores"] = cand_scores
                # D-11 never-narrow merge: rescore evidence replaces the stored
                # one only when that is missing, not a dict, of another version,
                # or covers no more observed days. The comparison is like for
                # like because every source counts observed_days by the same
                # evidence window contract.
                rescore_evidence = stats.get("evidence")
                if isinstance(rescore_evidence, dict):
                    stored_evidence = cand.get("evidence")
                    if self._should_replace_evidence(stored_evidence, rescore_evidence):
                        cand["evidence"] = rescore_evidence

            if self._candidates_record is not None:
                self._candidates_record["validated"] = True
                self._candidates_record["candidates"] = self.candidates
                await self._candidates_store.async_save(self._candidates_record)

            self._rescore_failures = 0
            c_data = dict(self.coordinator.data)
            c_data["rescore_failures"] = 0
            self.coordinator.async_set_updated_data(c_data)
        except Exception as err:
            _LOGGER.warning("Preview rescore failed: %s", err)
            self._rescore_failures += 1
            c_data = dict(self.coordinator.data)
            c_data["rescore_failures"] = self._rescore_failures
            self.coordinator.async_set_updated_data(c_data)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
            c_data = dict(self.coordinator.data)
            c_data["status"] = prev_status
            c_data["running_job"] = None
            self.coordinator.async_set_updated_data(c_data)
            # Persist only after the status is restored: a stored "running" would turn
            # into "failed / interrupted" on the next start (WR-02).
            await self._learning_store.async_save(self._build_learning_store_data())
            if prev_status in ("idle", "blocked"):
                await self._async_reevaluate_gates()
            # Release the slot (see _async_run's finally), then start a queued
            # union retrain, if any (D-13, T-04-25).
            self.run_task = None
            await self._async_maybe_refit()
            # This finally is the one site where the rescore has just had its turn:
            # rescore_attempted=True skips the rescore-due predicate, so evidence
            # follows even a failing rescore (which stays due and would otherwise
            # starve it, D-10).
            await self._async_maybe_evidence(rescore_attempted=True)

    async def async_request_refit(self, uuid: str) -> None:
        """Queue or start a local union retrain for a device (D-13, T-04-25).

        Single-flight with learning runs and rescores: while a background job
        occupies the slot the retrain is recorded as pending on the device (it
        survives a restart in the cold Store) and starts when the job ends or
        at the next startup; otherwise the slot is claimed and the retrain
        starts immediately as run_task with running_job "refit".
        """
        devices = self.hub.devices
        if devices.record(uuid) is None:
            return
        if self._job_active():
            await devices.async_mark_refit_pending(uuid)
            return
        self._start_pending = True
        try:
            await devices.async_mark_refit_pending(uuid)
            self.run_task = self.entry.async_create_background_task(
                self.hass,
                self._async_run_refit(uuid),
                "powerguesser_refit",
            )
        finally:
            self._start_pending = False

    async def _async_maybe_refit(self) -> None:
        """Start the first queued device retrain once the slot is free (D-13)."""
        if self._stopped or self._job_active():
            return
        self._start_pending = True
        try:
            await self._async_maybe_refit_claimed()
        finally:
            self._start_pending = False

    async def _async_maybe_refit_claimed(self) -> None:
        """Start the first pending device retrain; the caller holds the slot."""
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return
        pending = devices.refit_pending_uuids()
        if not pending:
            return
        uuid = pending[0]
        # Re-check at the point of task creation: the flag may have been latched
        # by async_stop while this coroutine was suspended in the awaits above
        # (WR-01 re-review).
        if self._stopped:
            return
        self.run_task = self.entry.async_create_background_task(
            self.hass,
            self._async_run_refit(uuid),
            "powerguesser_refit",
        )

    async def _async_run_refit(self, uuid: str) -> None:
        """Execute the local union retrain of one device in the niced worker.

        The refit stage rebuilds the episode table from the buffer with a
        fresh 70/30 split, refits each state from the union of the train
        episodes claimed by the device's active version and its merged
        candidates (alias_artefacts), and compares validation scores of the
        current and refitted parameters on the same validation days. An
        accepted refit becomes version n+1 with source "union" through
        async_apply_refit; a rejected one keeps the current version; either
        way the alias routes identity and uuid/unique_ids/entity_ids/energy
        totals are untouched. A failure records last_refit with the error and
        never touches the learning run's last_run, last_error or Repairs
        issues.
        """
        prev_status = self.coordinator.data.get("status", "idle")
        c_data = dict(self.coordinator.data)
        c_data["status"] = "running"
        c_data["running_job"] = "refit"
        self.coordinator.async_set_updated_data(c_data)

        stage_dir = Path(tempfile.mkdtemp(prefix="powerguesser-refit-"))
        try:
            devices = self.hub.devices
            record = devices.record(uuid)
            if record is None:
                return

            fresh_days = await self._async_get_usable_days()
            validated = fresh_days >= VALIDATED_DAYS

            def _prepare_refit() -> Path:
                stage_dir.mkdir(parents=True, exist_ok=True)
                job_extract, _cache_key, _window_end_ms = self._build_extract_job_data(
                    stage_dir
                )
                current = None
                for entry in record.get("versions") or []:
                    if isinstance(entry, Mapping) and entry.get("v") == record.get(
                        "active_v"
                    ):
                        current = entry.get("artefact")
                if not isinstance(current, Mapping):
                    raise LearningStageError("refit: device has no active artefact")
                alias_artefacts = record.get("alias_artefacts") or {}
                extra = [
                    dict(a)
                    for a in alias_artefacts.values()
                    if isinstance(a, Mapping)
                ]
                job_refit = {
                    **job_extract,
                    "stage_dir": str(stage_dir),
                    "validated": validated,
                    "current": dict(current),
                    "extra": extra,
                }
                job_path = stage_dir / "job_refit.json"
                job_path.write_text(json.dumps(job_refit, indent=2), encoding="utf-8")
                return job_path

            job_path = await self.hass.async_add_executor_job(_prepare_refit)
            res = _require_outputs(
                "refit", await self._async_run_stage("refit", job_path)
            )

            now = dt_util.utcnow().isoformat()
            if res["accepted"]:
                result_path = str(res["result"])

                def _read_refit_artefact() -> Mapping[str, Any]:
                    with open(result_path, "r", encoding="utf-8") as f:
                        refit_json = json.load(f)
                    # T-04-26: the artefact crosses the process boundary only
                    # through SignatureArtefact.from_dict; a parse failure is a
                    # refit error that keeps the current version.
                    return SignatureArtefact.from_dict(
                        refit_json["artefact"]
                    ).to_dict()

                artefact = await self.hass.async_add_executor_job(
                    _read_refit_artefact
                )
                await devices.async_apply_refit(uuid, artefact, trained_at=now)
            await devices.async_record_refit(
                uuid,
                {
                    "at": now,
                    "accepted": bool(res["accepted"]),
                    "current_S": float(res["current_S"]),
                    "new_S": float(res["new_S"]),
                },
            )
        except Exception as err:
            _LOGGER.warning("Union retrain of device %s failed: %s", uuid, err)
            await devices.async_record_refit(
                uuid,
                {
                    "at": dt_util.utcnow().isoformat(),
                    "error": f"{type(err).__name__}: {err}"[:200],
                },
            )
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
            c_data = dict(self.coordinator.data)
            c_data["status"] = prev_status
            c_data["running_job"] = None
            self.coordinator.async_set_updated_data(c_data)
            # Persist only after the status is restored (WR-02, same as rescore).
            await self._learning_store.async_save(self._build_learning_store_data())
            # Release the slot (see _async_run's finally), then start the next
            # queued device retrain, if any.
            self.run_task = None
            await self._async_maybe_refit()
            await self._async_maybe_evidence()

    # -- evidence refresh (D-10, D-11, D-12, D-18) ------------------------

    def _reset_evidence_failures(self) -> None:
        """Reset the evidence failure counter (a successful stage or a fresh run)."""
        self._evidence_failures = 0
        c_data = dict(self.coordinator.data or {})
        if c_data.get("evidence_failures") != 0:
            c_data["evidence_failures"] = 0
            self.coordinator.async_set_updated_data(c_data)

    def _evidence_targets(self, *, include_aged: bool) -> list[dict[str, Any]]:
        """Stored candidates whose evidence the refresh stage must (re)compute.

        Only signatures visible in Review are targets: owned, dismissed, merged
        and suppressed signatures are not. Iterates self.candidates -- never
        visible_candidates() itself, which also appends
        devices.unconfirmed_candidates(): those rows are device targets that can
        never gain candidate evidence, so a predicate over them would stay true
        forever (Pitfall 3). ``include_aged`` additionally selects evidence whose
        computed_at is missing, unparsable or older than EVIDENCE_REFRESH_AGE
        (the 6-hourly refresh only, D-11a).
        """
        visible_sigs = {c.get("signature_id") for c in self.visible_candidates()}
        targets: list[dict[str, Any]] = []
        for cand in self.candidates:
            sig_id = cand.get("signature_id")
            if not sig_id or sig_id not in visible_sigs:
                continue
            if self._evidence_needed(cand.get("evidence"), include_aged=include_aged):
                targets.append(cand)
        return targets

    @staticmethod
    def _evidence_needed(evidence: Any, *, include_aged: bool) -> bool:
        """True when a stored candidate evidence dict must be (re)computed.

        Needed when it is missing, not a dict, of another EVIDENCE_VERSION, or
        was computed by rescore (a narrower post-preview window that the stage
        then refreshes over the full window, D-11). With ``include_aged`` a
        stored entry whose computed_at is missing, unparsable or older than
        EVIDENCE_REFRESH_AGE is needed too (D-11a).
        """
        if not isinstance(evidence, dict) or evidence.get("v") != EVIDENCE_VERSION:
            return True
        if evidence.get("source") == "rescore":
            return True
        if include_aged:
            return _evidence_is_aged(evidence)
        return False

    @staticmethod
    def _should_replace_evidence(stored: Any, new: Mapping[str, Any]) -> bool:
        """D-11 never-narrow rule for a rescore evidence merge.

        The stored evidence is replaced only when it is missing, not a dict, of
        another EVIDENCE_VERSION, or covers no more observed days than the new
        one. Both sides measure observed_days by the same evidence window
        contract, so the comparison is like for like.
        """
        if not isinstance(stored, dict) or stored.get("v") != EVIDENCE_VERSION:
            return True
        stored_days = int((stored.get("window") or {}).get("observed_days") or 0)
        new_days = int((new.get("window") or {}).get("observed_days") or 0)
        return new_days >= stored_days

    async def _async_maybe_evidence(
        self, *, include_aged: bool = False, rescore_attempted: bool = False
    ) -> None:
        """Deferred single-flight entry to the evidence refresh stage (D-10, D-12).

        Defers (never raises already_running) while a job holds the slot, and
        stops once the failure cap is reached. A due rescore keeps priority
        (D-10): unless rescore_attempted, the claimed part starts nothing while
        a rescore is due, so evidence follows from that rescore's own finally.
        """
        if self._stopped or self._job_active():
            return
        if self._evidence_failures >= EVIDENCE_MAX_FAILURES:
            return
        self._start_pending = True
        try:
            await self._async_maybe_evidence_claimed(include_aged, rescore_attempted)
        finally:
            self._start_pending = False

    async def _async_maybe_evidence_claimed(
        self, include_aged: bool, rescore_attempted: bool
    ) -> None:
        """Start the evidence stage when targets exist; the caller holds the slot."""
        # Drop device_evidence for devices that are gone before computing targets:
        # a retired or removed device loses its entry at the next hook check,
        # even when no stage starts.
        self._prune_device_evidence()
        cand_targets = self._evidence_targets(include_aged=include_aged)
        device_targets = self._device_evidence_targets(include_aged=include_aged)
        if not cand_targets and not device_targets:
            return
        if not rescore_attempted and await self._async_rescore_due_bounds() is not None:
            # A due rescore takes the slot at its own trigger (restart or the
            # 6-hourly refresh); it never starts here, because that would change
            # when rescores begin and re-run a failing rescore from this finally.
            return
        if self._stopped:
            return
        self.run_task = self.entry.async_create_background_task(
            self.hass,
            self._async_run_evidence(
                [str(t.get("signature_id")) for t in cand_targets],
                device_sources=device_targets,
            ),
            "powerguesser_evidence",
        )

    def _device_evidence_targets(self, *, include_aged: bool) -> list[dict[str, Any]]:
        """evidence_sources() entries whose device evidence must be (re)computed.

        Needed when the stored entry is missing, not a dict, of another
        EVIDENCE_VERSION or computed for another active_v, or (``include_aged``
        only) aged. Reads only devices.evidence_sources(): it never touches a
        device record (D-07a/D-13).
        """
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return []
        targets: list[dict[str, Any]] = []
        for source in devices.evidence_sources():
            uuid = str(source.get("uuid"))
            stored = self._device_evidence.get(uuid)
            if not isinstance(stored, dict) or stored.get("v") != EVIDENCE_VERSION:
                targets.append(source)
                continue
            if int(stored.get("active_v") or 0) != int(source.get("active_v") or 0):
                targets.append(source)
                continue
            if include_aged and _evidence_is_aged(stored):
                targets.append(source)
        return targets

    def _prune_device_evidence(self) -> None:
        """Drop device_evidence entries whose uuid is no longer an evidence source.

        A retired or removed device loses its evidence entry at the next hook
        check even when no stage starts; the next learning Store save persists
        the pruned map. Not called from _build_learning_store_data: that is a
        plain serializer used at every save site, and a pruning side effect there
        would couple every save to the device view.
        """
        if not self._device_evidence:
            return
        devices = getattr(self.hub, "devices", None)
        if devices is None:
            return
        live = {str(source.get("uuid")) for source in devices.evidence_sources()}
        for uuid in [u for u in self._device_evidence if u not in live]:
            del self._device_evidence[uuid]

    def device_evidence(self, uuid: str) -> dict[str, Any] | None:
        """Stored device evidence, or the confirmed-from candidate's evidence (D-07a).

        The stored entry wins while it is a dict of the current
        EVIDENCE_VERSION (an entry for an older active_v keeps showing until the
        refresh replaces it). As a read-time fallback for a just-confirmed device
        (confirm starts no job, so no stage has run since), the evidence of the
        candidate whose signature_id is the record's signature_id is returned
        when it is a dict of the current version; nothing is copied into the
        record (D-13, resolve at read time).
        """
        stored = self._device_evidence.get(str(uuid))
        if isinstance(stored, dict) and stored.get("v") == EVIDENCE_VERSION:
            return stored
        devices = getattr(self.hub, "devices", None)
        record = devices.record(str(uuid)) if devices is not None else None
        sig_id = str((record or {}).get("signature_id") or "")
        if sig_id:
            for cand in self.candidates:
                if str(cand.get("signature_id")) != sig_id:
                    continue
                evidence = cand.get("evidence")
                if isinstance(evidence, dict) and evidence.get("v") == EVIDENCE_VERSION:
                    return evidence
                break
        return None

    async def _merge_evidence(
        self, evidence_map: Mapping[str, Any], device_sources: Sequence[Any]
    ) -> None:
        """Merge a stage result into candidates and device evidence (D-07a, T-07-11).

        Candidate evidence merges by signature_id only into candidates still
        present: a candidate removed mid-stage is never resurrected. Device
        entries merge only for the uuids the stage was sent (still among
        evidence_sources()), recording the active_v the artefact was computed
        for, so a later version change re-targets the entry.
        """
        changed = False
        device_by_key = {
            f"device:{source['uuid']}": (
                str(source["uuid"]),
                int(source.get("active_v") or 0),
            )
            for source in device_sources
        }
        for key, entry in evidence_map.items():
            if not isinstance(entry, dict):
                continue
            device = device_by_key.get(str(key))
            if device is not None:
                uuid, active_v = device
                stored = self._device_evidence.get(uuid)
                # An unparsable artefact yields an n_runs-0 entry (the deliberate
                # D-12b anti-loop mechanism). Keep a good stored entry rather than
                # clobber it, but ALWAYS replace a missing or stale one: a stale
                # entry (wrong EVIDENCE_VERSION or active_v) keeps
                # _device_evidence_targets true, so the success path would re-run
                # the stage from its finally without bound (MR !17 notes 4917/4921).
                if (
                    entry.get("n_runs")
                    or not isinstance(stored, dict)
                    or stored.get("v") != EVIDENCE_VERSION
                    or int(stored.get("active_v") or 0) != active_v
                ):
                    self._device_evidence[uuid] = {**entry, "active_v": active_v}
                    changed = True
        for cand in self.candidates:
            sig_id = str(cand.get("signature_id"))
            evidence = evidence_map.get(sig_id)
            if isinstance(evidence, dict):
                existing = cand.get("evidence")
                # Same rule as the device path: keep a good entry rather than
                # clobber it with an n_runs-0 entry, but always replace a stale one
                # (older EVIDENCE_VERSION, or source == "rescore") — otherwise
                # _evidence_needed stays true and the success path re-runs the
                # stage from its finally without bound (MR !17 notes 4917/4922).
                if (
                    evidence.get("n_runs")
                    or self._evidence_needed(existing, include_aged=False)
                ):
                    cand["evidence"] = evidence
                    changed = True
        self._prune_device_evidence()
        if changed and self._candidates_record is not None:
            self._candidates_record["candidates"] = self.candidates
            await self._candidates_store.async_save(self._candidates_record)

    async def _async_run_evidence(
        self, sig_ids: Sequence[str], *, device_sources: Sequence[Any] = ()
    ) -> None:
        """Run the worker evidence stage for the given candidate signatures.

        Mirrors _async_run_refit's job shape, except the stage directory is
        created and removed in the executor (never on the event loop, project
        constraint). D-18: status is untouched; only running_job becomes
        "evidence", so a restart during the stage never reads as failed.
        """
        sig_id_set = {str(s) for s in sig_ids}
        c_data = dict(self.coordinator.data)
        c_data["running_job"] = "evidence"
        self.coordinator.async_set_updated_data(c_data)

        stage_dir: Path | None = None
        evidence_succeeded = False
        try:
            # The executor thread must not read supervisor state the loop
            # mutates: build the plain deep-copied job inputs here, on the loop.
            cand_artefacts = [
                copy.deepcopy(c)
                for c in self.candidates
                if str(c.get("signature_id")) in sig_id_set
            ]
            device_artefacts = [
                dict(
                    copy.deepcopy(source["artefact"]),
                    signature_id=f"device:{source['uuid']}",
                )
                for source in device_sources
            ]
            now_iso = dt_util.utcnow().isoformat()

            def _prepare_evidence(
                cand_artefacts: list[dict[str, Any]],
                device_artefacts: list[dict[str, Any]],
                now_iso: str,
            ) -> tuple[Path, Path]:
                new_dir = Path(tempfile.mkdtemp(prefix="powerguesser-evidence-"))
                # Only job_extract is used, and never via shadow=True: the shadow
                # job empties metered_channels, which changes the residual and
                # would break the cross-plan evidence window contract.
                job_extract, _cache_key, _window_end_ms = self._build_extract_job_data(
                    new_dir
                )
                job = {
                    **job_extract,
                    "stage_dir": str(new_dir),
                    "candidates": cand_artefacts + device_artefacts,
                    "now_iso": now_iso,
                }
                job_path = new_dir / "job_evidence.json"
                job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
                return new_dir, job_path

            stage_dir, job_path = await self.hass.async_add_executor_job(
                _prepare_evidence, cand_artefacts, device_artefacts, now_iso
            )
            res = _require_outputs(
                "evidence", await self._async_run_stage("evidence", job_path)
            )

            def _read_evidence(path: str) -> dict[str, Any]:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

            evidence_map = await self.hass.async_add_executor_job(
                _read_evidence, str(res["result"])
            )

            await self._merge_evidence(evidence_map, device_sources)
            evidence_succeeded = True
            self._reset_evidence_failures()
        except Exception as err:  # noqa: BLE001 - a stage failure is counted, never raised
            _LOGGER.warning("Evidence refresh failed: %s", err)
            self._evidence_failures += 1
            c_data = dict(self.coordinator.data)
            c_data["evidence_failures"] = self._evidence_failures
            self.coordinator.async_set_updated_data(c_data)
        finally:
            if stage_dir is not None:
                await self.hass.async_add_executor_job(shutil.rmtree, stage_dir, True)
            c_data = dict(self.coordinator.data)
            c_data["running_job"] = None
            self.coordinator.async_set_updated_data(c_data)
            await self._learning_store.async_save(self._build_learning_store_data())
            self.run_task = None
            if evidence_succeeded:
                # D-12b: only a SUCCESS re-checks the hook from its own finally; a
                # failure waits for the next natural trigger, so the cap is spread
                # over time rather than spent in one burst.
                await self._async_maybe_evidence()

    async def _async_run_stage(
        self, stage: str, job_path: Path | str
    ) -> dict[str, Any]:
        """Spawn the worker subprocess for a stage and stream results."""
        argv = [
            sys.executable,
            "-P",
            str(WORKER_ENTRY),
            stage,
            "--job",
            str(job_path),
            "--watch-stdin",
        ]
        if IONICE is not None:
            argv = [IONICE, "-c3"] + argv

        env = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            close_fds=False,
        )
        self._proc = proc
        stderr_tail = bytearray()

        async def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            while True:
                chunk = await proc.stderr.read(1024)
                if not chunk:
                    break
                stderr_tail.extend(chunk)
                if len(stderr_tail) > 2048:
                    del stderr_tail[:-2048]

        drain_task = asyncio.create_task(_drain_stderr())

        async def _read_stdout() -> dict[str, Any]:
            done_payload: dict[str, Any] | None = None
            last_error: str | None = None
            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    event = json.loads(line_str)
                except Exception:
                    continue
                ev_type = event.get("event")
                if ev_type == "done":
                    done_payload = event
                elif ev_type == "error":
                    last_error = event.get("error", "Unknown worker error")
                elif ev_type == "progress" and event.get("step") == "skipped":
                    # IN-01: the reader drops every non-`done`/`error` event, so
                    # a per-artefact skip would otherwise leave no trace. Surface
                    # it in the HA log here; the batch still succeeds.
                    _LOGGER.warning(
                        "Learning stage %s skipped an unusable artefact (id=%s): %s",
                        stage,
                        event.get("id"),
                        event.get("error"),
                    )

            await proc.wait()
            await drain_task
            tail_text = stderr_tail.decode("utf-8", errors="replace").strip()
            if last_error is not None:
                raise LearningStageError(f"{stage}: {last_error}")
            if proc.returncode != 0:
                raise LearningStageError(
                    f"{stage} exited with code {proc.returncode}; stderr: {tail_text}"
                )
            if done_payload is None:
                raise LearningStageError(
                    f"{stage} produced no done event; exit code {proc.returncode}; stderr: {tail_text}"
                )

            stats = done_payload.get("stats", {})
            wall_s = float(stats.get("wall_s", 0.0))
            _LOGGER.info(
                "Learning stage %s finished in %.1f s, peak memory %s",
                stage,
                wall_s,
                peak_memory_text(stats),
            )
            outputs = done_payload.get("outputs", {})
            return {**outputs, "stats": stats}

        try:
            async with asyncio.timeout(STAGE_TIMEOUT_S):
                return await _read_stdout()
        except TimeoutError:
            tail_text = stderr_tail.decode("utf-8", errors="replace").strip()
            raise LearningStageError(
                f"{stage} timed out after {STAGE_TIMEOUT_S} s; stderr: {tail_text}"
            ) from None
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            if not drain_task.done():
                drain_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain_task

    def _record_model_mismatch(self, snapshot: RunSnapshot, returned: str) -> None:
        """Count a learning request answered by another model than requested.

        A router (for example LiteLLM) can silently fall back to another model, so
        the run records the requested and returned model and the number of
        affected requests, and warns once per run. The run is not aborted.
        """
        c_data = dict(self.coordinator.data)
        current = c_data.get("model_mismatch_run")
        if not current:
            _LOGGER.warning(
                "LLM endpoint %s answered a learning request for model %s with model %s; "
                "the router may have fallen back to another model",
                endpoint_location(snapshot.base_url).host,
                snapshot.model,
                returned,
            )
            current = {"requested": snapshot.model, "returned": returned, "requests": 0}
        c_data["model_mismatch_run"] = {
            **current,
            "requests": int(current.get("requests", 0)) + 1,
        }
        self.coordinator.async_set_updated_data(c_data)

    def _record_trimmed(self, paths: Sequence[str]) -> None:
        """Count free-text fields of an accepted proposal that were cut to their length cap.

        Providers do not reliably honour the caps (03-SPIKE.md follow-up 7), so over-long
        labels, rationales, analysis and merge reasons are trimmed locally instead of costing
        a repair turn. This is a soft note: logged at INFO (paths only, no text) and counted
        in the run, never an error.
        """
        if not paths:
            return
        _LOGGER.info(
            "Trimmed %d over-long free-text field(s) of the LLM proposal to their length caps: %s",
            len(paths),
            ", ".join(paths),
        )
        c_data = dict(self.coordinator.data)
        c_data["trimmed_run"] = int(c_data.get("trimmed_run") or 0) + len(paths)
        self.coordinator.async_set_updated_data(c_data)

    def _record_tokens(
        self,
        usage: Any,
        *,
        est_in: int,
        content_bytes: int,
        caps: Caps,
    ) -> tuple[int, int, bool]:
        """Record tokens and cost used by an LLM completion.

        Returns (prompt tokens, completion tokens, reported): reported is False when the
        counts are local estimates, which must never feed the truncation guard or its
        calibration (WR-13).
        """
        has_usage = False
        prompt_t = 0
        comp_t = 0
        if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
            p_val = usage.get("prompt_tokens")
            c_val = usage.get("completion_tokens")
            if p_val is not None and c_val is not None:
                prompt_t = int(p_val)
                comp_t = int(c_val)
                has_usage = True
        elif isinstance(usage, (tuple, list)) and len(usage) >= 2:
            prompt_t = int(usage[0])
            comp_t = int(usage[1])
            has_usage = True

        if not has_usage:
            prompt_t = est_in
            comp_t = output_estimate(content_bytes)

        self._add_tokens(prompt_t, comp_t, caps)
        return prompt_t, comp_t, has_usage

    def _add_tokens(self, prompt_t: int, comp_t: int, caps: Caps) -> None:
        """Add tokens (and cost) to the run and month counters and persist them soon.

        The delayed save keeps the month counters across a reload or restart that
        cancels the run, also for endpoints that report no usage (WR-04).
        """
        c_data = dict(self.coordinator.data)
        tokens_run = c_data.get("tokens_run", {"input": 0, "output": 0})
        tokens_run = {
            "input": int(tokens_run.get("input", 0)) + prompt_t,
            "output": int(tokens_run.get("output", 0)) + comp_t,
        }
        c_data["tokens_run"] = tokens_run

        curr_month_key = month_key(dt_util.now())
        month_info = c_data.get(
            "month",
            {"key": curr_month_key, "input": 0, "output": 0, "cost": None},
        )
        if month_info.get("key") != curr_month_key:
            month_info = {"key": curr_month_key, "input": 0, "output": 0, "cost": None}

        cost_add = request_cost(prompt_t, comp_t, caps)
        new_cost = month_info.get("cost")
        if cost_add is not None:
            new_cost = (float(new_cost) if new_cost is not None else 0.0) + cost_add

        month_info = {
            "key": curr_month_key,
            "input": int(month_info.get("input", 0)) + prompt_t,
            "output": int(month_info.get("output", 0)) + comp_t,
            "cost": new_cost,
        }
        c_data["month"] = month_info
        self.coordinator.async_set_updated_data(c_data)
        self._learning_store.async_delay_save(self._build_learning_store_data, 1)

    async def _async_ask(
        self,
        body: dict[str, Any],
        *,
        round_no: int,
        digest_bytes: int,
        cluster_ids: Sequence[str],
        plug_ids: Sequence[str],
        snapshot: RunSnapshot,
        cluster_kinds: Mapping[str, str] | None = None,
        stage_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Request completion from LLM with classification, validation, repair, and salvage.

        digest_bytes is the escaped digest's share of body's payload bytes; repair turns
        resend no digest and are checked with 0.
        """
        if stage_dir is None:
            stage_dir = Path(self.hass.config.path(BUFFER_DIR, LEARNING_DIR)) / getattr(
                self, "_current_run_id", "current"
            )

        client = LlmClient(self.hass, snapshot.base_url, snapshot.api_key)
        current_body = body
        turn = 1
        # The last parsed object, its trimmed paths and its own validation errors; salvage
        # must pair the object with the errors of the same turn (WR-06).
        last_parsed_obj: object = None
        last_trimmed: list[str] = []
        last_parsed_errors: list[str] = []

        calib_key = f"{endpoint_location(snapshot.base_url).host}|{snapshot.model}"
        calib_entry = self._calibration.get(calib_key)
        calib_ratio = (
            float(calib_entry["ratio"]) if calib_entry and "ratio" in calib_entry else None
        )
        calib_n = int(calib_entry["n"]) if calib_entry and "n" in calib_entry else 0

        while True:
            turn_no = turn
            req_to_write = current_body

            n_bytes = payload_bytes(current_body)
            est_in = estimate_tokens(n_bytes, calibrated_ratio=calib_ratio)

            usage = self._get_current_usage()
            cap = check_request(est_in, snapshot.max_tokens, usage, snapshot.caps)
            if cap is not None:
                limit = getattr(snapshot.caps, cap)
                raise LearningCapped(cap, limit)

            def _write_req() -> None:
                (stage_dir / f"request_{round_no}_{turn_no}.json").write_text(
                    json.dumps(req_to_write, indent=2), encoding="utf-8"
                )
                if turn_no == 1:
                    (stage_dir / f"request_{round_no}.json").write_text(
                        json.dumps(req_to_write, indent=2), encoding="utf-8"
                    )

            await self.hass.async_add_executor_job(_write_req)

            def _on_billable_failure(will_retry: bool, est_in: int = est_in) -> None:
                # A timed-out or 5xx attempt may have been billed without usage: count the
                # worst case, and refuse a retry that would exceed a cap (WR-04).
                self._add_tokens(est_in, snapshot.max_tokens, snapshot.caps)
                if will_retry:
                    retry_cap = check_request(
                        est_in, snapshot.max_tokens, self._get_current_usage(), snapshot.caps
                    )
                    if retry_cap is not None:
                        raise LearningCapped(retry_cap, getattr(snapshot.caps, retry_cap))

            resp = await client.async_complete(
                current_body, on_billable_failure=_on_billable_failure
            )
            self._last_llm_model = resp.model
            if not model_matches(snapshot.model, resp.model):
                self._record_model_mismatch(snapshot, resp.model)

            # The first successful LLM response of any run deletes ISSUE_LLM_AUTH
            ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LLM_AUTH)

            # Update tokens
            prompt_t, comp_t, usage_reported = self._record_tokens(
                resp.usage,
                est_in=est_in,
                content_bytes=resp.content_bytes,
                caps=snapshot.caps,
            )

            if usage_reported:
                if snapshot.truncation_guard and truncation_suspected(
                    prompt_t, n_bytes, digest_bytes
                ):
                    raise LearningRunError(
                        f"prompt_truncated: endpoint reported {prompt_t} prompt tokens for {n_bytes} bytes; if your provider counts tokens differently, turn off the truncation check in the PowerGuesser options, Learning"
                    )
                new_ratio, new_n = update_ratio(calib_ratio, calib_n, prompt_t, n_bytes)
                calib_ratio = new_ratio
                calib_n = new_n
                self._calibration[calib_key] = {"ratio": new_ratio, "n": new_n}
                await self._learning_store.async_save(self._build_learning_store_data())

            # Classify response
            if resp.finish_reason == "length":
                _LOGGER.warning(
                    "Learning reply hit the per-call completion limit of %d tokens "
                    "(output_truncated). Raise 'Max completion tokens per call' in the "
                    "PowerGuesser options, Learning; reasoning models spend hidden "
                    "reasoning tokens inside this limit.",
                    snapshot.max_tokens,
                )
                raise LearningRunError("output_truncated")

            msg = resp.message or {}
            refusal = msg.get("refusal")
            if refusal:
                raise LearningRunError("refused")

            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            if (
                content is None or (isinstance(content, str) and not content.strip())
            ) and not tool_calls:
                raise LearningRunError("empty_output")

            # Extract raw output text
            if isinstance(content, str) and content.strip():
                raw_text = content
            elif tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                raw_text = tool_calls[0].get("function", {}).get("arguments", "")
            else:
                raw_text = ""

            parsed_obj, _ = extract_json(msg)
            parsed_proposal, errors, trimmed = proposal.parse_proposal(
                msg, cluster_ids=cluster_ids, plug_ids=plug_ids, cluster_kinds=cluster_kinds
            )
            if parsed_obj is not None:
                last_parsed_obj = parsed_obj
                last_trimmed = trimmed
                last_parsed_errors = errors

            if not errors and parsed_proposal is not None:
                self._record_trimmed(trimmed)

                def _write_prop() -> None:
                    (stage_dir / f"proposal_{round_no}.json").write_text(
                        json.dumps(parsed_proposal, indent=2), encoding="utf-8"
                    )

                await self.hass.async_add_executor_job(_write_prop)
                return parsed_proposal

            if turn <= MAX_REPAIR_TURNS:
                current_body = prompts.build_repair_request(
                    model=snapshot.model,
                    profile=snapshot.profile,
                    max_tokens=snapshot.max_tokens,
                    cluster_ids=cluster_ids,
                    plug_ids=plug_ids,
                    invalid_output=raw_text[:REPAIR_OUTPUT_MAX_CHARS],
                    errors=sorted(errors),
                )
                digest_bytes = 0
                turn += 1
                continue

            # After MAX_REPAIR_TURNS repair requests failed, attempt salvage
            if last_parsed_obj is not None:
                salvaged = proposal.salvage(
                    last_parsed_obj,
                    last_parsed_errors,
                    cluster_ids=cluster_ids,
                    plug_ids=plug_ids,
                    cluster_kinds=cluster_kinds,
                )
                if salvaged is not None:
                    salvaged_proposal, _ = salvaged
                    self._record_trimmed(last_trimmed)

                    def _write_salvaged() -> None:
                        (stage_dir / f"proposal_{round_no}.json").write_text(
                            json.dumps(salvaged_proposal, indent=2), encoding="utf-8"
                        )

                    await self.hass.async_add_executor_job(_write_salvaged)
                    return salvaged_proposal

            raise LearningRunError("malformed_output")



async def async_remove_learning_data(hass: HomeAssistant) -> None:
    """Remove learning files and stores."""
    learning_dir = hass.config.path(BUFFER_DIR, LEARNING_DIR)
    await hass.async_add_executor_job(shutil.rmtree, learning_dir, True)
    for key in (
        LEARNING_STORE_KEY,
        CANDIDATES_STORE_KEY,
        CONSENT_STORE_KEY,
        REVIEW_STORE_KEY,
    ):
        await Store(hass, LEARNING_STORES_VERSION, key).async_remove()
    # Plug sessions (Plan 06-06, D-09) live in their own Store. Without this the
    # records survive entry removal, so a reinstall restores stale open sessions
    # and a ghost session plug is re-tracked and silently subtracted from the
    # remainder (REV-04).
    await Store(hass, SESSIONS_STORE_VERSION, SESSIONS_STORE_KEY).async_remove()
