"""Constants for the PowerGuesser integration."""

from datetime import timedelta

from homeassistant.const import Platform

DOMAIN = "powerguesser"
PLATFORMS = [Platform.SENSOR, Platform.BUTTON]
OWN_PLATFORM = DOMAIN

BUFFER_DIR = "powerguesser"
BUFFER_FILE = "buffer.sqlite"
FLUSH_INTERVAL = timedelta(seconds=60)
RETENTION_FIRST_RUN_S = 300
RETENTION_INTERVAL = timedelta(days=1)

SIGNAL_PREFS_UPDATED = "powerguesser_energy_prefs_updated"
SIGNAL_PROFILE_SOURCES = "powerguesser_profile_sources"
STRUCTURAL_GRACE_S = 60
TICK_INTERVAL = timedelta(seconds=10)

PROFILE_STORE_KEY = "powerguesser.profile"
PROFILE_STORE_VERSION = 1
PROFILE_SAVE_DELAY_S = 600

CONF_PHASE_L1 = "phase_l1"
CONF_PHASE_L2 = "phase_l2"
CONF_PHASE_L3 = "phase_l3"
PHASE_OPTION_ROLES = {
    CONF_PHASE_L1: "l1",
    CONF_PHASE_L2: "l2",
    CONF_PHASE_L3: "l3",
}
CONF_INVERT_PHASES = "invert_phases"

BOOTSTRAP_START_TIMEOUT_S = 900

CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_PROBE = "llm_probe"
# Options-flow form field only (never stored): drop the stored API key
CONF_LLM_CLEAR_KEY = "llm_clear_key"

LLM_TIMEOUT_S = 600
LLM_CONNECT_TIMEOUT_S = 15
PROBE_RUNG_TIMEOUT_S = 90
PROBE_MAX_TOKENS = 200
LLM_MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 2.0
RESPONSE_MAX_BYTES = 262_144
ERROR_BODY_MAX_CHARS = 500
# Secrets shorter than this are not redacted from provider error text (IN-09): such
# keys are placeholders for auth-less local endpoints (`x`, `ollama`, `EMPTY`) and
# protect nothing, while redacting them garbles the diagnostic text.
REDACT_MIN_SECRET_CHARS = 8

SERVICE_LEARN = "learn"
SERVICE_PREVIEW = "preview_learning_payload"
LEARNING_DIR = "learning"
PREPARE_DIR = "prepare"
LEARNING_STORE_KEY = "powerguesser.learning"
CANDIDATES_STORE_KEY = "powerguesser.candidates"
CONSENT_STORE_KEY = "powerguesser.consent"
REVIEW_STORE_KEY = "powerguesser.review"
LEARNING_STORES_VERSION = 1

# Plug sessions (Plan 06-06, D-09): the session records live in their own Store
# so a ReviewStore decision writer (each saves {"decisions": ...}) can never
# erase them (RESEARCH Pitfall 7). Version 1 is the contract Plan 06-09 reads.
SESSIONS_STORE_KEY = "powerguesser.review_sessions"
SESSIONS_STORE_VERSION = 1

# Capture records (Plan 08-01, D-05): captures live in their own Store so a
# ReviewStore decision writer can never erase them (RESEARCH Pitfall 7). The
# record schema and this v1 key are consumed by Phase 9 learning and the Phase
# 10 websocket contract, so they are frozen from day one.
CAPTURES_STORE_KEY = "powerguesser.captures"
CAPTURES_STORE_VERSION = 1

# Holdout validation Stores (Plan 05-01): the stored shadow set (D-01 contract;
# written by Plan 05-03) and the latest validation report plus the stored
# per-plug class map. Version 1 schemas are costly: readers and writers across
# Plans 05-01..05-06 pin them.
SHADOW_STORE_KEY = "powerguesser.shadow"
VALIDATION_STORE_KEY = "powerguesser.validation"
VALIDATION_STORES_VERSION = 1
VALIDATION_DIR = "validation"
SERVICE_RUN_VALIDATION = "run_validation"

# Capture admin services (Plan 08-01, CAP-06). SERVICE_CAPTURE_LIST is added by
# Plan 08-01 Task 2; the live capture actions by Plan 08-03 Task 3.
SERVICE_CAPTURE_WINDOW = "capture_window"
SERVICE_CAPTURE_LIST = "capture_list"
SERVICE_CAPTURE_START = "capture_start"
SERVICE_CAPTURE_STOP = "capture_stop"
SERVICE_CAPTURE_CANCEL = "capture_cancel"
# Plan 08-05 Task 2 adds the delete/re-assign actions and the orphan issue.
SERVICE_CAPTURE_DELETE = "capture_delete"
SERVICE_CAPTURE_REASSIGN = "capture_reassign"
ISSUE_CAPTURE_ORPHANS = "capture_orphans"

# Device lifecycle Stores (Plan 04-03): cold device records and hot runtime
# state (energy totals, detector state). Version 1 schemas are pinned by
# tests/ha/test_device_store_schema.py (cross-plan schema-drift guard, D-11).
DEVICES_STORE_KEY = "powerguesser.devices"
DEVICE_STATE_STORE_KEY = "powerguesser.device_state"
DEVICES_STORE_VERSION = 1
SUBENTRY_TYPE_DEVICE = "device"

# Online detector grid and hot-state save delays (Plan 04-04). GRID_GUARD_S is
# one hub tick: a grid point is only evaluated once the aligner had time to
# settle after the grid boundary (RESEARCH Pitfall 8). The hot Store is
# delay-saved; a device transition (ON set change) saves sooner than steady
# state so a crash loses at most the transition window.
GRID_GUARD_S = 10.0
DEVICE_STATE_SAVE_DELAY_S = 60
DEVICE_TRANSITION_SAVE_DELAY_S = 10
CONF_SHARE_NAMES = "share_entity_names"
CONF_MAX_TOKENS = "max_tokens_per_call"
CONF_RUN_INPUT_TOKENS = "run_input_tokens"
CONF_RUN_OUTPUT_TOKENS = "run_output_tokens"
CONF_MONTH_INPUT_TOKENS = "month_input_tokens"
CONF_MONTH_OUTPUT_TOKENS = "month_output_tokens"
CONF_PRICE_INPUT = "price_input_per_1m"
CONF_PRICE_OUTPUT = "price_output_per_1m"
CONF_MONTH_COST_LIMIT = "month_cost_limit"
CONF_SCHEDULE = "schedule"
CONF_SCHEDULE_WEEKDAY = "schedule_weekday"
CONF_SCHEDULE_TIME = "schedule_time"
CONF_TRUNCATION_GUARD = "truncation_guard"
# D-13: opt-in chain that runs the local holdout validation after every
# successful learning run (default off; sends nothing to the LLM).
CONF_VALIDATE_AFTER_LEARNING = "validate_after_learning"

# Folded todo 2026-11-18 (raise max tokens for reasoning models): reasoning models
# spend hidden reasoning tokens inside max_completion_tokens, so the old 16 000
# per-call default truncated kimi-for-coding (about 27 000 tokens needed). 32 000
# worked for the reference reasoning model. The per-run cap doubles to 256 000 so a
# run still admits 8 full-size calls, exactly as the old 16 000 / 128 000 pair did.
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_RUN_INPUT_TOKENS = 300_000
DEFAULT_RUN_OUTPUT_TOKENS = 256_000
DEFAULT_MONTH_INPUT_TOKENS = 2_000_000
DEFAULT_MONTH_OUTPUT_TOKENS = 600_000
DEFAULT_SCHEDULE = "off"
DEFAULT_SCHEDULE_WEEKDAY = "sun"
DEFAULT_SCHEDULE_TIME = "03:00:00"
DEFAULT_VALIDATE_AFTER_LEARNING = False
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

STAGE_TIMEOUT_S = 3600
MIN_LEARNING_DAYS = 7.0
VALIDATED_DAYS = 21.0
USABLE_DAYS_REFRESH = timedelta(hours=6)

# Evidence refresh failure cap (D-12/D-12a): shared by the refresh stage
# (Plan 07-04) and the Review text (Plan 07-05). Persisted beside
# rescore_failures in the learning Store; at the cap the check stops starting
# the stage until a learning run (or a successful stage) resets it.
EVIDENCE_MAX_FAILURES = 3

# D-11a: the age refresh threshold. The existing 6-hourly refresh recomputes
# candidate and device evidence older than this (no new timer).
EVIDENCE_REFRESH_AGE = timedelta(days=7)

# D-13: candidate-only decorations that must never enter a stored device
# artefact (versions, pending proposals, alias artefacts). "origin" is reserved
# for EVID-05 capture provenance (nothing sets it in this phase); "evidence" is
# the candidate evidence dict, which is display-only and lives outside every
# device record. One tuple keeps all three D-13 copy sites in lockstep.
CANDIDATE_DECORATION_KEYS: tuple[str, ...] = (
    "identity",
    "fingerprint",
    "first_label",
    "evidence",
    "origin",
)

PREPARE_STARTUP_DELAY_S = 300
PREPARE_DEBOUNCE_S = 30

ISSUE_LLM_AUTH = "llm_auth"
ISSUE_LEARNING_CONSENT = "learning_consent"
ISSUE_LEARNING_CAPPED = "learning_capped"
MAX_REPAIR_TURNS = 2
MAX_REFINE_ROUNDS = 2
REPAIR_OUTPUT_MAX_CHARS = 16_384


