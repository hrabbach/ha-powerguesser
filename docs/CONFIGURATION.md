<!-- generated-by: gsd-doc-writer -->
# Configuration

PowerGuesser is a Home Assistant custom integration. **It uses no environment variables and no config files of its own** — everything is configured in the Home Assistant UI through the integration's config flow and options flow, and persisted by Home Assistant in the config entry (`.storage/core.config_entries`) and in integration-owned storage files under `/config/.storage/`.

## Where configuration lives

| Location | Content |
| --- | --- |
| Config entry `data` | LLM endpoint settings: base URL, API key, model, probe profile. Written by the "LLM endpoint" options step after a successful probe. |
| Config entry `options` | Phase mapping, learning schedule, token/cost caps, name sharing, validation toggle. Written by the other options steps. |
| Config entry subentries | Confirmed devices (`device` subentries). Devices are created only by confirming a learning candidate under *Configure → Review candidates* — clicking the subentry "Add device" button aborts with the `use_review` reason directing you there. |
| `/config/.storage/powerguesser.*` | Runtime stores owned by the integration: energy profile, learning state, candidates, consent record, review decisions, plug sessions, device records, capture records (`powerguesser.captures`), shadow/validation data. These are managed by the integration; do not edit them by hand. |
| `/config/powerguesser/` | SQLite sample buffer (`buffer.sqlite`), plus the `learning/` working directory with `prepare/` and `validation/` nested inside it. |

Energy dashboard sources (grid / gross power sensors and metered devices) are **not** configured in PowerGuesser at all — they are read automatically from Home Assistant's Energy preferences. Setup aborts with an explanatory message if no usable gross source exists.

## Config flow (initial setup)

*Settings → Devices & services → Add integration → PowerGuesser.*

- Single-instance only: adding a second entry aborts with `single_instance_allowed`.
- Setup creates the entry with empty `data`; all real configuration happens afterwards in the options flow.
- Setup aborts early if the Energy dashboard has no usable power source (see the abort reasons in `strings.json`).

## Options flow

Open the integration's **Configure** button. The menu offers five areas: **Per-phase power (L1/L2/L3)**, **LLM endpoint**, **Learning**, **Review candidates**, and **Plug sessions**.

### Phase mapping (`phase_l1`, `phase_l2`, `phase_l3`, `invert_phases`)

Optional. Assign up to three power sensors (`sensor` with device class `power`) to phases L1–L3 so per-phase readings can be reconciled against the grid meter.

- Leaving all three empty clears the mapping.
- Validation rejects: the same entity in two fields (`duplicate_phase`), a PowerGuesser-owned sensor (`phase_self_reference`), and an entity that is already a gross/metered Energy source (`phase_is_source`).
- After saving, the live phase sum is compared with the grid reading. On a mismatch or inverted result you get a confirmation step where `invert_phases` can be toggled (pre-selected when the sum looks inverted); the check is non-blocking. The stored `invert_phases` flag is carried forward on later saves.
- The form suggests candidate sensors found on the grid meter's device.

### LLM endpoint (`llm_base_url`, `llm_api_key`, `llm_model`)

Required before any learning run. Points PowerGuesser at an OpenAI-compatible chat-completions endpoint.

| Field | Required | Notes |
| --- | --- | --- |
| `llm_base_url` | Yes (to use LLM features) | Must be `http` or `https`, no userinfo, query, or fragment. A trailing `/v1` is normalised to exactly one `/v1`; requests go to `{base_url}/chat/completions`. Clearing the field removes the whole LLM configuration. |
| `llm_api_key` | No | Sent as `Authorization: Bearer …` when non-empty; local endpoints without auth work with an empty key. A stored key is reused only when scheme, host, and port of the new URL match the stored one — it is never forwarded to a different origin. Tick `llm_clear_key` to drop the stored key. |
| `llm_model` | Yes (when a base URL is set) | Model name sent in the request body. |

Saving runs a capability probe (progress step, per-rung timeout 90 s) against the endpoint: it walks a strategy ladder, determines whether the endpoint wants `max_completion_tokens` or `max_tokens` and whether it accepts `temperature`, and verifies the model returns the expected JSON. The resulting profile is stored in the config entry as `llm_probe`. Failures surface as `invalid_auth` (401/403), `cannot_connect`, or `model_unusable`. If the endpoint answers with a different model than requested (router fallback), the mismatch is stored and logged but not failed.

### Learning (options)

Consent for sharing data with the LLM endpoint is not an options-flow field: it is granted through a Repairs/notification prompt the first time a learning run needs it, and is stored per endpoint origin (scheme + host + port) together with the name-sharing flag. Changing the endpoint origin or the `share_entity_names` flag asks again.

| Option key | Required | Default | Description |
| --- | --- | --- | --- |
| `share_entity_names` | Yes (form field) | `false` | Share entity/device names with the LLM. Off = anonymised labels. Changing it invalidates stored consent. |
| `max_tokens_per_call` | Yes | `32000` | Max output tokens per LLM call (256–1 000 000). Must not exceed `run_output_tokens`. Reasoning models need headroom — they spend hidden reasoning tokens inside this budget. |
| `run_input_tokens` | Yes | `300000` | Input-token cap per learning run (1 000–1 000 000 000). |
| `run_output_tokens` | Yes | `256000` | Output-token cap per learning run (1 000–1 000 000 000). |
| `month_input_tokens` | Yes | `2000000` | Monthly input-token cap (1 000–1 000 000 000). |
| `month_output_tokens` | Yes | `600000` | Monthly output-token cap (1 000–1 000 000 000). |
| `price_input_per_1m` | No | unset | Price per 1M input tokens (0–1000, step 0.001), in your own currency. Only stored when set. |
| `price_output_per_1m` | No | unset | Price per 1M output tokens. Only stored when set. |
| `month_cost_limit` | No | unset | Monthly cost cap (0.01–100 000). Requires both prices to be set (`cost_limit_needs_prices` otherwise). |
| `schedule` | Yes | `off` | `off` or `weekly` automatic learning runs. |
| `schedule_weekday` | Yes | `sun` | Weekday for the weekly run (`mon`–`sun`). |
| `schedule_time` | Yes | `03:00:00` | Time of day for the weekly run (local time). |
| `truncation_guard` | Yes | `true` | Guards against truncated LLM JSON responses. |
| `validate_after_learning` | Yes | `false` | After each successful learning run, run the local holdout validation automatically. Sends nothing to the LLM. |

Hitting a token or cost cap pauses learning and raises a `learning_capped` repair issue.

### Review candidates & plug sessions

These options steps manage learned devices rather than settings: confirming/dismissing candidates, accepting parameter updates, rolling back device versions, merging devices, renaming, and starting/ending plug sessions (assigning a plug power sensor to a candidate or device to gather validation data). No persistent options are written here.

Opening a candidate — or a confirmed device — shows an **evidence** block under its name, with run times in the Home Assistant time zone:

- a summary line with the total run count and the observation span;
- the newest 5 occurrences (date, start, duration, step W, Wh), then "…and N more";
- a **Rhythm** line (typical time of day and typical gap), suppressed below 5 runs;
- a **Profile** line (typical step W, duration, Wh per run, Wh per day, and the phase when phase channels exist) — any missing value prints `n/a`, never `0`;
- **Hints** under an explicit heuristic label (few runs, wide step spread, short coverage).

The hints describe the data behind the candidate; they are **heuristics, not an accuracy figure**, and no percentage is ever shown.

The evidence block has four states. It is **"being computed"** in the background right after a restart or an upgrade from an earlier version — reopen Review in a few minutes. It becomes **ready** once the background `evidence` worker stage has run. After a run of failed refresh attempts it reads "**Evidence unavailable, will retry at the next learning run**". A confirmed device shows the same block, replayed from its **active learned parameters**, so it is directly comparable to the candidate it came from. Review is read-only: it never starts a learning or evidence job, and the background refresh sends nothing to the LLM endpoint.

Plug sessions assign a plug power sensor to a candidate or device to gather validation data; a candidate-target plug reading is subtracted from the remainder while the session is active, a confirmed device's own sensors keep running. Sessions stay open until you end them or their end date passes.

## Required vs optional at a glance

- **Required for learning to run**: an LLM endpoint (base URL + model), a granted consent for that endpoint origin, and at least 7 days of usable history (`MIN_LEARNING_DAYS`). A missing LLM endpoint surfaces as the `llm_auth` repairs issue, and a missing consent as the `learning_consent` repairs prompt; insufficient history creates no repairs issue — it only blocks the run with the `insufficient_data` reason.
- **Optional**: phase mapping, API key (for auth-less local endpoints), prices and cost limit, weekly schedule (default off), name sharing (default off), validation-after-learning (default off).

## Defaults

Defaults are applied when an option has never been saved. The token/cost cap and schedule defaults are defined in `custom_components/powerguesser/const.py` (`DEFAULT_*` constants); the `share_entity_names` (`false`) and `truncation_guard` (`true`) defaults are inline literals in `config_flow.py`/`learning.py`. Token and cap defaults:

| Setting | Default |
| --- | --- |
| `max_tokens_per_call` | 32 000 |
| `run_input_tokens` | 300 000 |
| `run_output_tokens` | 256 000 |
| `month_input_tokens` | 2 000 000 |
| `month_output_tokens` | 600 000 |
| `schedule` | `off` (weekly default: Sunday 03:00:00) |
| `share_entity_names` | off |
| `truncation_guard` | on |
| `validate_after_learning` | off |

Fixed (non-configurable) limits from `const.py` include: LLM request timeout 600 s (connect 15 s), 3 attempts with 2 s backoff, 256 KiB max response size, 60 s buffer flush interval, 10 s hub tick.

## Per-environment overrides

None. Home Assistant config entries are per-instance: there is no dev/staging/production layering, no `.env` files, and no `NODE_ENV`-style switching. To replicate a setup on another Home Assistant instance, configure the same options in its UI; the stored LLM API key and consent records are per-instance and are never exported. A diagnostics download includes the latest LLM endpoint probe result, with the base URL and the API key **redacted**.

## Guided capture (admin actions)

Besides the options flow, PowerGuesser exposes seven **admin-only** capture actions
(call them as services, or from *Developer tools → Actions*). They teach the integration one
real device run without waiting for a learning pass. Captures stay local — no capture data is
sent to the LLM endpoint in this version (captures join the learning payload only in a later
version).

| Action | Fields | Purpose |
| --- | --- | --- |
| `capture_window` | `start`, `end` (local datetimes), `label`, `target_type`, `target`, `device_id` (optional) | Snap a past window to the run's real rise/fall edges. |
| `capture_start` | `label`, `target_type`, `target`, `device_id` (optional) | Start a live "on now" capture (one at a time). |
| `capture_stop` | `capture_id` (optional) | Stop the open live capture; returns at once with state `resolving`, and the result is read with `capture_list`. |
| `capture_cancel` | `capture_id` (optional) | Cancel an open capture (idempotent on a finished one). |
| `capture_list` | `capture_id` (optional) | Read every stored capture (or one) with its result, window and resolved target. |
| `capture_delete` | `capture_id` | Delete a stored capture. |
| `capture_reassign` | `capture_id`, `target_type`, `target`, `device_id` (optional) | Re-target a capture. |

`start`/`end` are read in the **Home Assistant time zone**. A time in a repeated hour (DST
fold) is refused with `capture_ambiguous_local_time` and a time in a skipped hour with
`capture_nonexistent_local_time`; a window can also be refused for `capture_invalid_window`,
`capture_future_end`, `capture_window_too_long`, `capture_too_short_for_rate`,
`capture_interval_too_coarse`, `capture_outside_buffer` or `capture_across_gap` (a gap in the
buffer inside the window). Every refusal names its reason and changes nothing.

`target_type` is one of:

- `new_device` — `target` is the new device's name; PowerGuesser fits a preview from it.
- `candidate` — `target` is an existing Review candidate's name or **signature id**.
- `device` — a confirmed device, chosen with the `device_id` device picker or named by its
  `target` (name or uuid); the result carries a read-only replay check ("would the active
  version have claimed this run?"). Nothing is written to the device.

Display names can change after an LLM relabel, so automations should address candidates by
**signature id** (and address a confirmed device by uuid or the device picker).

Each capture result carries a quality **verdict**: `clean` (teaches), `contaminated`,
`ambiguous`, or `none` (no pair). Only `clean` captures teach a preview.

A capture aimed at a new device or an existing preview appears in Review at once as a preview
card reading "from N captured examples". Backed by fewer than `CONFIRM_MIN_RUNS` clean runs it
cannot be confirmed by default; the confirmation form offers **Confirm anyway**, which shows a
warning, is logged, and is recorded as an attempted override on the decision. The gate never
blocks reconciliation with the LLM's candidate.

Dismissing a capture-backed preview keeps its captured examples; deleting a capture is the
separate `capture_delete` action. Captures whose target no longer resolves surface in Review
with a notice and as a non-fixable `capture_orphans` Repairs issue until they are reassigned or
deleted.

Captured windows are **excluded from holdout validation**: every capture record carries a
holdout-exclusion marker from day one, and `run_validation` drops the capture windows from the
validation metrics. The report's `holdout.excluded_capture` counts the capture windows whose
padded span overlaps the holdout (`windows` — not every stored capture) and the holdout samples
removed (`samples`).

### Capture tuning constants

These are **code-level defaults** carried by `CaptureConfig`
(`engine/capture/constants.py`); they have no options-flow setting and are measured and tuned
post-ship on the reference install. Values are the shipped defaults:

| Constant | Default | Meaning |
| --- | --- | --- |
| `SNAP_MARGIN_S` | 120 | Half-width (s) of the rise/fall search window around the requested instant. |
| `CONTEXT_BEFORE_S` | 86,400 | Residual context (s) loaded before the window so edge detection sees the learner's neighbourhood. |
| `CONTEXT_AFTER_S` | 3,600 | Residual context (s) loaded after the window. |
| `AMBIGUITY_RATIO` | 2.0 | Runner-up score factor above which a result is `ambiguous`. |
| `INSIDE_EDGE_REL` | 0.25 | Inside-edge contamination threshold as a fraction of the rise. |
| `MIN_RUN_SAMPLES` | 2 | Fewest plateau samples a snapped run must have. |
| `GAP_MAX_FLOOR_S` | 150 | Floor (s) of the inside-gap threshold. |
| `GAP_MAX_SAMPLES` | 5 | Sample-count multiplier of the inside-gap threshold. |
| `TRACE_MAX_POINTS` | 300 | Most int values kept in a stored trace. |
| `TRACE_PAD_S` | 300 | Trace padding (s) either side of the snapped run. |
| `LIVE_RISE_BEFORE_S` | 180 | Live rise search (s) before the press. |
| `LIVE_RISE_AFTER_S` | 120 | Live rise search (s) after the press. |
| `LIVE_FALL_BEFORE_S` | 180 | Live fall search (s) before the stop press. |
| `LIVE_STOP_SETTLE_SAMPLES` | 3 | Settle wait = clamp(this x effective sample interval, min, max). |
| `LIVE_STOP_SETTLE_MIN_S` | 90 | Minimum settle wait (s). |
| `LIVE_STOP_SETTLE_MAX_S` | 180 | Maximum settle wait (s). |
| `LIVE_STOP_MAX_WAIT_S` | 300 | One extension of the still-on wait, then `no_off_edge`. |
| `LIVE_STILL_ON_FRAC` | 0.7 | "Still on" fraction of the rise. |
| `PRE_LEVEL_SAMPLES` | 4 | Samples for the still-on pre-level median. |
| `LIVE_MAX_OPEN_S` | 3,600 | An open live capture auto-expires after this many seconds. |
| `PREVIEW_W_TOL` | 0.15 | Preview watts tolerance fraction, floored at `threshold_w` / 2. |
| `PREVIEW_DUR_TOL` | 0.30 | Preview duration tolerance fraction, floored at one grid step. |
| `CONFIRM_MIN_RUNS` | 3 | Clean runs below which a preview cannot be confirmed by default. |
| `ANCHOR_TOL_STEPS` | 1.5 | Grid steps within which a capture anchor maps to an episode row. |
| `HOLDOUT_PAD_STEPS` | 2 | Grid steps each side a holdout window is padded. |
| `REPLAY_PRE_ROLL_S` | 600 | Pre-roll (s) before the rise for the read-only replay check. |
| `MAX_CAPTURE_RECORDS` | 200 | Most capture records retained. |
