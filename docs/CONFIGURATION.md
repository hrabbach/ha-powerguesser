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

## Guided capture

Guided capture teaches PowerGuesser one real device run without waiting for a learning pass. A
captured run gives you an immediate local **preview candidate** in Review **and** a labelled
example for the next learning run. Captures stay local, except that **every settled capture
becomes a numeric example in the next learning run** — numbers only (step, duration, energy,
hour, weekday and cluster). The capture's **label** is sent only when you have opted in to
`share_entity_names` (default off). You can start, stop and review captures from the dashboard,
the options-flow **Capture** menu, or the admin services described at the end of this section.

### Capture from the dashboard

Four hub-level entities and one sensor drive a capture without any service call:

| Entity | Purpose |
| --- | --- |
| `sensor.powerguesser_capture_status` | Enum: `idle`, `recording` or `resolving`. |
| `button.powerguesser_start_capture` | Start a live capture using the select and label below. |
| `button.powerguesser_stop_capture` | Stop the open live capture. |
| `select.powerguesser_capture_target` | The kind of target: `new_device`, `candidate` or `device`. |
| `text.powerguesser_capture_label` | The target's name or id (max 64 characters). |

`select.powerguesser_capture_target` and `text.powerguesser_capture_label` restore their value
after a Home Assistant restart.

The **label** has two uses:

- For `new_device` it is the new device's name.
- For `candidate` and `device` it names the target, by display name or by signature id / device
  uuid, resolved the same way the services resolve it. An unknown or ambiguous name is refused
  with the error that lists the choices.

An **empty label** on `new_device` starts an unlabelled capture; it appears in Review at once as
"Unnamed capture" and is named later with the normal Review rename. An empty label on `candidate`
or `device` is refused. The label entity is cleared after a successful start; from the moment you
type it until then its state is kept in the recorder history so a rename and the capture agree.

The status sensor carries `capture_id`, `target_kind`, `target_name` and `started_at` for the open
capture, and a `last_result` summary (quality, reason, `near_twin`, `earlier_runs`) of the newest
finished capture. Nothing ticks: the state changes only on a transition.

### Capture menu

*Settings → Devices & services → PowerGuesser → Configure → Capture a device run* opens a Capture
menu with four entries:

- **Start a live capture** — a form (a new-device name, or an existing candidate or device) whose
  Submit commits at once.
- **Stop the live capture** — offered only while a live capture is recording.
- **Capture a past run** — a form with a local start and end time that snaps the window to the
  run's real edges.
- **Recent captures** — the newest ten captures with their result, the most similar entry, the
  earlier-runs count, the LLM's verdict and the replay-check evidence flag.

No step waits: every entry calls the capture manager and returns immediately with the result
text, exactly like the services below.

### Capture insight

A capture that snaps to a rise/fall pair carries two read-only insights:

- **Most similar** — up to three candidates and confirmed devices, each with a similarity score,
  its step (W) and duration, computed from the capture's own tolerance box and the learned boxes.
- **Near twin** — a plain flag when a candidate or device is the same load as the capture: their
  state boxes claim the capture's run, or their features sit within the tolerance distance, in
  either direction. The named twin may sit outside the short list.
- **Earlier runs matched** — how many runs already in the buffer the captured signature would have
  matched, with the **basis** naming the days actually covered (a shorter buffer names the shorter
  span). It is computed once when the capture settles, never recomputed.

The insight is **advisory only**. It never blocks Confirm, and it is not an accuracy figure.

### Captures in learning runs

Every capture that finishes with a snapped pair becomes one numeric **example** in the next
learning run. Each example line carries an opaque id (`e1`, `e2`, …), an opaque group token (`t1`,
…), a quality verdict (`clean`, `contaminated`, `ambiguous`), the rise and fall step, the duration,
the energy (Wh), the coarse hour and weekday and the digest cluster the run falls in. It carries
**no timestamp** and no capture id.

- **Labels** are sent only under the `share_entity_names` opt-in, in a separate `<example_names>`
  block; without the opt-in only the opaque ids appear. A label names the load; it never tells the
  model how many states or clusters the load has.
- **Budget.** Examples share a byte budget derived from the run's input-token cap. What does not
  fit is left out and counted: `examples_left_out n=K` in the payload, `captures_left_out` in the
  preview action's response and `captures_left_out_this_run` on the learning status sensor. The
  order is deterministic (one example per target first, then newest first, ties by capture id).
- **Verdict.** The model may return an optional per-example verdict `consistent`, `inconsistent`
  or `cannot_tell`, keyed by example id. It is stored and shown as `llm_verdict` on
  `capture_list` and in Recent captures; it never changes a candidate, a preview or a device.

The preview action's body is byte-identical to the body the learning run sends (tested), so what
you preview is exactly what leaves your network.

### Device updates from captures

Clean captures aimed at a **confirmed device** are batched into the next learning run:

- The run's `capture_refit` stage refits the device on its clean captured runs and the rows it
  already claims, then the existing **no-regression gate** decides: the refit is judged on
  validation days that **exclude every captured window**, so a capture can never validate the
  update it informed.
- At most one pending proposal exists per device. A newer gated update replaces an unaccepted one
  (the supersession is recorded on the device). Accept it from the device menu
  (*Configure → Review candidates → the device → Accept update*), which appends a new version that
  **rollback** can undo. Nothing changes in place without your Accept.
- **Many capture-driven refits end with no proposal.** A few extra runs rarely move the learned
  boxes (state boxes are fitted from percentiles widened by 5 %), so the value is mostly better
  coverage for a run the active version missed, not a new accuracy claim.
- **Contaminated and ambiguous captures never train** a proposal; they stay as labelled examples
  only, and the UI says so.
- The Phase 8 **replay check** ("would the active version have claimed this run?") is shown in the
  device menu and in Recent captures as **evidence only**; it never decides whether an update is
  proposed.

### Automation event and recipes

When a capture reaches a terminal state, PowerGuesser fires one `powerguesser` event with
`type: capture_finished`, after the record is saved. The states are `done`, `failed`, `cancelled`
and `interrupted`; a capture that times out is `interrupted` with `reason_code: timeout`, and a
restart fires one event per capture it marks `interrupted`.

| Event data key | Meaning |
| --- | --- |
| `type` | Always `capture_finished`. |
| `capture_id` | The capture's id; read the label with `powerguesser.capture_list`. |
| `state` | `done`, `failed`, `cancelled` or `interrupted`. |
| `quality` | `clean`, `contaminated`, `ambiguous` or `none` (null without a pair). |
| `reason_code` | The terminal or refusal reason (`timeout`, `settle_error`, …), else null. |
| `target_kind` | `new_device`, `candidate` or `device`. |
| `target_id` | The opaque target id (signature id or device uuid), else null. |
| `step_w` | The captured rise step in W, else null. |
| `duration_s` | The run duration in seconds, else null. |
| `energy_wh` | The run energy in Wh, else null. |
| `near_twin` | Whether the capture is a near twin of an existing candidate or device. |
| `earlier_runs` | How many earlier buffer runs the captured signature would have matched, else null. |

The event carries **no label and no target display name** — a bus event is recorded and visible to
every websocket subscriber. Read the label with `powerguesser.capture_list` using the event's
`capture_id`.

**NFC tag** — one tag starts a capture, a second stops it:

```yaml
- alias: Kettle capture — start (NFC)
  triggers:
    - trigger: tag
      tag_id: kettle-start
  actions:
    - action: powerguesser.capture_start
      data:
        label: Kettle
        target_type: new_device
        target: Kettle
```

```yaml
- alias: Kettle capture — stop (NFC)
  triggers:
    - trigger: tag
      tag_id: kettle-stop
  actions:
    - action: powerguesser.capture_stop
```

**Voice** — a spoken command starts and stops a capture:

```yaml
- alias: Kettle capture — voice
  triggers:
    - trigger: conversation
      command:
        - start kettle capture
  actions:
    - action: powerguesser.capture_start
      data:
        label: Kettle
        target_type: new_device
        target: Kettle
```

```yaml
- alias: Capture — stop by voice
  triggers:
    - trigger: conversation
      command:
        - stop capture
  actions:
    - action: powerguesser.capture_stop
```

**Physical button** — set the target and label entities, then press the start button:

```yaml
- alias: Capture — button start
  triggers:
    - trigger: state
      entity_id: sensor.zigbee_remote_action
      to: "start"
  actions:
    - action: select.select_option
      target:
        entity_id: select.powerguesser_capture_target
      data:
        option: new_device
    - action: text.set_value
      target:
        entity_id: text.powerguesser_capture_label
      data:
        value: Kettle
    - action: button.press
      target:
        entity_id: button.powerguesser_start_capture
```

```yaml
- alias: Capture — button stop
  triggers:
    - trigger: state
      entity_id: sensor.zigbee_remote_action
      to: "stop"
  actions:
    - action: button.press
      target:
        entity_id: button.powerguesser_stop_capture
```

**Notify when a capture finishes and read its label** — the label never travels on the bus, so
read it from `capture_list`:

```yaml
- alias: Capture finished — notify
  triggers:
    - trigger: event
      event_type: powerguesser
      event_data:
        type: capture_finished
        state: done
  actions:
    - action: powerguesser.capture_list
      data:
        capture_id: "{{ trigger.event.data.capture_id }}"
      response_variable: capture
    - action: persistent_notification.create
      data:
        title: Capture finished
        message: >
          {{ capture.captures[0].label or capture.captures[0].capture_id }} —
          {{ trigger.event.data.quality }}, {{ trigger.event.data.step_w }} W,
          {{ trigger.event.data.duration_s }} s,
          {{ trigger.event.data.energy_wh }} Wh
```

### Admin capture actions

Besides the dashboard and the options flow, PowerGuesser exposes seven **admin-only** capture
actions (call them as services, or from *Developer tools → Actions*).

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
