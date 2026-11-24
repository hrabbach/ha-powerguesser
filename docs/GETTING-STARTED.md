<!-- generated-by: gsd-doc-writer -->
# Getting started

PowerGuesser is a Home Assistant custom integration — it has **no standalone run mode** and is not installed with a package manager. It runs inside Home Assistant and is installed from the built-in Marketplace (HACS is part of Home Assistant since 2026.11). This guide takes you from a running Home Assistant instance to your first locally estimated device sensors.

## Prerequisites

- **Home Assistant 2026.11.0 or newer** (this is the version pinned in `hacs.json`).
- **Energy dashboard configured** with at least one gross power source — the grid `stat_rate` power entity. Solar and battery sources are used when present, but the grid source is the baseline. PowerGuesser reads these sources from Home Assistant's Energy preferences; it does not create them.
- **An OpenAI-compatible LLM endpoint** (base URL, API key if required, model id). The LLM is used **only during the learning phase**; day-to-day detection runs locally and deterministically. You can install and set up everything else before the endpoint exists — learning simply stays disabled until one is configured.
- **History**: learning needs at least **7 days** of usable history, and validated device candidates appear after roughly **three weeks** of data.
- **Meter sampling**: 30-second meters are the design baseline. Step detection needs an effective interval of **45 s or less**; coarser sampling disables it.

For contributors running the test suite locally instead: **Python >= 3.14.2** (the CI gate in `ci/pytest.sh` enforces this because `pytest-homeassistant==2026.11.1` requires it), plus `git`.

## Installation

End users install PowerGuesser from Home Assistant's Marketplace as a custom repository — there is nothing to clone or build:

1. In Home Assistant, open **Settings → Devices & Services → Add Integration → Marketplace** and choose **"Add from link"** (custom repository).
2. Enter the repository URL:

   ```
   https://github.com/hrabbach/ha-powerguesser
   ```

3. Install **PowerGuesser** and **restart Home Assistant**.

Adding the integration (next section) happens after the restart.

### Clone the repository (contributors only)

If you want to work on the code or run the test suite:

```bash
git clone https://github.com/hrabbach/ha-powerguesser.git
cd ha-powerguesser
```

## First run

The integration runs entirely inside Home Assistant; there is no local server or CLI to start. "First working output" means PowerGuesser installed, configured, and estimating a device's power. Follow these steps in the Home Assistant UI:

1. **Add the integration**: **Settings → Devices & Services → Add Integration → PowerGuesser**.
   Setup detects your gross power sources and metered devices automatically from the Energy dashboard and shows you what it found. A single instance is allowed — a second entry aborts with `single_instance_allowed`. Setup aborts early with an explanatory message if the Energy dashboard has no usable power source.
2. **Enter the LLM endpoint**: open the integration's **Configure** button and choose **LLM endpoint**. Enter the base URL (ending in `/v1`), an API key if the endpoint requires one, and the model id. Saving runs a short capability probe against the endpoint that sends only a fixed test prompt — no household data.
3. **Grant data-sharing consent**: the first time a learning run needs it, a Repairs/notification prompt asks you to confirm exactly what would be sent (destination host, local or remote, scheme, model, token and cost estimates, whether entity names are included). Consent is stored per endpoint origin; a transport downgrade requires fresh consent.
4. **Start a learning run** from **Developer Tools → Actions**:

   ```yaml
   action: powerguesser.learn
   ```

   The run needs the LLM endpoint, granted consent, and at least 7 days of usable history. To inspect exactly what the run would send without sending anything:

   ```yaml
   action: powerguesser.preview_learning_payload
   response_variable: payload
   ```

5. **Review candidates**: once learning produces validated candidates (after about three weeks of history), open the integration's **Configure → Review candidates** and confirm the devices you recognise. Confirming creates a device with power and energy sensors that PowerGuesser estimates **locally** — no cloud and no AI at runtime.

### Verifying the code instead (no Home Assistant instance)

If you only cloned the repo and want to check that everything works, run the test suite — it exercises the integration against `pytest-homeassistant` fixtures:

```bash
python3 -m pytest
```

The full setup CI uses is in `ci/pytest.sh` (installs `pytest-homeassistant==2026.11.1` plus the manifest requirements `numpy` and `scipy>=1.16.2,<2` under the HA constraint file). Deselect the slow worker-subprocess tests for a fast loop:

```bash
python3 -m pytest -m "not slow"
```

## Common setup issues

- **Setup aborts with "No grid, solar, battery, or home power source is configured in the Energy dashboard"** (`no_energy_prefs`): configure a power source in **Settings → Dashboards → Energy** first, then re-add the integration.
- **Setup aborts with "The energy source {source} does not have a power rate sensor configured"** (`missing_stat_rate`): the Energy source exists but its `stat_rate` power sensor is not set — add one in the Energy dashboard.
- **Setup aborts with "does not have a valid power unit of measurement"** (`invalid_unit`): the power entity must report in `W` or `kW` (or an equivalent power unit).
- **Setup aborts with "The power entity for {source} cannot be found"** (`source_not_found`): the entity listed in the Energy dashboard no longer exists in Home Assistant — verify the entity and retry.
- **LLM endpoint step fails**: the probe reports `invalid_auth` (HTTP 401/403 — check the API key), `cannot_connect` (check the base URL, scheme, and reachability), or `model_unusable` (the endpoint answered but the model did not return the expected JSON). Note a stored API key is only reused when scheme, host, and port of the new URL match — it is never forwarded to a different origin.
- **Learning does not start**: check the **Repairs** panel. Missing prerequisites surface as repair issues — `llm_auth` (no usable LLM endpoint), `learning_consent` (consent not granted for this endpoint origin), and `learning_capped` (a token or cost cap was hit, pausing learning).
- **No candidates after learning**: this is usually a history problem, not a bug — validated candidates need roughly three weeks of usable data; step detection silently stays off if the effective meter interval is above 45 s.
- **"Only a single instance of PowerGuesser is allowed"**: the integration is single-instance by design (`single_config_entry: true` in the manifest). Remove the existing entry first if you need to reconfigure from scratch.

## Next steps

- [Configuration](CONFIGURATION.md) — every config flow step, option, and default value, including the token and cost caps.
- [Architecture](ARCHITECTURE.md) — how the buffer, learning engine, and local detection fit together.
- [README](../README.md) — requirements, accuracy reporting, and the full service reference.
