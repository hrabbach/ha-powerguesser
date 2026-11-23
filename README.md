# PowerGuesser

![PowerGuesser](custom_components/powerguesser/brand/icon.png)

PowerGuesser is a Home Assistant custom integration for non-intrusive load monitoring (NILM).
It learns the power fingerprints of individual household devices from the whole-house signal you
already have configured in the Energy dashboard, then estimates and tracks the energy use of
consumers that cannot be metered directly.

Day-to-day detection runs **locally and deterministically** from the learned signatures: no cloud
and no AI. An OpenAI-compatible LLM endpoint is used **only during the learning
(pattern-detection) phase**.

## Requirements

- Home Assistant **2026.11.0** or newer.
- Energy dashboard sources configured, including at least the grid `stat_rate` power entity
  (solar and battery are used when present).
- An OpenAI-compatible LLM endpoint (base URL, API key, model) for the learning phase.
- About **three weeks** of data before validated candidates can be discovered.

## Installation

PowerGuesser is installed as a custom repository in Home Assistant's built-in Marketplace (HACS
is part of HA since 2026.11).

1. Open **Settings → Devices & Services → Add Integration → Marketplace** and choose
   **"Add from link"** (custom repository).
2. Enter the repository URL:

   ```
   https://github.com/hrabbach/ha-powerguesser
   ```

3. Install **PowerGuesser** and restart Home Assistant.
4. Add the integration: **Settings → Devices & Services → Add Integration → PowerGuesser**.

[Add PowerGuesser to Home Assistant](https://my.home-assistant.io/redirect/hacs_repository/?owner=hrabbach&repository=ha-powerguesser&category=integration)

Listing PowerGuesser in the default catalog follows once the 0.x line is stable.

## What is sent to the LLM

The learning phase sends the LLM **power-data summaries** (event and cluster features derived
from your own buffer), never your raw history. Entity names are sent only if you opt in; the
default is off. Before any request leaves your network you see a consent step that names the
destination host, whether it is local or remote, the scheme, the model, the token and cost
estimates, and whether entity names are included. A transport downgrade (for example HTTPS to
HTTP) requires fresh consent.

## Accuracy

<!-- accuracy-table:start -->
The accuracy report status is 'none', so every class reads not validated (no validation report has been published for this release).

| Class | Description | Status | Passing peers | Median energy error % | Max energy error % | Holdout events |
| --- | --- | --- | --- | --- | --- | --- |
| switched_under_0.5kw_1_level | Switched load, under 0.5 kW, single level | not validated |  |  |  |  |
| switched_under_0.5kw_multi_level | Switched load, under 0.5 kW, multi level | not validated |  |  |  |  |
| switched_0.5_1.5kw_1_level | Switched load, 0.5-1.5 kW, single level | not validated |  |  |  |  |
| switched_0.5_1.5kw_multi_level | Switched load, 0.5-1.5 kW, multi level | not validated |  |  |  |  |
| switched_1.5_3kw_1_level | Switched load, 1.5-3 kW, single level | not validated |  |  |  |  |
| switched_1.5_3kw_multi_level | Switched load, 1.5-3 kW, multi level | not validated |  |  |  |  |
| switched_over_3kw_1_level | Switched load, over 3 kW, single level | not validated |  |  |  |  |
| switched_over_3kw_multi_level | Switched load, over 3 kW, multi level | not validated |  |  |  |  |
| program_under_0.5kw_1_level | Program appliance, under 0.5 kW, single level | not validated |  |  |  |  |
| program_under_0.5kw_multi_level | Program appliance, under 0.5 kW, multi level | not validated |  |  |  |  |
| program_0.5_1.5kw_1_level | Program appliance, 0.5-1.5 kW, single level | not validated |  |  |  |  |
| program_0.5_1.5kw_multi_level | Program appliance, 0.5-1.5 kW, multi level | not validated |  |  |  |  |
| program_1.5_3kw_1_level | Program appliance, 1.5-3 kW, single level | not validated |  |  |  |  |
| program_1.5_3kw_multi_level | Program appliance, 1.5-3 kW, multi level | not validated |  |  |  |  |
| program_over_3kw_1_level | Program appliance, over 3 kW, single level | not validated |  |  |  |  |
| program_over_3kw_multi_level | Program appliance, over 3 kW, multi level | not validated |  |  |  |  |

Figures come from one household's holdout validation (plug-masked shadow models replayed on the most recent 7 days). A class reads validated only when at least two plug devices of that class pass; other installations may differ, and every unvalidated device carries `unvalidated: true`.
<!-- accuracy-table:end -->

## Known limitations

- This is a **0.x release**: the hardware UAT is still open, so version 1.0.0 comes later.
- **30 s meters are the design baseline**, not an edge case. Faster meters unlock finer features
  when available.
- Step detection needs an **effective interval of 45 s or less**. Coarser sampling disables it.
- program devices cannot be merged in v0.x: neither as a merge source nor as a merge target.

## Licence

PowerGuesser is licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
