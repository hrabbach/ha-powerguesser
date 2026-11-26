"""HA tests for the capture services contract (CAP-06).

Plan 08-05 Task 2: the seven capture services are registered admin-only, every
service and field in ``services.yaml`` has a name and description in
``strings.json``, every ``capture_*`` exception key the module raises exists under
``exceptions``, and ``strings.json`` equals ``translations/en.json`` byte for byte.
"""

from pathlib import Path
import json
import re
from typing import Any

import pytest
import yaml

from homeassistant.auth.models import User
from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import Unauthorized

from custom_components.powerguesser.const import DOMAIN, ISSUE_CAPTURE_ORPHANS
from tests.conftest import async_set_power, async_setup_via_flow

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "custom_components" / "powerguesser"

SERVICES = (
    "capture_window",
    "capture_list",
    "capture_start",
    "capture_stop",
    "capture_cancel",
    "capture_delete",
    "capture_reassign",
)


async def test_capture_services_are_admin_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_read_only_user: User,
) -> None:
    """All seven capture services are registered and raise Unauthorized for a non-admin."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    await async_setup_via_flow(hass)

    registered = hass.services.async_services().get(DOMAIN, {})
    for name in SERVICES:
        assert name in registered, name

    ctx = Context(user_id=hass_read_only_user.id)
    payloads: dict[str, dict[str, Any]] = {
        "capture_window": {
            "start": "2026-11-20T10:00:00+00:00",
            "end": "2026-11-20T10:05:00+00:00",
            "label": "l",
            "target_type": "new_device",
            "target": "Kettle",
        },
        "capture_list": {},
        "capture_start": {"label": "l", "target_type": "new_device", "target": "Kettle"},
        "capture_stop": {},
        "capture_cancel": {},
        "capture_delete": {"capture_id": "x"},
        "capture_reassign": {"capture_id": "x", "target_type": "new_device", "target": "K"},
    }
    for name, data in payloads.items():
        with pytest.raises(Unauthorized):
            await hass.services.async_call(
                DOMAIN, name, data, blocking=True, return_response=True, context=ctx
            )


def test_services_yaml_fields_have_strings() -> None:
    """Every service and field in services.yaml has a name and description in strings.json."""
    services_yaml = yaml.safe_load((PKG / "services.yaml").read_text(encoding="utf-8"))
    strings = json.loads((PKG / "strings.json").read_bytes())
    assert (PKG / "strings.json").read_bytes() == (PKG / "translations" / "en.json").read_bytes()

    s_services = strings["services"]
    for name, block in services_yaml.items():
        assert block.get("name"), name
        assert block.get("description"), name
        assert name in s_services, name
        assert s_services[name].get("name"), name
        assert s_services[name].get("description"), name
        for field, field_block in (block.get("fields") or {}).items():
            assert field_block.get("name"), f"{name}.{field}"
            assert field_block.get("description"), f"{name}.{field}"
            str_field = (s_services[name].get("fields") or {}).get(field)
            assert str_field, f"{name}.{field} missing in strings"
            assert str_field.get("name"), f"{name}.{field} name"
            assert str_field.get("description"), f"{name}.{field} description"


def test_capture_exception_keys_exist() -> None:
    """Every capture_* exception key literal in capture.py has an exceptions entry."""
    source = (PKG / "capture.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'translation_key="(capture_[a-z_]+)"', source))
    assert keys, "no capture_* exception keys found"
    strings = json.loads((PKG / "strings.json").read_bytes())
    exceptions = strings["exceptions"]
    for key in keys:
        if key == ISSUE_CAPTURE_ORPHANS:
            # An issue translation key, not an exception.
            assert key in strings["issues"], key
            continue
        assert key in exceptions, key
        assert exceptions[key].get("message")


def test_capture_orphan_issue_and_selector_translations() -> None:
    """issues.capture_orphans carries the locked wording and the selector options exist."""
    strings = json.loads((PKG / "strings.json").read_bytes())
    orphan = strings["issues"]["capture_orphans"]
    assert orphan["title"]
    assert "target no longer exists: reassign or delete" in orphan["description"]

    options = strings["selector"]["capture_target_type"]["options"]
    assert set(options) == {"new_device", "candidate", "device"}
