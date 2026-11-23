"""Tests for engine.prefs source model ingestion and power_config classification."""

import json
from pathlib import Path
import pytest

from custom_components.powerguesser.engine.model import (
    EntityFacts,
    RegistryFacts,
    Role,
    Severity,
    Transform,
)
from custom_components.powerguesser.engine.prefs import (
    ABORT_ORDER,
    build_source_model,
    classify_power_config,
    source_slug,
)


@pytest.fixture
def variants() -> dict:
    path = Path(__file__).parent.parent / "fixtures" / "prefs_variants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_abort_order_constant() -> None:
    """Verify ABORT_ORDER contains the exact expected tuple."""
    assert ABORT_ORDER == (
        "no_energy_prefs",
        "missing_stat_rate",
        "unsupported_power_config",
        "source_not_found",
        "invalid_unit",
    )


def test_source_slug() -> None:
    """Verify source_slug sanitization."""
    src = {"type": "grid", "stat_energy_from": "sensor.grid_import_energy"}
    assert source_slug(src, Role.GRID, 0) == "sensor_grid_import_energy"

    src_fallback = {"type": "grid"}
    assert source_slug(src_fallback, Role.GRID, 2) == "grid_2"


def test_classify_power_config_direct() -> None:
    src = {"type": "grid", "stat_rate": "sensor.grid_power"}
    ch = classify_power_config(src, Role.GRID, "grid_slug", "Grid (sensor.grid_power)")
    assert ch.transform == Transform.DIRECT
    assert ch.entity_ids == ("sensor.grid_power",)


def test_classify_power_config_inverted() -> None:
    src = {
        "type": "grid",
        "power_config": {"stat_rate_inverted": "sensor.grid_export_power"},
    }
    ch = classify_power_config(src, Role.GRID, "grid_slug", "Grid (export)")
    assert ch.transform == Transform.INVERTED
    assert ch.entity_ids == ("sensor.grid_export_power",)


def test_classify_power_config_two_sensor() -> None:
    src = {
        "type": "battery",
        "power_config": {
          "stat_rate_from": "sensor.battery_discharge_power",
          "stat_rate_to": "sensor.battery_charge_power",
        },
    }
    ch = classify_power_config(src, Role.BATTERY, "batt_slug", "Battery")
    assert ch.transform == Transform.FROM_MINUS_TO
    assert ch.entity_ids == (
        "sensor.battery_discharge_power",
        "sensor.battery_charge_power",
    )


def test_classify_power_config_empty_dict_with_stat_rate() -> None:
    src = {
        "type": "grid",
        "stat_rate": "sensor.grid_power",
        "power_config": {},
    }
    ch = classify_power_config(src, Role.GRID, "grid_slug", "Grid")
    assert ch.transform == Transform.DIRECT
    assert ch.entity_ids == ("sensor.grid_power",)


def test_classify_power_config_unsupported_mixed() -> None:
    src = {
        "type": "grid",
        "power_config": {
            "stat_rate": "sensor.grid_power",
            "stat_rate_inverted": "sensor.grid_export_power",
        },
    }
    prob = classify_power_config(src, Role.GRID, "grid_slug", "Grid")
    assert prob.kind == "unsupported_power_config"
    assert prob.issue_id == "unsupported_power_config_grid_slug"


def test_build_source_model_home_source(variants: dict) -> None:
    model = build_source_model(variants["home_source"])
    assert len(model.gross_channels) == 1
    assert model.gross_channels[0].role == Role.HOME
    assert model.gross_channels[0].entity_ids == ("sensor.home_power",)
    assert not model.problems


def test_build_source_model_home_without_stat_rate(variants: dict) -> None:
    model = build_source_model(variants["home_without_stat_rate"])
    assert len(model.gross_channels) == 1
    assert model.gross_channels[0].role == Role.GRID


def test_build_source_model_no_sources(variants: dict) -> None:
    model = build_source_model(variants["no_sources"])
    assert any(p.kind == "no_energy_prefs" for p in model.problems)
    assert model.fatal


def test_build_source_model_gas_only(variants: dict) -> None:
    model = build_source_model(variants["gas_only"])
    assert any(p.kind == "no_energy_prefs" for p in model.problems)
    assert model.fatal


def test_build_source_model_missing_stat_rate(variants: dict) -> None:
    model = build_source_model(variants["missing_stat_rate_grid"])
    assert any(p.kind == "missing_stat_rate" for p in model.problems)
    assert model.fatal


def test_build_source_model_order_independent(variants: dict) -> None:
    model1 = build_source_model(variants["multi_grid"])
    model2 = build_source_model(variants["multi_grid_reordered"])

    # Issue ids and problem kinds must be identical
    assert sorted(p.issue_id for p in model1.problems) == sorted(p.issue_id for p in model2.problems)

    # Gross channels set of keys must be identical
    assert {ch.key for ch in model1.gross_channels} == {ch.key for ch in model2.gross_channels}


def test_build_source_model_facts_checks(variants: dict) -> None:
    prefs = variants["standard"]
    # All 3 gross entities exist
    facts = RegistryFacts(
        entities={
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="dev1"),
            "sensor.solar_power": EntityFacts(exists=False, unit_ok=None, platform=None, device_id=None),
            "sensor.battery_power": EntityFacts(exists=True, unit_ok=False, platform="sensor", device_id="dev2"),
        },
        siblings={},
    )
    model = build_source_model(prefs, facts=facts)
    kinds = [p.kind for p in model.problems]
    assert "source_not_found" in kinds
    assert "invalid_unit" in kinds

    snf = next(p for p in model.problems if p.kind == "source_not_found")
    assert snf.structural is True

    iu = next(p for p in model.problems if p.kind == "invalid_unit")
    assert iu.structural is False
    assert model.fatal


def test_metered_sibling_resolution_and_warnings() -> None:
    """Sibling power entity resolution per D-03: exactly one wins, multiple/none leaves unresolved."""
    prefs = {
        "energy_sources": [
            {"type": "grid", "stat_rate": "sensor.grid_power", "stat_energy_from": "sensor.grid_energy"}
        ],
        "device_consumption": [
            {"stat_consumption": "sensor.plug_a_energy", "name": "Plug A"},
            {"stat_consumption": "sensor.plug_b_energy", "name": "Plug B"},
            {"stat_consumption": "sensor.plug_c_energy", "name": "Plug C"},
            {"stat_consumption": "external:device_energy", "name": "External"},
        ],
    }
    facts = RegistryFacts(
        entities={
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d0"),
            "sensor.plug_a_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d1"),
        },
        siblings={
            # Exactly one sibling
            "sensor.plug_a_energy": ("sensor.plug_a_power",),
            # Multiple candidates -> unresolved
            "sensor.plug_b_energy": ("sensor.plug_b_power1", "sensor.plug_b_power2"),
            # No siblings -> unresolved
            "sensor.plug_c_energy": (),
            # External statistic -> empty
            "external:device_energy": (),
        },
    )
    model = build_source_model(prefs, facts=facts)

    # Plug A got resolved
    assert len(model.metered_roots) == 1
    assert model.metered_roots[0].stat_consumption == "sensor.plug_a_energy"
    assert model.metered_roots[0].power_entity == "sensor.plug_a_power"

    # Plugs B, C, External have metered_no_power warnings
    kinds = [p.kind for p in model.problems]
    assert kinds.count("metered_no_power") == 3
    assert all(p.severity == Severity.WARNING for p in model.problems if p.kind == "metered_no_power")


def test_self_reference_exclusions() -> None:
    """Self-reference with platform powerguesser is excluded and flagged."""
    # 1. Metered device pointing to powerguesser entity -> ignored, WARNING
    prefs_metered = {
        "energy_sources": [
            {"type": "grid", "stat_rate": "sensor.grid_power", "stat_energy_from": "sensor.grid_energy"}
        ],
        "device_consumption": [
            {
                "stat_consumption": "sensor.plug_energy",
                "stat_rate": "sensor.powerguesser_remainder_power",
                "name": "Self Metered",
            }
        ],
    }
    facts_metered = RegistryFacts(
        entities={
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d0"),
            "sensor.powerguesser_remainder_power": EntityFacts(
                exists=True, unit_ok=True, platform="powerguesser", device_id="d_pg"
            ),
        },
        siblings={},
    )
    model_metered = build_source_model(prefs_metered, facts=facts_metered)
    # Metered root was not adopted (power_entity is None)
    assert len(model_metered.metered_roots) == 0
    assert any(p.kind == "self_reference" and p.severity == Severity.WARNING for p in model_metered.problems)

    # 2. Home source pointing to powerguesser entity -> dropped, formula applies, WARNING
    prefs_home = {
        "energy_sources": [
            {"type": "home", "stat_rate": "sensor.powerguesser_gross_power", "stat_energy_to": "sensor.home_energy"},
            {"type": "grid", "stat_rate": "sensor.grid_power", "stat_energy_from": "sensor.grid_energy"},
        ],
        "device_consumption": [],
    }
    facts_home = RegistryFacts(
        entities={
            "sensor.powerguesser_gross_power": EntityFacts(
                exists=True, unit_ok=True, platform="powerguesser", device_id="d_pg"
            ),
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d0"),
        },
        siblings={},
    )
    model_home = build_source_model(prefs_home, facts=facts_home)
    # Home channel was dropped, grid channel used instead
    assert len(model_home.gross_channels) == 1
    assert model_home.gross_channels[0].role == Role.GRID
    assert any(p.kind == "self_reference" and p.severity == Severity.WARNING for p in model_home.problems)

    # 3. Grid source pointing to powerguesser entity -> fatal ERROR
    prefs_grid = {
        "energy_sources": [
            {"type": "grid", "stat_rate": "sensor.powerguesser_gross_power", "stat_energy_from": "sensor.grid_energy"},
        ],
        "device_consumption": [],
    }
    facts_grid = RegistryFacts(
        entities={
            "sensor.powerguesser_gross_power": EntityFacts(
                exists=True, unit_ok=True, platform="powerguesser", device_id="d_pg"
            ),
        },
        siblings={},
    )
    model_grid = build_source_model(prefs_grid, facts=facts_grid)
    assert model_grid.fatal
    assert any(p.kind == "self_reference" and p.severity == Severity.ERROR for p in model_grid.problems)


def test_own_device_energy_ignored_silently(variants: dict) -> None:
    """Own-platform stat_consumption is skipped silently; the foreign unpowered plug still warns (OUT-02)."""
    prefs = variants["own_device_energy"]
    facts = RegistryFacts(
        entities={
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d0"),
            "sensor.solar_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d1"),
            "sensor.battery_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d2"),
        },
        siblings={},
        consumption_platforms={"sensor.oven_energy": "powerguesser"},
    )
    model = build_source_model(prefs, facts=facts)

    # The own energy sensor is not a MeteredDevice root
    assert all(d.stat_consumption != "sensor.oven_energy" for d in model.metered_roots)

    # ... and raises no metered_no_power or self_reference of its own
    assert not any(
        p.kind in ("metered_no_power", "self_reference") and "oven_energy" in p.issue_id
        for p in model.problems
    )

    # The foreign plug without a power entity still gets metered_no_power
    plug = [p for p in model.problems if p.kind == "metered_no_power"]
    assert len(plug) == 1
    assert plug[0].issue_id == "metered_no_power_sensor_foreign_plug_energy"
    assert plug[0].severity == Severity.WARNING


def test_child_of_own_device_energy_is_root() -> None:
    """A powered child whose included_in_stat names an own energy sensor is treated as a root."""
    prefs = {
        "energy_sources": [
            {"type": "grid", "stat_rate": "sensor.grid_power", "stat_energy_from": "sensor.grid_energy"}
        ],
        "device_consumption": [
            {"stat_consumption": "sensor.oven_energy", "name": "Oven"},
            {
                "stat_consumption": "sensor.oven_child_energy",
                "stat_rate": "sensor.oven_child_power",
                "included_in_stat": "sensor.oven_energy",
                "name": "Oven Child",
            },
        ],
    }
    facts = RegistryFacts(
        entities={
            "sensor.grid_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d0"),
            "sensor.oven_child_power": EntityFacts(exists=True, unit_ok=True, platform="sensor", device_id="d1"),
        },
        siblings={},
        consumption_platforms={"sensor.oven_energy": "powerguesser"},
    )
    model = build_source_model(prefs, facts=facts)

    # The skipped own sensor is no hierarchy parent: the child is a root
    assert [d.stat_consumption for d in model.metered_roots] == ["sensor.oven_child_energy"]
    assert not any(p.kind == "metered_no_power" for p in model.problems)
