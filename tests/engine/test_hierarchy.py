"""Tests for engine.hierarchy power root resolution and unresolved devices."""

import itertools
import pytest

from custom_components.powerguesser.engine.hierarchy import (
    power_roots,
    unresolved_devices,
)
from custom_components.powerguesser.engine.model import MeteredDevice


def test_empty_devices() -> None:
    """Empty device list yields empty roots and empty unresolved."""
    assert power_roots([]) == []
    assert unresolved_devices([]) == []


def test_single_powered_device() -> None:
    """Single powered device is a root."""
    d = MeteredDevice(
        stat_consumption="sensor.tv_energy",
        name="TV",
        power_entity="sensor.tv_power",
    )
    assert power_roots([d]) == [d]
    assert unresolved_devices([d]) == []


def test_single_unpowered_device() -> None:
    """Single unpowered device is unresolved and not a root."""
    d = MeteredDevice(
        stat_consumption="sensor.ac_energy",
        name="AC",
        power_entity=None,
    )
    assert power_roots([d]) == []
    assert unresolved_devices([d]) == [d]


def test_powered_parent_with_powered_children() -> None:
    """Powered parent with powered children -> only the parent is subtracted."""
    parent = MeteredDevice(
        stat_consumption="sensor.ac_total_energy",
        name="AC Total",
        power_entity="sensor.ac_total_power",
    )
    child1 = MeteredDevice(
        stat_consumption="sensor.ac_living_energy",
        name="AC Living",
        included_in_stat="sensor.ac_total_energy",
        power_entity="sensor.ac_living_power",
    )
    child2 = MeteredDevice(
        stat_consumption="sensor.ac_bed_energy",
        name="AC Bed",
        included_in_stat="sensor.ac_total_energy",
        power_entity="sensor.ac_bed_power",
    )
    devices = [parent, child1, child2]

    roots = power_roots(devices)
    assert roots == [parent]
    assert unresolved_devices(devices) == []


def test_unpowered_parent_with_powered_children() -> None:
    """Unpowered parent with powered children -> children are roots, parent not unresolved."""
    parent = MeteredDevice(
        stat_consumption="sensor.ac_total_energy",
        name="AC Total",
        power_entity=None,
    )
    child1 = MeteredDevice(
        stat_consumption="sensor.ac_living_energy",
        name="AC Living",
        included_in_stat="sensor.ac_total_energy",
        power_entity="sensor.ac_living_power",
    )
    child2 = MeteredDevice(
        stat_consumption="sensor.ac_bed_energy",
        name="AC Bed",
        included_in_stat="sensor.ac_total_energy",
        power_entity="sensor.ac_bed_power",
    )
    devices = [parent, child1, child2]

    roots = power_roots(devices)
    assert roots == [child1, child2]
    # Parent has powered descendants, so it is NOT in unresolved_devices
    assert unresolved_devices(devices) == []


def test_dangling_parent_treated_as_root() -> None:
    """A device whose included_in_stat points to a non-existent statistic is a root."""
    d = MeteredDevice(
        stat_consumption="sensor.heatpump_energy",
        name="Heatpump",
        included_in_stat="sensor.nonexistent_parent_energy",
        power_entity="sensor.heatpump_power",
    )
    assert power_roots([d]) == [d]
    assert unresolved_devices([d]) == []


def test_self_parent_cycle() -> None:
    """A device naming itself as parent terminates and is subtracted once."""
    d = MeteredDevice(
        stat_consumption="sensor.loop_energy",
        name="Loop",
        included_in_stat="sensor.loop_energy",
        power_entity="sensor.loop_power",
    )
    assert power_roots([d]) == [d]
    assert unresolved_devices([d]) == []


def test_two_device_cycle() -> None:
    """Two devices pointing to each other terminate within len(devices) hops."""
    d1 = MeteredDevice(
        stat_consumption="sensor.a_energy",
        name="A",
        included_in_stat="sensor.b_energy",
        power_entity="sensor.a_power",
    )
    d2 = MeteredDevice(
        stat_consumption="sensor.b_energy",
        name="B",
        included_in_stat="sensor.a_energy",
        power_entity="sensor.b_power",
    )
    # The first one walked finds the other as ancestor; both are powered.
    # Exactly one or both (as long as finite and at most once)
    roots = power_roots([d1, d2])
    assert len(roots) >= 1
    assert len(roots) <= 2
    assert len(set(roots)) == len(roots)
    assert unresolved_devices([d1, d2]) == []


def test_permutation_invariance() -> None:
    """Order of devices in the list does not affect the set of power roots."""
    # 4-device hierarchy: Root (unpowered) -> Branch (powered) -> Leaf1 (powered), Leaf2 (unpowered)
    d_root = MeteredDevice("sensor.root_energy", name="Root", power_entity=None)
    d_branch = MeteredDevice(
        "sensor.branch_energy",
        name="Branch",
        included_in_stat="sensor.root_energy",
        power_entity="sensor.branch_power",
    )
    d_leaf1 = MeteredDevice(
        "sensor.leaf1_energy",
        name="Leaf1",
        included_in_stat="sensor.branch_energy",
        power_entity="sensor.leaf1_power",
    )
    d_leaf2 = MeteredDevice(
        "sensor.leaf2_energy",
        name="Leaf2",
        included_in_stat="sensor.branch_energy",
        power_entity=None,
    )

    devices = [d_root, d_branch, d_leaf1, d_leaf2]
    expected_root_set = {d_branch}

    for perm in itertools.permutations(devices):
        roots = power_roots(list(perm))
        assert set(roots) == expected_root_set
        unresolved = unresolved_devices(list(perm))
        # d_root has powered descendants (d_branch, d_leaf1), so only d_leaf2 is unresolved
        assert set(unresolved) == {d_leaf2}
