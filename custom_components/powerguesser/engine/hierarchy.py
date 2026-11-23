"""Hierarchy resolution for Energy dashboard metered devices."""

from collections.abc import Sequence

from .model import MeteredDevice


def _leads_to(start: str, target: str, by_stat: dict[str, MeteredDevice], max_hops: int) -> bool:
    """Return True if walking included_in_stat from start reaches target within max_hops."""
    cur = start
    hops = 0
    visited = set()
    while cur and hops < max_hops:
        if cur == target:
            return True
        if cur in visited:
            break
        visited.add(cur)
        parent = by_stat.get(cur)
        if parent is None or not parent.included_in_stat:
            break
        cur = parent.included_in_stat
        hops += 1
    return False


def power_roots(devices: Sequence[MeteredDevice]) -> list[MeteredDevice]:
    """Find the top-most powered devices across all branches in the hierarchy.

    Mirrors the Home Assistant frontend chain-walk:
    A device is a root if it has a power_entity and has no powered ancestor.
    Cycle-safe and order-independent.
    """
    by_stat = {d.stat_consumption: d for d in devices}
    max_hops = len(devices)
    roots: list[MeteredDevice] = []

    for d in devices:
        if d.power_entity is None:
            continue

        has_powered_ancestor = False
        cur = d.included_in_stat
        hops = 0
        visited = {d.stat_consumption}

        while cur and hops < max_hops:
            if cur in visited:
                break
            visited.add(cur)

            parent = by_stat.get(cur)
            if parent is None:
                # Dangling parent: statistic not in list, treat as no powered ancestor
                break

            if parent.power_entity is not None:
                # Check for cycle between d and parent
                if _leads_to(parent.stat_consumption, d.stat_consumption, by_stat, max_hops):
                    # In a cycle of powered devices, deterministically designate
                    # the one with lowest stat_consumption as cycle root
                    if d.stat_consumption < parent.stat_consumption:
                        # d is the cycle representative, parent does not disqualify d
                        pass
                    else:
                        has_powered_ancestor = True
                        break
                else:
                    has_powered_ancestor = True
                    break

            cur = parent.included_in_stat
            hops += 1

        if not has_powered_ancestor:
            roots.append(d)

    return roots


def unresolved_devices(devices: Sequence[MeteredDevice]) -> list[MeteredDevice]:
    """Return unpowered devices that have no powered descendants anywhere in their branch."""
    by_stat = {d.stat_consumption: d for d in devices}
    children: dict[str, list[MeteredDevice]] = {d.stat_consumption: [] for d in devices}
    for d in devices:
        if (
            d.included_in_stat
            and d.included_in_stat in children
            and d.included_in_stat != d.stat_consumption
        ):
            children[d.included_in_stat].append(d)

    unresolved: list[MeteredDevice] = []
    for d in devices:
        if d.power_entity is not None:
            continue

        has_powered_descendant = False
        queue = list(children.get(d.stat_consumption, []))
        visited = {d.stat_consumption}

        while queue:
            child = queue.pop(0)
            if child.stat_consumption in visited:
                continue
            visited.add(child.stat_consumption)

            if child.power_entity is not None:
                has_powered_descendant = True
                break

            queue.extend(children.get(child.stat_consumption, []))

        if not has_powered_descendant:
            unresolved.append(d)

    return unresolved
