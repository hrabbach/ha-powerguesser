"""Switched-attribution characterization snapshot (Plan 06-01, RESEARCH B2).

Recorded from the UNCHANGED OnlineDetector before any program-device edit and
committed alone.  Every scenario runs switched-only devices, so after Plan
06-01 the snapshot must still match bit-for-bit: program runtime additions
must not alter switched attribution while no program model is present.

Projection contract (pinned by the plan's grep gate): this test projects only
the NAMED StepResult fields and drained episode tuples below.  It never calls
the detector's state method and never converts a whole StepResult (or any
dataclass) into a dict, so StepResult fields or state keys added later cannot
widen this snapshot.
"""

import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from custom_components.powerguesser.engine.detect.online import (
    DeviceModel,
    OnlineDetector,
)
from tests.fixtures.detect_traces import add_run, cycling_run, grid_trace, switched_artefact

GRID_S = 30
THRESHOLD = 120.0


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(AmberSnapshotExtension)


def _models() -> list[DeviceModel]:
    specs = [
        ("kettle", (1900.0, 2000.0, 2100.0)),
        ("oven", (1400.0, 1500.0, 1600.0)),
        ("heater", (700.0, 800.0, 900.0)),
    ]
    return [
        DeviceModel.from_artefact(
            sig,
            switched_artefact(
                sig,
                rise=box,
                fall=(-box[2], -box[1], -box[0]),
                dur=(300.0, 600.0, 3600.0),
                plateau=box,
            ),
        )
        for sig, box in specs
    ]


def _scenario(name: str) -> np.ndarray:
    if name == "isolated":
        # three switched loads at separate times
        r = grid_trace(400)
        add_run(r, 10, 20, 2000.0)
        add_run(r, 100, 40, 1500.0)
        add_run(r, 200, 30, 800.0)
        return r
    if name == "nested":
        # the kettle inside an oven run
        r = grid_trace(300)
        add_run(r, 20, 100, 1500.0)
        add_run(r, 50, 20, 2000.0)
        return r
    if name == "crossing":
        # kettle and heater ON in the same sample, OFF in the same later sample
        r = grid_trace(300)
        add_run(r, 20, 40, 2000.0)
        add_run(r, 20, 40, 800.0)
        return r
    if name == "cycling":
        # thermostat-style heater cycles
        r = grid_trace(400)
        cycling_run(r, 10, 8, 8, 5, 800.0)
        return r
    if name == "gap":
        # oven ON with NaN runs of 1 and 5 samples inside the ON period
        r = grid_trace(300)
        add_run(r, 20, 100, 1500.0)
        r[40] = np.nan
        r[60:65] = np.nan
        return r
    raise AssertionError(f"unknown scenario {name!r}")


def _drive_projection(det: OnlineDetector, r: np.ndarray) -> dict:
    """Step the detector and project only the named fields (see module docstring)."""
    steps = []
    for i in range(len(r)):
        res = det.step(float(i) * GRID_S, float(r[i]))
        steps.append(
            (
                round(res.grid_ts, 3),
                sorted((k, round(v, 3)) for k, v in res.device_w.items()),
                sorted(res.on),
                sorted(res.unknown),
                sorted((k, round(v, 6)) for k, v in res.energy_wh_delta.items()),
                res.unattributed_total,
                res.forced_off,
                round(res.estimated_w, 3),
            )
        )
    det.flush()
    episodes = {
        key: [
            (ep.start_i, ep.end_i, round(ep.step_w, 3), round(ep.energy_wh, 6))
            for ep in lst
        ]
        for key, lst in sorted(det.drain_episodes().items())
    }
    return {
        "steps": steps,
        "final_energy_wh": {k: round(v, 6) for k, v in sorted(det.energy_wh.items())},
        "episodes": episodes,
    }


@pytest.mark.parametrize(
    "scenario",
    ["isolated", "nested", "crossing", "cycling", "gap"],
    ids=["isolated", "nested", "crossing", "cycling", "gap"],
)
def test_switched_attribution_characterization(snapshot, scenario: str) -> None:
    det = OnlineDetector(
        _models(),
        grid_s=GRID_S,
        threshold_w=THRESHOLD,
        suppress_alignment=True,
    )
    snapshot.assert_match(_drive_projection(det, _scenario(scenario)))
