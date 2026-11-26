"""Engine tests for the D-04 read-only replay check (CAP-03, D-04).

Plan 08-05 Task 1: ``replay_check`` replays the device's active artefact over the
capture window and reports whether it would have claimed the run; a program
device is never matched to a single run.
"""

from pathlib import Path

from custom_components.powerguesser.engine.capture import replay_check
from tests.conftest import make_switched_candidate
from tests.fixtures.learning.synth import SynthSpec, build_buffer

END_MS = 1_700_000_000_000
GRID_S = 30
BASIS = "active version alone on the capture window"


def _context(buffer_path: str) -> dict:
    return {
        "buffer_path": buffer_path,
        "grid_s": GRID_S,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "session_windows": [],
        "tz_name": "UTC",
    }


def test_replay_check_claims_own_run(tmp_path: Path) -> None:
    """A kettle artefact claims its run; a heavy artefact does not; a program never does."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=3, interval_s=(30.0, 30.0)), END_MS)
    context = _context(path)
    on_ms, off_ms = data.kettle_runs[-1]

    kettle = make_switched_candidate("kettle", "Kettle", rise_w=2000.0)
    owned = replay_check.replay_check(context, kettle, rise_ms=on_ms, fall_ms=off_ms)
    assert owned["would_claim"] is True
    assert owned["n_episodes"] >= 1
    assert owned["basis"] == BASIS

    heavy = make_switched_candidate("heavy", "Heavy", rise_w=9000.0)
    assert (
        replay_check.replay_check(context, heavy, rise_ms=on_ms, fall_ms=off_ms)[
            "would_claim"
        ]
        is False
    )

    program = {**make_switched_candidate("prog", "Program"), "kind": "program"}
    out = replay_check.replay_check(context, program, rise_ms=on_ms, fall_ms=off_ms)
    assert out["would_claim"] is False
    assert out["n_episodes"] == 0
    assert "program" in out["basis"]
