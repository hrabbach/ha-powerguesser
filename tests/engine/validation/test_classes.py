"""Device class derivation and the class pass bar (Plan 05-02, Task 3; D-06, D-08, D-21).

Class = kind + band of the largest state rise_w.p50 (A1) + level count, with
15 % hysteresis at band edges against the class stored at the previous
validation (D-06). class_passes applies the D-08 bar over every peer with a
figure (A2).
"""

from typing import Any

from custom_components.powerguesser.engine.validation.classes import (
    class_passes,
    derive_class,
)
from tests.fixtures.detect_traces import switched_artefact


def _artefact(p50: float, states: int = 1) -> dict[str, Any]:
    """A switched artefact whose states carry the given rise p50 values."""
    arte = switched_artefact(
        "sig-x",
        rise=(p50 * 0.9, p50, p50 * 1.1),
        fall=(-p50 * 1.1, -p50, -p50 * 0.9),
        plateau=(p50 * 0.9, p50, p50 * 1.1),
    )
    for _ in range(states - 1):
        arte["states"].append(dict(arte["states"][0]))
    return arte


def _program_artefact(levels: list[float]) -> dict[str, Any]:
    """A program artefact following the Plan 06-01 session["stages"] contract.

    Program states are degenerate rise boxes around 0 W (RESEARCH Pitfall 1), so
    the class must come from the stage template. Built as a plain dict rather
    than through ``tests/fixtures/programs.py`` (created by Plan 06-01) so this
    plan has no code dependency on that one.
    """
    stages = [
        {
            "name": f"s{i + 1}",
            "level_w": {"lo": p * 0.95, "p50": p, "hi": p * 1.05},
            "entry_w": {"lo": p * 0.9, "p50": p, "hi": p * 1.1},
            "dur_s": {"lo": 30.0, "p50": 600.0, "hi": 1800.0},
            "optional": False,
        }
        for i, p in enumerate(levels)
    ]
    return {
        "schema_version": 1,
        "signature_id": "sig-program",
        "kind": "program",
        "label": "Program",
        "category": "other",
        "extraction": {"grid_s": 30, "edge_threshold_w": 100.0},
        "states": [
            {
                "rise_w": {"lo": -20.0, "p50": 0.0, "hi": 20.0},
                "fall_w": {"lo": -20.0, "p50": 0.0, "hi": 20.0},
                "dur_s": {"lo": 30.0, "p50": 600.0, "hi": 1800.0},
                "plateau_w": {"lo": 0.0, "p50": 0.0, "hi": 0.0},
                "phase_share": None,
            }
        ],
        "session": {
            "stages": stages,
            "terminal": {"fall_w": {"lo": -2000.0, "p50": -2000.0, "hi": -2000.0}},
        },
        "time_prior": {},
        "constraints": {},
        "provenance": {},
        "scores": {},
    }


def test_derive_class_single_state_2000w() -> None:
    """A single-state 2000 W artefact lands in the 1.5-3 kW band, one level."""
    assert derive_class(_artefact(2000.0)) == "switched_1.5_3kw_1_level"


def test_derive_class_multi_state_largest_p50() -> None:
    """The band comes from the largest rise p50 across states (A1)."""
    arte = _artefact(2000.0, states=2)
    arte["states"][1]["rise_w"] = {"lo": 3000.0, "p50": 3200.0, "hi": 3400.0}
    assert derive_class(arte) == "switched_over_3kw_multi_level"


def test_derive_class_hysteresis_keeps_previous_band_above_edge() -> None:
    """3300 W stays in 1.5_3kw: inside 3000 x 1.15 against the previous class."""
    previous = "switched_1.5_3kw_1_level"
    assert derive_class(_artefact(3300.0), previous) == previous


def test_derive_class_hysteresis_moves_beyond_edge() -> None:
    """3500 W is outside 3000 x 1.15: the class moves to over_3kw."""
    assert (
        derive_class(_artefact(3500.0), "switched_1.5_3kw_1_level")
        == "switched_over_3kw_1_level"
    )


def test_derive_class_hysteresis_keeps_previous_band_below_edge() -> None:
    """480 W stays in 0.5_1.5kw: inside 500 x 0.85 against the previous class."""
    previous = "switched_0.5_1.5kw_1_level"
    assert derive_class(_artefact(480.0), previous) == previous


def test_derive_class_ignores_incompatible_previous() -> None:
    """Hysteresis only honours the same kind and level count (T-05-24).

    A corrupt or stale stored class cannot pin an arbitrary band: different
    kind, different level count and non-adjacent bands are all ignored.
    """
    arte = _artefact(3300.0)
    # different level count
    assert derive_class(arte, "switched_1.5_3kw_multi_level") == "switched_over_3kw_1_level"
    # not adjacent (under_0.5kw vs over_3kw)
    assert derive_class(arte, "switched_under_0.5kw_1_level") == "switched_over_3kw_1_level"
    # unparseable garbage
    assert derive_class(arte, "not-a-class") == "switched_over_3kw_1_level"
    # different kind: a program artefact with no stage template has no class,
    # so its degenerate ~0 W rise boxes can never pin a band (P5 D-08). The
    # kind-mismatch hysteresis case for a real program template is covered by
    # test_derive_class_program_hysteresis_ignores_other_kind below.
    program = {**arte, "kind": "program"}
    assert derive_class(program, "switched_1.5_3kw_1_level") is None


def test_derive_class_program_dishwasher() -> None:
    """RESEARCH Pitfall 1: a program class comes from its stage power.

    A dishwasher with 2000 W heat stages reads program_1.5_3kw_multi_level from
    its largest stage level_w.p50, never under_0.5kw from its degenerate rise
    boxes around 0 W.
    """
    arte = _program_artefact([2000.0, 150.0, 2000.0, 80.0])
    assert derive_class(arte) == "program_1.5_3kw_multi_level"


def test_derive_class_program_missing_template_is_none() -> None:
    """A program without a usable stage template has no class (P5 D-08).

    No session, session None, no stages, a non-list stages value and stages
    that carry no numeric level p50 all return None, so the device can never
    be counted under an unrelated passing class (T-06-28).
    """
    arte = _program_artefact([2000.0])
    no_session = dict(arte)
    no_session.pop("session")
    assert derive_class(no_session) is None
    assert derive_class({**arte, "session": None}) is None
    assert derive_class({**arte, "session": {}}) is None
    assert derive_class({**arte, "session": {"stages": []}}) is None
    assert derive_class({**arte, "session": {"stages": "junk"}}) is None
    bare = {
        "kind": "program",
        "states": arte["states"],
        "session": {"stages": [{"name": "s1", "entry_w": {}, "dur_s": {}}]},
    }
    assert derive_class(bare) is None


def test_derive_class_program_single_stage_and_washer() -> None:
    """The level count comes from the stage count, the band from the max p50."""
    assert derive_class(_program_artefact([900.0])) == "program_0.5_1.5kw_1_level"
    assert (
        derive_class(_program_artefact([2000.0, 300.0, 500.0]))
        == "program_1.5_3kw_multi_level"
    )


def test_derive_class_program_never_under_0_5kw_from_degenerate_state() -> None:
    """RESEARCH Pitfall 1: the 0 W degenerate rise box never decides the band."""
    arte = _program_artefact([2000.0])
    assert arte["states"][0]["rise_w"]["p50"] == 0.0
    assert derive_class(arte) == "program_1.5_3kw_1_level"


def test_derive_class_program_hysteresis_keeps_previous_band() -> None:
    """1450 W stays in 1.5_3kw: inside 1500 x 0.85 with same kind and levels."""
    previous = "program_1.5_3kw_multi_level"
    assert derive_class(_program_artefact([1450.0, 300.0]), previous) == previous


def test_derive_class_program_hysteresis_ignores_other_kind() -> None:
    """A switched previous class never pins a program band (T-05-24)."""
    arte = _program_artefact([1450.0, 300.0])
    assert (
        derive_class(arte, "switched_1.5_3kw_multi_level")
        == "program_0.5_1.5kw_multi_level"
    )


def test_derive_class_program_skips_non_finite_and_malformed_stages() -> None:
    """NaN/Inf levels and non-mapping stages are skipped before any band lookup.

    Same rejection semantics as the detector-side _check_finite: a NaN p50 must
    never reach _band_of (L286@8b3be9c, T-06-29).
    """
    arte = _program_artefact([2000.0])
    arte["session"]["stages"].extend(
        [
            "junk",
            {"name": "nan", "level_w": {"lo": 1.0, "p50": float("nan"), "hi": 2.0}},
            {"name": "inf", "level_w": {"lo": 1.0, "p50": float("inf"), "hi": 2.0}},
            {"name": "text", "level_w": {"lo": 1.0, "p50": "high", "hi": 2.0}},
            {"name": "absent", "level_w": {"lo": 1.0, "hi": 2.0}},
            {"name": "no-level", "entry_w": {}},
        ]
    )
    assert derive_class(arte) == "program_1.5_3kw_1_level"


def test_class_passes_bar() -> None:
    """D-08: >= 2 passing peers, median <= 25 %, no peer above 35 % (A2 peers)."""
    # two passing peers (10 %, 18 %) and a non-passing peer at 30 %: passes
    assert class_passes([10.0, 18.0, 30.0], n_passing=2) is True
    # one passing peer: fails the count
    assert class_passes([10.0, 30.0], n_passing=1) is False
    # a peer above 35 %: fails the peer cap
    assert class_passes([10.0, 18.0, 36.0], n_passing=2) is False
    # median above 25 %: fails the median cap
    assert class_passes([10.0, 26.0, 30.0], n_passing=2) is False
