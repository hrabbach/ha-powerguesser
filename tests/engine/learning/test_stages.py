"""Stage-template fit behavior cases (Plan 06-01, Task 2, D-27/LRN-03)."""

import json
import math

import numpy as np

from custom_components.powerguesser.engine.learning import compile as compile_mod
from custom_components.powerguesser.engine.learning import prompts
from custom_components.powerguesser.engine.learning.compile import compile_proposal
from custom_components.powerguesser.engine.learning.stages import (
    STAGE_OPTIONAL_FRACTION,
    expand_rle,
    fit_stages,
)
from tests.fixtures.programs import (
    dishwasher_stages,
    dryer_stages,
    hob_session_stages,
    program_episode_table,
    program_rle,
    washer_stages,
)

GRID_S = 30
THRESHOLD = 120.0


def _members(stages, n_runs, *, seed=11, level_jitter=0.03, dur_jitter=0.10, skip=()):
    rng = np.random.default_rng(seed)
    return [
        program_rle(
            stages,
            rng=rng,
            level_jitter=level_jitter,
            dur_jitter=dur_jitter,
            skip=set(skip),
        )
        for _ in range(n_runs)
    ]


def _expanded(members):
    return [expand_rle(levels, counts) for levels, counts in members]


def test_fit_stages_dishwasher_jitter() -> None:
    """12 jittered dishwasher runs -> 4 stages near the preset levels (unit form)."""
    template = fit_stages(_expanded(_members(dishwasher_stages(), 12)), THRESHOLD, GRID_S)
    assert template is not None
    stages = template["stages"]
    assert len(stages) == 4
    for st, want in zip(stages, [2000.0, 150.0, 2000.0, 200.0]):
        assert abs(st["level_w"]["p50"] - want) <= 0.05 * want
    terminal = template["terminal"]["fall_w"]
    assert terminal["lo"] <= -200.0 <= terminal["hi"]


def test_fit_stages_washer_optional_spin() -> None:
    """4 of 12 runs skip the 500 W spin stage -> that stage is optional only."""
    members = _members(washer_stages(), 8)
    rng = np.random.default_rng(99)
    for _ in range(4):
        members.append(program_rle(washer_stages(), rng=rng, level_jitter=0.03, dur_jitter=0.10, skip={2}))
    template = fit_stages(_expanded(members), THRESHOLD, GRID_S)
    assert template is not None
    stages = template["stages"]
    assert len(stages) == 3
    assert [st["optional"] for st in stages] == [False, False, True]


def test_fit_stages_glitch_merged() -> None:
    """A 1-sample +400 W glitch inside a stage never becomes a stage of its own."""
    rng = np.random.default_rng(5)
    members = [
        expand_rle(*program_rle(dishwasher_stages(), rng=rng, glitch=(k == 0)))
        for k in range(12)
    ]
    template = fit_stages(members, THRESHOLD, GRID_S)
    assert template is not None
    assert len(template["stages"]) == 4


def test_fit_stages_preset_counts() -> None:
    """Stage count equals the generator's count for every preset."""
    for preset, want in (
        (dishwasher_stages(), 4),
        (washer_stages(), 3),
        (dryer_stages(), 2),
        (hob_session_stages(), 3),
    ):
        template = fit_stages(_expanded(_members(preset, 12)), THRESHOLD, GRID_S)
        assert template is not None, preset
        assert len(template["stages"]) == want, preset


def test_fit_stages_too_few_members_returns_none() -> None:
    """Fewer than three members: no template."""
    members = _members(dishwasher_stages(), 2)
    assert fit_stages(_expanded(members), THRESHOLD, GRID_S) is None
    assert fit_stages([], THRESHOLD, GRID_S) is None


def test_fit_stages_one_sample_members_do_not_crash() -> None:
    """Three one-sample members (one-sample program session episodes): no
    member is segmentable, so there is no reference and no template (CR-01)."""
    assert fit_stages([np.array([5.0])] * 3, THRESHOLD, GRID_S) is None


def test_fit_stages_one_sample_member_is_never_the_reference() -> None:
    """A one-sample member is never the reference, even when every member has a
    single segment: the merge loop needs a neighbour, and a one-sample member
    has no stage structure to segment (CR-01: segmenting one raised IndexError
    at ``segs[target]`` for every member, not just the reference)."""
    members = [
        np.array([5.0]),  # the median-energy member: previously the reference
        np.zeros(10),
        np.ones(20),
    ]
    template = fit_stages(members, THRESHOLD, GRID_S)
    assert template is not None
    # One whole-run segment: the members are flat, so there is nothing to split.
    assert len(template["stages"]) == 1
    assert (
        template["stages"][0]["level_w"]["lo"]
        <= 1.0
        <= template["stages"][0]["level_w"]["hi"]
    )


def test_fit_stages_aligned_one_sample_member_alongside_runs() -> None:
    """A degenerate one-sample run among full members is aligned, never the
    reference: the template is still fitted (CR-01)."""
    members = _expanded(_members(dishwasher_stages(), 12)) + [np.array([2000.0])]
    template = fit_stages(members, THRESHOLD, GRID_S)
    assert template is not None
    assert len(template["stages"]) == 4


def test_fit_stages_skipped_stage_member_falls_back_to_aligned_samples() -> None:
    """WR-02: exactly two members align into a segment with medians far apart
    (the second member "skipped a stage" and only its tail aligned), so no
    member is within ``half`` of the centre. The stage is fitted from every
    aligned sample instead of raising IndexError on an empty percentile."""
    first = np.concatenate([np.full(10, 100.0), np.full(10, 170.0)])
    # A one-sample member, excluded from every alignment (CR-01): it keeps the
    # aligned set at two members, an even count, so the centre is their mean.
    one_sample = np.array([400.0])
    second = np.concatenate([np.full(10, 400.0), np.full(10, 470.0)])
    template = fit_stages([first, one_sample, second], THRESHOLD, GRID_S)
    assert template is not None
    stages = template["stages"]
    assert len(stages) == 2
    # Each stage's median sits between the two members' levels: both members'
    # aligned samples were used (the numbers are 250 and 320).
    assert 100.0 < stages[0]["level_w"]["p50"] < 400.0
    assert 170.0 < stages[1]["level_w"]["p50"] < 470.0


def test_compile_omits_stages_when_fit_returns_none(monkeypatch) -> None:
    """compile_proposal leaves stages/terminal out when fit_stages finds no template."""
    monkeypatch.setattr(compile_mod, "fit_stages", lambda *a, **k: None)
    members = _members(dishwasher_stages(), 12)
    table = program_episode_table(members)
    proposal = {
        "devices": [
            {
                "ref": "dev1",
                "member_clusters": ["c1"],
                "tracker_kind": "program",
                "label": "Dishwasher",
                "category": "dishwasher",
            }
        ]
    }
    artefacts, unresolved = compile_proposal(
        proposal,
        table,
        {"c1": list(range(12))},
        run_id="test-run",
        round_no=1,
        model="fake-model",
        input_digest="digest",
        extraction={"grid_s": GRID_S, "edge_threshold_w": THRESHOLD},
        threshold_w=THRESHOLD,
    )
    assert unresolved == []
    session = artefacts[0].to_dict()["session"]
    assert "stages" not in session
    assert "terminal" not in session


def test_fit_stages_member_order_independent() -> None:
    """The same members in a different order produce an identical template."""
    members = _members(dishwasher_stages(), 12)
    forward = fit_stages(_expanded(members), THRESHOLD, GRID_S)
    backward = fit_stages(_expanded(list(reversed(members))), THRESHOLD, GRID_S)
    assert forward == backward


def _walk_boxes(node, path=""):
    if isinstance(node, dict):
        if set(node) == {"lo", "p50", "hi"}:
            yield path, node
        else:
            for k, v in node.items():
                yield from _walk_boxes(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_boxes(v, f"{path}[{i}]")


def test_fit_stages_boxes_sane() -> None:
    """No NaN/inf anywhere; every box has lo <= p50 <= hi (D-27 local numbers)."""

    def clean(node) -> bool:
        if isinstance(node, dict):
            return all(clean(v) for v in node.values())
        if isinstance(node, list):
            return all(clean(v) for v in node)
        if isinstance(node, (int, float)):
            return math.isfinite(float(node))
        return isinstance(node, str)

    template = fit_stages(_expanded(_members(dishwasher_stages(), 12)), THRESHOLD, GRID_S)
    assert template is not None
    assert clean(template)
    for path, box in _walk_boxes(template):
        assert box["lo"] <= box["p50"] <= box["hi"], path


def test_proposal_schema_has_no_sequence() -> None:
    """D-27/LRN-03 pin: stage structure never comes from the LLM proposal."""
    assert "sequence" not in json.dumps(prompts.PROPOSAL_SCHEMA)
