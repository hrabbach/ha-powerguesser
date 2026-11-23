"""Pure stage-tracking rules for program appliances (HA-free).

Implements the D-01 measured-level clamp, the D-03 recognition-lag cap
(promotion after stage 1 is held inside its level box), D-04 divergence and
overdue predicates, and the D-22 constant set.  Constants are conservative
initial values tuned post-ship on real data (D-22).
"""

import math

PROGRAM_PROMOTE_MIN_STEPS = 2  # D-03/D-22: recognition lag never below two samples
PROGRAM_PROMOTE_FRACTION = 1.0  # D-03: lag cap = stage-1 dur_s.lo (conservative initial)
PROGRAM_DIVERGE_STEPS = 4  # D-04: same value as online.LOW_RESIDUAL_STEPS
PROGRAM_ADVANCE_STEPS = 2  # D-04: level-based stage advance persistence


def in_box(value: float, box: tuple[float, float, float]) -> bool:
    """Inclusive lo/hi containment."""
    lo, _, hi = box
    return lo <= value <= hi


def reported_level(measured: float, box: tuple[float, float, float]) -> float:
    """D-01: the measured value inside the stage box, else the box p50."""
    lo, p50, hi = box
    return measured if lo <= measured <= hi else p50


def promote_steps(stage1_dur_lo_s: float, grid_s: float) -> int:
    """D-03: samples stage 1 must be held before promotion, bounded below by 2."""
    frac = PROGRAM_PROMOTE_FRACTION * stage1_dur_lo_s / max(float(grid_s), 1e-9)
    return max(PROGRAM_PROMOTE_MIN_STEPS, int(math.ceil(frac)))


def next_stage_indices(optional_flags: tuple[bool, ...], k: int) -> tuple[int, ...]:
    """Stages reachable from stage k: k+1, plus k+2 when stage k+1 is optional."""
    n = len(optional_flags)
    if k + 1 >= n:
        return ()
    out = [k + 1]
    if optional_flags[k + 1] and k + 2 < n:
        out.append(k + 2)
    return tuple(out)


def stage_due(time_in_stage_s: float, dur_box: tuple[float, float, float]) -> bool:
    """The next-stage entry explanation is offered once the duration lo is met."""
    return time_in_stage_s >= dur_box[0]


def stage_overdue(
    time_in_stage_s: float,
    dur_box: tuple[float, float, float],
    stuck_factor: float,
) -> bool:
    """A stage held beyond stuck_factor x dur_s.hi is stuck (D-04)."""
    return time_in_stage_s > stuck_factor * dur_box[2]
