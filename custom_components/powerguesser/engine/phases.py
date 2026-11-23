"""Phase sum and sign check calculations for PowerGuesser."""

from collections.abc import Sequence
from enum import StrEnum

PHASE_TOLERANCE_W = 200.0
PHASE_TOLERANCE_FRAC = 0.10


class PhaseSumResult(StrEnum):
    """Result of comparing per-phase power sum against grid power."""

    OK = "ok"
    INVERTED = "inverted"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"


def check_phase_sum(
    phase_values_w: Sequence[float | None],
    grid_w: float | None,
) -> PhaseSumResult:
    """Compare sum of mapped per-phase active power values against total grid power.

    Returns SKIPPED if grid_w is None, phase_values_w is empty, or any phase value is None.
    Otherwise evaluates sum within max(200 W, 10% of abs(grid_w)):
    - OK if abs(sum - grid_w) <= tol
    - INVERTED if abs(sum + grid_w) <= tol
    - MISMATCH otherwise
    """
    if grid_w is None or not phase_values_w or any(v is None for v in phase_values_w):
        return PhaseSumResult.SKIPPED

    s = sum(v for v in phase_values_w if v is not None)
    tol = max(PHASE_TOLERANCE_W, PHASE_TOLERANCE_FRAC * abs(grid_w))

    if abs(s - grid_w) <= tol:
        return PhaseSumResult.OK
    if abs(s + grid_w) <= tol:
        return PhaseSumResult.INVERTED
    return PhaseSumResult.MISMATCH
