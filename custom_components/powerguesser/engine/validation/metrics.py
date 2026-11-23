"""Holdout validation metrics (energy half plus the NDE/MAE half, Plan 05-02).

numpy and the standard library only; this module must stay HA-free
(tests/engine/test_no_ha_imports.py). Every metric returns None for an empty
denominator (RESEARCH "Code Examples": refuse, never 0 %).
"""

from collections.abc import Sequence

import numpy as np


def energy_error_pct(e_true_wh: float, e_est_wh: float) -> float | None:
    """Energy error (%) of an estimate against a reference.

    Returns round(100 x |e_est - e_true| / e_true, 1), or None when the
    reference energy is not positive: with nothing to compare against the
    figure is refused, never reported as 0 %.
    """
    if e_true_wh <= 0.0:
        return None
    return round(100.0 * abs(e_est_wh - e_true_wh) / e_true_wh, 1)


def nde(y: Sequence[float], y_hat: Sequence[float]) -> float | None:
    """Normalized disaggregation error (A7): sum((y - y_hat)^2) / sum(y^2).

    Returns round(..., 4), or None when the reference energy sum(y^2) is zero:
    a zero denominator means nothing was consumed, so there is no error to
    normalize — refused, never 0.
    """
    arr = np.asarray(y, dtype=np.float64)
    est = np.asarray(y_hat, dtype=np.float64)
    denom = float(np.sum(arr * arr))
    if denom <= 0.0:
        return None
    return round(float(np.sum((arr - est) ** 2)) / denom, 4)


def mae(y: Sequence[float], y_hat: Sequence[float]) -> float | None:
    """Mean absolute error in watts: mean |y - y_hat| over the given samples.

    Returns round(..., 1), or None for an empty input (no samples to average).
    """
    arr = np.asarray(y, dtype=np.float64)
    est = np.asarray(y_hat, dtype=np.float64)
    if len(arr) == 0:
        return None
    return round(float(np.mean(np.abs(arr - est))), 1)


def baseline(y: Sequence[float], mean_w: float, step_s: float) -> dict[str, float | None]:
    """Naive baseline (A8): a constant equal to the plug's mean power over the
    shadow training block (shadow plugs[entity_id].train_mean_w), replayed over
    the same samples as the estimate, so the comparison covers identical data."""
    arr = np.asarray(y, dtype=np.float64)
    y_hat = np.full(len(arr), float(mean_w), dtype=np.float64)
    e_true_wh = float(np.sum(arr)) * step_s / 3600.0
    e_base_wh = float(mean_w) * len(arr) * step_s / 3600.0
    return {
        "energy_error_pct": energy_error_pct(e_true_wh, e_base_wh),
        "nde": nde(arr, y_hat),
    }
