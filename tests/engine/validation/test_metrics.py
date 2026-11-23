"""Edge tests for the holdout metrics (Plan 05-02, Task 2).

None, never 0, for empty denominators (RESEARCH "Code Examples"); the NDE of
identical series is exactly 0; the training-mean baseline of a constant
series equal to the mean is exactly right on both figures.
"""

import numpy as np
import pytest

from custom_components.powerguesser.engine.validation.metrics import (
    baseline,
    energy_error_pct,
    mae,
    nde,
)


def test_energy_error_pct_none_for_zero_reference() -> None:
    """A zero (or negative) reference energy refuses, never reports 0 %."""
    assert energy_error_pct(0.0, 5.0) is None
    assert energy_error_pct(-1.0, 5.0) is None


def test_energy_error_pct_value() -> None:
    """100 Wh reference, 90 Wh estimate: exactly 10.0 %."""
    assert energy_error_pct(100.0, 90.0) == 10.0


def test_nde_identical_arrays_is_zero() -> None:
    """The estimate matches the reference: NDE is exactly 0."""
    y = np.array([0.0, 500.0, 2000.0, 0.0, 800.0])
    assert nde(y, y.copy()) == 0.0


def test_nde_all_zero_reference_is_none() -> None:
    """sum(y^2) == 0 means nothing was consumed: refused, never 0."""
    y = np.zeros(5)
    assert nde(y, np.full(5, 100.0)) is None


def test_mae_empty_input_is_none() -> None:
    """No samples to average: refused, never 0."""
    assert mae(np.array([]), np.array([])) is None


def test_baseline_constant_series_matches_mean() -> None:
    """A constant series equal to the training mean: both baseline figures are 0."""
    y = np.full(120, 500.0)
    result = baseline(y, 500.0, 30.0)
    assert result["nde"] == 0.0
    assert result["energy_error_pct"] == 0.0
