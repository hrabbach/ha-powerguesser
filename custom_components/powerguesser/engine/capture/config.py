"""Capture tuning parameters as a frozen, slotted dataclass (D-02, D-09).

Every engine capture function takes a ``config: CaptureConfig`` argument
(defaulting to :data:`DEFAULT_CONFIG`) instead of reading the module constants
directly, so post-ship tuning can vary a value without editing a module. The
defaults are read from :mod:`constants` (each an INITIAL VALUE there).
"""

from dataclasses import dataclass

from . import constants as c


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Tuning parameters for the capture snap and preview fit."""

    snap_margin_s: int = c.SNAP_MARGIN_S
    context_before_s: int = c.CONTEXT_BEFORE_S
    context_after_s: int = c.CONTEXT_AFTER_S
    ambiguity_ratio: float = c.AMBIGUITY_RATIO
    inside_edge_rel: float = c.INSIDE_EDGE_REL
    min_run_samples: int = c.MIN_RUN_SAMPLES
    gap_max_floor_s: int = c.GAP_MAX_FLOOR_S
    gap_max_samples: int = c.GAP_MAX_SAMPLES
    trace_max_points: int = c.TRACE_MAX_POINTS
    trace_pad_s: int = c.TRACE_PAD_S
    live_rise_before_s: int = c.LIVE_RISE_BEFORE_S
    live_rise_after_s: int = c.LIVE_RISE_AFTER_S
    live_fall_before_s: int = c.LIVE_FALL_BEFORE_S
    live_stop_settle_samples: int = c.LIVE_STOP_SETTLE_SAMPLES
    live_stop_settle_min_s: int = c.LIVE_STOP_SETTLE_MIN_S
    live_stop_settle_max_s: int = c.LIVE_STOP_SETTLE_MAX_S
    live_stop_max_wait_s: int = c.LIVE_STOP_MAX_WAIT_S
    live_still_on_frac: float = c.LIVE_STILL_ON_FRAC
    pre_level_samples: int = c.PRE_LEVEL_SAMPLES
    live_max_open_s: int = c.LIVE_MAX_OPEN_S
    preview_w_tol: float = c.PREVIEW_W_TOL
    preview_dur_tol: float = c.PREVIEW_DUR_TOL
    confirm_min_runs: int = c.CONFIRM_MIN_RUNS
    anchor_tol_steps: float = c.ANCHOR_TOL_STEPS
    holdout_pad_steps: int = c.HOLDOUT_PAD_STEPS
    replay_pre_roll_s: int = c.REPLAY_PRE_ROLL_S
    max_capture_records: int = c.MAX_CAPTURE_RECORDS
    similar_top_k: int = c.SIMILAR_TOP_K
    earlier_runs_lookback_days: int = c.EARLIER_RUNS_LOOKBACK_DAYS


DEFAULT_CONFIG = CaptureConfig()
