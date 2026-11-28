"""Named capture constants and vocabularies (D-02, D-03, D-09).

Every numeric constant below is an INITIAL VALUE (D-09): Phase 8 ships the
research defaults and each is carried by :class:`CaptureConfig` so post-ship
tuning can vary it without editing modules. Sources are the Phase 8 spike
(S1-S5) and research Pitfalls 2 and 7.
"""

# --- Snap margins and context (Spike S2, Pattern 1) ---
# INITIAL VALUE: the half-width of the rise/fall search window around the
# user's start/end (Spike S2: 120 s keeps >= 98 % correct at +-120 s timing
# error while ambiguity stays ~2 %).
SNAP_MARGIN_S = 120
# INITIAL VALUE: the residual context loaded before the window start (24 h) so
# edge_threshold and the pairing see the same neighbourhood the learner does.
CONTEXT_BEFORE_S = 86400
# INITIAL VALUE: the residual context loaded after the window end (1 h).
CONTEXT_AFTER_S = 3600

# --- Verdict thresholds (Pattern 5, Spike S5) ---
# INITIAL VALUE: the runner-up score must exceed the best score by this factor
# for the result to stay clean; otherwise it is ambiguous (Pattern 5).
# (Reserved: the ambiguous/contaminated rules land in Plan 08-03.)
AMBIGUITY_RATIO = 2.0
# INITIAL VALUE: an inside edge counts as contamination at or above
# max(threshold_w, INSIDE_EDGE_REL x rise_w) (Spike S5: 0.25 is the knee).
INSIDE_EDGE_REL = 0.25
# INITIAL VALUE: a snapped run needs at least this many plateau samples; no
# 1-sample run survives pairing (Spike S3).
MIN_RUN_SAMPLES = 2
# INITIAL VALUE: an inside native gap counts as a gap above this floor (S1);
# the effective threshold is max(this, GAP_MAX_SAMPLES x effective_s).
GAP_MAX_FLOOR_S = 150
# INITIAL VALUE: the sample-count multiplier for the inside-gap threshold.
GAP_MAX_SAMPLES = 5

# --- Trace (Pattern 2) ---
# INITIAL VALUE: a stored trace is decimated to at most this many int values.
TRACE_MAX_POINTS = 300
# INITIAL VALUE: the trace extends this far either side of the snapped run.
TRACE_PAD_S = 300

# --- Live capture (Pitfall 2, Pitfall 7; Plan 08-03 consumes these) ---
# INITIAL VALUE: live rise search before the press (180 s).
LIVE_RISE_BEFORE_S = 180
# INITIAL VALUE: live rise search after the press (120 s).
LIVE_RISE_AFTER_S = 120
# INITIAL VALUE: live fall search before the stop press (180 s).
LIVE_FALL_BEFORE_S = 180
# INITIAL VALUE: settle wait = clamp(this x effective_s, MIN, MAX) (S1).
LIVE_STOP_SETTLE_SAMPLES = 3
LIVE_STOP_SETTLE_MIN_S = 90
LIVE_STOP_SETTLE_MAX_S = 180
# INITIAL VALUE: one extension of the still-on wait, then fail with no_off_edge.
LIVE_STOP_MAX_WAIT_S = 300
# INITIAL VALUE: "still on" when the residual stays above this fraction of rise.
LIVE_STILL_ON_FRAC = 0.7
# INITIAL VALUE: the still_on pre-level is the median of this many finite
# residual samples ending at the rise edge (Plan 08-03 Task 2 consumes it).
PRE_LEVEL_SAMPLES = 4
# INITIAL VALUE: an open live capture auto-expires after this many seconds.
LIVE_MAX_OPEN_S = 3600

# --- Preview tolerance (D-02) ---
# INITIAL VALUE: watts tolerance fraction, floored at threshold_w / 2.
PREVIEW_W_TOL = 0.15
# INITIAL VALUE: duration tolerance fraction, floored at one grid step.
PREVIEW_DUR_TOL = 0.30
# INITIAL VALUE: D-03 confirm gate: fewer clean runs than this cannot confirm.
CONFIRM_MIN_RUNS = 3

# --- Reconciliation / replay (Plan 08-05, 08-06; declared now) ---
# INITIAL VALUE: an anchor maps to an episode row within this many grid steps.
ANCHOR_TOL_STEPS = 1.5
# INITIAL VALUE: a holdout window is padded this many grid steps each side.
HOLDOUT_PAD_STEPS = 2
# INITIAL VALUE: the D-04 replay check pre-roll before the rise.
REPLAY_PRE_ROLL_S = 600
# INITIAL VALUE: at most this many capture records are retained (Pitfall 8).
MAX_CAPTURE_RECORDS = 200

# --- Insight (Plan 09-02, D-11/D-12) ---
# INITIAL VALUE: how many most-similar candidates/devices a capture shows. This
# is the DISPLAY list only: the near-twin flag is computed over the full scored
# pool, never this list (D-11).
SIMILAR_TOP_K = 3
# INITIAL VALUE: how far back of the captured rise the earlier-runs replay looks,
# clipped to the buffer (D-12).
EARLIER_RUNS_LOOKBACK_DAYS = 28

# --- Vocabularies ---
# INITIAL VALUE: every capture record state (Pattern 2; OPEN_STATES is its
# subset that survives only in memory).
CAPTURE_STATES: tuple[str, ...] = (
    "recording",
    "resolving",
    "done",
    "failed",
    "cancelled",
    "interrupted",
)
OPEN_STATES: tuple[str, ...] = ("recording", "resolving")
TERMINAL_STATES: tuple[str, ...] = ("done", "failed", "cancelled", "interrupted")
# INITIAL VALUE: the snap quality verdicts (Pattern 5).
VERDICTS: tuple[str, ...] = ("clean", "contaminated", "ambiguous", "none")
# INITIAL VALUE: a new-device preview's signature id prefix.
PREVIEW_ID_PREFIX = "cap_"
