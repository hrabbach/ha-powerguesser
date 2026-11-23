"""Holdout validation contract constants (Plan 05-01).

Each constant names its decision. Plan 05-02 appends the D-06/D-07/D-08/D-12
pass-bar and data-gate thresholds as initial values calibrated on the first
real holdout run (CONTEXT Specific Ideas). Stdlib only; HA-free.
"""

# D-11/D-18: the holdout is the most recent 7 calendar days, gaps included.
SHADOW_HOLDOUT_DAYS = 7

# D-04: the shadow pass re-runs when the stored set is older than 30 days;
# validate flags an older set (shadow_stale) but still measures it.
SHADOW_MAX_AGE_DAYS = 30

# D-01: the powerguesser.shadow Store contract version (costly: read by
# ValidationManager.async_load and written by Plan 05-03's shadow pass).
SHADOW_SCHEMA_VERSION = 1

# D-01: the run_validation report contract version (costly: every report
# carries it; Plans 05-02..05-06 extend the report under this version).
REPORT_SCHEMA_VERSION = 1

# --- Pass bars and data gates (Plan 05-02) ---
# All of the following are INITIAL VALUES, to be calibrated on the first real
# holdout run on the reference install (CONTEXT Specific Ideas). Each names its
# decision. Stdlib only; HA-free.

# D-12: fewer usable days than this in the current window refuse the report.
MIN_USABLE_DAYS = 28

# D-12: fewer holdout days at >= 50 % coverage than this refuse the report.
MIN_COVERED_HOLDOUT_DAYS = 5

# D-12: a plug with fewer ON sessions than this in the holdout gets no figure.
MIN_EVENTS_FIGURE = 5

# D-07: a plug device passes only with at least this many holdout ON sessions.
MIN_EVENTS_PASS = 8

# D-07: a plug device passes only at or below this energy error (%).
MAX_DEVICE_ERROR_PCT = 20.0

# D-08: a class passes with at least this many passing plug devices.
CLASS_MIN_PASSING = 2

# D-08: a class passes only at or below this median peer energy error (%).
CLASS_MAX_MEDIAN_PCT = 25.0

# D-08: a class passes only when no peer is above this energy error (%).
CLASS_MAX_PEER_PCT = 35.0

# D-06: rise-p50 band edges in watts: under 0.5 kW, 0.5-1.5 kW, 1.5-3 kW, over 3 kW.
BAND_EDGES_W = (500.0, 1500.0, 3000.0)

# D-06: hysteresis at band edges so a device does not flip class on noise.
BAND_HYSTERESIS = 0.15

# --- Shadow linking (Plan 05-03) ---
# D-03/A12: a shadow artefact links to a plug only at or above this Jaccard
# overlap of their train-block ON masks (both sides gap-merged at PLUG_GAP_S).
# INITIAL VALUE, to be calibrated on the first real holdout run.
LINK_MIN_OVERLAP = 0.5

# --- Live drift guard (Plan 05-05, D-15) ---
# All of the following are INITIAL VALUES per D-15, to be calibrated on the
# first real holdout run on the reference install. Stdlib only; HA-free, so
# LiveDrift is safe inside the DeviceRuntime @callback detector drain path.
# The detector's ClosedEpisode stream feeds the window; one evaluation runs
# when an ON episode closes, never per sample.

# D-15: evaluation window = the last 20 closed ON episodes ...
LIVE_WINDOW_EPISODES = 20

# D-15: ... at least 8 of them required, else the device sits in insufficient
# evidence (drift false, shown as drift_evidence "insufficient").
LIVE_MIN_EPISODES = 8

# D-15: ... capped at 14 days (episodes older than the newest minus this age
# out of the window; the baseline survives ageing by design).
LIVE_WINDOW_MAX_DAYS = 14

# D-15: the baseline is the median match_score and refused ratio of the first
# 10 episodes after confirm or after a version activation (D-22 reset).
LIVE_BASELINE_EPISODES = 10

# D-15: drift sets when the window median match_score is below 0.6 ...
MATCH_SCORE_MIN = 0.6

# D-15: ... or more than 0.2 below the baseline median.
MATCH_SCORE_MAX_DROP = 0.2

# D-15: ... or the refused-edge ratio is above 0.30 ...
REFUSED_RATIO_MAX = 0.30

# D-15: ... and at least 0.15 above the baseline ratio.
REFUSED_RATIO_MAX_RISE = 0.15

# D-15: hysteresis - drift sets after 2 consecutive evaluations over the
# threshold; the over counter resets on any non-over or insufficient one.
DRIFT_SET_AFTER = 2

# D-15: drift clears after 1 evaluation that is 0.05 clear of triggering, in
# each metric's own units, applied asymmetrically by rule shape: the OR rule
# (match_score_low) clears only when BOTH of its bounds clear, the AND rule
# (edges_refused) clears when EITHER of its bounds clears (cross-AI review
# Round 1 made this reading explicit).
DRIFT_CLEAR_MARGIN = 0.05

# RESEARCH Pattern 4: at most this many closed episodes per device persist in
# the hot Store (plus the baseline and the hysteresis state).
EVIDENCE_MAX = 40

# --- Live drift for program devices (Plan 06-11, D-26) ---
# All of the following are INITIAL VALUES per D-26, to be calibrated like the
# P5 values on the first real holdout run on the reference install. A program
# device (washer, dryer, dishwasher) used a few times a week never reaches the
# D-15 8 episodes in 14 days, so its live guard would sit in insufficient
# evidence forever (RESEARCH C6); a longer window and a smaller baseline make a
# verdict reachable from a few weeks of normal use. LIVE_WINDOW_EPISODES stays
# shared by both kinds. Stdlib only; HA-free.

# D-26: at least this many program window episodes for a verdict ...
PROGRAM_LIVE_MIN_EPISODES = 3

# D-26: ... the baseline is the first this many episodes after confirm or a
# D-22 version reset (the switched baseline of 10 is unreachable for a rare
# device) ...
PROGRAM_LIVE_BASELINE_EPISODES = 3

# D-26: ... and the window is capped at this many days (the switched 14-day cap
# would age out a rarely run device's whole window).
PROGRAM_LIVE_WINDOW_MAX_DAYS = 42

# --- Holdout-basis drift (Plan 05-06, D-15/D-23) ---
# INITIAL VALUES per D-15, to be calibrated on the first real holdout run.
# Pure stdlib; HA-free, so evaluate_holdout_drift stays testable without HA.

# D-15: holdout drift sets when the latest error is at least this many
# percentage points above the best earlier error of the SAME shadow set ...
HOLDOUT_DRIFT_MIN_PP = 5.0

# D-15: ... and at least this factor times the best earlier error ...
HOLDOUT_DRIFT_FACTOR = 1.5

# D-15: ... on 2 consecutive distinct holdout windows (the hysteresis shape
# DRIFT_SET_AFTER/DRIFT_CLEAR_MARGIN above is reused; the clear margin is
# applied as HOLDOUT_DRIFT_MIN_PP of energy error, D-15's 0.05 margin in
# error-fraction units).

# T-05-22: at most this many history entries (across all shadow sets) are
# kept per plug entity id or "class:<class>" key; the oldest holdout_end is
# dropped first.
HISTORY_MAX = 20

# --- Overlap attribution report (Plan 06-08, D-08/D-28) ---
# D-08: below this many ticks an overlap figure is null, never 0 (the report's
# refusal-without-figures rule, 05-LEARNINGS.md). INITIAL VALUE, to be
# calibrated on the first real holdout run on the reference install. A pair's
# ticks aggregate across the WHOLE holdout window (all overlap episodes summed),
# so the threshold applies to that total, never per episode.
OVERLAP_MIN_TICKS = 10
