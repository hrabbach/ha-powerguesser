"""Box-based online attribution of confirmed switching loads (HA-free).

Implements D-06..D-09 (bounded power, precision-first ambiguity refusal,
unmatched edges leave state unchanged, one-sample confirmation with
back-dated monotonic energy), the D-28/D-29/D-30 unknown state with its
entry rules and evidence exits, the D-31 one-shot post-gap plausibility
check for OFF devices, the fall-time verification against the complete
learning box rule (DET-01), and a JSON-serialisable state with covered-keys
-only restore (D-30b) and the changed-grid rebuild rule (D-26/D-29a).

Plan 06-05 (DET-03) adds the joint search (D-06/D-07/D-36/D-41): every edge is
decided against ONE candidate list of singletons and small subsets under one
runner-up margin, so a single device never wins by search order and anything
ambiguous is refused and counted as ``overlap_refused``.
"""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import statistics

from ..learning.artefact import SignatureArtefact
from ..learning.episodes import MAX_PAIR_S, PAIR_ABS_W, PAIR_TOL
from .joint import (
    JOINT_PREFILTER,
    Explanation,
    best_subsets,
    norm_distance,
    pick,
    prefilter,
    split_delta,
)
from .program import (
    PROGRAM_ADVANCE_STEPS,
    PROGRAM_DIVERGE_STEPS,
    in_box,
    next_stage_indices,
    promote_steps,
    reported_level,
    stage_due,
    stage_overdue,
)
from .stream import StreamEdges

AMBIGUITY_MARGIN = 0.25
FALL_MARGIN = 0.05
GAP_HOLD_STEPS = 2
STUCK_FACTOR = 1.5
LOW_RESIDUAL_FRACTION = 0.5
LOW_RESIDUAL_STEPS = 4

# Plan 05-05 (D-15 live proxy): the closed-episode evidence log is bounded so
# the detector can run forever in HA without growth (RESEARCH Pitfall 7);
# DeviceRuntime drains it every batch.
CLOSED_LOG_MAX = 256


@dataclass(frozen=True, slots=True)
class StateModel:
    """Compiled float-tuple state box set (rise/fall/plateau lo-p50-hi, dur lo-hi)."""

    name: str
    rise: tuple[float, float, float]
    fall: tuple[float, float, float]
    dur: tuple[float, float]
    plateau: tuple[float, float, float]
    phase_share: tuple[tuple[str, float, float], ...] | None


def _check_finite(values: Iterable[float], what: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{what} bound {v!r} is not finite")


@dataclass(frozen=True, slots=True)
class DeviceModel:
    """A confirmed device's compiled detection model."""

    key: str
    states: tuple[StateModel, ...]
    grid_s: int
    threshold_w: float
    max_on_s: float
    min_step_w: float

    @classmethod
    def from_artefact(cls, key: str, artefact: Mapping) -> "DeviceModel":
        """Compile a stored SignatureArtefact v1 dict, rejecting corrupt boxes."""
        art = SignatureArtefact.from_dict(artefact)
        if art.kind != "switched":
            raise ValueError(f"artefact kind {art.kind!r} is not detectable in Phase 4")
        if not art.states:
            raise ValueError("artefact has no states")

        states: list[StateModel] = []
        for st in art.states:
            rise = (st.rise_w.lo, st.rise_w.p50, st.rise_w.hi)
            fall = (st.fall_w.lo, st.fall_w.p50, st.fall_w.hi)
            dur = (st.dur_s.lo, st.dur_s.hi)
            plateau = (st.plateau_w.lo, st.plateau_w.p50, st.plateau_w.hi)
            _check_finite(rise, "rise_w")
            _check_finite(fall, "fall_w")
            _check_finite(dur, "dur_s")
            _check_finite(plateau, "plateau_w")
            if rise[0] > rise[2] or fall[0] > fall[2] or plateau[0] > plateau[2]:
                raise ValueError("box with lo > hi")
            if dur[0] > dur[1]:
                raise ValueError("dur_s box with lo > hi")
            if st.dur_s.hi <= 0:
                raise ValueError("dur_s.hi must be > 0")
            phase_share = None
            if st.phase_share:
                ps: list[tuple[str, float, float]] = []
                for ph in ("l1", "l2", "l3"):
                    if ph in st.phase_share:
                        plo, phi = st.phase_share[ph]
                        _check_finite((plo, phi), f"phase_share {ph}")
                        ps.append((ph, plo, phi))
                phase_share = tuple(ps) if ps else None
            states.append(
                StateModel(
                    name=st.name,
                    rise=rise,
                    fall=fall,
                    dur=dur,
                    plateau=plateau,
                    phase_share=phase_share,
                )
            )

        extraction = art.extraction
        grid_s = int(extraction.get("grid_s", 0) or 0)
        threshold_w = float(extraction.get("edge_threshold_w", 0.0) or 0.0)
        max_dur_hi = max(st.dur[1] for st in states)
        max_on_s = min(STUCK_FACTOR * max_dur_hi, float(MAX_PAIR_S))
        min_step_w = min(st.rise[1] for st in states)
        return cls(
            key=str(key),
            states=tuple(states),
            grid_s=grid_s,
            threshold_w=threshold_w,
            max_on_s=max_on_s,
            min_step_w=min_step_w,
        )

    @property
    def plateau_cap(self) -> int:
        """Hard cap on the per-device plateau list."""
        return int(self.max_on_s / self.grid_s) + 2 if self.grid_s > 0 else 2


@dataclass(frozen=True, slots=True)
class StageModel:
    """One compiled program stage (level/entry lo-p50-hi boxes, dur, flag)."""

    name: str
    level: tuple[float, float, float]
    entry: tuple[float, float, float]
    dur: tuple[float, float, float]
    optional: bool


@dataclass(frozen=True, slots=True)
class ProgramModel:
    """A confirmed program appliance's compiled stage-tracking model (DET-02)."""

    key: str
    stages: tuple[StageModel, ...]
    terminal_fall: tuple[float, float, float]
    grid_s: int
    threshold_w: float
    max_run_s: float
    min_step_w: float
    promote_steps: int

    @property
    def optional_flags(self) -> tuple[bool, ...]:
        return tuple(st.optional for st in self.stages)

    @property
    def last_required_index(self) -> int:
        required = [k for k, st in enumerate(self.stages) if not st.optional]
        return required[-1] if required else len(self.stages) - 1

    @classmethod
    def from_artefact(cls, key: str, artefact: Mapping) -> "ProgramModel":
        """Compile a stored SignatureArtefact v1 program dict, rejecting corrupt boxes."""
        art = SignatureArtefact.from_dict(artefact)
        if art.kind != "program":
            raise ValueError(f"artefact kind {art.kind!r} is not a program")
        session = art.session or {}
        raw_stages = session.get("stages")
        if not raw_stages:
            raise ValueError("program artefact has no stages")

        def _box(d: Mapping, what: str) -> tuple[float, float, float]:
            b = (float(d["lo"]), float(d["p50"]), float(d["hi"]))
            _check_finite(b, what)
            if b[0] > b[2]:
                raise ValueError("box with lo > hi")
            return b

        stages: list[StageModel] = []
        for rs in raw_stages:
            level = _box(rs["level_w"], "level_w")
            entry = _box(rs["entry_w"], "entry_w")
            dur = _box(rs["dur_s"], "dur_s")
            if dur[2] <= 0:
                raise ValueError("dur_s.hi must be > 0")
            stages.append(
                StageModel(
                    name=str(rs.get("name", f"s{len(stages) + 1}")),
                    level=level,
                    entry=entry,
                    dur=dur,
                    optional=bool(rs.get("optional", False)),
                )
            )
        term_raw = (session.get("terminal") or {}).get("fall_w")
        if term_raw is None:
            raise ValueError("program artefact has no terminal fall_w")
        terminal = _box(term_raw, "terminal fall_w")
        dur_s_box = session.get("dur_s") or {}
        try:
            run_dur_hi = float(dur_s_box["hi"])
        except (KeyError, TypeError):
            raise ValueError("program artefact session dur_s.hi missing") from None
        _check_finite((run_dur_hi,), "session dur_s")
        extraction = art.extraction
        grid_s = int(extraction.get("grid_s", 0) or 0)
        threshold_w = float(extraction.get("edge_threshold_w", 0.0) or 0.0)
        return cls(
            key=str(key),
            stages=tuple(stages),
            terminal_fall=terminal,
            grid_s=grid_s,
            threshold_w=threshold_w,
            max_run_s=STUCK_FACTOR * run_dur_hi,
            min_step_w=stages[0].entry[1],  # D-43: stage-1 entry p50
            promote_steps=promote_steps(stages[0].dur[0], grid_s),
        )


def compile_model(key: str, artefact: Mapping) -> DeviceModel | ProgramModel:
    """Compile a stored artefact into its detection model by kind."""
    kind = artefact.get("kind") if isinstance(artefact, Mapping) else None
    if kind == "switched":
        return DeviceModel.from_artefact(key, artefact)
    if kind == "program":
        return ProgramModel.from_artefact(key, artefact)
    raise ValueError(f"artefact kind {kind!r} is not detectable")


def _fall_box_match(box: tuple[float, float, float], delta: float) -> bool:
    """score.py negative-bound fall rule (inclusive bounds)."""
    lo, _, hi = box
    if hi <= 0:
        return (lo <= delta <= hi) or (lo <= -abs(delta) <= hi)
    return lo <= abs(delta) <= hi


def _norm_distance(delta: float, box: tuple[float, float, float]) -> float:
    """|delta - p50| / max(side half-width, 1.0) for the side p50 lies on.

    One definition shared with the joint subset scorer (``joint.norm_distance``),
    so a subset is scored in exactly the same form as a singleton.
    """
    return norm_distance(delta, box)


def _match_score(box: tuple[float, float, float], rise_delta: float) -> float:
    """Rise-box fit score shared by StepResult and the closed-episode log (D-15).

    One helper feeds both paths so a drift window sees exactly the score the
    live match_score attribute reports.
    """
    return 1.0 - min(1.0, _norm_distance(rise_delta, box))


def state_keys(data: Mapping | None) -> frozenset[str]:
    """Device keys and parked keys a state() dict covers (D-30b)."""
    if not isinstance(data, Mapping):
        return frozenset()
    devices = data.get("devices")
    if not isinstance(devices, Mapping):
        return frozenset()
    return frozenset(str(k) for k in devices)


@dataclass(frozen=True, slots=True)
class StepResult:
    """One grid step's attribution outcome."""

    grid_ts: float
    device_w: dict[str, float]
    on: frozenset[str]
    unknown: frozenset[str]
    match_score: dict[str, float]
    energy_wh_delta: dict[str, float]
    unattributed_edges: dict[str, int]
    unverified_episodes: dict[str, int]
    unattributed_total: int
    forced_off: int
    estimated_w: float
    # Plan 06-05 (D-07/D-36/D-41): cumulative overlap counters. Both default to
    # 0 so every existing caller and comparison keeps working.
    overlap_refused: int = 0
    joint_attributed: int = 0


@dataclass(frozen=True, slots=True)
class ClosedEpisode:
    """One attributed ON episode that just closed (Plan 05-05, D-15 live proxy).

    Recorded at every attributed fall whose runtime has a known state
    (``state_index >= 0``), verified or not: a drifted box shows up first as
    rises near the box edge whose falls may no longer verify, so the score
    must be captured for unverified closes too. A close without a known state
    records nothing, so no fallback score (0.0 or any other) can pollute a
    drift window; its refused edges still reach the next recorded close
    through the cumulative ``unattributed`` counter.
    """

    key: str
    end_ts: float
    verified: bool
    match_score: float
    unattributed: int


class _Runtime:
    """Mutable per-device detector state (fixed-size containers only)."""

    __slots__ = (
        "on",
        "unknown",
        "state_index",
        "on_w",
        "edge_ts",
        "seq",
        "unattributed",
        "unverified",
        "low_count",
        "plateau",
        "pre_edge",
        "rise_delta",
        "rise_index",
        "rise_shares",
        "nan_seen",
        "just_on",
        "episode_start_wh",
        # program runtime (DET-02): scalars and counters only (T-06-02)
        "program_phase",
        "stage_index",
        "stage_since_ts",
        "run_start_index",
        "run_start_ts",
        "base_w",
        "pending_wh",
        "committed_wh",
        "level_w",
        "confirm_steps",
        "out_steps",
        "advance_steps",
        "run_ticks",
        "in_box_ticks",
        # Plan 06-05 (D-07/D-36/D-41): overlap counters, never reset by
        # clear_run (lifetime counters, like `unattributed`).
        "overlap_refused",
        "joint_attributed",
    )

    def __init__(self) -> None:
        self.on = False
        self.unknown = False
        self.state_index = -1
        self.on_w = 0.0
        self.edge_ts: float | None = None
        self.seq = 0
        self.unattributed = 0
        self.unverified = 0
        self.low_count = 0
        self.plateau: list[float] = []
        self.pre_edge = math.nan
        self.rise_delta = 0.0
        self.rise_index: int | None = None
        self.rise_shares: tuple[float, float, float] | None = None
        self.nan_seen = False
        self.just_on = False
        self.episode_start_wh = 0.0
        self.program_phase = "idle"
        self.stage_index = 0
        self.stage_since_ts = 0.0
        self.run_start_index: int | None = None
        self.run_start_ts = 0.0
        self.base_w = 0.0
        self.pending_wh = 0.0
        self.committed_wh = 0.0
        self.level_w = 0.0
        self.confirm_steps = 0
        self.out_steps = 0
        self.advance_steps = 0
        self.run_ticks = 0
        self.in_box_ticks = 0
        self.overlap_refused = 0
        self.joint_attributed = 0

    def clear_run(self) -> None:
        """Reset every ON/unknown transition counter and episode buffer."""
        self.on = False
        self.on_w = 0.0
        self.state_index = -1
        self.low_count = 0
        self.plateau = []
        self.rise_index = None
        self.rise_shares = None
        self.nan_seen = False
        self.just_on = False
        self.program_phase = "idle"
        self.stage_since_ts = 0.0
        self.run_start_index = None
        self.run_start_ts = 0.0
        self.base_w = 0.0
        self.pending_wh = 0.0
        self.committed_wh = 0.0
        self.level_w = 0.0
        self.confirm_steps = 0
        self.out_steps = 0
        self.advance_steps = 0
        self.run_ticks = 0
        self.in_box_ticks = 0


class OnlineDetector:
    """Streaming box-based attribution over a gridded residual (D-06..D-09).

    ``unknown=`` seeds the D-28/D-29 unknown flag (parked for keys without a
    model).  One private helper ``_enter_unknown`` is the only way into the
    unknown state; one helper ``_low_residual_due`` owns the D-29b/D-30a
    predicate; one helper ``_post_gap_check`` is the one-shot D-31 check.
    """

    def __init__(
        self,
        models,
        *,
        grid_s: int,
        threshold_w: float,
        suppress_alignment: bool = True,
        unknown: Iterable[str] = (),
    ) -> None:
        self._grid_s = int(grid_s)
        self._threshold_w = float(threshold_w)
        self._edges = StreamEdges(self._threshold_w, suppress_alignment)
        self._models: dict[str, DeviceModel] = {}
        self._programs: dict[str, ProgramModel] = {}
        for m in sorted(models, key=lambda m: m.key):
            if isinstance(m, ProgramModel):
                self._programs[m.key] = m
            else:
                self._models[m.key] = m
        self._runtimes: dict[str, _Runtime] = {}
        self._energy: dict[str, float] = {}
        self._unattributed_total = 0
        self._forced_off = 0
        self._overlap_refused_total = 0
        self._joint_attributed_total = 0
        self._post_gap_armed = False
        self._gap_nan = 0
        self._last_grid_ts: float | None = None
        self._index = 0
        self._seq = 0
        self._ring: deque[tuple[int, float, float, tuple | None]] = deque(maxlen=4)
        self._episode_log: dict[str, list] = {}
        self._closed_log: deque[ClosedEpisode] = deque(maxlen=CLOSED_LOG_MAX)
        for key in unknown:
            self._runtime(str(key)).unknown = True

    # -- internal helpers -------------------------------------------------

    def _runtime(self, key: str) -> _Runtime:
        rt = self._runtimes.get(key)
        if rt is None:
            rt = _Runtime()
            self._runtimes[key] = rt
            self._energy.setdefault(key, 0.0)
        return rt

    def _enter_unknown(self, key: str) -> None:
        """Single entry point into the D-28/D-29/D-42 unknown state."""
        rt = self._runtime(key)
        was_on = rt.on
        rt.clear_run()
        rt.program_phase = "unknown"
        if not rt.unknown:
            rt.unknown = True
            if was_on:
                self._forced_off += 1

    def _low_residual_due(self, key: str, value: float, step_w: float) -> bool:
        """One predicate+persistence for the D-29b (ON) and D-30a (unknown) rules."""
        rt = self._runtime(key)
        if math.isnan(value) or value >= LOW_RESIDUAL_FRACTION * step_w:
            rt.low_count = 0
            return False
        rt.low_count += 1
        return rt.low_count >= LOW_RESIDUAL_STEPS

    def _post_gap_check(self, value: float) -> None:
        """One-shot D-31 plausibility check for plain-OFF modelled devices."""
        for key, model in self._models.items():
            rt = self._runtimes.get(key)
            if rt is not None and (rt.on or rt.unknown):
                continue
            if value >= LOW_RESIDUAL_FRACTION * model.min_step_w:
                self._enter_unknown(key)
        self._post_gap_armed = False

    def _ring_at(self, index: int) -> tuple[float, float, tuple | None] | None:
        for entry in self._ring:
            if entry[0] == index:
                return entry
        return None

    def _edge_phase_ok(self, state: StateModel, edge_index: int) -> bool:
        """Phase-share gate mirroring compute_features/score.replay."""
        if state.phase_share is None:
            return True
        pre = self._ring_at(edge_index)
        post = self._ring_at(edge_index + 1)
        if pre is None or post is None or pre[3] is None or post[3] is None:
            return True  # no live phases: check skipped
        tot = 0.0
        comps = []
        for k in range(3):
            d = abs(float(post[3][k]) - float(pre[3][k]))
            comps.append(d)
            tot += d
        if tot <= 0.0:
            return True  # shares NaN: constraint skipped, as score.replay
        shares = {"l1": comps[0] / tot, "l2": comps[1] / tot, "l3": comps[2] / tot}
        for ph, plo, phi in state.phase_share:
            sh = shares[ph]
            if not (plo <= sh <= phi):
                return False
        return True

    # -- program runtime (DET-02, D-01..D-04 as amended by D-42) -------------

    def _program_start(self, key: str, model: ProgramModel, edge, delta: float) -> None:
        """IDLE -> PENDING on a matched stage-1 entry (D-02: OFF at 0 W, nothing committed)."""
        rt = self._runtime(key)
        pre = self._ring_at(edge.index)
        rt.unknown = False
        rt.program_phase = "pending"
        rt.stage_index = 0
        rt.stage_since_ts = pre[1] if pre is not None else float(self._last_grid_ts or 0.0)
        rt.run_start_index = edge.index
        rt.run_start_ts = rt.stage_since_ts
        rt.base_w = pre[2] if pre is not None else math.nan
        rt.pending_wh = 0.0
        rt.committed_wh = 0.0
        rt.level_w = 0.0
        rt.confirm_steps = 0
        rt.out_steps = 0
        rt.advance_steps = 0
        rt.run_ticks = 0
        rt.in_box_ticks = 0
        rt.pre_edge = rt.base_w
        rt.rise_delta = delta
        rt.rise_index = edge.index
        rt.rise_shares = None
        rt.edge_ts = rt.stage_since_ts
        rt.low_count = 0
        rt.just_on = False
        rt.episode_start_wh = self._energy.get(key, 0.0)

    def _promote_program(
        self, key: str, model: ProgramModel, rt: _Runtime, measured: float, grid_ts: float
    ) -> None:
        """PENDING -> RUNNING: commit the pending energy back-dated to the run start."""
        st = model.stages[rt.stage_index]
        rt.program_phase = "running"
        rt.on = True
        rt.on_w = reported_level(measured, st.level)
        rt.level_w = measured
        rt.confirm_steps = 0
        self._seq += 1
        rt.seq = self._seq
        rt.just_on = True  # _report back-dates one sample (D-02, D-09)
        rt.edge_ts = grid_ts
        self._energy[key] = self._energy.get(key, 0.0) + rt.pending_wh
        rt.committed_wh = rt.pending_wh
        rt.pending_wh = 0.0

    def _program_advance(
        self, key: str, model: ProgramModel, rt: _Runtime, new_stage: int, since_ts: float, measured: float
    ) -> None:
        """RUNNING stage k -> stage k+1 (or k+2 across an optional stage)."""
        rt.stage_index = new_stage
        rt.stage_since_ts = since_ts
        rt.confirm_steps = 0
        rt.out_steps = 0
        rt.advance_steps = 0
        rt.on_w = reported_level(measured, model.stages[new_stage].level)
        rt.level_w = measured

    def _program_diverge(self, key: str, model: ProgramModel, rt: _Runtime) -> None:
        """D-04/D-42: a diverging, abandoned or decaying run enters the unknown state."""
        score = rt.in_box_ticks / rt.run_ticks if rt.run_ticks > 0 else 0.0
        self._closed_log.append(
            ClosedEpisode(
                key=key,
                end_ts=float(self._last_grid_ts or 0.0),
                verified=False,
                match_score=score,
                unattributed=int(rt.unattributed),
            )
        )
        self._enter_unknown(key)

    def _program_end(self, key: str, model: ProgramModel, rt: _Runtime, edge) -> None:
        """Matched terminal fall (D-42 proven-off): verified close, then OFF at 0 W."""
        from .replay import OnlineEpisode

        score = rt.in_box_ticks / rt.run_ticks if rt.run_ticks > 0 else 0.0
        energy_now = self._energy.get(key, 0.0)
        self._episode_log.setdefault(key, []).append(
            OnlineEpisode(
                start_i=int(rt.run_start_index or 0),
                end_i=int(edge.index),
                step_w=float(rt.on_w),
                energy_wh=float(energy_now - rt.episode_start_wh),
            )
        )
        self._closed_log.append(
            ClosedEpisode(
                key=key,
                end_ts=float(self._last_grid_ts or 0.0),
                verified=True,
                match_score=score,
                unattributed=int(rt.unattributed),
            )
        )
        rt.clear_run()

    def _apply_program_rise(
        self, key: str, edge, delta: float, *, stage: int | None = None
    ) -> None:
        """Apply the winning rise explanation to a program device.

        ``stage`` is the target stage index the winning explanation carries
        (D-41), so a jointly attributed edge advances the program to the stage
        whose entry box was part of the winning subset; without it the target is
        re-derived from the whole edge delta (the singleton path).
        """
        model = self._programs[key]
        rt = self._runtime(key)
        pre = self._ring_at(edge.index)
        if rt.program_phase == "idle":
            self._program_start(key, model, edge, delta)
            return
        if stage is not None:
            nxt = [stage]
        else:
            nxt = [
                n
                for n in next_stage_indices(model.optional_flags, rt.stage_index)
                if model.stages[n].entry[0] <= delta <= model.stages[n].entry[2]
            ]
        if not nxt:
            return  # defensive: the ranked decision already chose this key
        n = nxt[0]
        since = pre[1] if pre is not None else float(self._last_grid_ts or 0.0)
        post_level = (pre[2] if pre is not None else math.nan) + delta - rt.base_w
        if rt.program_phase == "pending":
            # a matched stage-2 entry confirms the program: promote and advance
            rt.stage_index = n
            rt.stage_since_ts = since
            rt.out_steps = 0
            rt.advance_steps = 0
            self._promote_program(key, model, rt, post_level, float(self._last_grid_ts or 0.0))
            return
        self._program_advance(key, model, rt, n, since, post_level)

    def _program_rise_candidates(
        self,
        delta: float,
        singletons: list[tuple[float, str, bool]],
        explanations: list[Explanation],
    ) -> None:
        """Program explanations join the switched candidates in ONE ranked list.

        An idle program offers its stage-1 entry box; a pending or running
        program whose current stage is due offers the next-stage entry boxes
        (D-41; single candidate list with the switched devices).
        """
        for key, model in self._programs.items():
            rt = self._runtime(key)
            if rt.unknown:
                continue  # an unknown program offers no rise explanations
            if rt.program_phase == "idle":
                boxes = [(0, model.stages[0].entry)]
            elif rt.program_phase in ("pending", "running"):
                time_in = float(self._last_grid_ts or 0.0) - rt.stage_since_ts
                if not stage_due(time_in, model.stages[rt.stage_index].dur):
                    continue
                boxes = [
                    (n, model.stages[n].entry)
                    for n in next_stage_indices(model.optional_flags, rt.stage_index)
                ]
            else:
                continue
            cands = [
                (_norm_distance(delta, box), n, box)
                for n, box in boxes
                if box[0] <= delta <= box[2]
            ]
            if cands:
                dist, n, box = min(cands, key=lambda c: (c[0], c[1]))
                singletons.append((dist, key, True))
                explanations.append((key, n, box))
            else:
                # D-41: the entry box still joins the subset pool (the
                # prefilter drops it when it cannot contribute to a containing
                # sum), it just cannot explain the edge on its own.
                dist, n, box = min(
                    ((_norm_distance(delta, b), n, b) for n, b in boxes),
                    key=lambda c: (c[0], c[1]),
                )
                explanations.append((key, n, box))

    def _step_programs(self, grid_ts: float, value: float) -> None:
        """Per-tick program stage rules (D-01..D-04 as amended by D-42).

        A running program reports the measured residual clamped to the stage
        box (D-01), never reaches 0 W without proven-off evidence (D-42): only
        its matched terminal fall ends it directly; every other end (divergence,
        abandonment, slow decay, overdue stage, max run time) enters the unknown
        state first and returns to 0 W only through the D-30a sustained
        low-remainder exit once the device has read unavailable.
        """
        if not self._programs:
            return
        prev = self._ring[-2] if len(self._ring) >= 2 else None
        dt_h = 0.0
        if prev is not None and prev[0] == self._index - 2 and math.isfinite(prev[2]):
            dt_h = (grid_ts - prev[1]) / 3600.0
        for key, model in self._programs.items():
            rt = self._runtimes.get(key)
            if rt is None or (rt.program_phase == "idle" and not rt.unknown):
                continue
            if rt.unknown:
                # D-42 proven-off exit: the P4 D-30a sustained-low-remainder
                # rule, counted from the unknown entry (clear_run resets it).
                if self._low_residual_due(key, value, model.min_step_w):
                    rt.unknown = False
                    rt.clear_run()
                continue
            measured = value - rt.base_w
            st = model.stages[rt.stage_index]
            rt.run_ticks += 1
            if rt.program_phase == "pending":
                # D-33: the pending device is OFF at 0 W; energy accrues at the
                # stage p50 so promotion can back-date it to the first edge.
                if dt_h > 0.0 and math.isfinite(rt.base_w):
                    rt.pending_wh += st.level[1] * dt_h
                if in_box(measured, st.level):
                    rt.confirm_steps += 1
                    rt.in_box_ticks += 1
                    rt.out_steps = 0
                else:
                    rt.out_steps += 1
                time_in = grid_ts - rt.stage_since_ts
                if rt.confirm_steps >= model.promote_steps:
                    self._promote_program(key, model, rt, measured, grid_ts)
                elif (
                    rt.out_steps >= PROGRAM_DIVERGE_STEPS
                    or stage_overdue(time_in, st.dur, STUCK_FACTOR)
                    or grid_ts - rt.run_start_ts > model.max_run_s
                ):
                    rt.clear_run()  # pending dropped: energy stays in the remainder (D-08)
                continue
            # running
            rt.level_w = measured
            time_in = grid_ts - rt.stage_since_ts
            if stage_overdue(time_in, st.dur, STUCK_FACTOR) or (
                grid_ts - rt.run_start_ts > model.max_run_s
            ):
                self._program_diverge(key, model, rt)
                continue
            if in_box(measured, st.level):
                rt.in_box_ticks += 1
                rt.out_steps = 0
                rt.advance_steps = 0
                rt.on_w = reported_level(measured, st.level)
                continue
            nxt = [
                n
                for n in next_stage_indices(model.optional_flags, rt.stage_index)
                if in_box(measured, model.stages[n].level)
            ]
            if nxt:
                rt.advance_steps += 1
                rt.out_steps = 0
                if rt.advance_steps >= PROGRAM_ADVANCE_STEPS:
                    self._program_advance(key, model, rt, nxt[0], grid_ts, measured)
                    continue
            else:
                rt.out_steps += 1
                rt.advance_steps = 0
                if rt.out_steps >= PROGRAM_DIVERGE_STEPS:
                    self._program_diverge(key, model, rt)
                    continue
            rt.on_w = reported_level(measured, st.level)

    # -- edge attribution --------------------------------------------------

    def _best_rise_state(
        self, model: DeviceModel, edge_index: int, delta: float
    ) -> tuple[int, tuple[float, float, float]] | None:
        """The best-fitting phase-passing rise state (subset explanation box).

        A device that passes the phase gate on no state offers no subset
        explanation at all, so a phase-gated edge can never be attributed
        through a subset either.
        """
        cands = [
            (_norm_distance(delta, st.rise), si, st.rise)
            for si, st in enumerate(model.states)
            if self._edge_phase_ok(st, edge_index)
        ]
        if not cands:
            return None
        _dist, si, box = min(cands, key=lambda c: (c[0], c[1]))
        return si, box

    def _attribute_rise(
        self,
        key: str,
        edge,
        delta: float,
        *,
        on_w: float | None = None,
        rise_delta: float | None = None,
    ) -> None:
        """Turn a switched device ON with today's singleton rise bookkeeping.

        ``on_w`` overrides the reported step (the D-36 split share for a jointly
        attributed member); ``rise_delta`` overrides the stored rise delta used
        by the fall-time verification.
        """
        model = self._models[key]
        rt = self._runtime(key)
        rt.unknown = False
        rt.on = True
        # Power reference state: nearest rise p50 over all states (D-06).  The
        # measured step is reported when it lies inside that state's box,
        # otherwise the box p50 bounds the reported power.
        ref_si = min(
            range(len(model.states)),
            key=lambda si: (abs(delta - model.states[si].rise[1]), si),
        )
        rt.state_index = ref_si
        lo, p50, hi = model.states[ref_si].rise
        rt.on_w = delta if (lo <= delta <= hi) else p50
        if on_w is not None:
            rt.on_w = float(on_w)
        pre = self._ring_at(edge.index)
        rt.pre_edge = pre[2] if pre is not None else math.nan
        rt.edge_ts = pre[1] if pre is not None else None
        rt.rise_delta = delta if rise_delta is None else float(rise_delta)
        rt.rise_index = edge.index
        rt.rise_shares = self._rise_shares(edge.index)
        rt.nan_seen = not math.isfinite(rt.pre_edge)
        rt.low_count = 0
        rt.just_on = True
        rt.plateau = []
        rt.episode_start_wh = self._energy.get(key, 0.0)
        self._seq += 1
        rt.seq = self._seq

    def _refuse_edge(self, competing: tuple[tuple[str, ...], ...]) -> None:
        """One refused edge: P4 per-key counters, D-07 overlap counter.

        Every competing containing key is counted exactly once per edge (a key
        that belongs to several competing subsets is deduplicated), and the
        overlap-refusal counter fires when two or more containing entries
        compete; a refusal caused only by a non-containing runner-up keeps
        exactly the P4 counters (D-41).
        """
        keys = sorted({key for entry in competing for key in entry})
        for key in keys:
            self._runtime(key).unattributed += 1
        if len(competing) >= 2:
            self._overlap_refused_total += 1
            for key in keys:
                self._runtime(key).overlap_refused += 1
        self._unattributed_total += 1

    def _process_rise(self, edge) -> None:
        delta = edge.delta_w
        # ONE candidate list (D-41): the existing singleton loop stays the
        # producer of the singleton entries (k = 1, no penalty), the subset
        # search adds every containing 2..3 explanation of the prefiltered pool.
        singletons: list[tuple[float, str, bool]] = []
        explanations: list[Explanation] = []
        for key, model in self._models.items():
            rt = self._runtimes.get(key)
            if rt is not None and rt.on:
                continue
            # Normalised distance for EVERY non-ON (OFF or unknown) device, and
            # the set of states whose rise box contains the delta (phase-gated).
            dist = min(
                (_norm_distance(delta, st.rise) for st in model.states),
                default=math.inf,
            )
            states = [
                si
                for si, st in enumerate(model.states)
                if st.rise[0] <= delta <= st.rise[2]
                and self._edge_phase_ok(st, edge.index)
            ]
            singletons.append((dist, key, bool(states)))
            best = self._best_rise_state(model, edge.index, delta)
            if best is not None:
                si, box = best
                explanations.append((key, si, box))

        if self._programs:
            self._program_rise_candidates(delta, singletons, explanations)

        pool = prefilter(delta, explanations, JOINT_PREFILTER)
        by_key = {exp[0]: exp for exp in explanations}
        subsets = best_subsets(delta, pool)
        action, payload = pick(singletons, subsets, AMBIGUITY_MARGIN)
        if action == "none":
            self._unattributed_total += 1
            return
        if action == "refuse":
            self._refuse_edge(payload)
            return

        keys = payload[0]
        if len(keys) == 1:
            target_key = keys[0]
            if target_key in self._programs:
                self._apply_program_rise(
                    target_key, edge, delta, stage=by_key[target_key][1]
                )
                return
            self._attribute_rise(target_key, edge, delta)
            return

        # A uniquely explained edge is split across its subset (D-36): each
        # member reports its p50-proportional share, clamped to its own box.
        members = [by_key[key] for key in keys]
        shares = dict(split_delta(delta, members))
        for key in keys:
            self._runtime(key).joint_attributed += 1
            if key in self._programs:
                self._apply_program_rise(key, edge, delta, stage=by_key[key][1])
            else:
                self._attribute_rise(key, edge, delta, on_w=shares[key])
        self._joint_attributed_total += 1

    def _rise_shares(self, edge_index: int) -> tuple[float, float, float] | None:
        pre = self._ring_at(edge_index)
        post = self._ring_at(edge_index + 1)
        if pre is None or post is None or pre[3] is None or post[3] is None:
            return None
        tot = 0.0
        comps = []
        for k in range(3):
            d = abs(float(post[3][k]) - float(pre[3][k]))
            comps.append(d)
            tot += d
        if tot <= 0.0:
            return None
        return (comps[0] / tot, comps[1] / tot, comps[2] / tot)

    def _process_fall(self, edge) -> None:
        delta = edge.delta_w
        fall_abs = abs(delta)
        singletons: list[tuple[float, str, bool]] = []
        explanations: list[Explanation] = []
        unknown_candidates: list[str] = []
        for key, model in self._models.items():
            rt = self._runtimes.get(key)
            if rt is None:
                continue
            if rt.on:
                # D-41: every ON device offers its own fall box to the subset
                # search (minus on_w widened by the pairing tolerance), while
                # the singleton entry keeps today's fall-box + tolerance gate.
                tol = max(PAIR_TOL * rt.on_w, PAIR_ABS_W)
                explanations.append(
                    (key, rt.state_index, (-(rt.on_w + tol), -rt.on_w, -(rt.on_w - tol)))
                )
                st = model.states[rt.state_index]
                if not _fall_box_match(st.fall, delta):
                    continue
                if abs(fall_abs - rt.on_w) > tol:
                    continue
                score = abs(fall_abs - rt.on_w) / rt.on_w if rt.on_w > 0 else math.inf
                singletons.append((score, key, True))
            elif rt.unknown:
                if any(_fall_box_match(st.fall, delta) for st in model.states):
                    unknown_candidates.append(key)

        for key, model in self._programs.items():
            rt = self._runtimes.get(key)
            if rt is None:
                continue
            if rt.program_phase == "running" and rt.on:
                # D-42 proven-off evidence: the matched terminal fall, valid
                # only in the last required stage or a later optional stage
                if rt.stage_index >= model.last_required_index:
                    # D-41: the end box joins the subset pool even when it does
                    # not contain the whole edge (a program ending together with
                    # a switched load is one edge); the singleton entry keeps
                    # today's matched-terminal-fall gate.
                    explanations.append((key, rt.stage_index, model.terminal_fall))
                    if _fall_box_match(model.terminal_fall, delta):
                        level_p50 = model.stages[rt.stage_index].level[1]
                        denom = max(level_p50, 1.0)
                        score = abs(fall_abs - level_p50) / denom
                        singletons.append((score, key, True))
            elif rt.unknown and not rt.on:
                # a matched terminal fall in the last required stage or a later
                # optional stage ends the unknown state (back to 0 W)
                if rt.stage_index >= model.last_required_index and _fall_box_match(
                    model.terminal_fall, delta
                ):
                    unknown_candidates.append(key)

        pool = prefilter(delta, explanations, JOINT_PREFILTER)
        subsets = best_subsets(delta, pool, relative=True)
        candidates_n = len(singletons) + len(unknown_candidates) + len(subsets)
        if candidates_n == 0:
            self._unattributed_total += 1
            return

        if unknown_candidates:
            # D-07 literal: a fall with an unknown device among >= 2 candidates
            # (a containing subset counts as a candidate, D-41) is refused; a
            # sole unknown candidate ends the unknown state.
            if candidates_n >= 2:
                keys = sorted(
                    {key for _s, key, _c in singletons}
                    | set(unknown_candidates)
                    | {key for _s, members in subsets for key in members}
                )
                for key in keys:
                    self._runtime(key).unattributed += 1
                self._unattributed_total += 1
                return
            rt = self._runtime(unknown_candidates[0])
            rt.unknown = False
            rt.clear_run()
            return

        # Falls mirror the rise rule with FALL_MARGIN, but keep the Phase 3/P4
        # singletons margin rule (an edge no subset ties decides exactly as
        # before, D-41 must-have), so the strict two-containing rule is off.
        action, payload = pick(singletons, subsets, FALL_MARGIN, strict_singletons=False)
        if action == "none":
            self._unattributed_total += 1
            return
        if action == "refuse":
            self._refuse_edge(payload)
            return

        keys = payload[0]
        if len(keys) == 1:
            best_key = keys[0]
            if best_key in self._programs:
                self._program_end(
                    best_key, self._programs[best_key], self._runtime(best_key), edge
                )
                return
            self._finish_on_episode(best_key, edge)
            return

        # Joint fall (D-36/D-41): every member turns OFF with its own share of
        # the edge used for its fall-time verification.
        by_key = {exp[0]: exp for exp in explanations}
        members = [by_key[key] for key in keys]
        shares = dict(split_delta(delta, members))
        for key in keys:
            self._runtime(key).joint_attributed += 1
            if key in self._programs:
                self._program_end(key, self._programs[key], self._runtime(key), edge)
            else:
                self._finish_on_episode(key, edge, fall_delta=shares[key])
        self._joint_attributed_total += 1

    def _finish_on_episode(self, key: str, edge, *, fall_delta: float | None = None) -> None:
        """Attributed fall: verify against the full box rule, then turn OFF.

        ``fall_delta`` is the D-36 split share of a jointly attributed fall;
        without it the whole edge delta is verified, exactly as before.
        """
        rt = self._runtime(key)
        model = self._models[key]
        verified = self._verify_episode(rt, model, edge, fall_delta=fall_delta)
        if verified:
            energy_now = self._energy.get(key, 0.0)
            ep = self._make_episode(rt, edge, energy_now - rt.episode_start_wh)
            self._episode_log.setdefault(key, []).append(ep)
        else:
            rt.unverified += 1
        if rt.state_index >= 0:
            # Plan 05-05 (D-15 live proxy): record every attributed close with
            # a known state, verified or not, with the same rise-box score
            # StepResult.match_score uses (via the shared _match_score
            # helper). A close without a known state (e.g. a runtime restored
            # mid-episode without one) records nothing: no fallback score can
            # enter a drift window, and its refused edges still reach the next
            # recorded close through the cumulative counter.
            self._closed_log.append(
                ClosedEpisode(
                    key=key,
                    end_ts=float(self._last_grid_ts or 0.0),
                    verified=bool(verified),
                    match_score=_match_score(
                        model.states[rt.state_index].rise, rt.rise_delta
                    ),
                    unattributed=int(rt.unattributed),
                )
            )
        rt.clear_run()

    def _make_episode(self, rt: _Runtime, edge, energy_wh: float):
        from .replay import OnlineEpisode

        return OnlineEpisode(
            start_i=int(rt.rise_index),
            end_i=int(edge.index),
            step_w=float(rt.on_w),
            energy_wh=float(energy_wh),
        )

    def _verify_episode(
        self,
        rt: _Runtime,
        model: DeviceModel,
        edge,
        *,
        fall_delta: float | None = None,
    ) -> bool:
        """Full score.replay switched rule over every state (first match wins).

        ``fall_delta`` is the D-36 share of a jointly attributed fall edge, so a
        jointly closed episode is verified against the member's own part of the
        edge rather than the summed delta no single box explains.
        """
        delta = edge.delta_w if fall_delta is None else float(fall_delta)
        if rt.nan_seen or rt.rise_index is None:
            return False
        dur = (edge.index - rt.rise_index) * self._grid_s
        if dur <= 0 or dur > MAX_PAIR_S:
            return False
        post = rt.plateau
        if not post:
            plateau_w = (rt.rise_delta + abs(delta)) / 2.0
        else:
            keep = edge.index - rt.rise_index
            if len(post) > keep:
                post = post[:keep]
            plateau_w = statistics.median(post) - rt.pre_edge
        for st in model.states:
            lo, p50, hi = st.rise
            if not (lo <= rt.rise_delta <= hi):
                continue
            if not _fall_box_match(st.fall, delta):
                continue
            if not (st.dur[0] <= dur <= st.dur[1]):
                continue
            if not (st.plateau[0] <= plateau_w <= st.plateau[2]):
                continue
            if st.phase_share is not None and rt.rise_shares is not None:
                shares = {"l1": rt.rise_shares[0], "l2": rt.rise_shares[1], "l3": rt.rise_shares[2]}
                ok = True
                for ph, plo, phi in st.phase_share:
                    if not (plo <= shares[ph] <= phi):
                        ok = False
                        break
                if not ok:
                    continue
            return True
        return False

    def _process_edge(self, edge) -> None:
        if edge.delta_w > 0:
            self._process_rise(edge)
        elif edge.delta_w < 0:
            self._process_fall(edge)

    # -- public API ---------------------------------------------------------

    @property
    def models(self) -> tuple[DeviceModel | ProgramModel, ...]:
        """Configured device models (switched and program), sorted by key."""
        return tuple(
            sorted(
                (*self._models.values(), *self._programs.values()),
                key=lambda m: m.key,
            )
        )

    @property
    def unknown(self) -> frozenset[str]:
        """Current D-28/D-29/D-30/D-31 unknown set, parked keys included."""
        return frozenset(k for k, rt in self._runtimes.items() if rt.unknown)

    @property
    def energy_wh(self) -> Mapping[str, float]:
        """Committed per-device energy (Wh), monotonic."""
        return dict(self._energy)

    def drain_episodes(self) -> dict[str, list]:
        """Return verified episodes accumulated since the previous drain."""
        out = self._episode_log
        self._episode_log = {}
        return out

    def drain_closed(self) -> list[ClosedEpisode]:
        """Return closed-episode evidence since the previous drain (D-15).

        DeviceRuntime drains this every detector batch in live use so the log
        stays empty between batches; replay_online never calls it, so replay
        semantics are unchanged.
        """
        out = list(self._closed_log)
        self._closed_log.clear()
        return out

    def step(self, grid_ts: float, value: float, phases: tuple | None = None) -> StepResult:
        """Advance the detector one grid sample."""
        grid_ts = float(grid_ts)
        value = float(value)
        if self._last_grid_ts is not None and grid_ts <= self._last_grid_ts:
            return self._report(grid_ts, value, energy=False)
        self._last_grid_ts = grid_ts

        if math.isnan(value):
            return self._step_nan(grid_ts, value, phases)
        return self._step_finite(grid_ts, value, phases)

    def _step_nan(self, grid_ts: float, value: float, phases: tuple | None) -> StepResult:
        edges = self._edges.push(value)
        for edge in edges:
            self._process_edge(edge)
        self._ring.append((self._index, grid_ts, value, phases))
        self._index += 1
        self._gap_nan += 1
        if self._gap_nan > GAP_HOLD_STEPS:
            for key, rt in list(self._runtimes.items()):
                if rt.on:
                    self._enter_unknown(key)
            self._post_gap_armed = True
        for key, rt in self._runtimes.items():
            if rt.on:
                rt.nan_seen = True  # the ON period spans a blind sample
        return self._report(grid_ts, math.nan, energy=False)

    def _step_finite(self, grid_ts: float, value: float, phases: tuple | None) -> StepResult:
        edges = self._edges.push(value)
        for edge in edges:
            self._process_edge(edge)
        self._ring.append((self._index, grid_ts, value, phases))
        self._index += 1
        self._gap_nan = 0

        # stuck-ON duration guard (evidence-free: enters unknown, D-29a)
        for key, model in self._models.items():
            rt = self._runtimes.get(key)
            if rt is not None and rt.on and rt.edge_ts is not None:
                if grid_ts - rt.edge_ts >= model.max_on_s:
                    self._enter_unknown(key)

        # program stage rules (divergence, advance, promotion, D-30a exit)
        if self._programs:
            self._step_programs(grid_ts, value)

        if self._post_gap_armed:
            self._post_gap_check(value)

        # low-residual rules (D-29b for ON, D-30a for unknown), one predicate
        for key, model in self._models.items():
            rt = self._runtimes.get(key)
            if rt is None:
                continue
            if rt.on:
                if self._low_residual_due(key, value, rt.on_w):
                    rt.clear_run()  # evidence the load is gone: plain OFF at 0 W
                    self._forced_off += 1
            elif rt.unknown:
                if self._low_residual_due(key, value, model.min_step_w):
                    rt.unknown = False
                    rt.clear_run()

        return self._report(grid_ts, value, energy=True)

    def _report(self, grid_ts: float, value: float, *, energy: bool) -> StepResult:
        """Build the StepResult for the current state (clamp, energy, scores)."""
        all_keys = sorted(set(self._runtimes) | set(self._models) | set(self._programs))
        device_w: dict[str, float] = {}
        on_keys: list[str] = []
        for key in all_keys:
            rt = self._runtimes.get(key)
            if rt is not None and rt.on:
                on_keys.append(key)
                device_w[key] = rt.on_w
            else:
                device_w[key] = 0.0

        # D-06 clamp: the sum never exceeds max(0, residual); the excess is
        # clipped from the most recently attributed device first, so the
        # oldest devices keep their power and the newest lose theirs.
        budget = max(0.0, value) if not math.isnan(value) else math.inf
        for key in sorted(on_keys, key=lambda k: self._runtimes[k].seq):
            p = device_w[key]
            if p > budget:
                device_w[key] = max(0.0, budget)
                budget = 0.0
            else:
                budget -= p

        match_score: dict[str, float] = {}
        for key in on_keys:
            rt = self._runtimes[key]
            model = self._models.get(key)
            if model is None or rt.state_index < 0:
                continue
            match_score[key] = _match_score(
                model.states[rt.state_index].rise, rt.rise_delta
            )

        energy_delta = {key: 0.0 for key in all_keys}

        # plateau accumulation for the fall-time check (switched only)
        for key in on_keys:
            if key in self._programs:
                continue  # programs need no plateau buffer (T-06-02)
            rt = self._runtimes[key]
            model = self._models.get(key)
            cap = model.plateau_cap if model is not None else 64
            cur = self._ring[-1]
            if rt.just_on and len(self._ring) >= 2:
                for entry in (self._ring[-2], self._ring[-1]):
                    if math.isfinite(entry[2]) and len(rt.plateau) < cap:
                        rt.plateau.append(entry[2])
            elif math.isfinite(cur[2]) and len(rt.plateau) < cap:
                rt.plateau.append(cur[2])

        if energy:
            prev = self._ring[-2] if len(self._ring) >= 2 else None
            if prev is not None and prev[0] == self._index - 2 and math.isfinite(prev[2]):
                dt_h = (grid_ts - prev[1]) / 3600.0
                for key in on_keys:
                    rt = self._runtimes[key]
                    if rt.just_on:
                        intervals = 2  # back-date to the pre-edge sample (D-09)
                        rt.just_on = False
                    else:
                        intervals = 1
                    if not math.isfinite(rt.pre_edge):
                        continue
                    if value - rt.pre_edge < 0.5 * rt.on_w:
                        continue  # fall seen with lag: no energy past last high sample
                    add = device_w[key] * dt_h * intervals
                    if add > 0.0:
                        self._energy[key] = self._energy.get(key, 0.0) + add
                        energy_delta[key] = add
            for key in on_keys:
                rt = self._runtimes[key]
                if rt.committed_wh > 0.0:
                    # promotion committed the pending energy back-dated to the
                    # first matched edge (D-02); report it on this tick
                    energy_delta[key] += rt.committed_wh
                    rt.committed_wh = 0.0
        else:
            for key in on_keys:
                self._runtimes[key].just_on = False

        return StepResult(
            grid_ts=grid_ts,
            device_w=device_w,
            on=frozenset(on_keys),
            unknown=frozenset(k for k, rt in self._runtimes.items() if rt.unknown),
            match_score=match_score,
            energy_wh_delta=energy_delta,
            unattributed_edges={
                k: (self._runtimes[k].unattributed if k in self._runtimes else 0)
                for k in all_keys
            },
            unverified_episodes={
                k: (self._runtimes[k].unverified if k in self._runtimes else 0)
                for k in all_keys
            },
            unattributed_total=self._unattributed_total,
            forced_off=self._forced_off,
            estimated_w=sum(device_w.values()),
            overlap_refused=self._overlap_refused_total,
            joint_attributed=self._joint_attributed_total,
        )

    def flush(self) -> None:
        """Finalise trailing stream edges (end-of-segment decision)."""
        for edge in self._edges.flush():
            self._process_edge(edge)

    def set_models(self, models) -> None:
        """Replace the model set, parking flags of dropped ON/unknown keys."""
        new_models = {m.key: m for m in models}
        for key in list(self._models):
            if key in new_models:
                continue
            rt = self._runtimes.get(key)
            if rt is not None and (rt.on or rt.unknown):
                self._enter_unknown(key)  # parked unknown flag (D-29a)
            else:
                self._runtimes.pop(key, None)
                self._energy.pop(key, None)
        for key in list(self._programs):
            if key in new_models:
                continue
            rt = self._runtimes.get(key)
            if rt is not None and (rt.on or rt.unknown):
                self._enter_unknown(key)  # parked unknown flag (D-29a)
            else:
                self._runtimes.pop(key, None)
                self._energy.pop(key, None)
        self._models = {
            k: m for k, m in sorted(new_models.items()) if not isinstance(m, ProgramModel)
        }
        self._programs = {
            k: m for k, m in sorted(new_models.items()) if isinstance(m, ProgramModel)
        }

    def state(self) -> dict:
        """JSON-safe detector state (D-31 armed flag included).

        Program keys carry a ``program`` sub-dict only while a run is RUNNING;
        a pending run is never persisted (it restores idle and its pending
        energy stays in the remainder, D-08).
        """
        devices = {}
        for key, rt in (
            (k, self._runtimes.get(k))
            for k in sorted(set(self._runtimes) | set(self._models) | set(self._programs))
        ):
            if rt is not None:
                d: dict[str, object] = {
                    "on": rt.on,
                    "unknown": rt.unknown,
                    "state_index": rt.state_index,
                    "on_w": rt.on_w,
                    "edge_ts": rt.edge_ts,
                    "attributed_seq": rt.seq,
                    "unattributed": rt.unattributed,
                    "unverified": rt.unverified,
                }
                if key in self._programs and rt.program_phase == "running":
                    d["program"] = {
                        "phase": "running",
                        "stage_index": rt.stage_index,
                        "stage_since_ts": rt.stage_since_ts,
                        "base_w": rt.base_w,
                        "run_start_ts": rt.run_start_ts,
                    }
                devices[key] = d
            else:
                devices[key] = {
                    "on": False,
                    "unknown": False,
                    "state_index": -1,
                    "on_w": 0.0,
                    "edge_ts": None,
                    "attributed_seq": 0,
                    "unattributed": 0,
                    "unverified": 0,
                }
        return {
            "grid_s": self._grid_s,
            "last_grid_ts": self._last_grid_ts,
            "unattributed_total": self._unattributed_total,
            "forced_off": self._forced_off,
            "post_gap_armed": self._post_gap_armed,
            "attributed_seq_max": self._seq,
            # Plan 06-05 (D-07/D-36): the overlap counters survive a restart;
            # only keys with a refusal are listed, so the state stays compact
            # and a missing key restores as 0.
            "overlap_refused_total": self._overlap_refused_total,
            "joint_attributed_total": self._joint_attributed_total,
            "overlap_refused_by_device": {
                key: rt.overlap_refused
                for key, rt in sorted(self._runtimes.items())
                if rt.overlap_refused
            },
            "devices": devices,
            "energy_wh": {
                key: self._energy.get(key, 0.0)
                for key in sorted(set(self._runtimes) | set(self._models) | set(self._programs))
            },
        }

    def _restore_program(self, key: str, rt: _Runtime, d: Mapping, on: bool) -> None:
        """Restore a program key: running at its stored stage, pending as idle."""
        model = self._programs[key]
        info = d.get("program")
        rt.unknown = bool(d.get("unknown", False))
        if not on or not isinstance(info, Mapping) or info.get("phase") != "running":
            # pending runs are never persisted: they restore idle, so the
            # not-yet-committed pending energy stays in the remainder (D-08)
            rt.clear_run()
            rt.unknown = bool(d.get("unknown", False))
            return
        rt.program_phase = "running"
        si = info.get("stage_index", 0)
        rt.stage_index = min(int(si) if si is not None else 0, len(model.stages) - 1)
        rt.stage_since_ts = float(info.get("stage_since_ts", 0.0) or 0.0)
        rt.run_start_ts = float(info.get("run_start_ts", 0.0) or 0.0)
        rt.run_start_index = None  # episode buffers are not persisted
        rt.base_w = float(info.get("base_w", 0.0) or 0.0)
        rt.on = True
        rt.on_w = model.stages[rt.stage_index].level[1]
        rt.level_w = rt.on_w
        rt.pre_edge = rt.base_w
        rt.edge_ts = rt.stage_since_ts
        rt.rise_delta = rt.on_w
        rt.rise_index = None
        rt.rise_shares = None
        rt.nan_seen = False
        rt.just_on = False
        rt.episode_start_wh = self._energy.get(key, 0.0)

    def restore(self, data: Mapping | None, now_ts: float) -> None:
        """Restore covered keys only (D-30b); a changed grid is a rebuild (D-26)."""
        covered = state_keys(data)
        if not covered:
            return
        assert data is not None
        # restore global counters first: the device loop below may increment
        # them (a grid rebuild counts a forced OFF per stored ON device)
        self._unattributed_total = int(data.get("unattributed_total", 0) or 0)
        self._forced_off = int(data.get("forced_off", 0) or 0)
        self._overlap_refused_total = int(data.get("overlap_refused_total", 0) or 0)
        self._joint_attributed_total = int(data.get("joint_attributed_total", 0) or 0)
        by_device = data.get("overlap_refused_by_device")
        if not isinstance(by_device, Mapping):
            by_device = {}
        grid_match = data.get("grid_s") == self._grid_s
        last_grid_ts = data.get("last_grid_ts")
        gap = last_grid_ts is not None and (
            float(now_ts) - float(last_grid_ts) > GAP_HOLD_STEPS * self._grid_s
        )
        devices = data.get("devices", {})
        for key in covered:
            d = devices.get(key)
            if not isinstance(d, Mapping):
                continue
            rt = self._runtime(key)
            rt.unattributed = int(d.get("unattributed", 0) or 0)
            rt.unverified = int(d.get("unverified", 0) or 0)
            rt.overlap_refused = int(by_device.get(key, 0) or 0)
            rt.seq = int(d.get("attributed_seq", 0) or 0)
            rt.low_count = 0
            on = bool(d.get("on", False))
            if on and (gap or not grid_match):
                rt.on = True  # the stored ON period ends here: count it (D-29a)
                self._enter_unknown(key)
                continue
            if key in self._programs:
                self._restore_program(key, rt, d, on)
                continue
            rt.unknown = bool(d.get("unknown", False))
            rt.on = on
            rt.on_w = float(d.get("on_w", 0.0) or 0.0)
            rt.edge_ts = d.get("edge_ts")
            # Do NOT use `or -1` here: a stored state_index of 0 is falsy but
            # valid (state 0 of the model); `or` would map it to -1 and make
            # _process_fall verify the fall against the LAST state (WR-02).
            _si = d.get("state_index", -1)
            rt.state_index = -1 if _si is None else int(_si)
            if on:
                # episode buffers are not persisted: the episode counts as
                # unverified at its fall
                rt.rise_delta = rt.on_w
                rt.rise_index = None
                rt.pre_edge = math.nan
                rt.rise_shares = None
                rt.nan_seen = False
                rt.just_on = False
                rt.episode_start_wh = self._energy.get(key, 0.0)
            else:
                rt.clear_run()
                rt.unknown = bool(d.get("unknown", False))
        energy = data.get("energy_wh", {})
        if isinstance(energy, Mapping):
            for key in covered:
                v = energy.get(key)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    self._energy[key] = max(self._energy.get(key, 0.0), float(v))
        stored_armed = bool(data.get("post_gap_armed", False))
        self._post_gap_armed = stored_armed or gap
        # Restore the global attribution counter so the D-06 clamp keeps
        # oldest-first ordering across a restart: without this, the counter
        # restarts at 0 and the next attribution sorts as "oldest" ahead of a
        # device that stayed ON through the restart (WR-03).
        self._seq = max(
            self._seq,
            int(data.get("attributed_seq_max", 0) or 0),
            max((rt.seq for rt in self._runtimes.values()), default=0),
        )
        if last_grid_ts is not None:
            ts = float(last_grid_ts)
            if self._last_grid_ts is None or ts > self._last_grid_ts:
                self._last_grid_ts = ts