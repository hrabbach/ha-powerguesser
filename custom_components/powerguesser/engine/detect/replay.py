"""Drive OnlineDetector over a gridded residual array (numpy batch replay).

Used for parity checks against score.replay, constant tuning, the D-16
validation value and Plan 04-07/04-12's worker stages.  Keys are
signature_ids; only verified episodes are reported (DET-01).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from ..learning.episodes import MIN_EDGE_W
from .online import OnlineDetector, compile_model


@dataclass(frozen=True, slots=True)
class OnlineEpisode:
    """One verified switched episode reported by the online detector."""

    start_i: int
    end_i: int
    step_w: float
    energy_wh: float


@dataclass(frozen=True, slots=True)
class OnlineReplay:
    """Aggregated replay result: verified episodes and committed energy.

    power_w and energy_step_wh carry per-step device power (W) and committed
    energy (Wh) per signature id, collected only when replay_online runs with
    collect_power=True (Plan 05-02's holdout figures); they are empty otherwise
    so the default path stays allocation-free (replay parity, T-05-23).

    overlap_refused and joint_attributed are the D-41 totals (refused ambiguous
    edges and jointly attributed edges); their per-step cumulative arrays are
    collected under the same collect_power flag and stay None otherwise.
    """

    episodes: dict[str, list[OnlineEpisode]] = field(default_factory=dict)
    energy_wh: dict[str, float] = field(default_factory=dict)
    unattributed_total: int = 0
    forced_off: int = 0
    unverified: int = 0
    power_w: dict[str, np.ndarray] = field(default_factory=dict)
    energy_step_wh: dict[str, np.ndarray] = field(default_factory=dict)
    # Plan 06-05 (D-07/D-36/D-41): the overlap counters. The totals always
    # carry the detector's final values; the per-step cumulative arrays are
    # filled only with collect_power=True, so the default path stays
    # allocation-free (replay parity, T-05-23).
    overlap_refused: int = 0
    joint_attributed: int = 0
    overlap_refused_cum: np.ndarray | None = None
    joint_attributed_cum: np.ndarray | None = None


def replay_online(
    artefacts: Sequence[Mapping],
    r: np.ndarray,
    step_s: float,
    *,
    phases: Mapping[str, np.ndarray] | None = None,
    threshold_w: float | None = None,
    suppress_alignment: bool = True,
    collect_power: bool = False,
) -> OnlineReplay:
    """Replay artefacts over a gridded residual array through OnlineDetector.

    With collect_power=True, additionally record the per-step device power and
    committed energy the detector reports for every model, so holdout metrics
    can be restricted to exactly the samples both sides cover (A6, RESEARCH
    Pattern 3). With the default False nothing else changes.
    """
    models = [
        compile_model(str(a.get("signature_id", f"sig{k}")), a)
        for k, a in enumerate(artefacts)
    ]
    if threshold_w is None:
        thresholds = [m.threshold_w for m in models if m.threshold_w > 0.0]
        threshold_w = max(MIN_EDGE_W, min(thresholds) if thresholds else MIN_EDGE_W)
    grid_s = int(round(float(step_s)))
    detector = OnlineDetector(
        models,
        grid_s=grid_s,
        threshold_w=float(threshold_w),
        suppress_alignment=suppress_alignment,
    )

    arr = np.asarray(r, dtype=np.float64)
    n = len(arr)
    phase_arr: Mapping[str, np.ndarray] | None = None
    if phases and all(k in phases for k in ("l1", "l2", "l3")):
        phase_arr = phases

    # T-05-23: two float64 arrays per model only inside the niced validate
    # stage that opts in; live detection and replay parity never collect.
    keys = [str(m.key) for m in models]
    power_w = {k: np.zeros(n, dtype=np.float64) for k in keys} if collect_power else {}
    energy_step = {k: np.zeros(n, dtype=np.float64) for k in keys} if collect_power else {}
    overlap_cum = np.zeros(n, dtype=np.int64) if collect_power else None
    joint_cum = np.zeros(n, dtype=np.int64) if collect_power else None

    for i in range(n):
        ph = None
        if phase_arr is not None:
            ph = (
                float(phase_arr["l1"][i]),
                float(phase_arr["l2"][i]),
                float(phase_arr["l3"][i]),
            )
        step_result = detector.step(float(i) * float(step_s), float(arr[i]), ph)
        if collect_power:
            for k in keys:
                power_w[k][i] = float(step_result.device_w.get(k, 0.0))
                energy_step[k][i] = float(step_result.energy_wh_delta.get(k, 0.0))
            assert overlap_cum is not None and joint_cum is not None
            overlap_cum[i] = int(step_result.overlap_refused)
            joint_cum[i] = int(step_result.joint_attributed)
    detector.flush()

    episodes = {k: list(v) for k, v in detector.drain_episodes().items()}
    final_state = detector.state()
    unverified = 0
    for dev in final_state.get("devices", {}).values():
        if isinstance(dev, Mapping):
            unverified += int(dev.get("unverified", 0) or 0)
    energy = {k: float(v) for k, v in detector.energy_wh.items()}
    return OnlineReplay(
        episodes=episodes,
        energy_wh=energy,
        unattributed_total=int(final_state.get("unattributed_total", 0) or 0),
        forced_off=int(final_state.get("forced_off", 0) or 0),
        unverified=unverified,
        power_w=power_w,
        energy_step_wh=energy_step,
        overlap_refused=int(final_state.get("overlap_refused_total", 0) or 0),
        joint_attributed=int(final_state.get("joint_attributed_total", 0) or 0),
        overlap_refused_cum=overlap_cum,
        joint_attributed_cum=joint_cum,
    )


def validation_error_pct(
    artefact: Mapping,
    r: np.ndarray,
    step_s: float,
    reference_wh: float,
    *,
    phases: Mapping[str, np.ndarray] | None = None,
    suppress_alignment: bool = True,
) -> float | None:
    """D-16 learning-split energy error (%) of one artefact against the offline claims.

    E_online is the sum of OnlineEpisode.energy_wh over the detector's verified
    episodes for this artefact alone (each passed score.replay's switched match
    rule at its fall, Plan 04-01) — never OnlineReplay.energy_wh, which also
    holds committed-but-unverified and forced-OFF energy the offline side never
    claims. reference_wh is the summed offline energy of the validation episodes
    score.replay claims for the artefact, so both sides cover the same episode
    population and an unclaimed episode (e.g. an oven preheat the extraction does
    not claim) inflates neither sum (D-16).

    Returns round(100 x |E_online - reference_wh| / reference_wh, 1), or None
    when the reference energy is 0 (nothing claimed) or the artefact is not a
    switched load (preview candidates and program candidates carry no figure).
    """
    if reference_wh <= 0.0 or artefact.get("kind") != "switched":
        return None
    result = replay_online(
        [artefact],
        r,
        step_s,
        phases=phases,
        suppress_alignment=suppress_alignment,
    )
    sig = str(artefact.get("signature_id", ""))
    e_online = sum(ep.energy_wh for ep in result.episodes.get(sig, []))
    return round(100.0 * abs(e_online - reference_wh) / reference_wh, 1)
