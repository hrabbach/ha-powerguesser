"""Synthetic multi-stage program appliances: generators and program artefact dicts.

Extends tests/fixtures/detect_traces.py with piecewise-constant program runs
(dishwasher, washer, dryer, hob session) used by the Plan 06-01 stage-fit and
program-tracker tests.
"""

from collections.abc import Mapping, Sequence

import numpy as np

GRID_S = 30


def dishwasher_stages() -> list[tuple[float, int]]:
    """Two heat bursts, wash motor, drying (drying stays above the edge threshold)."""
    return [(2000.0, 20), (150.0, 30), (2000.0, 16), (200.0, 20)]


def washer_stages() -> list[tuple[float, int]]:
    return [(2000.0, 30), (300.0, 80), (500.0, 20)]


def dryer_stages() -> list[tuple[float, int]]:
    return [(2200.0, 60), (900.0, 60)]


def hob_session_stages() -> list[tuple[float, int]]:
    return [(1800.0, 10), (900.0, 20), (1800.0, 8)]


def program_run(
    r: np.ndarray,
    start_i: int,
    stages: Sequence[tuple[float, int]],
    *,
    rng: np.random.Generator | None = None,
    level_jitter: float = 0.0,
    dur_jitter: float = 0.0,
    skip: set[int] | frozenset[int] = frozenset(),
    glitch: bool = False,
) -> float:
    """Add a piecewise-constant program run; return the run's true energy (Wh).

    ``start_i`` is the pre-edge sample (add_run style: the run occupies
    samples ``start_i + 1 .. start_i + total``).  ``skip`` drops stage indices
    for this run (an optional stage some members lack); ``glitch`` inserts a
    one-sample +400 W spike inside the second kept stage.
    """
    levels, counts = program_rle(
        stages,
        rng=rng,
        level_jitter=level_jitter,
        dur_jitter=dur_jitter,
        skip=skip,
        glitch=glitch,
    )
    n = int(np.sum(counts))
    r[start_i + 1 : start_i + 1 + n] += np.repeat(levels, counts)
    return float(np.sum(levels * counts)) * GRID_S / 3600.0


def program_rle(
    stages: Sequence[tuple[float, int]],
    *,
    rng: np.random.Generator | None = None,
    level_jitter: float = 0.0,
    dur_jitter: float = 0.0,
    skip: set[int] | frozenset[int] = frozenset(),
    glitch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (levels, counts) RLE of one program run with optional jitter."""
    levels: list[float] = []
    counts: list[int] = []
    kept = 0
    for k, (w, n) in enumerate(stages):
        if k in skip:
            continue
        lw = float(w)
        ln = int(n)
        if rng is not None and level_jitter:
            lw = float(w) * (1.0 + rng.normal(0.0, level_jitter))
        if rng is not None and dur_jitter:
            ln = max(1, int(round(n * (1.0 + rng.normal(0.0, dur_jitter)))))
        levels.append(lw)
        counts.append(ln)
        kept += 1
        if glitch and kept == 2 and ln >= 2:
            levels[-1] = lw + 400.0
            counts[-1] = 1
            levels.append(lw)
            counts.append(ln - 1)
    return np.asarray(levels, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def kettle_overlay(r: np.ndarray, start_i: int, n_samples: int = 6, w: float = 2000.0) -> None:
    """Overlay a short kettle boil on top of whatever else is running."""
    r[start_i + 1 : start_i + 1 + n_samples] += w


def program_artefact(
    sig_id: str,
    stages: Sequence[Mapping],
    *,
    grid_s: int = GRID_S,
    threshold_w: float = 120.0,
    optional: Sequence[int] = (),
    session_energy_wh: Sequence[float] = (400.0, 700.0, 900.0),
    session_dur_s: Sequence[float] = (1800.0, 2580.0, 3200.0),
) -> dict:
    """Build a SignatureArtefact v1 program dict with a hand-written stage template.

    ``stages`` is a list of (level_w, n_samples) pairs; boxes are widened by
    ``max(threshold_w / 2, 0.05 x |p50|)`` with duration boxes of +-20 %.
    """
    out_stages = []
    prev_p50 = 0.0
    for k, st in enumerate(stages):
        w = float(st[0])
        n = int(st[1])
        widen = max(threshold_w / 2.0, 0.05 * abs(w))
        dur_lo = max(float(grid_s), 0.8 * n * grid_s)
        dur_hi = 1.2 * n * grid_s
        entry = w - prev_p50
        entry_widen = max(threshold_w / 2.0, 0.05 * abs(entry))
        out_stages.append(
            {
                "name": f"s{k + 1}",
                "level_w": {
                    "lo": round(w - widen, 1),
                    "p50": round(w, 1),
                    "hi": round(w + widen, 1),
                },
                "entry_w": {
                    "lo": round(entry - entry_widen, 1),
                    "p50": round(entry, 1),
                    "hi": round(entry + entry_widen, 1),
                },
                "dur_s": {
                    "lo": round(dur_lo, 1),
                    "p50": round(n * grid_s, 1),
                    "hi": round(dur_hi, 1),
                },
                "optional": k in optional,
            }
        )
        prev_p50 = w
    last_required = max(
        (k for k, st in enumerate(out_stages) if not st["optional"]),
        default=len(out_stages) - 1,
    )
    last = out_stages[last_required]["level_w"]
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "program",
        "label": sig_id,
        "category": "program",
        "extraction": {"grid_s": int(grid_s), "edge_threshold_w": float(threshold_w)},
        "states": [
            {
                "name": "session",
                "rise_w": {"lo": -threshold_w / 2, "p50": 0.0, "hi": threshold_w / 2},
                "fall_w": {"lo": -threshold_w / 2, "p50": 0.0, "hi": threshold_w / 2},
                "dur_s": {"lo": 1.0, "p50": 1.0, "hi": float(grid_s)},
                "plateau_w": {"lo": 0.0, "p50": 0.0, "hi": 0.0},
                "phase_share": None,
            }
        ],
        "session": {
            "energy_wh": {
                "lo": float(session_energy_wh[0]),
                "p50": float(session_energy_wh[1]),
                "hi": float(session_energy_wh[2]),
            },
            "dur_s": {
                "lo": float(session_dur_s[0]),
                "p50": float(session_dur_s[1]),
                "hi": float(session_dur_s[2]),
            },
            "n_segments": {"lo": 1.0, "p50": 1.0, "hi": 60.0},
            "exemplars": [],
            "stages": out_stages,
            "terminal": {
                "fall_w": {
                    "lo": round(-last["hi"], 1),
                    "p50": round(-last["p50"], 1),
                    "hi": round(-last["lo"], 1),
                }
            },
        },
        "time_prior": {},
        "constraints": {},
        "provenance": {},
        "scores": {"train": {"share": 0.5, "n": 5}, "val": {"share": 0.5, "n": 2}},
    }


def program_episode_table(
    runs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    grid_s: int = GRID_S,
) -> dict[str, np.ndarray]:
    """Build the minimal episode table compile_proposal needs for kind-1 session rows."""
    n = len(runs)
    rle_offset = np.zeros(n + 1, dtype=np.int64)
    rle_level: list[float] = []
    rle_n: list[int] = []
    energy = np.zeros(n, dtype=np.float64)
    dur = np.zeros(n, dtype=np.float64)
    for i, (levels, counts) in enumerate(runs):
        rle_offset[i + 1] = rle_offset[i] + len(levels)
        rle_level.extend(float(v) for v in levels)
        rle_n.extend(int(v) for v in counts)
        energy[i] = float(np.sum(levels * counts)) * grid_s / 3600.0
        dur[i] = float(np.sum(counts)) * grid_s
    return {
        "start_i": np.arange(n, dtype=np.int64),
        "end_i": np.arange(n, dtype=np.int64),
        "block": np.zeros(n, dtype=np.int32),
        "hour": np.full(n, 12, dtype=np.int64),
        "dow": (np.arange(n) % 7).astype(np.int64),
        "energy_wh": energy,
        "dur_s": dur,
        "rise_w": np.zeros(n, dtype=np.float64),
        "fall_w": np.zeros(n, dtype=np.float64),
        "plateau_w": np.zeros(n, dtype=np.float64),
        "rle_offset": rle_offset,
        "rle_level_w": np.asarray(rle_level, dtype=np.float64),
        "rle_n": np.asarray(rle_n, dtype=np.int64),
    }
