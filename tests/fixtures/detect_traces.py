"""Synthetic gridded residual builders and switched artefact dicts for detector tests."""

from collections.abc import Mapping, Sequence

import numpy as np


def grid_trace(n: int, base_w: float = 250.0, noise_w: float = 15.0, seed: int = 7) -> np.ndarray:
    """Noisy base-load residual series of n 30 s grid samples."""
    rng = np.random.default_rng(seed)
    return base_w + rng.normal(0.0, noise_w, size=n)


def random_series(seed: int, n: int, base_w: float = 250.0) -> np.ndarray:
    """Random series with steps, split steps, opposite spikes and NaN runs."""
    rng = np.random.default_rng(seed)
    r = grid_trace(n, base_w=base_w, noise_w=20.0, seed=seed)
    n_steps = int(rng.integers(1, 6))
    for _ in range(n_steps):
        start = int(rng.integers(0, max(1, n - 6)))
        w = float(rng.uniform(50.0, 3000.0))
        kind = int(rng.integers(0, 4))
        if kind == 0 and start + 2 < n:
            # split step across two samples
            r[start + 1] += w * 0.45
            r[start + 2] += w * 0.55
            end = int(rng.integers(start + 3, min(n, start + 40)))
            r[end:] -= w
        elif kind == 1 and start + 3 < n:
            # single-sample opposite-sign alignment spike
            r[start + 1] += w
            r[start + 2 : start + 3] -= w
        elif kind == 2 and start + 2 < n:
            # NaN run of 1..4 samples
            gap = int(rng.integers(1, 5))
            r[start + 1 : start + 1 + gap] = np.nan
        else:
            dur = int(rng.integers(1, 30))
            r[start + 1 : min(n, start + 1 + dur)] += w
            end = min(n - 1, start + 1 + dur)
            r[end:] -= w
    return r


def add_run(r: np.ndarray, start_i: int, n_samples: int, w: float) -> None:
    """Add a rectangular run of w W so start_i is the pre-edge sample (pair_switched style)."""
    r[start_i + 1 : start_i + 1 + n_samples] += w


def cycling_run(
    r: np.ndarray,
    start_i: int,
    on_samples: int,
    off_samples: int,
    cycles: int,
    w: float,
) -> int:
    """Add thermostat-style on/off cycles; return the index after the last cycle."""
    i = start_i
    for _ in range(cycles):
        if i + 1 >= len(r):
            break
        add_run(r, i, on_samples, w)
        i += on_samples
        i += off_samples
    return i


def switched_artefact(
    sig_id: str,
    rise: Sequence[float] = (1900.0, 2000.0, 2100.0),
    fall: Sequence[float] = (-2100.0, -2000.0, -1900.0),
    dur: Sequence[float] = (120.0, 180.0, 300.0),
    plateau: Sequence[float] = (1900.0, 2000.0, 2100.0),
    *,
    grid_s: int = 30,
    threshold_w: float = 120.0,
    phase_share: Mapping[str, tuple[float, float]] | None = None,
) -> dict:
    """Build a SignatureArtefact v1 switched dict (scores with train/val shares)."""
    states = [
        {
            "name": "on",
            "rise_w": {"lo": float(rise[0]), "p50": float(rise[1]), "hi": float(rise[2])},
            "fall_w": {"lo": float(fall[0]), "p50": float(fall[1]), "hi": float(fall[2])},
            "dur_s": {"lo": float(dur[0]), "p50": float(dur[1]), "hi": float(dur[2])},
            "plateau_w": {
                "lo": float(plateau[0]),
                "p50": float(plateau[1]),
                "hi": float(plateau[2]),
            },
            "phase_share": (
                {k: [float(v[0]), float(v[1])] for k, v in phase_share.items()}
                if phase_share is not None
                else None
            ),
        }
    ]
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "switched",
        "label": sig_id,
        "category": "switching",
        "extraction": {"grid_s": int(grid_s), "edge_threshold_w": float(threshold_w)},
        "states": states,
        "session": None,
        "time_prior": {},
        "constraints": {},
        "provenance": {},
        "scores": {"train": {"share": 0.5, "n": 5}, "val": {"share": 0.5, "n": 2}},
    }
