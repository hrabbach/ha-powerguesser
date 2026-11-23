"""Per-tick cost is bounded: fixed-size containers and a generous median.

The blocking gate is the deterministic bound (plateau lists and rings stay
within their fixed caps) plus a best-of-three median under a large margin,
because a p95 wall-clock assertion is unstable on shared CI runners; SC5's
real evidence is the post-ship Pi measurement (DET-04, RESEARCH Pattern 6).
"""

import time

import numpy as np

from custom_components.powerguesser.engine.detect.online import (
    DeviceModel,
    OnlineDetector,
    ProgramModel,
    compile_model,
)
from custom_components.powerguesser.engine.detect.stats import TickTimer
from tests.fixtures.detect_traces import random_series, switched_artefact
from tests.fixtures.programs import dishwasher_stages, program_artefact, washer_stages

GRID_S = 30
STEPS = 2880


def _busy_detector() -> OnlineDetector:
    models = []
    for k in range(20):
        p50 = 300.0 + 150.0 * k
        arte = switched_artefact(
            f"d{k:02d}",
            rise=(p50 - 80.0, p50, p50 + 80.0),
            plateau=(p50 - 80.0, p50, p50 + 80.0),
            dur=(60.0, 120.0, 6000.0),
        )
        models.append(DeviceModel.from_artefact(f"d{k:02d}", arte))
    return OnlineDetector(models, grid_s=GRID_S, threshold_w=120.0)


def _simultaneous_trace(n: int = STEPS) -> np.ndarray:
    """A trace rich in simultaneous edges: device pairs switch on and off together.

    Each event's rise is the sum of its two neighbouring boxes, so the joint
    search runs on a containing subset (not just a singleton) for every edge.
    """
    r = np.zeros(n, dtype=np.float64) + 400.0
    for pair in range(0, 20, 2):
        a = 300.0 + 150.0 * pair
        b = 300.0 + 150.0 * (pair + 1)
        for cycle in range(6):
            start = 60 + pair * 20 + cycle * 300
            if start + 12 >= n:
                break
            r[start + 1 : start + 13] += a + b
    return r


def test_tick_budget() -> None:
    """20 devices over 2880 busy steps: bounds hold, best-of-3 p50 < 2 ms."""
    r = random_series(5, STEPS)
    summaries = []
    for _ in range(3):
        det = _busy_detector()
        timer = TickTimer(maxlen=256)
        for i in range(STEPS):
            t0 = time.perf_counter_ns()
            det.step(float(i) * GRID_S, float(r[i]))
            timer.record(time.perf_counter_ns() - t0)
        s = timer.summary()
        assert set(s) == {"p50_ms", "p95_ms", "max_ms", "ticks"}
        assert s["ticks"] == STEPS
        # every per-device plateau list stays inside its compiled cap
        for key, model in det._models.items():
            rt = det._runtimes.get(key)
            if rt is not None:
                assert len(rt.plateau) <= model.plateau_cap
        assert len(det._ring) <= 4
        summaries.append(s)

    best_p50 = min(s["p50_ms"] for s in summaries)
    p95 = max(s["p95_ms"] for s in summaries)
    print(f"tick budget: best-of-3 p50={best_p50} ms, worst p95={p95} ms over {STEPS} ticks")
    assert best_p50 < 2.0


def test_tick_budget_with_program_devices() -> None:
    """20 switched + 2 program devices keep the bound (T-06-02, D-22)."""
    models = []
    for k in range(20):
        p50 = 300.0 + 150.0 * k
        arte = switched_artefact(
            f"d{k:02d}",
            rise=(p50 - 80.0, p50, p50 + 80.0),
            plateau=(p50 - 80.0, p50, p50 + 80.0),
            dur=(60.0, 120.0, 6000.0),
        )
        models.append(DeviceModel.from_artefact(f"d{k:02d}", arte))
    models.append(compile_model("dishwasher", program_artefact("dishwasher", dishwasher_stages())))
    models.append(compile_model("washer", program_artefact("washer", washer_stages())))
    assert any(isinstance(m, ProgramModel) for m in models)

    r = random_series(5, STEPS)
    summaries = []
    for _ in range(3):
        det = OnlineDetector(models, grid_s=GRID_S, threshold_w=120.0)
        timer = TickTimer(maxlen=256)
        for i in range(STEPS):
            t0 = time.perf_counter_ns()
            det.step(float(i) * GRID_S, float(r[i]))
            timer.record(time.perf_counter_ns() - t0)
        s = timer.summary()
        assert s["ticks"] == STEPS
        # every per-device plateau list stays inside its compiled cap and the
        # program runtimes hold only scalars and counters (no list growth)
        for key, model in det._models.items():
            rt = det._runtimes.get(key)
            if rt is not None:
                assert len(rt.plateau) <= model.plateau_cap
        for key in det._programs:
            rt = det._runtimes.get(key)
            if rt is not None:
                assert rt.plateau == []
        assert len(det._ring) <= 4
        summaries.append(s)

    best_p50 = min(s["p50_ms"] for s in summaries)
    p95 = max(s["p95_ms"] for s in summaries)
    print(
        f"tick budget with programs: best-of-3 p50={best_p50} ms, "
        f"worst p95={p95} ms over {STEPS} ticks"
    )
    assert best_p50 < 2.0


def test_tick_budget_with_simultaneous_edges() -> None:
    """A trace rich in simultaneous edges keeps the bound (D-41/T-06-12).

    The prefiltered subset search now runs on every edge, so the busiest case is
    a stream where each edge is a two-device sum: the pool is prefiltered to
    JOINT_PREFILTER candidates and at most 35 subsets are scored per edge.
    """
    models = []
    for k in range(20):
        p50 = 300.0 + 150.0 * k
        arte = switched_artefact(
            f"d{k:02d}",
            rise=(p50 - 80.0, p50, p50 + 80.0),
            plateau=(p50 - 80.0, p50, p50 + 80.0),
            dur=(60.0, 120.0, 6000.0),
        )
        models.append(DeviceModel.from_artefact(f"d{k:02d}", arte))
    models.append(compile_model("dishwasher", program_artefact("dishwasher", dishwasher_stages())))
    models.append(compile_model("washer", program_artefact("washer", washer_stages())))

    r = _simultaneous_trace()
    summaries = []
    for _ in range(3):
        det = OnlineDetector(models, grid_s=GRID_S, threshold_w=120.0)
        timer = TickTimer(maxlen=256)
        for i in range(STEPS):
            t0 = time.perf_counter_ns()
            det.step(float(i) * GRID_S, float(r[i]))
            timer.record(time.perf_counter_ns() - t0)
        s = timer.summary()
        assert s["ticks"] == STEPS
        for key, model in det._models.items():
            rt = det._runtimes.get(key)
            if rt is not None:
                assert len(rt.plateau) <= model.plateau_cap
        for key in det._programs:
            rt = det._runtimes.get(key)
            if rt is not None:
                assert rt.plateau == []
        assert len(det._ring) <= 4
        summaries.append(s)

    # the simultaneous edges really were attributed jointly (not just refused)
    assert det.state()["joint_attributed_total"] > 0
    best_p50 = min(s["p50_ms"] for s in summaries)
    p95 = max(s["p95_ms"] for s in summaries)
    print(
        f"tick budget with simultaneous edges: best-of-3 p50={best_p50} ms, "
        f"worst p95={p95} ms over {STEPS} ticks"
    )
    assert best_p50 < 2.0
