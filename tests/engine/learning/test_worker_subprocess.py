"""Tests for out-of-process worker execution and isolation."""

from datetime import datetime, timezone
import io
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import types

import pytest
from typing import Any

from custom_components.powerguesser.engine.learning import worker
from custom_components.powerguesser.engine.learning.worker import _start_stdin_watchdog
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds


@pytest.mark.slow
def test_extract_subprocess_finds_kettle_cluster(tmp_path: Path) -> None:
    """A real subprocess runs extract stage, writing atomic outputs and finding kettle cluster."""
    # 2026-11-01T00:00:00Z in epoch ms
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buffer_path = tmp_path / "buffer.sqlite"
    stage_dir = tmp_path / "stage"
    build_buffer(buffer_path, SynthSpec(days=days), end_ms)

    # Sunday is weekday 6
    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    job_path = tmp_path / "job.json"
    job_data = {
        "schema": 1,
        "buffer_path": str(buffer_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f)

    worker_entry = (
        Path(__file__).parent.parent.parent.parent
        / "custom_components"
        / "powerguesser"
        / "engine"
        / "worker_entry.py"
    )
    assert worker_entry.exists(), f"worker_entry.py not found at {worker_entry}"

    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    cmd = [
        sys.executable,
        "-P",
        str(worker_entry),
        "extract",
        "--job",
        str(job_path),
        "--watch-stdin",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    events: list[dict] = []
    t_start = time.monotonic()
    done_or_error = False

    while time.monotonic() - t_start < 120:
        line = proc.stdout.readline()
        if not line:
            break
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            ev = json.loads(line_clean)
            events.append(ev)
            if ev.get("event") in ("done", "error"):
                done_or_error = True
                break
        except json.JSONDecodeError:
            continue

    rc = proc.wait(timeout=10)
    if proc.stdin:
        proc.stdin.close()

    assert rc == 0, f"Worker failed with returncode {rc}, stderr: {proc.stderr.read()}"
    assert done_or_error, f"Never received done or error event: {events}"

    # Verify env event
    env_events = [e for e in events if e.get("event") == "env"]
    assert env_events, "No env event emitted"
    env_ev = env_events[0]
    assert env_ev["nice"] == 19
    assert env_ev["ha_imported"] is False
    assert env_ev["omp"] == "1"

    # Verify done event and stats
    done_events = [e for e in events if e.get("event") == "done"]
    assert done_events, "No done event emitted"
    done_ev = done_events[0]
    stats = done_ev.get("stats", {})
    assert stats.get("wall_s", 0) > 0
    assert stats.get("max_rss_kb", 0) > 0
    assert stats.get("max_rss_source") == "vmhwm"

    # Verify output files exist and no .tmp files remain
    digest_path = stage_dir / "digest.json"
    meta_path = stage_dir / "digest_meta.json"
    episodes_path = stage_dir / "episodes.npz"

    assert digest_path.is_file(), "digest.json missing"
    assert meta_path.is_file(), "digest_meta.json missing"
    assert episodes_path.is_file(), "episodes.npz missing"

    tmp_files = list(stage_dir.glob("*.tmp"))
    assert not tmp_files, f"Unexpected .tmp files left behind: {tmp_files}"

    with open(digest_path, "r", encoding="utf-8") as f:
        digest = json.load(f)

    # Check for switched kettle cluster: rise_w.p50 within [1800, 2200] and n >= 3
    clusters = digest.get("clusters", [])
    kettle_clusters = [
        c
        for c in clusters
        if c.get("kind") == "switched"
        and 1800 <= c.get("rise_w", {}).get("p50", 0) <= 2200
        and c.get("n", 0) >= 3
    ]
    assert len(kettle_clusters) >= 1, f"No kettle cluster found in digest: {digest}"


@pytest.mark.slow
def test_evidence_stage_subprocess_writes_evidence_json(tmp_path: Path) -> None:
    """The evidence stage runs as a real out-of-process worker and writes evidence.json (D-10)."""
    from tests.conftest import make_switched_candidate

    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = 8
    day_bounds_ms = utc_day_bounds(end_ms, days)

    buffer_path = tmp_path / "buffer_evidence.sqlite"
    stage_dir = tmp_path / "stage_evidence"
    build_buffer(buffer_path, SynthSpec(days=days), end_ms)

    start_dt = datetime.fromtimestamp(day_bounds_ms[0] / 1000.0, tz=timezone.utc)
    first_weekday = start_dt.weekday()

    candidates = [
        make_switched_candidate("cand-2kw", "Kettle", rise_w=2000.0),
        make_switched_candidate("cand-9kw", "Big load", rise_w=9000.0),
    ]

    job_path = tmp_path / "job_evidence.json"
    job_data = {
        "schema": 1,
        "buffer_path": str(buffer_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds_ms,
        "first_weekday": first_weekday,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "candidates": candidates,
        "now_iso": "2026-11-25T07:00:00+00:00",
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f)

    worker_entry = (
        Path(__file__).parent.parent.parent.parent
        / "custom_components"
        / "powerguesser"
        / "engine"
        / "worker_entry.py"
    )
    assert worker_entry.exists(), f"worker_entry.py not found at {worker_entry}"

    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    cmd = [
        sys.executable,
        "-P",
        str(worker_entry),
        "evidence",
        "--job",
        str(job_path),
        "--watch-stdin",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    events: list[dict] = []
    t_start = time.monotonic()
    done_or_error = False

    while time.monotonic() - t_start < 120:
        line = proc.stdout.readline()
        if not line:
            break
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            ev = json.loads(line_clean)
            events.append(ev)
            if ev.get("event") in ("done", "error"):
                done_or_error = True
                break
        except json.JSONDecodeError:
            continue

    rc = proc.wait(timeout=10)
    if proc.stdin:
        proc.stdin.close()

    assert rc == 0, f"Worker failed with returncode {rc}, stderr: {proc.stderr.read()}"
    assert done_or_error, f"Never received done or error event: {events}"

    env_events = [e for e in events if e.get("event") == "env"]
    assert env_events, "No env event emitted"
    assert env_events[0]["nice"] == 19
    assert env_events[0]["ha_imported"] is False

    done_events = [e for e in events if e.get("event") == "done"]
    assert done_events, "No done event emitted"
    outputs = done_events[0]["outputs"]
    assert outputs["n"] == 2
    assert outputs["result"].endswith("evidence.json")

    evidence_path = stage_dir / "evidence.json"
    assert evidence_path.is_file(), "evidence.json missing"
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence_map = json.load(f)

    assert set(evidence_map) == {"cand-2kw", "cand-9kw"}
    for entry in evidence_map.values():
        assert entry["v"] == 1
        assert entry["source"] == "refresh"

    assert not list(stage_dir.glob("*.tmp")), "unexpected .tmp files left behind"


def test_watchdog_exits_on_eof() -> None:
    """Stdin watchdog invokes exit_fn(3) when stream reaches EOF."""
    exit_codes: list[int] = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)

    stream = io.BytesIO(b"")
    t = _start_stdin_watchdog(stream, fake_exit)
    t.join(timeout=2.0)

    assert exit_codes == [3]


def test_main_refuses_inside_home_assistant_process(tmp_path: Path) -> None:
    """Worker main() returns 4 immediately when homeassistant is in sys.modules."""
    # homeassistant is imported by pytest in this process
    assert "homeassistant" in sys.modules

    nice_before = os.nice(0) if hasattr(os, "nice") else 0
    job_path = tmp_path / "dummy_job.json"
    job_path.write_text("{}", encoding="utf-8")

    rc = worker.main(["extract", "--job", str(job_path)])
    nice_after = os.nice(0) if hasattr(os, "nice") else 0

    assert rc == 4
    assert nice_before == nice_after


def test_atomic_write_temp_name_is_per_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Temp files carry the writer's pid, so two workers in one stage dir never share one (WR-08)."""
    seen: list[str] = []
    real_replace = os.replace

    def _spy(src: Any, dst: Any) -> None:
        seen.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(worker.os, "replace", _spy)
    worker._write_atomic(tmp_path / "digest.json", b"{}")
    worker._write_atomic_npz(tmp_path / "episodes.npz", {"a": __import__("numpy").zeros(1)})

    assert all(f".{os.getpid()}.tmp" in s for s in seen), seen
    assert (tmp_path / "digest.json").read_bytes() == b"{}"
    assert not list(tmp_path.glob("*.tmp"))


def _rusage_stub(ru_maxrss: int) -> types.SimpleNamespace:
    """A stand-in for the resource module whose getrusage reports a fixed ru_maxrss."""
    return types.SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: types.SimpleNamespace(ru_maxrss=ru_maxrss),
    )


def test_peak_rss_reads_vmhwm_line(tmp_path: Path) -> None:
    """_peak_rss picks VmHWM, not VmPeak or VmRSS, and labels it as such."""
    path = tmp_path / "status"
    path.write_text(
        "Name:\tpython3\n"
        "VmPeak:\t  999999 kB\n"
        "VmHWM:\t   11048 kB\n"
        "VmRSS:\t    9000 kB\n",
        encoding="utf-8",
    )
    assert worker._peak_rss(str(path)) == (11048, "vmhwm")


def test_peak_rss_falls_back_to_rusage_without_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a status file the helper falls back to ru_maxrss (kB on Linux) and says so."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(worker, "resource", _rusage_stub(4242))
    assert worker._peak_rss(str(tmp_path / "missing")) == (4242, "ru_maxrss")


def test_peak_rss_falls_back_when_vmhwm_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status file without a VmHWM line falls back to ru_maxrss."""
    path = tmp_path / "status"
    path.write_text("Name:\tpython3\nVmRSS:\t    9000 kB\n", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(worker, "resource", _rusage_stub(4242))
    assert worker._peak_rss(str(path)) == (4242, "ru_maxrss")


def test_peak_rss_fallback_converts_macos_bytes_to_kb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS reports ru_maxrss in bytes; the figure is normalised to kB (IN-11)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(worker, "resource", _rusage_stub(4242 * 1024))
    assert worker._peak_rss(str(tmp_path / "missing")) == (4242, "ru_maxrss")


def test_peak_rss_unavailable_without_proc_or_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without /proc and without the resource module (Windows) the figure is 0, unavailable."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(worker, "resource", None)
    assert worker._peak_rss(str(tmp_path / "missing")) == (0, "unavailable")


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"max_rss_kb": 12 * 1024, "max_rss_source": "vmhwm"}, "12 MB"),
        ({"max_rss_kb": 12 * 1024}, "12 MB"),
        (
            {"max_rss_kb": 12 * 1024, "max_rss_source": "ru_maxrss"},
            "12 MB (ru_maxrss fallback, may include the parent's peak)",
        ),
        ({"max_rss_kb": 0, "max_rss_source": "unavailable"}, "unavailable"),
    ],
    ids=["vmhwm", "no-source", "fallback", "unavailable"],
)
def test_peak_memory_text_marks_fallback(stats: dict[str, Any], expected: str) -> None:
    """The peak-memory text says when the figure is not VmHWM (IN-11)."""
    assert worker.peak_memory_text(stats) == expected
