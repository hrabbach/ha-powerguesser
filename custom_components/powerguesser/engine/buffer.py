"""SQLite buffer storage and channel lifecycle for PowerGuesser."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import asyncio
import glob
import json
import logging
import os
import shutil
import sqlite3
import threading
import time

from .coverage import ChannelCoverage, channel_coverage, segments, usable_days
from .model import SourceModel
from .reader import read_range

_LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
JOURNAL_SIZE_LIMIT_BYTES = 4_194_304
RETENTION_DAYS = 90
RETENTION_CHUNK_MS = 3_600_000
KIND_GROSS = "gross"
KIND_METERED = "metered"
KIND_PHASE = "phase"
# Session retention (Plan 06-06, the contract Plan 06-09 reads):
# there is one session channel per plug entity, (entity_id, "session"), reused by
# every session of that plug; the session record's window, not the channel,
# delimits a session. When a session ends, the next rebuild drops its
# ChannelSpec: the writer's set_active writes a NULL removal marker at that
# moment and the channel leaves writer.held_channels (reassigned from scratch
# every rebuild, never accumulated). The channel row and every sample stay in
# the buffer, because ensure_channels only upserts and never deletes a row;
# samples age out only through the normal RETENTION_DAYS (90) sample retention,
# so an ended session stays readable retrospectively, and one older than the
# retention has no data and must be refused with insufficient_data, never read
# as empty-but-valid.
KIND_SESSION = "session"
ROLE_METERED = "metered"
ROLE_SESSION = "session"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Specification of an input channel to be registered."""

    entity_id: str
    role: str
    kind: str


@dataclass(frozen=True, slots=True)
class GrossLayoutEntry:
    """Specification of gross channel composition."""

    role: str
    transform: str
    entity_ids: tuple[str, ...]


def _assert_off_loop() -> None:
    """Assert that current thread is not running an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("BufferDB called from an event loop thread")


def channel_specs_from_model(
    model: SourceModel,
) -> tuple[list[ChannelSpec], list[GrossLayoutEntry]]:
    """Extract channel specs and gross layout entries from a SourceModel."""
    specs: list[ChannelSpec] = []
    seen: set[tuple[str, str]] = set()

    for channel in model.gross_channels:
        role_str = str(channel.role.value)
        for entity_id in channel.entity_ids:
            key = (entity_id, role_str)
            if key not in seen:
                seen.add(key)
                specs.append(ChannelSpec(entity_id, role_str, KIND_GROSS))

    layout = [
        GrossLayoutEntry(
            role=str(channel.role.value),
            transform=str(channel.transform.value),
            entity_ids=channel.entity_ids,
        )
        for channel in model.gross_channels
    ]

    for entity_id in model.metered_entity_ids:
        key = (entity_id, ROLE_METERED)
        if key not in seen:
            seen.add(key)
            specs.append(ChannelSpec(entity_id, ROLE_METERED, KIND_METERED))

    return specs, layout


class BufferDB:
    """Thread-safe SQLite storage for the rolling sample buffer."""

    def __init__(self, path: str) -> None:
        """Initialize the buffer database."""
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self.reset_backup: str | None = None

    def open(self) -> None:
        """Open the database, set WAL pragmas, and create schema if missing."""
        _assert_off_loop()
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            try:
                self._init_conn_locked()
            except sqlite3.DatabaseError as err:
                err_msg = str(err).lower()
                if "malformed" in err_msg or "not a database" in err_msg:
                    epoch = int(time.time())
                    suffix = f".corrupt-{epoch}"
                    # Backup -wal and -shm before closing connection in case SQLite deletes them on close
                    for extra in ("-wal", "-shm"):
                        fpath = self.path + extra
                        if os.path.exists(fpath):
                            try:
                                shutil.copyfile(fpath, fpath + suffix)
                            except OSError:
                                pass

                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                        self._conn = None

                    renamed_main = None
                    for fpath in (self.path, self.path + "-wal", self.path + "-shm"):
                        target = fpath + suffix
                        if fpath == self.path:
                            if os.path.exists(fpath):
                                os.rename(fpath, target)
                                renamed_main = os.path.basename(target)
                        else:
                            if os.path.exists(target):
                                if os.path.exists(fpath):
                                    try:
                                        os.remove(fpath)
                                    except OSError:
                                        pass
                            elif os.path.exists(fpath):
                                try:
                                    os.rename(fpath, target)
                                except OSError:
                                    pass

                    self.reset_backup = renamed_main or f"{os.path.basename(self.path)}{suffix}"
                    self._init_conn_locked()
                else:
                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                        self._conn = None
                    raise

    def _init_conn_locked(self) -> None:
        """Establish connection, pragmas, schema, and hard-cut recovery."""
        self._conn = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA journal_size_limit = 4194304")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS samples("
            "ts INTEGER NOT NULL, "
            "ch INTEGER NOT NULL, "
            "w INTEGER, "
            "PRIMARY KEY(ts, ch)) WITHOUT ROWID"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS channels("
            "id INTEGER PRIMARY KEY, "
            "entity_id TEXT NOT NULL, "
            "role TEXT NOT NULL, "
            "kind TEXT NOT NULL, "
            "UNIQUE(entity_id, role)"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL"
            ")"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

        # Hard-cut recovery (D-03): schema exists first, then check clean_stop and last_flush_ms
        cursor = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'clean_stop'"
        )
        clean_stop_row = cursor.fetchone()
        clean_stop = clean_stop_row[0] if clean_stop_row is not None else None

        cursor = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'last_flush_ms'"
        )
        last_flush_row = cursor.fetchone()
        last_flush_ms = last_flush_row[0] if last_flush_row is not None else None

        if clean_stop != "1" and last_flush_ms is not None:
            recovery_ts = int(last_flush_ms) + 1
            cursor = self._conn.execute("SELECT id FROM channels")
            channel_ids = [row[0] for row in cursor.fetchall()]
            for ch_id in channel_ids:
                self._conn.execute(
                    "INSERT OR IGNORE INTO samples(ts, ch, w) VALUES(?, ?, NULL)",
                    (recovery_ts, ch_id),
                )

        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("clean_stop", "0"),
        )

    def delete_older_than(
        self, cutoff_ms: int, chunk_ms: int = RETENTION_CHUNK_MS
    ) -> int:
        """Delete sample rows older than cutoff_ms in time chunks."""
        _assert_off_loop()
        deleted = 0
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            row = self._conn.execute("SELECT MIN(ts) FROM samples").fetchone()
            lo = row[0] if row is not None else None

        while lo is not None and lo < cutoff_ms:
            step = min(lo + chunk_ms, cutoff_ms)
            with self._lock:
                assert self._conn is not None, "BufferDB not open"
                res = self._conn.execute(
                    "DELETE FROM samples WHERE ts < ?", (step,)
                )
                deleted += res.rowcount
            lo = step
        return deleted

    def ensure_channels(
        self,
        specs: Sequence[ChannelSpec],
        layout: Sequence[GrossLayoutEntry],
        *,
        phase_invert: bool = False,
    ) -> dict[tuple[str, str], int]:
        """Ensure channels exist in database and record gross layout and phase inversion in meta."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            self._conn.execute("BEGIN")
            try:
                for spec in specs:
                    self._conn.execute(
                        "INSERT INTO channels(entity_id, role, kind) VALUES(?, ?, ?) "
                        "ON CONFLICT(entity_id, role) DO UPDATE SET kind = excluded.kind",
                        (spec.entity_id, spec.role, spec.kind),
                    )
                cursor = self._conn.execute(
                    "SELECT entity_id, role, id FROM channels"
                )
                id_map: dict[tuple[str, str], int] = {
                    (row[0], row[1]): row[2] for row in cursor.fetchall()
                }

                gross_layout_data = []
                for entry in layout:
                    ch_ids = [id_map[(e, entry.role)] for e in entry.entity_ids]
                    gross_layout_data.append(
                        {
                            "role": entry.role,
                            "transform": entry.transform,
                            "channels": ch_ids,
                        }
                    )

                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("gross_layout", json.dumps(gross_layout_data)),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("phase_invert", "1" if phase_invert else "0"),
                )
                self._conn.execute("COMMIT")
                return id_map
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def write_batch(self, rows: Sequence[tuple[int, int, int | None]]) -> None:
        """Write a batch of samples and update meta entries."""
        _assert_off_loop()
        if not rows:
            return
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO samples(ts, ch, w) VALUES(?, ?, ?)",
                    rows,
                )
                max_ts = max(r[0] for r in rows)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("last_flush_ms", str(max_ts)),
                )
                ch_min_ts: dict[int, int] = {}
                for ts, ch, _ in rows:
                    if ch not in ch_min_ts or ts < ch_min_ts[ch]:
                        ch_min_ts[ch] = ts

                for ch, min_ts in ch_min_ts.items():
                    key = "live_start_ms:" + str(ch)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                        (key, str(min_ts)),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def insert_bootstrap(
        self,
        rows: Sequence[tuple[int, int, int | None]],
        ch: int,
        cursor_ms: int,
    ) -> None:
        """Insert historical samples without overwriting existing rows, and update cursor."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            self._conn.execute("BEGIN")
            try:
                if rows:
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO samples(ts, ch, w) VALUES(?, ?, ?)",
                        rows,
                    )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    (f"bootstrap_cursor_ms:{ch}", str(cursor_ms)),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_meta(self, key: str) -> str | None:
        """Get a metadata value by key."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            cursor = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            return row[0] if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                (key, value),
            )

    def channel_rows(self) -> list[tuple[int, str, str, str]]:
        """Return all channels as (id, entity_id, role, kind)."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            cursor = self._conn.execute(
                "SELECT id, entity_id, role, kind FROM channels ORDER BY id"
            )
            return cursor.fetchall()

    def coverage_for(
        self,
        windows: Mapping[int, tuple[int, int]],
        roles: Mapping[int, str],
    ) -> tuple[float, dict[int, ChannelCoverage]]:
        """Compute usable days and per-channel coverage for specified channel windows.

        Acquires the buffer lock once per channel read and releases it before
        computing segments and channel coverage, allowing flushes to interleave.
        """
        _assert_off_loop()
        start_time = time.monotonic()
        coverages: dict[int, ChannelCoverage] = {}
        segments_by_channel: dict[int, list[tuple[int, int]]] = {}

        for ch, (start_ms, end_ms) in windows.items():
            with self._lock:
                assert self._conn is not None, "BufferDB not open"
                rows = read_range(self._conn, [ch], start_ms, end_ms).get(ch, [])

            cov = channel_coverage(rows, start_ms, end_ms)
            segs = segments(rows, start_ms, end_ms)
            coverages[ch] = cov
            segments_by_channel[ch] = segs

        gross_roles = dict(roles)
        usable = usable_days(segments_by_channel, gross_roles)
        duration = time.monotonic() - start_time
        _LOGGER.debug(
            "BufferDB coverage scan took %.3f s for %d channels (usable: %.2f days)",
            duration,
            len(windows),
            usable,
        )
        return usable, coverages

    def stats(self) -> dict[str, Any]:
        """Return aggregated database statistics and per-channel summaries.

        Aggregates only: counts, spans, and metadata without raw sample rows.
        """
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            meta_dict = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
            schema_version_raw = meta_dict.get("schema_version")
            schema_version = int(schema_version_raw) if schema_version_raw else SCHEMA_VERSION

            file_bytes = 0
            for path in [self.path, self.path + "-wal", self.path + "-shm"]:
                try:
                    file_bytes += os.path.getsize(path)
                except OSError:
                    pass

            last_flush_raw = meta_dict.get("last_flush_ms")
            last_flush_ms = int(last_flush_raw) if last_flush_raw else None
            clean_stop = meta_dict.get("clean_stop") == "1"

            cursor = self._conn.execute(
                "SELECT c.id, c.entity_id, c.role, c.kind, "
                "COALESCE(s.rows_count, 0), COALESCE(s.null_count, 0), s.min_ts, s.max_ts "
                "FROM channels c "
                "LEFT JOIN ("
                "    SELECT ch, COUNT(*) AS rows_count, "
                "    SUM(CASE WHEN w IS NULL THEN 1 ELSE 0 END) AS null_count, "
                "    MIN(ts) AS min_ts, MAX(ts) AS max_ts "
                "    FROM samples GROUP BY ch"
                ") s ON c.id = s.ch "
                "ORDER BY c.id"
            )
            channels = [
                {
                    "id": row[0],
                    "entity_id": row[1],
                    "role": row[2],
                    "kind": row[3],
                    "rows": row[4],
                    "null_rows": row[5],
                    "first_ms": row[6],
                    "last_ms": row[7],
                }
                for row in cursor.fetchall()
            ]

            return {
                "schema_version": schema_version,
                "file_bytes": file_bytes,
                "last_flush_ms": last_flush_ms,
                "clean_stop": clean_stop,
                "retention_days": RETENTION_DAYS,
                "channels": channels,
            }

    def rename_entity(self, old_entity_id: str, new_entity_id: str) -> int:
        """Rename an entity across its channels, skipping roles already registered for target."""
        _assert_off_loop()
        with self._lock:
            assert self._conn is not None, "BufferDB not open"
            res = self._conn.execute(
                "UPDATE channels SET entity_id = ? "
                "WHERE entity_id = ? "
                "AND NOT EXISTS ("
                "SELECT 1 FROM channels AS c2 "
                "WHERE c2.entity_id = ? AND c2.role = channels.role"
                ")",
                (new_entity_id, old_entity_id, new_entity_id),
            )
            return res.rowcount

    def close_clean(self) -> None:
        """Mark clean stop in meta and close database connection."""
        _assert_off_loop()
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                        ("clean_stop", "1"),
                    )
                except Exception:
                    pass
                self._conn.close()
                self._conn = None


def remove_buffer_files(db_path: str) -> list[str]:
    """Delete buffer database files, WAL/SHM, corrupt backups, and containing directory if empty."""
    _assert_off_loop()
    deleted: list[str] = []
    targets = [db_path, db_path + "-wal", db_path + "-shm"]
    targets.extend(glob.glob(db_path + ".corrupt-*"))
    for fpath in targets:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted.append(os.path.basename(fpath))
        except FileNotFoundError:
            pass
        except OSError as err:
            _LOGGER.warning("Failed to remove buffer file %s: %s", fpath, err)

    folder = os.path.dirname(os.path.abspath(db_path))
    try:
        os.rmdir(folder)
    except (FileNotFoundError, OSError):
        pass

    return deleted

