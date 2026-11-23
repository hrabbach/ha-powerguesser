"""Unit tests for BufferDB SQLite storage, schema, hard-cut recovery, and off-loop guards."""

import asyncio
import os
import sqlite3
import pytest

from custom_components.powerguesser.engine.buffer import (
    BufferDB,
    ChannelSpec,
    GrossLayoutEntry,
    KIND_SESSION,
    SCHEMA_VERSION,
    remove_buffer_files,
)


def test_open_fresh_file_creates_schema(tmp_path) -> None:
    """open() on a non-existent path creates folder, schema_version 1, clean_stop 0, and tables."""
    db_path = tmp_path / "deep" / "nested" / "buffer.sqlite"
    assert not db_path.parent.exists()

    db = BufferDB(str(db_path))
    db.open()
    try:
        assert db_path.exists()
        assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert db.get_meta("clean_stop") == "0"
        assert db.get_meta("last_flush_ms") is None

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        tables = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "samples" in tables
        assert "channels" in tables
        assert "meta" in tables

        # Zero sample rows
        count = cur.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        assert count == 0
        conn.close()
    finally:
        db.close_clean()


def test_open_existing_file_and_reopen_keeps_rows(tmp_path) -> None:
    """WAL pragma, synchronous=NORMAL, duplicate batch handling, and live_start_ms persistence."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        with db._lock:
            assert db._conn is not None
            journal_mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
            sync_mode = db._conn.execute("PRAGMA synchronous").fetchone()[0]
        assert journal_mode.lower() == "wal"
        assert sync_mode == 1  # 1 == NORMAL

        specs = [ChannelSpec("sensor.grid", "grid", "gross")]
        layout = [GrossLayoutEntry("grid", "direct", ("sensor.grid",))]
        id_map = db.ensure_channels(specs, layout)
        ch_id = id_map[("sensor.grid", "grid")]

        # Write batch with duplicate (ts, ch) -> keeps the last row
        db.write_batch([
            (1000, ch_id, 500),
            (1000, ch_id, 600),
            (2000, ch_id, 700),
        ])
        assert db.get_meta("last_flush_ms") == "2000"
        assert db.get_meta(f"live_start_ms:{ch_id}") == "1000"

        # Second batch with earlier or later timestamp doesn't change live_start_ms
        db.write_batch([(3000, ch_id, 800)])
        assert db.get_meta(f"live_start_ms:{ch_id}") == "1000"
    finally:
        db.close_clean()

    # Reopening keeps rows
    db2 = BufferDB(str(db_path))
    db2.open()
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT ts, ch, w FROM samples ORDER BY ts").fetchall()
        conn.close()
        assert rows == [(1000, ch_id, 600), (2000, ch_id, 700), (3000, ch_id, 800)]
    finally:
        db2.close_clean()


def test_open_recovery_after_hard_cut(tmp_path) -> None:
    """open() after unclean stop writes a NULL row at last_flush_ms + 1 for every channel."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()

    specs = [
        ChannelSpec("sensor.grid", "grid", "gross"),
        ChannelSpec("sensor.battery", "battery", "gross"),
    ]
    layout = [
        GrossLayoutEntry("grid", "direct", ("sensor.grid",)),
        GrossLayoutEntry("battery", "direct", ("sensor.battery",)),
    ]
    id_map = db.ensure_channels(specs, layout)
    ch_grid = id_map[("sensor.grid", "grid")]
    ch_bat = id_map[("sensor.battery", "battery")]

    db.write_batch([
        (5000, ch_grid, 1000),
        (5000, ch_bat, -200),
    ])
    assert db.get_meta("last_flush_ms") == "5000"
    assert db.get_meta("clean_stop") == "0"

    # Simulate hard cut: close connection directly without close_clean()
    with db._lock:
        assert db._conn is not None
        db._conn.close()
        db._conn = None

    # Reopen -> hard-cut recovery triggers
    db_recovered = BufferDB(str(db_path))
    db_recovered.open()
    try:
        assert db_recovered.get_meta("clean_stop") == "0"
        conn = sqlite3.connect(str(db_path))
        null_rows = conn.execute(
            "SELECT ts, ch, w FROM samples WHERE ts = 5001 ORDER BY ch"
        ).fetchall()
        conn.close()
        assert null_rows == [(5001, ch_grid, None), (5001, ch_bat, None)]
    finally:
        db_recovered.close_clean()

    # Reopening after clean stop does NOT add another recovery row
    db_clean = BufferDB(str(db_path))
    db_clean.open()
    try:
        assert db_clean.get_meta("clean_stop") == "0"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM samples WHERE ts > 5001").fetchone()[0]
        conn.close()
        assert count == 0
    finally:
        db_clean.close_clean()


def test_ensure_channels_lifecycle(tmp_path) -> None:
    """ensure_channels returns the same id across calls, updates kind, and rewrites gross_layout."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        specs1 = [ChannelSpec("sensor.grid", "grid", "gross")]
        layout1 = [GrossLayoutEntry("grid", "direct", ("sensor.grid",))]
        map1 = db.ensure_channels(specs1, layout1)
        grid_id = map1[("sensor.grid", "grid")]

        # Update kind of same channel and add another
        specs2 = [
            ChannelSpec("sensor.grid", "grid", "phase"),
            ChannelSpec("sensor.solar", "solar", "gross"),
        ]
        layout2 = [
            GrossLayoutEntry("grid", "direct", ("sensor.grid",)),
            GrossLayoutEntry("solar", "direct", ("sensor.solar",)),
        ]
        map2 = db.ensure_channels(specs2, layout2)
        assert map2[("sensor.grid", "grid")] == grid_id
        solar_id = map2[("sensor.solar", "solar")]
        assert solar_id != grid_id

        # Verify kind was updated
        rows = db.channel_rows()
        kind_map = {row[1]: row[3] for row in rows}
        assert kind_map["sensor.grid"] == "phase"
        assert kind_map["sensor.solar"] == "gross"
    finally:
        db.close_clean()


@pytest.mark.asyncio
async def test_off_loop_buffer_raises_on_event_loop(tmp_path) -> None:
    """Every BufferDB method raises RuntimeError when called from an event loop thread."""
    db = BufferDB(str(tmp_path / "buffer.sqlite"))

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.open()

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.ensure_channels([], [])

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.write_batch([])

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.get_meta("key")

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.set_meta("key", "val")

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.channel_rows()

    with pytest.raises(RuntimeError, match="BufferDB called from an event loop thread"):
        db.close_clean()


def test_delete_older_than(tmp_path) -> None:
    """delete_older_than deletes rows older than cutoff in chunks and keeps cutoff and newer rows."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        # Empty table returns 0
        assert db.delete_older_than(1_000_000) == 0

        specs = [ChannelSpec("sensor.grid", "grid", "gross")]
        layout = [GrossLayoutEntry("grid", "direct", ("sensor.grid",))]
        id_map = db.ensure_channels(specs, layout)
        ch_id = id_map[("sensor.grid", "grid")]

        # Insert rows spread over 3 days (72 hours), one per hour: ts = h * 3_600_000
        hour_ms = 3_600_000
        rows = [(h * hour_ms, ch_id, 100 + h) for h in range(73)]  # 0 to 72 hours
        db.write_batch(rows)

        # Cutoff at 36 hours (36 * hour_ms)
        cutoff = 36 * hour_ms
        deleted = db.delete_older_than(cutoff, chunk_ms=hour_ms)
        # Should delete rows 0 to 35 (36 rows)
        assert deleted == 36

        conn = sqlite3.connect(str(db_path))
        remaining = [r[0] for r in conn.execute("SELECT ts FROM samples ORDER BY ts").fetchall()]
        conn.close()

        # Rows from cutoff (36 hours) to 72 hours should remain (37 rows)
        assert len(remaining) == 37
        assert remaining[0] == cutoff
        assert remaining[-1] == 72 * hour_ms
    finally:
        db.close_clean()


def test_open_corrupt_file_resets_and_sets_backup(tmp_path) -> None:
    """open() on a corrupt file renames it to .corrupt-<epoch> and creates fresh schema."""
    db_path = tmp_path / "buffer.sqlite"
    wal_path = tmp_path / "buffer.sqlite-wal"
    shm_path = tmp_path / "buffer.sqlite-shm"

    # Write corrupt data
    db_path.write_bytes(os.urandom(256))
    wal_path.write_bytes(os.urandom(64))
    shm_path.write_bytes(os.urandom(64))

    db = BufferDB(str(db_path))
    db.open()
    try:
        assert db.reset_backup is not None
        assert db.reset_backup.startswith("buffer.sqlite.corrupt-")
        corrupt_backup = tmp_path / db.reset_backup
        assert corrupt_backup.exists()
        epoch_suffix = db.reset_backup[len("buffer.sqlite"):]
        assert (tmp_path / f"buffer.sqlite-wal{epoch_suffix}").exists()
        assert (tmp_path / f"buffer.sqlite-shm{epoch_suffix}").exists()

        # New file has fresh valid schema
        assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert db.get_meta("clean_stop") == "0"
    finally:
        db.close_clean()

    # Reopening healthy file leaves reset_backup None
    db2 = BufferDB(str(db_path))
    db2.open()
    try:
        assert db2.reset_backup is None
    finally:
        db2.close_clean()


def test_rename_entity_skips_existing_entity_role(tmp_path) -> None:
    """rename_entity skips updating when target entity already has a channel with the same role."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        # Case 1: (sensor.a, metered) and (sensor.b, metered) present
        specs = [
            ChannelSpec("sensor.a", "metered", "metered"),
            ChannelSpec("sensor.b", "metered", "metered"),
        ]
        layout = []
        id_map = db.ensure_channels(specs, layout)
        id_a = id_map[("sensor.a", "metered")]
        id_b = id_map[("sensor.b", "metered")]

        # rename sensor.a -> sensor.b: collision on role metered -> returns 0, no error
        assert db.rename_entity("sensor.a", "sensor.b") == 0

        rows = db.channel_rows()
        row_map = {r[0]: (r[1], r[2]) for r in rows}
        assert row_map[id_a] == ("sensor.a", "metered")
        assert row_map[id_b] == ("sensor.b", "metered")

        # Case 2: (sensor.c, grid) and (sensor.c, metered) where sensor.b has only metered
        specs2 = [
            ChannelSpec("sensor.c", "grid", "gross"),
            ChannelSpec("sensor.c", "metered", "metered"),
        ]
        id_map2 = db.ensure_channels(specs2, layout)
        id_c_grid = id_map2[("sensor.c", "grid")]
        id_c_metered = id_map2[("sensor.c", "metered")]

        # rename sensor.c -> sensor.b: grid succeeds, metered collides with sensor.b metered
        count = db.rename_entity("sensor.c", "sensor.b")
        assert count == 1

        rows2 = db.channel_rows()
        row_map2 = {r[0]: (r[1], r[2]) for r in rows2}
        assert row_map2[id_c_grid] == ("sensor.b", "grid")
        assert row_map2[id_c_metered] == ("sensor.c", "metered")
    finally:
        db.close_clean()


def test_ensure_channels_session_channel_is_distinct_and_kept(tmp_path) -> None:
    """A session channel exists beside the metered channel of the same entity."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        specs = [
            ChannelSpec("sensor.plug", "metered", "metered"),
            ChannelSpec("sensor.plug", KIND_SESSION, KIND_SESSION),
        ]
        id_map = db.ensure_channels(specs, [])
        metered_id = id_map[("sensor.plug", "metered")]
        session_id = id_map[("sensor.plug", KIND_SESSION)]
        assert metered_id != session_id

        kinds = {(row[1], row[2]): row[3] for row in db.channel_rows()}
        assert kinds[("sensor.plug", KIND_SESSION)] == KIND_SESSION
        assert kinds[("sensor.plug", "metered")] == "metered"
    finally:
        db.close_clean()


def test_ensure_channels_keeps_dropped_session_channel(tmp_path) -> None:
    """A session channel dropped from the next spec set keeps its row and samples.

    The session retention contract (Plan 06-06, read by Plan 06-09):
    ensure_channels only upserts and never deletes a channel row, so an ended
    session's samples stay readable until the normal 90-day sample retention
    removes them.
    """
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        id_map = db.ensure_channels(
            [ChannelSpec("sensor.plug", KIND_SESSION, KIND_SESSION)], []
        )
        channel_id = id_map[("sensor.plug", KIND_SESSION)]
        db.write_batch([(1000, channel_id, 400), (2000, channel_id, 450)])

        # The session ended: the next rebuild no longer carries its spec.
        db.ensure_channels([], [])

        rows = {row[0]: (row[1], row[2], row[3]) for row in db.channel_rows()}
        assert rows[channel_id] == ("sensor.plug", KIND_SESSION, KIND_SESSION)

        conn = sqlite3.connect(str(db_path))
        samples = conn.execute(
            "SELECT ts, w FROM samples WHERE ch = ? ORDER BY ts", (channel_id,)
        ).fetchall()
        conn.close()
        assert samples == [(1000, 400), (2000, 450)]
    finally:
        db.close_clean()


def test_remove_buffer_files(tmp_path) -> None:
    """remove_buffer_files deletes db, -wal, -shm, .corrupt-* and removes folder if empty."""
    folder = tmp_path / "powerguesser"
    folder.mkdir()
    db_file = folder / "buffer.sqlite"
    db_file.write_text("db")
    (folder / "buffer.sqlite-wal").write_text("wal")
    (folder / "buffer.sqlite-shm").write_text("shm")
    (folder / "buffer.sqlite.corrupt-12345").write_text("corrupt1")
    (folder / "buffer.sqlite.corrupt-67890").write_text("corrupt2")

    deleted = remove_buffer_files(str(db_file))
    assert len(deleted) == 5
    assert not db_file.exists()
    assert not (folder / "buffer.sqlite-wal").exists()
    assert not (folder / "buffer.sqlite-shm").exists()
    assert not (folder / "buffer.sqlite.corrupt-12345").exists()
    assert not (folder / "buffer.sqlite.corrupt-67890").exists()
    assert not folder.exists()

    # Calling on non-existent path returns empty list without error
    assert remove_buffer_files(str(db_file)) == []


def test_coverage_for_missing_gross_channel_yields_zero_usable(tmp_path) -> None:
    """When a required gross role channel is missing from windows, usable days is 0.0."""
    db_path = tmp_path / "buffer.sqlite"
    db = BufferDB(str(db_path))
    db.open()
    try:
        specs = [
            ChannelSpec("sensor.grid", "grid", "gross"),
            ChannelSpec("sensor.battery", "battery", "gross"),
        ]
        layout = [
            GrossLayoutEntry("grid", "direct", ("sensor.grid",)),
            GrossLayoutEntry("battery", "direct", ("sensor.battery",)),
        ]
        id_map = db.ensure_channels(specs, layout)
        grid_ch = id_map[("sensor.grid", "grid")]
        batt_ch = id_map[("sensor.battery", "battery")]

        day_ms = 86_400_000
        db.write_batch([(0, grid_ch, 500), (2 * day_ms, grid_ch, 500)])

        roles = {grid_ch: "grid", batt_ch: "battery"}
        windows = {grid_ch: (0, 2 * day_ms)}

        usable, covs = db.coverage_for(windows, roles)
        assert usable == 0.0
        assert grid_ch in covs
        assert batt_ch not in covs
    finally:
        db.close_clean()



