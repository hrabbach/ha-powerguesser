"""HA-free capture engine package.

Every module in this package is importable without Home Assistant (enforced by
``tests/engine/test_no_ha_imports.py``), so the capture snap and preview fit can
be unit-tested offline and called from the executor thread. This module holds no
imports so an existing engine module (e.g. ``engine.learning.worker``) can import
individual capture modules later without creating an import cycle.
"""
