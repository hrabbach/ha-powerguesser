"""Tests for the GitLab -> GitHub allowlist export and its leak scan (Plan 06-10).

Plan 06-10 T1 (D-23, T-06-24) and the plan's DIST-01 prohibition
``MUST NOT publish planning notes, private hostnames, credentials or the private
git history to the public repository`` carry ``verification: test``, but until now
the only check was a one-shot manual ``sh ci/export_github.sh dry-run``. This file
turns that into a repeatable behavioural test that can fail.

Three things are pinned, each against the real script (never a re-implementation):

* ``dry-run`` on the current HEAD succeeds and its file list excludes the private
  trees (``.planning/``, ``.claude/``, ``.memdb/``, ``.memtrace/``, ``spike-out/``),
  ``.gitlab-ci.yml`` and the export script itself, while the allowlisted
  ``custom_components`` tree is present.
* a planted, token-shaped string in an allowlisted file makes the export refuse
  (non-zero exit) and the matched text never reaches stdout or stderr (T-06-23).
* a planted, mixed-case private hostname and a planted PEM private-key header
  each make the export refuse, and the matched text is never printed.

The planted strings are assembled at run time (``"ghp_" + "A" * 30`` and friends).
That is load-bearing, not stylistic: this test file lives in ``tests/``, which is
inside the allowlist, so a literal token or hostname in the source would itself be
found by the positive-case leak scan and fail this very suite.

``git`` and a POSIX ``sh`` are required. Both are guaranteed in the project's
canonical test environment (WSL AlmaLinux-10 locally, Linux in ``ci/pytest.sh``);
elsewhere the module skips with a reason instead of erroring.

The planted tree is built with git plumbing (a throwaway index via
``GIT_INDEX_FILE``), so no ref, branch or working-tree file is ever touched.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "ci" / "export_github.sh"

_SH = shutil.which("sh")
_GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    _SH is None or _GIT is None,
    reason="needs a POSIX sh and git (WSL/Linux test environment)",
)

# Assembled at run time: see the module docstring. A contiguous literal would be
# caught by the positive-case dry-run of this same suite.
_TOKEN = "ghp_" + "A" * 30
_PRIVATE_HOST = "GiTlAb." + "RaBbIt-" + "SyStEmS" + ".CoM"
_PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY" + "-----"


def _script_copy(tmp_path: Path) -> Path:
    """The script as a LF-only copy, matching ci/pytest.sh's CR-stripping precedent."""
    dest = tmp_path / "export_github.sh"
    dest.write_bytes(_SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
    return dest


def _env_with(export_ref: str | None) -> dict[str, str]:
    env = dict(os.environ)
    # Never let a leftover index from a caller leak into the script's own git calls.
    env.pop("GIT_INDEX_FILE", None)
    if export_ref is not None:
        env["EXPORT_REF"] = export_ref
    return env


def _dry_run(tmp_path: Path, export_ref: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run `sh ci/export_github.sh dry-run <dest>` and return the finished process."""
    script = _script_copy(tmp_path)
    dest = tmp_path / "out"
    return subprocess.run(
        [_SH, str(script), "dry-run", str(dest)],
        cwd=str(_REPO_ROOT),
        env=_env_with(export_ref),
        capture_output=True,
        text=True,
        check=False,
    )


def _tree_with(tmp_path: Path, rel_path: str, content: str) -> str:
    """A tree object equal to HEAD plus one planted file, built with a throwaway index."""
    index = tmp_path / "index"
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = str(index)

    read_tree = subprocess.run(
        [_GIT, "read-tree", "HEAD"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert read_tree.returncode == 0, read_tree.stderr

    blob = subprocess.run(
        [_GIT, "hash-object", "-w", "--stdin"],
        cwd=str(_REPO_ROOT),
        env=env,
        input=content,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blob.returncode == 0, blob.stderr

    update = subprocess.run(
        [
            _GIT,
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{blob.stdout.strip()},{rel_path}",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert update.returncode == 0, update.stderr

    write = subprocess.run(
        [_GIT, "write-tree"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert write.returncode == 0, write.stderr
    return write.stdout.strip()


def test_dry_run_succeeds_and_keeps_private_trees_out(tmp_path: Path) -> None:
    """HEAD exports green, and the built tree holds the allowlist but no private tree."""
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stderr

    listed = [line for line in proc.stdout.splitlines() if line]
    assert listed, "dry-run printed no file list"

    for path in listed:
        for forbidden in (".planning", ".claude", ".memdb", ".memtrace", "spike-out"):
            assert not path.startswith(f"{forbidden}/"), f"{forbidden} leaked into {path}"
    assert ".gitlab-ci.yml" not in listed
    assert "ci/export_github.sh" not in listed
    assert "ci/release_check.sh" in listed
    assert "custom_components/powerguesser/manifest.json" in listed


def test_planted_token_makes_the_export_refuse_without_printing_it(tmp_path: Path) -> None:
    """A token-shaped string in an allowlisted file stops the export (T-06-23)."""
    tree = _tree_with(tmp_path, "tests/leak_probe.txt", f"token = {_TOKEN}\n")
    proc = _dry_run(tmp_path, export_ref=tree)

    assert proc.returncode != 0
    assert "token shape" in proc.stderr
    assert "tests/leak_probe.txt" in proc.stderr
    assert _TOKEN not in proc.stdout
    assert _TOKEN not in proc.stderr


def test_planted_mixed_case_private_host_makes_the_export_refuse(tmp_path: Path) -> None:
    """The private-domain scan is case-insensitive and never echoes the match."""
    tree = _tree_with(tmp_path, "tests/host_probe.txt", f"host: {_PRIVATE_HOST}\n")
    proc = _dry_run(tmp_path, export_ref=tree)

    assert proc.returncode != 0
    assert "private host domain" in proc.stderr
    assert "tests/host_probe.txt" in proc.stderr
    assert _PRIVATE_HOST not in proc.stdout
    assert _PRIVATE_HOST.lower() not in (proc.stdout + proc.stderr).lower()


def test_planted_private_key_header_makes_the_export_refuse(tmp_path: Path) -> None:
    """A PEM private-key header in an allowlisted file stops the export."""
    tree = _tree_with(tmp_path, "tests/key_probe.txt", f"{_PEM_HEADER}\n")
    proc = _dry_run(tmp_path, export_ref=tree)

    assert proc.returncode != 0
    assert "private key header" in proc.stderr
    assert "tests/key_probe.txt" in proc.stderr
    assert _PEM_HEADER not in proc.stdout
    assert _PEM_HEADER not in proc.stderr
