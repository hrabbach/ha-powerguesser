"""Worker entrypoint script for PowerGuesser out-of-process engine stages.

Loads engine/ under the private package name 'powerguesser_engine' to avoid
importing custom_components.powerguesser.__init__ and Home Assistant.
"""

import importlib.util
from pathlib import Path
import sys

_ENGINE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "powerguesser_engine",
    _ENGINE / "__init__.py",
    submodule_search_locations=[str(_ENGINE)],
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load engine spec from {_ENGINE}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules["powerguesser_engine"] = _mod
_spec.loader.exec_module(_mod)

from powerguesser_engine.learning.worker import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
