"""Shared interactive-app test configuration.

The shared tests import ``code_interactive`` by default. Set
``INTERACTIVE_TEST_PACKAGE=code_interactive_v2`` to run the same tests against
the v2 candidate package without maintaining a duplicate test tree.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _select_interactive_package() -> None:
    target = os.environ.get("INTERACTIVE_TEST_PACKAGE", "code_interactive").strip()
    if not target:
        target = "code_interactive"
    if target not in {"code_interactive", "code_interactive_v2"}:
        raise RuntimeError(
            "INTERACTIVE_TEST_PACKAGE must be 'code_interactive' or "
            f"'code_interactive_v2', got {target!r}."
        )
    if target == "code_interactive":
        return

    for name in list(sys.modules):
        if name == "code_interactive" or name.startswith("code_interactive."):
            del sys.modules[name]

    target_module = importlib.import_module(target)
    sys.modules["code_interactive"] = target_module


_select_interactive_package()
