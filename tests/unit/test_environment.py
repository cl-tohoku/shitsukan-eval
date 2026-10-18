"""Checks of the installed model libraries.

The other tests replace the model libraries with fakes. This module checks the
real ``transformers`` installation instead, so that an incompatible version is
reported before an evaluation run: for example, transformers 5 removed
``AutoModelForVision2Seq``, which the Idefics3 wrapper imports. The check is
skipped where ``transformers`` is not installed.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Runs in a separate interpreter so that the real libraries never mix with the
# fakes used by the other tests. Prints the names that are missing.
_CHECK_SCRIPT = """
import importlib
import sys

missing = []
for qualified_name in sys.argv[1:]:
    module_name, _, attribute = qualified_name.rpartition(".")
    if not hasattr(importlib.import_module(module_name), attribute):
        missing.append(qualified_name)
print(" ".join(missing))
sys.exit(1 if missing else 0)
"""


def transformers_names_imported_by_the_wrappers() -> list[str]:
    """Return every ``transformers`` name that ``models/`` imports, as ``module.name``."""
    names = set()
    for path in sorted((REPOSITORY_ROOT / "models").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "transformers":
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(names)


def test_the_wrappers_import_the_idefics3_loader_from_transformers():
    # Guards the check below against silently checking nothing.
    names = transformers_names_imported_by_the_wrappers()

    assert "transformers.AutoModelForVision2Seq" in names
    assert "transformers.image_utils.load_image" in names


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is not installed")
def test_installed_transformers_provides_every_name_the_wrappers_import():
    names = transformers_names_imported_by_the_wrappers()

    result = subprocess.run([sys.executable, "-c", _CHECK_SCRIPT, *names], capture_output=True, text=True, check=False, timeout=900)

    assert result.returncode == 0, f"Missing in the installed transformers: {result.stdout.strip()}\n{result.stderr[-2000:]}"
