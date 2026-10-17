"""Golden-record helpers for the characterization tests.

A golden record is a JSON file under ``tests/golden`` that stores what the code
observably did for a scenario: recorded calls into the (fake) model libraries,
return values, files written, standard output, and raised errors. The records were
generated from the original implementation and act as observational records
(characterization tests) while the code is refactored.

Set the environment variable ``UPDATE_GOLDEN=1`` to (re)write the records instead of
comparing against them. Only do this for intentional behavior changes, and review
the resulting diff.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from tests.fakes.recorder import snapshot

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_ENVIRONMENT_VARIABLE = "UPDATE_GOLDEN"

# Floats are compared with a relative tolerance so that records stay valid across
# NumPy/SciPy versions whose results may differ in the last bits.
FLOAT_RELATIVE_TOLERANCE = 1e-9
FLOAT_ABSOLUTE_TOLERANCE = 1e-12
_FLOAT_IN_TEXT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
_SIGNIFICANT_DIGITS_IN_TEXT = ".10g"


def normalize_float_text(text: str) -> str:
    """Round every decimal number inside ``text`` to 10 significant digits.

    Example: ``"accuracy": 0.6666666666666666`` becomes ``"accuracy": 0.6666666667``.
    """
    return _FLOAT_IN_TEXT.sub(lambda match: format(float(match.group()), _SIGNIFICANT_DIGITS_IN_TEXT), text)


def to_record(value: Any, replacements: dict[str, str] | None = None) -> Any:
    """Convert ``value`` into plain JSON data, replacing machine-specific substrings.

    Args:
        value: Data to convert; fake objects are snapshotted.
        replacements: Substrings to replace in every string, typically the absolute
            temporary directory of the test mapped to ``"<tmp>"``.
    """
    plain = json.loads(json.dumps(snapshot(value), ensure_ascii=False))
    if not replacements:
        return plain
    return _replace_in_strings(plain, replacements)


def _replace_in_strings(value: Any, replacements: dict[str, str]) -> Any:
    """Apply ``replacements`` to every string nested in ``value``."""
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_in_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_in_strings(item, replacements) for key, item in value.items()}
    return value


def text_lines(text: str) -> list[str]:
    """Split captured text into lines with decimal numbers normalized."""
    return [normalize_float_text(line) for line in text.splitlines()]


def collect_files(root: Path) -> dict[str, list[str]]:
    """Return every file below ``root`` as ``{relative path: normalized lines}``."""
    return {path.relative_to(root).as_posix(): text_lines(path.read_text(encoding="utf-8")) for path in sorted(root.rglob("*")) if path.is_file()}


def group_events_by_call(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group events by call name, preserving the order within each call name.

    Comparing grouped events checks every argument and the number and order of
    calls of each kind (for example, one ``generate`` per dataset item in dataset
    order) without freezing how unrelated calls interleave.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["call"], []).append(event)
    return grouped


def assert_equivalent(expected: Any, actual: Any, where: str = "$") -> None:
    """Assert that two JSON values are equal, comparing floats with a tolerance.

    Raises:
        AssertionError: With the JSON path of the first difference.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert expected == actual, f"{where}: expected {expected!r}, got {actual!r}"
    elif isinstance(expected, int | float) and isinstance(actual, int | float):
        assert math.isclose(expected, actual, rel_tol=FLOAT_RELATIVE_TOLERANCE, abs_tol=FLOAT_ABSOLUTE_TOLERANCE), (
            f"{where}: expected {expected!r}, got {actual!r}"
        )
    elif isinstance(expected, dict) and isinstance(actual, dict):
        assert expected.keys() == actual.keys(), (
            f"{where}: keys differ; missing {sorted(expected.keys() - actual.keys())}, unexpected {sorted(actual.keys() - expected.keys())}"
        )
        for key in expected:
            assert_equivalent(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual), f"{where}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            assert_equivalent(expected_item, actual_item, f"{where}[{index}]")
    else:
        assert expected == actual, f"{where}: expected {expected!r}, got {actual!r}"


def _comparable(record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record`` with its ``events`` grouped by call name."""
    comparable = dict(record)
    if "events" in comparable:
        comparable["events"] = group_events_by_call(comparable["events"])
    return comparable


def check_golden(name: str, observed: dict[str, Any]) -> None:
    """Compare ``observed`` with the golden record ``name`` (or write it when updating).

    Args:
        name: Record path below ``tests/golden`` without the ``.json`` suffix.
        observed: JSON-compatible observation of one scenario.

    Raises:
        AssertionError: If the record is missing or differs from ``observed``.
    """
    path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get(UPDATE_ENVIRONMENT_VARIABLE) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(observed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    assert path.exists(), f"Golden record {path} is missing; run the tests with {UPDATE_ENVIRONMENT_VARIABLE}=1 on the reference code."
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert_equivalent(_comparable(expected), _comparable(observed))
