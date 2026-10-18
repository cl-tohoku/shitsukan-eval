"""Creation of module objects for the fake libraries."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def make_module(name: str, **attributes: Any) -> ModuleType:
    """Return a new module called ``name`` with the given attributes."""
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module
