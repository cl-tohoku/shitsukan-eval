"""Shared pytest fixtures: fake third-party libraries and fresh repository imports."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest

from tests.fakes import fake_modules, purge_repository_modules, reset_fakes


@pytest.fixture
def fake_libraries(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[None]:
    """Replace heavy dependencies with recording fakes for one test.

    Parametrize indirectly with ``True`` to simulate an installed ``flash_attn``;
    by default it is missing. Repository modules are purged before and after the
    test so that every test imports them freshly against the fakes, and the OpenAI
    API key is removed from the environment.
    """
    flash_attn_available = getattr(request, "param", False)
    reset_fakes()
    for name, module in fake_modules(flash_attn_available).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    purge_repository_modules()
    yield
    purge_repository_modules()
    reset_fakes()


@pytest.fixture
def import_fresh(fake_libraries: None) -> Callable[[str], ModuleType]:
    """Return a function that imports a repository module against the fakes."""
    return importlib.import_module
