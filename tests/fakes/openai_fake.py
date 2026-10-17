"""Minimal stand-in for the :mod:`openai` client used by the characterization tests.

``chat.completions.create`` records its arguments and answers with the next
scripted response. Tests can make the first calls fail by listing exception class
names in ``SETTINGS["openai_failures"]``. The API key is never recorded; events only
show whether one was provided.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

from tests.fakes.recorder import RECORDER, RESPONDER, SETTINGS, prompt_text, snapshot


class APIError(Exception):
    """Base class of the fake OpenAI errors."""


class APIConnectionError(APIError):
    """Fake connection error."""


class APIStatusError(APIError):
    """Fake error for a non-success HTTP status."""


class RateLimitError(APIStatusError):
    """Fake HTTP 429 error."""


class NotFoundError(APIStatusError):
    """Fake HTTP 404 error."""


ERRORS = {cls.__name__: cls for cls in (APIConnectionError, APIStatusError, RateLimitError, NotFoundError)}


class _Completions:
    """Fake ``client.chat.completions`` resource."""

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Record the request; fail or answer with the next scripted response."""
        RECORDER.record("chat.completions.create", kwargs=kwargs)
        failures = SETTINGS["openai_failures"]
        if failures:
            name = failures.pop(0)
            raise ERRORS[name](f"fake {name}")
        RESPONDER.remember_prompt(prompt_text(snapshot(kwargs.get("messages"))))
        message = SimpleNamespace(content=RESPONDER.next())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AsyncOpenAI:
    """Fake asynchronous OpenAI client."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Record the client creation without recording the key itself."""
        RECORDER.record("AsyncOpenAI", api_key_provided=api_key is not None, kwargs=kwargs)
        self.chat = SimpleNamespace(completions=_Completions())


def build_modules() -> dict[str, ModuleType]:
    """Return a fresh ``openai`` module."""
    openai = ModuleType("openai")
    openai.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    for name, cls in ERRORS.items():
        setattr(openai, name, cls)
    openai.APIError = APIError  # type: ignore[attr-defined]
    return {"openai": openai}
