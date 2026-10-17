"""Call recorder and scripted responses shared by the fake third-party libraries.

The fakes in this package replace heavy dependencies (``torch``, ``transformers``,
``openai``, ``vllm``, ...) during the characterization tests. Every call that can
influence a model's output is appended to :data:`RECORDER` as a JSON-serializable
event, so a test can compare the full interaction with a stored golden record.
The text that a fake model "generates" comes from :data:`RESPONDER`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


class Snapshottable:
    """Mixin for fake objects that know how to describe themselves in a record."""

    def snapshot(self) -> Any:
        """Return a JSON-serializable description of the object."""
        raise NotImplementedError


def snapshot(value: Any) -> Any:
    """Convert ``value`` into a JSON-serializable structure for the golden record.

    Dictionaries keep their keys as strings, sequences become lists, fake objects
    describe themselves through :meth:`Snapshottable.snapshot`, and any other object
    is represented by its type name so that memory addresses never leak into records.
    """
    if isinstance(value, Snapshottable):
        return value.snapshot()
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [snapshot(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return {"object": type(value).__name__}


def prompt_text(value: Any) -> str:
    """Join every string nested in ``value`` with newlines.

    Used to show a response rule the text of a prompt regardless of whether it
    arrives as a plain string or as chat messages.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(prompt_text(item) for item in value.values())
    if isinstance(value, list | tuple):
        return "\n".join(prompt_text(item) for item in value)
    return ""


def digest(value: Any) -> str:
    """Return a short, stable SHA-256 digest of the snapshot of ``value``."""
    payload = json.dumps(snapshot(value), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def text_digest(text: str | None) -> str | None:
    """Return a SHA-256 digest of ``text`` (``None`` stays ``None``).

    Used for long strings such as chat templates, where any change matters but the
    full text would make golden records unreadable.
    """
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Recorder:
    """Ordered log of the events emitted by the fakes."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.events: list[dict[str, Any]] = []
        self._prompt_count = 0

    def record(self, call: str, **details: Any) -> None:
        """Append one event named ``call`` with its snapshotted ``details``."""
        self.events.append({"call": call, **{key: snapshot(item) for key, item in details.items()}})

    def next_prompt_marker(self) -> str:
        """Return a marker for the next rendered prompt: ``<prompt 1>``, ``<prompt 2>``, ...

        The marker links a recorded chat-template call to the later calls that
        receive its output, without depending on machine-specific paths.
        """
        self._prompt_count += 1
        return f"<prompt {self._prompt_count}>"

    def reset(self) -> None:
        """Forget all recorded events and restart prompt numbering."""
        self.events.clear()
        self._prompt_count = 0


RECORDER = Recorder()


class Responder:
    """Source of the texts returned by fake generation calls.

    A test either scripts an exact sequence of responses or installs a rule that
    derives the response from the most recent prompt seen by a fake.
    """

    def __init__(self) -> None:
        """Create a responder without scripted responses."""
        self._script: deque[str] = deque()
        self._rule: Callable[[str | None], str] | None = None
        self.last_prompt: str | None = None

    def script(self, responses: Iterable[str]) -> None:
        """Return ``responses`` in order from the following generation calls."""
        self._script = deque(responses)
        self._rule = None

    def use_rule(self, rule: Callable[[str | None], str]) -> None:
        """Derive every response from the latest prompt with ``rule``."""
        self._rule = rule
        self._script.clear()

    def remember_prompt(self, prompt: str) -> None:
        """Store the text of the prompt that the next response answers."""
        self.last_prompt = prompt

    def next(self) -> str:
        """Return the next response.

        Raises:
            AssertionError: If the code under test asks for more responses than the
                test scripted, which usually means an unexpected extra model call.
        """
        if self._rule is not None:
            return self._rule(self.last_prompt)
        if not self._script:
            raise AssertionError("The fake model was asked for more responses than the test scripted.")
        return self._script.popleft()

    def reset(self) -> None:
        """Forget scripted responses, the rule, and the last prompt."""
        self._script.clear()
        self._rule = None
        self.last_prompt = None


RESPONDER = Responder()

# Knobs that let a test choose how the fakes behave (e.g. whether a tokenizer
# already has a chat template). ``reset_settings`` restores the defaults.
DEFAULT_SETTINGS: dict[str, Any] = {
    "tokenizer_chat_template": None,
    "tokenizer_pad_token_id": 0,
    "config_has_vision_attributes": True,
    "openai_failures": [],
    "requests_status_code": 200,
    "requests_raise": False,
    "record_rng_state": False,
}
SETTINGS: dict[str, Any] = {}


def reset_settings() -> None:
    """Restore :data:`SETTINGS` to :data:`DEFAULT_SETTINGS`."""
    SETTINGS.clear()
    SETTINGS.update({key: list(value) if isinstance(value, list) else value for key, value in DEFAULT_SETTINGS.items()})


reset_settings()

_FRESH_PYTHON_STATE = random.Random(42).getstate()
_FRESH_NUMPY_STATE = np.random.RandomState(42).get_state()


def _numpy_state_is_fresh() -> bool:
    """Return whether NumPy's global RNG is exactly in the state after ``seed(42)``."""
    current = np.random.get_state()
    return current[0] == _FRESH_NUMPY_STATE[0] and np.array_equal(current[1], _FRESH_NUMPY_STATE[1]) and current[2:] == _FRESH_NUMPY_STATE[2:]


def rng_summary(torch_module: Any) -> dict[str, Any] | None:
    """Describe the global RNG state when :data:`SETTINGS` asks for it.

    The summary tells whether Python's and NumPy's global generators are in the
    state produced by seeding with 42 and which seed the fake ``torch`` received.
    Reading the state does not consume random numbers.
    """
    if not SETTINGS["record_rng_state"]:
        return None
    return {
        "python_random_fresh_seed_42": random.getstate() == _FRESH_PYTHON_STATE,
        "numpy_random_fresh_seed_42": _numpy_state_is_fresh(),
        "torch_seed": torch_module.SEED_STATE["torch"],
        "cuda_seed": torch_module.SEED_STATE["cuda"],
        "draws_since_torch_seed": torch_module.SEED_STATE["draws_since_seed"],
        "cudnn_deterministic": torch_module.backends.cudnn.deterministic,
    }
