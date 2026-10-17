"""Fake third-party libraries for hermetic characterization tests.

The evaluation code imports ``torch``, ``transformers``, ``openai``, ``vllm``, and
other heavy packages at module level. :func:`fake_modules` builds lightweight
replacements that record every interaction relevant to the generated text, so
the tests run without GPUs, model weights, network access, or API keys.
"""

from __future__ import annotations

import sys
from types import ModuleType

from tests.fakes import misc_fakes, openai_fake, torch_fake, transformers_fake
from tests.fakes.recorder import RECORDER, RESPONDER, reset_settings

# Top-level packages of this repository. They are purged from ``sys.modules`` so
# that each test imports them again against the fakes.
REPOSITORY_PACKAGES = ("models", "shitsukan_eval", "tools")


def fake_modules(flash_attn_available: bool) -> dict[str, ModuleType | None]:  # noqa: FBT001
    """Return the fake modules keyed by the import name they replace."""
    modules: dict[str, ModuleType | None] = {"torch": torch_fake}
    modules.update(transformers_fake.build_modules())
    modules.update(openai_fake.build_modules())
    modules.update(misc_fakes.build_modules(flash_attn_available))
    return modules


def purge_repository_modules() -> None:
    """Remove this repository's modules from :data:`sys.modules`."""
    for name in list(sys.modules):
        if name.split(".")[0] in REPOSITORY_PACKAGES:
            del sys.modules[name]


def reset_fakes() -> None:
    """Reset recorded events, scripted responses, settings, and fake torch state."""
    RECORDER.reset()
    RESPONDER.reset()
    reset_settings()
    torch_fake.reset()
