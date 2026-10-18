"""Stand-ins for ``vllm``, ``qwen_vl_utils``, ``requests``, ``tqdm``, and ``flash_attn``."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from types import ModuleType, SimpleNamespace
from typing import Any

from tests.fakes import torch_fake
from tests.fakes.module_factory import make_module
from tests.fakes.recorder import RECORDER, RESPONDER, SETTINGS, Snapshottable, rng_summary, snapshot
from tests.fakes.transformers_fake import FakeImage


# --- vllm -----------------------------------------------------------------------------
class SamplingParams(Snapshottable):
    """Fake ``vllm.SamplingParams`` that keeps its keyword arguments."""

    def __init__(self, **kwargs: Any) -> None:
        """Record the sampling parameters."""
        RECORDER.record("SamplingParams", kwargs=kwargs)
        self.kwargs = kwargs

    def snapshot(self) -> dict[str, Any]:
        """Describe the parameters by their values."""
        return {"SamplingParams": snapshot(self.kwargs)}


class LLM:
    """Fake ``vllm.LLM`` engine."""

    def __init__(self, **kwargs: Any) -> None:
        """Record the engine arguments."""
        RECORDER.record("LLM", kwargs=kwargs, rng=rng_summary(torch_fake))

    def generate(self, prompts: Any, sampling_params: Any = None, **kwargs: Any) -> list[SimpleNamespace]:
        """Record the request and answer with the next scripted response."""
        RECORDER.record("LLM.generate", prompts=prompts, sampling_params=sampling_params, kwargs=kwargs, rng=rng_summary(torch_fake))
        torch_fake.consume_random_draw()
        completion = SimpleNamespace(text=RESPONDER.next())
        return [SimpleNamespace(outputs=[completion])]


# --- qwen_vl_utils ----------------------------------------------------------------------
def process_vision_info(conversations: list[dict[str, Any]]) -> tuple[list[FakeImage] | None, None]:
    """Record the call and return the images found in the message contents.

    Mirrors the real contract: ``(image_inputs, video_inputs)`` where each element is
    ``None`` when the conversation contains no such inputs.
    """
    RECORDER.record("process_vision_info", conversations=conversations)
    images = []
    for message in conversations:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image":
                source = part["image"].source if isinstance(part["image"], FakeImage) else str(part["image"])
                images.append(FakeImage(f"processed:{source}"))
    return (images or None), None


# --- requests ---------------------------------------------------------------------------
class RequestException(Exception):  # noqa: N818 - mirrors the real ``requests`` name.
    """Fake base exception of ``requests``."""


def _response(url: str) -> SimpleNamespace:
    """Return a response whose status code comes from ``SETTINGS``."""
    if SETTINGS["requests_raise"]:
        raise RequestException(f"fake failure for {url}")
    return SimpleNamespace(status_code=SETTINGS["requests_status_code"], content=f"bytes-of:{url}".encode())


def requests_get(url: str, **kwargs: Any) -> SimpleNamespace:
    """Record ``requests.get`` and return a fake response."""
    RECORDER.record("requests.get", url=url, kwargs=kwargs)
    return _response(url)


def requests_head(url: str, **kwargs: Any) -> SimpleNamespace:
    """Record ``requests.head`` and return a fake response."""
    RECORDER.record("requests.head", url=url, kwargs=kwargs)
    return _response(url)


# --- tqdm -------------------------------------------------------------------------------
class tqdm:  # noqa: N801 - mirrors the real ``tqdm.tqdm`` name.
    """Progress-bar stand-in: iterates transparently and prints ``write`` to stdout."""

    def __init__(self, iterable: Iterable[Any], **kwargs: Any) -> None:
        """Wrap ``iterable``; progress-bar options are ignored."""
        self.iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the wrapped iterable."""
        return iter(self.iterable)

    @staticmethod
    def write(text: str, file: Any = None, end: str = "\n", nolock: bool = False) -> None:
        """Write ``text`` like the real ``tqdm.write`` (stdout unless ``file`` is given)."""
        stream = file if file is not None else sys.stdout
        stream.write(text)
        stream.write(end)


def build_modules(flash_attn_available: bool) -> dict[str, ModuleType | None]:
    """Return fresh fake modules; ``flash_attn`` maps to ``None`` when unavailable.

    A ``None`` entry in :data:`sys.modules` makes ``import flash_attn`` raise
    :class:`ImportError`, which is how the wrappers detect a missing installation.
    """
    vllm = make_module("vllm", LLM=LLM, SamplingParams=SamplingParams)
    qwen_vl_utils = make_module("qwen_vl_utils", process_vision_info=process_vision_info)
    requests = make_module("requests", get=requests_get, head=requests_head, RequestException=RequestException)
    tqdm_module = make_module("tqdm", tqdm=tqdm)

    return {
        "vllm": vllm,
        "qwen_vl_utils": qwen_vl_utils,
        "requests": requests,
        "tqdm": tqdm_module,
        "flash_attn": ModuleType("flash_attn") if flash_attn_available else None,
    }
