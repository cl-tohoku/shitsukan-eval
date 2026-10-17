"""Minimal stand-in for :mod:`torch` used by the characterization tests.

Only the attributes touched by the evaluation code are provided. Tensors are
symbolic: they carry a name and a shape, and every operation that matters for the
generated text (moving inputs to a device, slicing off the prompt tokens) is
recorded instead of computed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tests.fakes.recorder import RECORDER, Snapshottable


class dtype(Snapshottable):  # noqa: N801 - mirrors the real ``torch.dtype`` name.
    """Symbolic tensor data type such as ``torch.bfloat16``."""

    def __init__(self, name: str) -> None:
        """Create a dtype called ``name``."""
        self.name = name

    def snapshot(self) -> str:
        """Describe the dtype as ``torch.<name>``."""
        return f"torch.{self.name}"


bfloat16 = dtype("bfloat16")
float16 = dtype("float16")
float32 = dtype("float32")


class device(Snapshottable):  # noqa: N801 - mirrors the real ``torch.device`` name.
    """Symbolic device such as ``cuda``."""

    def __init__(self, spec: Any) -> None:
        """Create a device from ``spec`` (a string or another device)."""
        self.spec = spec.spec if isinstance(spec, device) else str(spec)

    def snapshot(self) -> str:
        """Describe the device as ``device(<spec>)``."""
        return f"device({self.spec})"


def _describe_index(key: Any) -> str:
    """Render a tensor index such as ``(slice(None), slice(3, None))`` as ``:, 3:``."""
    parts = key if isinstance(key, tuple) else (key,)
    rendered = []
    for part in parts:
        if isinstance(part, slice):
            start = "" if part.start is None else str(part.start)
            stop = "" if part.stop is None else str(part.stop)
            rendered.append(f"{start}:{stop}")
        else:
            rendered.append(str(part))
    return ", ".join(rendered)


class Tensor(Snapshottable):
    """Symbolic tensor that records the operations applied to it."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        """Create a tensor called ``name`` with the given ``shape``."""
        self.name = name
        self.shape = shape

    def snapshot(self) -> dict[str, Any]:
        """Describe the tensor by name and shape."""
        return {"tensor": self.name, "shape": list(self.shape)}

    def size(self, dim: int) -> int:
        """Return the length of dimension ``dim``."""
        return self.shape[dim]

    def to(self, *args: Any, **kwargs: Any) -> Tensor:
        """Record a device/dtype move and return the same tensor."""
        RECORDER.record("Tensor.to", tensor=self.name, args=args, kwargs=kwargs)
        return self

    def unsqueeze(self, dim: int) -> Tensor:
        """Record ``unsqueeze`` and return a tensor with an extra leading dimension."""
        RECORDER.record("Tensor.unsqueeze", tensor=self.name, dim=dim)
        return Tensor(f"{self.name}.unsqueeze({dim})", (1, *self.shape))

    def __getitem__(self, key: Any) -> Tensor:
        """Record the index expression and return a symbolic slice."""
        description = _describe_index(key)
        RECORDER.record("Tensor.__getitem__", tensor=self.name, index=description)
        return Tensor(f"{self.name}[{description}]", self.shape)


class no_grad:  # noqa: N801 - mirrors the real ``torch.no_grad`` name.
    """Context manager standing in for :class:`torch.no_grad`."""

    def __enter__(self) -> None:
        """Record entering the no-grad block."""
        RECORDER.record("torch.no_grad.enter")

    def __exit__(self, *exc_info: object) -> None:
        """Record leaving the no-grad block."""
        RECORDER.record("torch.no_grad.exit")


class autocast:  # noqa: N801 - mirrors the real ``torch.autocast`` name.
    """Context manager standing in for :class:`torch.autocast`."""

    def __init__(self, **kwargs: Any) -> None:
        """Record the autocast arguments."""
        RECORDER.record("torch.autocast", kwargs=kwargs)

    def __enter__(self) -> None:
        """Enter the autocast block."""

    def __exit__(self, *exc_info: object) -> None:
        """Leave the autocast block."""


def manual_seed(seed: int) -> None:
    """Record a call to :func:`torch.manual_seed` and restart the draw counter."""
    RECORDER.record("torch.manual_seed", seed=seed)
    SEED_STATE["torch"] = seed
    SEED_STATE["draws_since_seed"] = 0


def consume_random_draw() -> None:
    """Count one sampling step, as real generation consumes the torch RNG."""
    SEED_STATE["draws_since_seed"] += 1


def _cuda_manual_seed_all(seed: int) -> None:
    """Record a call to :func:`torch.cuda.manual_seed_all`."""
    RECORDER.record("torch.cuda.manual_seed_all", seed=seed)
    SEED_STATE["cuda"] = seed


def use_deterministic_algorithms(mode: bool) -> None:  # noqa: FBT001 - mirrors the real signature.
    """Record a call to :func:`torch.use_deterministic_algorithms`."""
    RECORDER.record("torch.use_deterministic_algorithms", mode=mode)


# Seeds most recently passed to the fake RNG functions and the number of sampling
# steps since the last ``manual_seed``; inspected by tests to check seeding order.
SEED_STATE: dict[str, int | None] = {"torch": None, "cuda": None, "draws_since_seed": 0}

cuda = SimpleNamespace(manual_seed_all=_cuda_manual_seed_all)
backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False))
ORIGINAL_USE_DETERMINISTIC_ALGORITHMS = use_deterministic_algorithms


def reset() -> None:
    """Restore the mutable module state changed by the code under test."""
    global use_deterministic_algorithms  # noqa: PLW0603 - the code under test assigns this attribute.
    use_deterministic_algorithms = ORIGINAL_USE_DETERMINISTIC_ALGORITHMS
    backends.cudnn.deterministic = False
    SEED_STATE.update({"torch": None, "cuda": None, "draws_since_seed": 0})
