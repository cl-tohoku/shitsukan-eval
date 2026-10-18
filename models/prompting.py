"""Chat-message helpers shared by the model wrappers.

The tasks call every wrapper with the same inputs: an optional system prompt,
optional few-shot examples, and a query ``text`` with optional ``images``. A
few-shot example is a mapping ``{"user": str, "assistant": str, "images":
list[str] | None}``. The helpers here turn these inputs into the message lists
that the wrappers pass to chat templates. They are pure: images are referenced,
never loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

IMAGE_PLACEHOLDER = "<image>"
"""Placeholder that a query text may contain to mark where an image belongs."""


def ensure_enough_images(text: str, images: Sequence[Any], placeholder: str = IMAGE_PLACEHOLDER) -> int:
    """Check that ``text`` does not reference more images than ``images`` provides.

    Args:
        text: Query text that may contain ``placeholder``.
        images: Images passed together with ``text``.
        placeholder: Marker counted in ``text``.

    Returns:
        The number of placeholders in ``text``.

    Raises:
        ValueError: If ``text`` contains more placeholders than there are images.
    """
    placeholder_count = text.count(placeholder)
    if placeholder_count > len(images):
        raise ValueError(
            f"Error! The number of image tokens (img_token_num={placeholder_count}) is greater than the number of input images (len(images)={len(images)})."
        )
    return placeholder_count


def remove_image_placeholders(text: str, placeholder: str = IMAGE_PLACEHOLDER) -> str:
    """Return ``text`` without ``placeholder`` (the wrappers attach images separately)."""
    return text.replace(placeholder, "")


def text_message(role: str, text: str) -> dict[str, Any]:
    """Return a message whose content is a plain string."""
    return {"role": role, "content": text}


def image_placeholder_parts(images: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Return one ``{"type": "image"}`` content part per image (none for ``None`` or empty)."""
    return [{"type": "image"} for _ in images or ()]


def content_message(role: str, text: str, leading_parts: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Return a message whose content is a list of parts ending with ``text``.

    Args:
        role: Message role such as ``"user"``.
        text: Text placed in the last content part.
        leading_parts: Parts placed before the text, typically images.
    """
    return {"role": role, "content": [*leading_parts, {"type": "text", "text": text}]}
