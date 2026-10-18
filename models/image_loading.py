"""Loading the images that dataset records reference by URL or relative path."""

from __future__ import annotations

import os
import pathlib
from typing import Any, cast

from transformers.image_utils import load_image

from tools.utils.data_utils import is_url


def load_images(image_paths: list[str] | None, image_dir: str | None) -> list[Any] | None:
    """Load images given as URLs or as paths relative to ``image_dir``.

    Args:
        image_paths: ``http(s)`` URLs or paths relative to ``image_dir``, or ``None``.
        image_dir: Directory that relative paths are resolved against. Required
            when ``image_paths`` contains a relative path.

    Returns:
        The loaded images in input order, or ``None`` when ``image_paths`` is ``None``.

    Raises:
        FileNotFoundError: If a relative path does not exist under ``image_dir``.
        TypeError: If a relative path is given while ``image_dir`` is ``None``.
    """
    if image_paths is None:
        return None
    images = []
    for image_path in image_paths:
        if is_url(image_path):
            images.append(load_image(image_path))
            continue
        # ``pathlib`` raises TypeError for a missing ``image_dir``; the cast only informs the type checker.
        local_path = str(pathlib.Path(cast("str", image_dir)).joinpath(image_path))
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"The file path does not exist: {image_path}")
        images.append(load_image(local_path))
    return images
