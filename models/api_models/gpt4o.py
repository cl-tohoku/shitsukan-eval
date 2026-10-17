"""Wrapper for OpenAI chat models (GPT-4o and other GPT-4 variants).

The API key is read from the ``OPENAI_API_KEY`` environment variable when this
module is imported. Requests that fail with connection, rate-limit, or status
errors are retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import base64
import os
import pathlib
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

import requests
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from models.prompting import IMAGE_PLACEHOLDER, content_message, ensure_enough_images, text_message
from tools.utils.data_utils import is_url, set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

T = TypeVar("T")

API_KEY = os.getenv("OPENAI_API_KEY")

MAX_NUM_TRIALS = 5
FIRST_WAIT_TIME_SECONDS = 10
HTTP_OK = 200

# Request parameters used when the task configuration does not set them. The
# fixed seed asks the API for best-effort deterministic sampling; determinism is
# not guaranteed (see ``system_fingerprint`` in the API reference).
DEFAULT_REQUEST_KWARGS: dict[str, Any] = {
    "max_completion_tokens": 1024,
    "n": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "logit_bias": None,
    "logprobs": False,
    "top_logprobs": None,
    "seed": 42,
    "temperature": 0.0,
    "top_p": 1.0,
}


class GPT4o:
    """OpenAI chat model for text and image inputs.

    Images are sent inline as base64 data URLs. ``<image>`` placeholders in the
    query text are validated against the number of images but kept in the text.

    Attributes:
        pretrained: OpenAI model name, such as ``"gpt-4o"``.
        image_token: Placeholder counted in the query text.
        image_dir: Directory that relative image paths are resolved against.

    Example:
        >>> model = GPT4o(pretrained="gpt-4o", image_dir="data")
        >>> model.generate_response("Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    def __init__(self, pretrained: str = "gpt-4o", image_dir: str | None = None, **kwargs: Any) -> None:
        """Configure the model; no request is sent here.

        Args:
            pretrained: OpenAI model name, such as ``"gpt-4o"``.
            image_dir: Directory that relative image paths are resolved against.
            **kwargs: Ignored.
        """
        self.pretrained = pretrained
        self.image_token = IMAGE_PLACEHOLDER
        self.image_dir = image_dir

    @staticmethod
    def rename_key_for_openai_api(gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate ``transformers``-style settings into OpenAI request parameters.

        ``max_new_tokens`` becomes ``max_completion_tokens``; ``do_sample`` and
        ``num_beams`` are dropped. ``gen_kwargs`` is modified in place and returned.

        Raises:
            KeyError: If ``max_new_tokens`` is missing.
        """
        gen_kwargs["max_completion_tokens"] = gen_kwargs.pop("max_new_tokens")
        gen_kwargs.pop("do_sample", None)
        gen_kwargs.pop("num_beams", None)
        return gen_kwargs

    @staticmethod
    def image_to_base64(input_path: str, image_dir: str | None) -> str:
        """Return the base64 encoding of an image given by URL or by relative path.

        Args:
            input_path: ``http(s)`` URL or path relative to ``image_dir``.
            image_dir: Directory that relative paths are resolved against.

        Returns:
            The image bytes encoded as base64 text.

        Raises:
            Exception: If downloading the URL does not return HTTP 200.
            FileNotFoundError: If the local file does not exist (its path is printed first).
        """
        if is_url(input_path):
            # No timeout is set, as in the runs reported in the paper.
            response = requests.get(input_path)
            if response.status_code != HTTP_OK:
                raise Exception(f"Failed to retrieve the image from URL. Status code: {response.status_code}") from None
            image_data = response.content
        else:
            # ``pathlib`` raises TypeError for a missing ``image_dir``; the cast only informs the type checker.
            local_path = pathlib.Path(cast("str", image_dir)).joinpath(input_path)
            if not os.path.exists(str(local_path)):
                print(str(local_path))
                raise FileNotFoundError(f"The file path does not exist: {input_path}") from None
            with open(str(local_path), "rb") as image_file:
                image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")

    @staticmethod
    def _image_parts(images: list[str] | None, image_dir: str | None) -> list[dict[str, Any]]:
        """Return one ``image_url`` content part per image, embedding it as a JPEG data URL."""
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{GPT4o.image_to_base64(image, image_dir)}"}} for image in images or ()]

    @staticmethod
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """Return the system message for ``system_prompt``."""
        return [text_message("system", system_prompt)]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any], img_dir: str | None = None) -> list[dict[str, Any]]:
        """Return the user message with the images of ``zero_shot_data`` followed by its text.

        Args:
            zero_shot_data: ``{"user": text, "images": image paths or None}``.
            img_dir: Directory that relative image paths are resolved against.
        """
        return [content_message("user", zero_shot_data["user"], GPT4o._image_parts(zero_shot_data.get("images"), img_dir))]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]], img_dir: str | None = None) -> list[dict[str, Any]]:
        """Return alternating user and assistant messages for the few-shot examples.

        Args:
            few_shot_data: Examples with ``"user"``, ``"assistant"``, and optional ``"images"``.
            img_dir: Directory that relative image paths are resolved against.
        """
        messages = []
        for example in few_shot_data:
            messages.append(content_message("user", example["user"], GPT4o._image_parts(example.get("images"), img_dir)))
            messages.append(content_message("assistant", example["assistant"]))
        return messages

    @staticmethod
    async def retry_on_error(
        openai_call: Callable[[], Awaitable[T]],
        max_num_trials: int = MAX_NUM_TRIALS,
        first_wait_time: int = FIRST_WAIT_TIME_SECONDS,
    ) -> T:
        """Await ``openai_call``, retrying transient API errors with exponential backoff.

        The wait before retry ``i`` (0-based) is ``first_wait_time * 2**i`` seconds.

        Args:
            openai_call: Function that starts one API request.
            max_num_trials: Maximum number of attempts; must be at least 1.
            first_wait_time: Wait in seconds before the first retry.

        Returns:
            The result of the first successful attempt.

        Raises:
            APIConnectionError: If the last attempt fails to connect.
            RateLimitError: If the last attempt is rate limited.
            APIStatusError: If the last attempt fails with another HTTP status.
            ValueError: If the API reports that the model was not found, or if
                ``max_num_trials`` is less than 1.
        """
        for attempt in range(max_num_trials):
            try:
                return await openai_call()
            except (APIConnectionError, RateLimitError, APIStatusError) as error:
                if attempt == max_num_trials - 1:
                    raise
                print(f"Error received: {error}")
                if error.__class__.__name__ == "NotFoundError":
                    raise ValueError("The specified model may not be supported.") from None
                wait_time_seconds = first_wait_time * (2**attempt)
                print(f"Waiting for {wait_time_seconds} seconds before retrying...")
                await asyncio.sleep(wait_time_seconds)
        raise ValueError(f"max_num_trials must be at least 1, got {max_num_trials}.")

    async def _async_batch_run_chatgpt(self, messages: list[list[dict[str, Any]]], **gen_kwargs: Any) -> list[str]:
        """Send one chat request per message list concurrently and return the answers in order."""
        client = AsyncOpenAI(api_key=API_KEY)
        requests_in_flight = [
            GPT4o.retry_on_error(openai_call=lambda conversation=conversation: client.chat.completions.create(messages=conversation, **gen_kwargs))
            for conversation in messages
        ]
        completions = await asyncio.gather(*requests_in_flight)
        return [completion.choices[0].message.content for completion in completions]

    def batch_run_chatgpt(self, messages: list[list[dict[str, Any]]], **gen_kwargs: Any) -> list[str]:
        """Synchronously run :meth:`_async_batch_run_chatgpt`.

        Args:
            messages: One message list (conversation) per request.
            **gen_kwargs: Request parameters such as ``model`` and ``temperature``.

        Returns:
            The answer text of each request, in input order.
        """
        return asyncio.run(self._async_batch_run_chatgpt(messages, **gen_kwargs))

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate the model's answer to ``text`` and ``images``.

        Args:
            text: Query text. It may contain ``<image>`` placeholders, at most one
                per image; they are sent unchanged.
            images: Image URLs or paths relative to :attr:`image_dir`.
            system_prompt: Task instructions sent as the system message.
            few_shot_data: Examples with ``"user"``, ``"assistant"``, and optional
                ``"images"`` entries.
            gen_kwargs: Request settings. ``transformers``-style keys are
                translated (see :meth:`rename_key_for_openai_api`) and missing keys
                are filled from :data:`DEFAULT_REQUEST_KWARGS`.

        Returns:
            The answer text.

        Raises:
            ValueError: If ``text`` has more ``<image>`` placeholders than images,
                or if the model is not found.
        """
        gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        if gen_kwargs.get("max_new_tokens", None):
            gen_kwargs = GPT4o.rename_key_for_openai_api(gen_kwargs)

        if images is not None:
            ensure_enough_images(text, images, self.image_token)

        message = []
        if system_prompt:
            message += GPT4o.create_system_prompt(system_prompt)
        if few_shot_data:
            message += GPT4o.create_few_shot_prompt(few_shot_data, self.image_dir)
        message += GPT4o.create_zero_shot_prompt({"user": text, "images": images}, self.image_dir)

        gen_kwargs.setdefault("model", self.pretrained)
        for key, value in DEFAULT_REQUEST_KWARGS.items():
            gen_kwargs.setdefault(key, value)

        return self.batch_run_chatgpt(messages=[message], **gen_kwargs)[0]
