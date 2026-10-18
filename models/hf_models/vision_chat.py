"""Base class for image-text chat models driven by a ``transformers`` processor."""

from __future__ import annotations

from typing import Any

from models.hf_models.base import HfModelWrapper, apply_default_generation_kwargs
from models.image_loading import load_images
from models.prompting import content_message, ensure_enough_images, image_placeholder_parts, remove_image_placeholders


class HfVisionChatModel(HfModelWrapper):
    """Image-text chat model whose processor renders messages with image parts.

    :meth:`generate_response` validates the ``<image>`` placeholders, builds chat
    messages with one ``{"type": "image"}`` part per image, renders them with the
    processor's chat template, loads the images, and decodes the new tokens.
    Subclasses load ``_model`` and ``_processor`` and override the hooks
    :meth:`_prepare_text`, :meth:`_render_prompt`, :meth:`_to_device`, and
    :meth:`_apply_token_defaults` where a model differs.

    Attributes:
        image_dir: Directory that relative image paths are resolved against.
    """

    _processor: Any

    def __init__(
        self,
        pretrained: str,
        image_dir: str | None,
        device: str,
        device_map: str | dict[str, Any],
        batch_size: int,
        use_cache: bool,
    ) -> None:
        """Store the settings shared by the image-text wrappers.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path.
            image_dir: Directory that relative image paths are resolved against.
            device: Device that inputs are moved to, such as ``"cuda"``.
            device_map: Device placement passed to ``from_pretrained``.
            batch_size: Batch size reported by :attr:`batch_size`.
            use_cache: Default for the ``use_cache`` generation setting.
        """
        super().__init__(pretrained, device=device, device_map=device_map, batch_size=batch_size, use_cache=use_cache)
        self.image_dir = image_dir

    @staticmethod
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """Return the system message for ``system_prompt``."""
        return [content_message("system", system_prompt)]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the user message with one image part per entry of ``zero_shot_data["images"]``."""
        return [content_message("user", zero_shot_data["user"], image_placeholder_parts(zero_shot_data.get("images")))]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return alternating user and assistant messages for the few-shot examples.

        A user message gets one image part per entry of the example's ``"images"``;
        the images themselves are not loaded.
        """
        messages = []
        for example in few_shot_data:
            messages.append(content_message("user", example["user"], image_placeholder_parts(example.get("images"))))
            messages.append(content_message("assistant", example["assistant"]))
        return messages

    def build_messages(
        self, text: str, images: list[str] | None, system_prompt: str | None, few_shot_data: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Return the chat messages: system prompt, few-shot examples, then the query with its images."""
        messages = []
        if system_prompt:
            messages += self.create_system_prompt(system_prompt)
        if few_shot_data:
            messages += self.create_few_shot_prompt(few_shot_data)
        messages += self.create_zero_shot_prompt({"user": text, "images": images})
        return messages

    def _prepare_text(self, text: str, images: list[str] | None) -> str:
        """Validate and remove ``<image>`` placeholders; the images become separate message parts.

        Raises:
            ValueError: If ``text`` has more placeholders than there are images.
        """
        if images is None:
            return text
        ensure_enough_images(text, images)
        return remove_image_placeholders(text)

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Render ``messages`` with the processor's chat template."""
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _to_device(self, inputs: Any) -> Any:
        """Move the processed inputs to :attr:`device`."""
        return inputs.to(self._device)

    def _apply_token_defaults(self, gen_kwargs: dict[str, Any]) -> None:
        """Default the padding and end-of-sequence token ids to the tokenizer's."""
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

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
                per image.
            images: Image URLs or paths relative to :attr:`image_dir`.
            system_prompt: Task instructions placed before the examples.
            few_shot_data: Examples with ``"user"``, ``"assistant"``, and optional
                ``"images"`` entries.
            gen_kwargs: Settings for ``generate``. Missing keys are filled with the
                defaults (1024 new tokens, sampling with temperature 0.05, ...). The dictionary is not modified.

        Returns:
            The generated text without the prompt.

        Raises:
            ValueError: If ``text`` has more ``<image>`` placeholders than images.
            FileNotFoundError: If a local image does not exist.
        """
        # Work on a copy: the tasks pass the same dictionary for every record.
        gen_kwargs = dict(gen_kwargs) if gen_kwargs is not None else {}
        text = self._prepare_text(text, images)
        messages = self.build_messages(text, images, system_prompt, few_shot_data)
        prompt = self._render_prompt(messages)
        visuals = load_images(images, self.image_dir)
        inputs = self._to_device(self._processor(text=prompt, images=visuals, padding=True, return_tensors="pt"))

        apply_default_generation_kwargs(gen_kwargs, self.use_cache)
        self._apply_token_defaults(gen_kwargs)

        return self._generate_and_decode(inputs, gen_kwargs, decoder=self._processor)
