"""Shared implementation of the Idefics2 and Idefics3 wrappers."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers import AutoProcessor

from models.hf_models.base import BEST_FIT_ATTN_IMPLEMENTATION, resolve_torch_dtype
from models.hf_models.vision_chat import HfVisionChatModel
from models.prompting import ensure_enough_images

END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"


class IdeficsFamilyHf(HfVisionChatModel):
    """Idefics model whose processor expands ``<image>`` placeholders in the text.

    Unlike the other wrappers, placeholders stay in the query text. Images without
    a placeholder get an image part in the query message instead.

    Attributes:
        default_pretrained: Model loaded when ``pretrained`` is not given.
        model_class: ``transformers`` class used to load the model.
        default_do_image_splitting: Default for ``do_image_splitting``.
    """

    default_pretrained: ClassVar[str]
    model_class: ClassVar[Any]
    default_do_image_splitting: ClassVar[bool]

    def __init__(
        self,
        pretrained: str | None = None,
        image_dir: str | None = None,
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        low_cpu_mem_usage: bool = True,
        batch_size: int = 1,
        trust_remote_code: bool | None = False,
        attn_implementation: str | None = BEST_FIT_ATTN_IMPLEMENTATION,
        device_map: str | dict[str, Any] = "",
        use_cache: bool = True,
        do_image_splitting: bool | None = None,
    ) -> None:
        """Load the model and the processor.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path
                (default: :attr:`default_pretrained`).
            image_dir: Directory that relative image paths are resolved against.
            revision: Model revision (branch, tag, or commit).
            device: Device that inputs are moved to.
            dtype: Weight dtype name such as ``"bfloat16"``, a ``torch.dtype``, or ``"auto"``.
            low_cpu_mem_usage: Accepted for a uniform interface; not passed to
                ``from_pretrained``.
            batch_size: Batch size reported by :attr:`batch_size` (converted to ``int``).
            trust_remote_code: Allow custom code from the model repository.
            attn_implementation: ``"flash_attention_2"`` or ``"eager"``.
            device_map: Device placement passed to ``from_pretrained``.
            use_cache: Default for the ``use_cache`` generation setting.
            do_image_splitting: Whether the processor splits images into sub-images
                (default: :attr:`default_do_image_splitting`).
        """
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.default_do_image_splitting
        super().__init__(pretrained, image_dir=image_dir, device=device, device_map=device_map, batch_size=int(batch_size), use_cache=use_cache)

        self._model = self.model_class.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            do_image_splitting=do_image_splitting,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

    def build_messages(
        self, text: str, images: list[str] | None, system_prompt: str | None, few_shot_data: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Return the chat messages with image parts only for images lacking a placeholder.

        Raises:
            ValueError: If ``text`` has more ``<image>`` placeholders than images.
        """
        messages = []
        if system_prompt:
            messages += self.create_system_prompt(system_prompt)
        if few_shot_data:
            messages += self.create_few_shot_prompt(few_shot_data)
        messages += self.create_zero_shot_prompt({"user": text, "images": self._images_without_placeholder(text, images)})
        return messages

    @staticmethod
    def _images_without_placeholder(text: str, images: list[str] | None) -> list[str] | None:
        """Return the leading images that need an image part (``None`` if all have placeholders)."""
        if images is None:
            return None
        placeholder_count = ensure_enough_images(text, images)
        if placeholder_count == len(images):
            return None
        return images[: len(images) - placeholder_count]

    def _prepare_text(self, text: str, images: list[str] | None) -> str:
        """Keep the placeholders: the processor replaces them with image tokens."""
        return text

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Render ``messages`` with the processor's chat template."""
        return self._processor.apply_chat_template(messages, add_generation_prompt=True)

    def _to_device(self, inputs: Any) -> Any:
        """Move every processed tensor to :attr:`device`."""
        return {key: value.to(self._device) for key, value in inputs.items()}

    def _apply_token_defaults(self, gen_kwargs: dict[str, Any]) -> None:
        """Stop at the EOS token or at ``<end_of_utterance>``; no padding token is set."""
        tokenizer = self._processor.tokenizer
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE_TOKEN)]
        gen_kwargs.setdefault("eos_token_id", eos_token_ids)
