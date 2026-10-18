"""Wrapper for Llama 3.2 Vision instruct (Mllama) models from the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

from models.hf_models.base import BEST_FIT_ATTN_IMPLEMENTATION, resolve_torch_dtype
from models.hf_models.vision_chat import HfVisionChatModel
from models.prompting import content_message


class MllamaInstructHf(HfVisionChatModel):
    """Llama 3.2 Vision instruct model for image-text chat generation.

    The system prompt is sent as a user message (see :meth:`create_system_prompt`).

    Example:
        >>> model = MllamaInstructHf(image_dir="data", attn_implementation="eager", device_map={"": 0})
        >>> model.generate_response("Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "meta-llama/Llama-3.2-11B-Vision-Instruct"

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
        device_map: str | dict[str, Any] = "auto",
        use_cache: bool = True,
    ) -> None:
        """Load the model and the processor.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path
                (default: :attr:`default_pretrained`).
            image_dir: Directory that relative image paths are resolved against.
            revision: Model revision (branch, tag, or commit).
            device: Device that inputs are moved to.
            dtype: Weight dtype name such as ``"bfloat16"``, a ``torch.dtype``, or ``"auto"``.
            low_cpu_mem_usage: Passed to ``from_pretrained``.
            batch_size: Batch size reported by :attr:`batch_size`.
            trust_remote_code: Allow custom code from the model repository.
            attn_implementation: ``"flash_attention_2"`` or ``"eager"``. The evaluator
                passes ``"eager"`` because ``MllamaForConditionalGeneration`` did not
                support FlashAttention-2.
            device_map: Device placement passed to ``from_pretrained``.
            use_cache: Default for the ``use_cache`` generation setting.
        """
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        super().__init__(pretrained, image_dir=image_dir, device=device, device_map=device_map, batch_size=batch_size, use_cache=use_cache)

        self._model = MllamaForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

    @staticmethod
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """Return ``system_prompt`` as a user message, as these experiments send it to Mllama."""
        return [content_message("user", system_prompt)]
