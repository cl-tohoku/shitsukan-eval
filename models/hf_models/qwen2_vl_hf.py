"""Wrapper for Qwen2-VL models from the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from models.hf_models.base import BEST_FIT_ATTN_IMPLEMENTATION, resolve_torch_dtype
from models.hf_models.vision_chat import HfVisionChatModel
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

END_OF_TEXT_TOKEN = "<|endoftext|>"


class Qwen2VLHf(HfVisionChatModel):
    """Qwen2-VL model for image-text chat generation.

    Generation stops at the tokenizer's end-of-sequence token or at
    ``<|endoftext|>``.

    Example:
        >>> model = Qwen2VLHf(pretrained="Qwen/Qwen2-VL-7B-Instruct", image_dir="data", device_map={"": 0})
        >>> model.generate_response("Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "Qwen/Qwen2-VL-7B-Instruct"

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
        """
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        super().__init__(pretrained, image_dir=image_dir, device=device, device_map=device_map, batch_size=int(batch_size), use_cache=use_cache)

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

    def _apply_token_defaults(self, gen_kwargs: dict[str, Any]) -> None:
        """Default the padding token and stop at the EOS token or ``<|endoftext|>``."""
        tokenizer = self._processor.tokenizer
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(END_OF_TEXT_TOKEN)]
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", eos_token_ids)
