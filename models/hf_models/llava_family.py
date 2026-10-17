"""Shared implementation of the LLaVA wrappers (LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision)."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers import AutoConfig, AutoProcessor

from models.hf_models.base import BEST_FIT_ATTN_IMPLEMENTATION, reject_unexpected_kwargs, resolve_torch_dtype
from models.hf_models.vision_chat import HfVisionChatModel

DEFAULT_PATCH_SIZE = 14
DEFAULT_VISION_FEATURE_SELECT_STRATEGY = "default"


class LlavaFamilyHf(HfVisionChatModel):
    """LLaVA-style model whose processor needs the vision tower's patch settings.

    Subclasses choose the checkpoint and the ``transformers`` model class.

    Attributes:
        default_pretrained: Model loaded when ``pretrained`` is not given.
        model_class: ``transformers`` class used to load the model.
        patch_size: Patch size of the vision tower, read from the model config.
        vision_feature_select_strategy: ``"default"`` drops the CLS token from the
            vision features; ``"full"`` keeps all of them.
    """

    default_pretrained: ClassVar[str]
    model_class: ClassVar[Any]

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
        **kwargs: Any,
    ) -> None:
        """Load the model configuration, the model, and the processor.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path
                (default: :attr:`default_pretrained`).
            image_dir: Directory that relative image paths are resolved against.
            revision: Model revision (branch, tag, or commit).
            device: Device recorded as :attr:`device`.
            dtype: Weight dtype name such as ``"bfloat16"``, a ``torch.dtype``, or ``"auto"``.
            low_cpu_mem_usage: Passed to ``from_pretrained``.
            batch_size: Batch size reported by :attr:`batch_size`.
            trust_remote_code: Allow custom code from the model repository.
            attn_implementation: ``"flash_attention_2"`` or ``"eager"``.
            device_map: Device placement passed to ``from_pretrained``.
            use_cache: Default for the ``use_cache`` generation setting.
            **kwargs: Must be empty; present so that unsupported options fail loudly.

        Raises:
            AssertionError: If unsupported keyword arguments are given.
        """
        reject_unexpected_kwargs(kwargs)
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        super().__init__(pretrained, image_dir=image_dir, device=device, device_map=device_map, batch_size=batch_size, use_cache=use_cache)

        config = AutoConfig.from_pretrained(pretrained)
        self.patch_size = getattr(config.vision_config, "patch_size", DEFAULT_PATCH_SIZE)
        self.vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", DEFAULT_VISION_FEATURE_SELECT_STRATEGY)

        self._model = self.model_class.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            patch_size=self.patch_size,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
        )
        # Left padding keeps every prompt directly before its generated tokens in a batch.
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

    def _to_device(self, inputs: Any) -> Any:
        """Move the processed inputs to the model's device and cast them to its dtype."""
        return inputs.to(self.model.device, self.model.dtype)
