"""Building blocks shared by the Hugging Face ``transformers`` model wrappers.

Every wrapper implements ``generate_response(text, images, system_prompt,
few_shot_data, gen_kwargs) -> str``. This module provides the parts that are
identical across wrappers: attention-implementation detection, dtype resolution,
default generation settings, accessors, and the generate-then-decode step.
"""

from __future__ import annotations

from typing import Any

import torch


def _detect_attn_implementation() -> str:
    """Return ``"flash_attention_2"`` if the ``flash_attn`` package imports, else ``"eager"``."""
    try:
        import flash_attn  # noqa: F401, PLC0415 - imported only to probe the installation.
    except ImportError:
        return "eager"
    return "flash_attention_2"


BEST_FIT_ATTN_IMPLEMENTATION = _detect_attn_implementation()
"""Default ``attn_implementation``, decided once when this module is imported."""

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_SAMPLING_TEMPERATURE = 0.05
DEFAULT_SAMPLING_TOP_P = 1.0


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    """Turn a dtype name such as ``"bfloat16"`` into the ``torch`` dtype.

    ``"auto"``, ``None``, and ``torch.dtype`` values are returned unchanged.
    """
    if isinstance(dtype, str) and dtype != "auto":
        return getattr(torch, dtype)
    return dtype


def apply_default_generation_kwargs(gen_kwargs: dict[str, Any], use_cache: bool) -> None:
    """Fill in the generation settings shared by the ``transformers`` wrappers.

    Keys already present in ``gen_kwargs`` are kept; the dictionary is updated in
    place. Without sampling, ``temperature`` and ``top_p`` default to ``None``.

    Args:
        gen_kwargs: Keyword arguments for ``generate``.
        use_cache: Default for ``use_cache``.
    """
    gen_kwargs.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    gen_kwargs.setdefault("do_sample", True)
    gen_kwargs.setdefault("temperature", DEFAULT_SAMPLING_TEMPERATURE if gen_kwargs["do_sample"] else None)
    gen_kwargs.setdefault("top_p", DEFAULT_SAMPLING_TOP_P if gen_kwargs["do_sample"] else None)
    gen_kwargs.setdefault("num_beams", 1)
    gen_kwargs.setdefault("use_cache", use_cache)


class HfModelWrapper:
    """Accessors and helpers shared by the wrappers around ``transformers`` models.

    Subclasses call :meth:`__init__` first and then load ``_model``,
    ``_tokenizer``, and ``_config``.

    Attributes:
        pretrained: Model name on the Hugging Face Hub or local path.
        device_map: Device placement passed to ``from_pretrained``.
        batch_size_per_gpu: Batch size reported by :attr:`batch_size`.
        use_cache: Default for the ``use_cache`` generation setting.
    """

    _model: Any
    _tokenizer: Any
    _config: Any

    def __init__(self, pretrained: str, device: str, device_map: str | dict[str, Any], batch_size: int, use_cache: bool) -> None:
        """Store the settings shared by all wrappers.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path.
            device: Device that inputs are moved to, such as ``"cuda"``.
            device_map: Device placement passed to ``from_pretrained``. With
                ``"auto"`` on several GPUs, sampling may fail with "probability
                tensor contains either `inf`, `nan` or element < 0"; use a single
                GPU such as ``{"": 0}`` in that case.
            batch_size: Batch size reported by :attr:`batch_size`.
            use_cache: Default for the ``use_cache`` generation setting.
        """
        self.pretrained = pretrained
        self._device = torch.device(device)
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size
        self.use_cache = use_cache

    @property
    def config(self) -> Any:
        """Model configuration."""
        return self._config

    @property
    def tokenizer(self) -> Any:
        """Tokenizer used for encoding and decoding text."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Loaded model."""
        return self._model

    @property
    def eot_token_id(self) -> int:
        """End-of-text token id of the tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self) -> int:
        """Batch size per GPU."""
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        """Device that inputs are moved to."""
        return self._device

    def tok_encode(self, string: str, left_truncate_len: int | None = None, add_special_tokens: bool | None = None) -> list[int]:
        """Encode ``string`` into token ids.

        Args:
            string: Text to encode.
            left_truncate_len: If set, keep only the last ``left_truncate_len`` ids.
            add_special_tokens: Whether to add special tokens (default: ``False``).

        Returns:
            The token ids.
        """
        add_special_tokens = add_special_tokens if add_special_tokens is not None else False
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode token ids into text."""
        return self.tokenizer.decode(tokens)

    def _generate(self, inputs: Any, gen_kwargs: dict[str, Any]) -> Any:
        """Call ``generate`` with the processed inputs; override if a model needs other arguments."""
        return self.model.generate(**inputs, **gen_kwargs)

    def _generate_and_decode(self, inputs: Any, gen_kwargs: dict[str, Any], decoder: Any) -> str:
        """Generate a continuation of ``inputs`` and decode only the new tokens.

        Args:
            inputs: Processed model inputs that include ``input_ids``.
            gen_kwargs: Keyword arguments for ``generate``.
            decoder: Tokenizer or processor whose ``batch_decode`` turns ids into text.

        Returns:
            The decoded text of the first (and only) sequence.
        """
        with torch.no_grad():
            generated_ids = self._generate(inputs, gen_kwargs)
            new_token_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        return decoder.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
