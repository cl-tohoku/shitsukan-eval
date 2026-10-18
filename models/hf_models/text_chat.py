"""Base class for text-only chat models loaded with ``AutoModelForCausalLM``."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.hf_models.base import (
    BEST_FIT_ATTN_IMPLEMENTATION,
    HfModelWrapper,
    apply_default_generation_kwargs,
    resolve_torch_dtype,
)
from models.prompting import text_message


class HfTextChatModel(HfModelWrapper):
    """Text-only chat model: renders a chat template, generates, and decodes.

    Images passed to :meth:`generate_response` are ignored, so text-only models
    can run the image tasks with the text part of the prompt only.

    Subclasses set the class attributes below and may override
    :meth:`build_messages` or :meth:`_generate`.

    Attributes:
        default_pretrained: Model loaded when ``pretrained`` is not given.
        fallback_chat_template: Jinja chat template installed when the tokenizer
            has none.
        replace_tokenizer_chat_template: Install ``fallback_chat_template`` even
            when the tokenizer has a chat template.
    """

    default_pretrained: ClassVar[str]
    fallback_chat_template: ClassVar[str]
    replace_tokenizer_chat_template: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: str | None = None,
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
        """Load the model and the tokenizer.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path
                (default: :attr:`default_pretrained`).
            revision: Model revision (branch, tag, or commit).
            device: Device that inputs are moved to.
            dtype: Weight dtype name such as ``"bfloat16"``, a ``torch.dtype``, or ``"auto"``.
            low_cpu_mem_usage: Passed to ``from_pretrained``.
            batch_size: Batch size reported by :attr:`batch_size`.
            trust_remote_code: Allow custom code from the model repository.
            attn_implementation: ``"flash_attention_2"`` or ``"eager"``.
            device_map: Device placement passed to ``from_pretrained``.
            use_cache: Default for the ``use_cache`` generation setting.
        """
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        super().__init__(pretrained, device=device, device_map=device_map, batch_size=batch_size, use_cache=use_cache)

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        if self.replace_tokenizer_chat_template or self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = self.fallback_chat_template
        self._config = self._model.config

    @staticmethod
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """Return the system message for ``system_prompt``."""
        return [text_message("system", system_prompt)]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the user message for the query in ``zero_shot_data["user"]``."""
        return [text_message("user", zero_shot_data["user"])]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return alternating user and assistant messages for the few-shot examples."""
        messages = []
        for example in few_shot_data:
            messages.append(text_message("user", example["user"]))
            messages.append(text_message("assistant", example["assistant"]))
        return messages

    def build_messages(self, text: str, system_prompt: str | None, few_shot_data: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Return the chat messages: system prompt, few-shot examples, then the query."""
        messages = []
        if system_prompt:
            messages += self.create_system_prompt(system_prompt)
        if few_shot_data:
            messages += self.create_few_shot_prompt(few_shot_data)
        messages += self.create_zero_shot_prompt({"user": text})
        return messages

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate the model's answer to ``text``.

        Args:
            text: Query text.
            images: Ignored; text-only models do not see images.
            system_prompt: Task instructions placed before the examples.
            few_shot_data: Examples with ``"user"`` and ``"assistant"`` texts.
            gen_kwargs: Settings for ``generate``. Missing keys are filled with the
                defaults (1024 new tokens, sampling with temperature 0.05, ...). The dictionary is not modified.

        Returns:
            The generated text without the prompt.
        """
        # Work on a copy: the tasks pass the same dictionary for every record.
        gen_kwargs = dict(gen_kwargs) if gen_kwargs is not None else {}
        messages = self.build_messages(text, system_prompt, few_shot_data)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text=prompt, return_tensors="pt").to(self._device)

        apply_default_generation_kwargs(gen_kwargs, self.use_cache)
        gen_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)

        return self._generate_and_decode(inputs, gen_kwargs, decoder=self._tokenizer)
