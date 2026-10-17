"""Wrapper for LLM-jp-3 instruct models from the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

from models.hf_models.text_chat import HfTextChatModel
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

LLM_JP_3_CHAT_TEMPLATE = "{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\n### 指示:\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}{% endfor %}"
"""LLM-jp-3 chat template, modified so that the system prompt can be set per request; always installed."""


class LlmJp3InstructHf(HfTextChatModel):
    """LLM-jp-3 instruct model for text-only chat generation.

    The bundled chat template always replaces the tokenizer's template, and
    ``generate`` receives only the input ids.

    Example:
        >>> model = LlmJp3InstructHf(pretrained="llm-jp/llm-jp-3-13b-instruct", device_map={"": 0})
        >>> model.generate_response("Where is the capital city of Japan?")
    """

    default_pretrained = "llm-jp/llm-jp-3-13b-instruct"
    fallback_chat_template = LLM_JP_3_CHAT_TEMPLATE
    replace_tokenizer_chat_template = True

    def _generate(self, inputs: Any, gen_kwargs: dict[str, Any]) -> Any:
        """Call ``generate`` with the input ids only (without the attention mask)."""
        return self.model.generate(inputs["input_ids"], **gen_kwargs)
