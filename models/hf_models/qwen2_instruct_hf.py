"""Wrapper for Qwen2 instruct models from the Hugging Face Hub."""

from __future__ import annotations

from models.hf_models.text_chat import HfTextChatModel
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

QWEN2_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
"""Qwen2 instruct chat template (ChatML format); installed when the tokenizer has none."""


class Qwen2InstructHf(HfTextChatModel):
    """Qwen2 instruct model for text-only chat generation.

    Example:
        >>> model = Qwen2InstructHf(pretrained="Qwen/Qwen2-7B-Instruct", device_map={"": 0})
        >>> model.generate_response("Where is the capital city of Japan?")
    """

    default_pretrained = "Qwen/Qwen2-7B-Instruct"
    fallback_chat_template = QWEN2_CHAT_TEMPLATE
