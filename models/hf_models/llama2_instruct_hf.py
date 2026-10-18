"""Wrapper for Llama 2 chat models from the Hugging Face Hub."""

from __future__ import annotations

from models.hf_models.text_chat import HfTextChatModel

LLAMA2_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
"""Llama 2 chat template (``[INST]`` / ``<<SYS>>`` format); installed when the tokenizer has none."""


class Llama2InstructHf(HfTextChatModel):
    """Llama 2 chat model for text-only chat generation.

    Example:
        >>> model = Llama2InstructHf(pretrained="meta-llama/Llama-2-7b-chat-hf", device_map={"": 0})
        >>> model.generate_response("Where is the capital city of Japan?")
    """

    default_pretrained = "meta-llama/Llama-2-7b-chat-hf"
    fallback_chat_template = LLAMA2_CHAT_TEMPLATE
