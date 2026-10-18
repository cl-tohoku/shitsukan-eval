"""Wrapper for Gemma 2 instruct models from the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

from models.hf_models.text_chat import HfTextChatModel

GEMMA2_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
"""Gemma 2 instruct chat template (no system role); installed when the tokenizer has none."""


class Gemma2InstructHf(HfTextChatModel):
    """Gemma 2 instruct model for text-only chat generation.

    The system prompt is merged into the first user turn (see :meth:`build_messages`).

    Example:
        >>> model = Gemma2InstructHf(pretrained="google/gemma-2-2b-jpn-it", device_map={"": 0})
        >>> model.generate_response("Where is the capital city of Japan?")
    """

    default_pretrained = "google/gemma-2-2b-jpn-it"
    fallback_chat_template = GEMMA2_CHAT_TEMPLATE

    def build_messages(self, text: str, system_prompt: str | None, few_shot_data: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Return the chat messages, merging the system prompt into the first user turn.

        Gemma 2's chat template rejects the ``system`` role, so the system prompt is
        prepended (followed by a blank line) to the first user message: the first
        few-shot example if there is one, otherwise the query.
        """
        if not system_prompt:
            return super().build_messages(text, None, few_shot_data)
        if few_shot_data:
            first_example, *other_examples = few_shot_data
            merged_first_example = {**first_example, "user": system_prompt + "\n\n" + first_example["user"]}
            return self.create_few_shot_prompt([merged_first_example, *other_examples]) + self.create_zero_shot_prompt({"user": text})
        return self.create_zero_shot_prompt({"user": system_prompt + "\n\n" + text})
