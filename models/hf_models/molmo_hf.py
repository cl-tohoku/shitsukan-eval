"""Wrapper for Molmo models (custom code on the Hugging Face Hub)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from models.hf_models.base import (
    BEST_FIT_ATTN_IMPLEMENTATION,
    HfModelWrapper,
    apply_default_generation_kwargs,
    resolve_torch_dtype,
)
from models.image_loading import load_images
from models.prompting import ensure_enough_images, remove_image_placeholders

END_OF_TEXT_TOKEN = "<|endoftext|>"


class MolmoHf(HfModelWrapper):
    """Molmo model for image-text generation from a plain-text prompt.

    Molmo has no chat template: the system prompt, the few-shot examples, and the
    query are concatenated into one string (see :meth:`build_prompt`), and images
    are passed to the processor separately.

    Attributes:
        image_dir: Directory that relative image paths are resolved against.

    Example:
        >>> model = MolmoHf(image_dir="data", dtype="auto", device_map="auto", attn_implementation="eager")
        >>> model.generate_response("Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "allenai/Molmo-7B-D-0924"

    def __init__(
        self,
        pretrained: str | None = None,
        image_dir: str | None = None,
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        low_cpu_mem_usage: bool = True,
        batch_size: int = 1,
        trust_remote_code: bool | None = True,
        attn_implementation: str | None = BEST_FIT_ATTN_IMPLEMENTATION,
        device_map: str | dict[str, Any] = "auto",
        use_cache: bool = True,
    ) -> None:
        """Load the model and the processor (Molmo requires ``trust_remote_code``).

        When running Molmo, make a single GPU visible (for example with
        ``CUDA_VISIBLE_DEVICES=0``); the evaluator loads it with ``dtype="auto"``,
        ``device_map="auto"``, and eager attention.

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
        """
        pretrained = pretrained if pretrained is not None else self.default_pretrained
        super().__init__(pretrained, device=device, device_map=device_map, batch_size=batch_size, use_cache=use_cache)
        self.image_dir = image_dir

        self._model = AutoModelForCausalLM.from_pretrained(
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
    def create_system_prompt(system_prompt: str) -> str:
        """Return ``system_prompt`` followed by a blank line."""
        return system_prompt + "\n\n"

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> str:
        """Return the query text ``zero_shot_data["user"]``."""
        return zero_shot_data["user"]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> str:
        """Return each example as its user text, a line break, its answer, and a blank line.

        Example images are ignored.
        """
        return "".join(example["user"] + "\n" + example["assistant"] + "\n\n" for example in few_shot_data)

    def build_prompt(self, text: str, system_prompt: str | None, few_shot_data: list[dict[str, Any]] | None) -> str:
        """Return the prompt string: system prompt, few-shot examples, then the query."""
        prompt = ""
        if system_prompt:
            prompt += self.create_system_prompt(system_prompt)
        if few_shot_data:
            prompt += self.create_few_shot_prompt(few_shot_data)
        prompt += self.create_zero_shot_prompt({"user": text})
        return prompt

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate the model's answer to ``text`` and ``images``.

        Args:
            text: Query text. ``<image>`` placeholders (at most one per image) are
                removed.
            images: Image URLs or paths relative to :attr:`image_dir`.
            system_prompt: Task instructions placed before the examples.
            few_shot_data: Examples with ``"user"`` and ``"assistant"`` texts.
            gen_kwargs: Settings for ``GenerationConfig``. Missing keys are filled
                with the defaults (1024 new tokens, sampling with temperature 0.05,
                stopping at ``<|endoftext|>``, ...). The dictionary is not modified.

        Returns:
            The generated text without the prompt.

        Raises:
            ValueError: If ``text`` has more ``<image>`` placeholders than images.
            FileNotFoundError: If a local image does not exist.
        """
        # Work on a copy: the tasks pass the same dictionary for every record.
        gen_kwargs = dict(gen_kwargs) if gen_kwargs is not None else {}
        if images is not None:
            ensure_enough_images(text, images)
            text = remove_image_placeholders(text)
        prompt = self.build_prompt(text, system_prompt, few_shot_data)
        visuals = load_images(images, self.image_dir)
        inputs = self._processor.process(text=prompt, images=visuals, return_tensors="pt")
        # The processor returns unbatched tensors; add the batch dimension.
        inputs = {key: value.to(self._model.device).unsqueeze(0) for key, value in inputs.items()}

        apply_default_generation_kwargs(gen_kwargs, self.use_cache)
        gen_kwargs.setdefault("stop_strings", END_OF_TEXT_TOKEN)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(inputs, GenerationConfig(**gen_kwargs), tokenizer=self._tokenizer)
        new_tokens = output[0, inputs["input_ids"].size(1) :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
