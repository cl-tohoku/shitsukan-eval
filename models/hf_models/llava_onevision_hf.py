"""Wrapper for LLaVA-OneVision models from the Hugging Face Hub."""

from __future__ import annotations

from transformers import LlavaOnevisionForConditionalGeneration

from models.hf_models.llava_family import LlavaFamilyHf


class LlavaOnevisionHf(LlavaFamilyHf):
    """LLaVA-OneVision model for image-text chat generation.

    Example:
        >>> model = LlavaOnevisionHf(pretrained="llava-hf/llava-onevision-qwen2-7b-ov-hf", image_dir="data", device_map={"": 0})
        >>> model.generate_response("Describe this image: <image>", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    model_class = LlavaOnevisionForConditionalGeneration
