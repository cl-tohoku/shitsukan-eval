"""Wrapper for LLaVA-NeXT (LLaVA-1.6) models from the Hugging Face Hub."""

from __future__ import annotations

from transformers import LlavaNextForConditionalGeneration

from models.hf_models.llava_family import LlavaFamilyHf
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)


class LlavaNextHf(LlavaFamilyHf):
    """LLaVA-NeXT (LLaVA-1.6) model for image-text chat generation.

    Example:
        >>> model = LlavaNextHf(pretrained="llava-hf/llava-v1.6-mistral-7b-hf", image_dir="data", device_map={"": 0})
        >>> model.generate_response("Describe this image: <image>", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_class = LlavaNextForConditionalGeneration
