"""Wrapper for Idefics2 models from the Hugging Face Hub."""

from __future__ import annotations

from transformers import Idefics2ForConditionalGeneration

from models.hf_models.idefics_family import IdeficsFamilyHf


class Idefics2Hf(IdeficsFamilyHf):
    """Idefics2 model for image-text chat generation (image splitting off by default).

    Example:
        >>> model = Idefics2Hf(pretrained="HuggingFaceM4/idefics2-8b", image_dir="data", device_map={"": 0})
        >>> model.generate_response("<image>Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "HuggingFaceM4/idefics2-8b"
    model_class = Idefics2ForConditionalGeneration
    default_do_image_splitting = False
