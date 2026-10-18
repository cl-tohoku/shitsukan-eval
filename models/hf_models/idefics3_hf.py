"""Wrapper for Idefics3 models from the Hugging Face Hub."""

from __future__ import annotations

from transformers import AutoModelForVision2Seq

from models.hf_models.idefics_family import IdeficsFamilyHf


class Idefics3Hf(IdeficsFamilyHf):
    """Idefics3 model for image-text chat generation (image splitting on by default).

    Example:
        >>> model = Idefics3Hf(pretrained="HuggingFaceM4/Idefics3-8B-Llama3", image_dir="data", device_map={"": 0})
        >>> model.generate_response("<image>Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    default_pretrained = "HuggingFaceM4/Idefics3-8B-Llama3"
    model_class = AutoModelForVision2Seq
    default_do_image_splitting = True
