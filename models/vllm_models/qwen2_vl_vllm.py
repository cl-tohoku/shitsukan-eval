"""Wrapper for Qwen2-VL served by the vLLM inference engine."""

from __future__ import annotations

from typing import Any

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from models.hf_models.base import resolve_torch_dtype
from models.image_loading import load_images
from models.prompting import content_message, ensure_enough_images, remove_image_placeholders
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

END_OF_TEXT_TOKEN = "<|endoftext|>"
# Resolution bounds for each image, passed to ``qwen_vl_utils.process_vision_info``.
MIN_PIXELS = 224 * 224
MAX_PIXELS = 1280 * 28 * 28
DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 10, "video": 1}

# Sampling parameters used when the task configuration does not set them.
DEFAULT_SAMPLING_KWARGS: dict[str, Any] = {
    "max_tokens": 1024,
    "temperature": 0,
    "top_p": 1.0,
    "n": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repetition_penalty": 1.0,
    "logit_bias": None,
    "logprobs": False,
    "seed": 42,
}


def _image_part(image: Any) -> dict[str, Any]:
    """Return an image content part with the resolution bounds used for Qwen2-VL."""
    return {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}


class Qwen2VLVllm:
    """Qwen2-VL model for image-text chat generation with vLLM.

    Attributes:
        pretrained: Model name on the Hugging Face Hub or local path.
        image_dir: Directory that relative image paths are resolved against.

    Example:
        >>> model = Qwen2VLVllm(pretrained="Qwen/Qwen2-VL-7B-Instruct", tokenizer="Qwen/Qwen2-VL-7B-Instruct", image_dir="data")
        >>> model.generate_response("Describe this image.", ["images/coco/train2017/000000000071.jpg"])
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        tokenizer: str = "Qwen/Qwen2-VL-7B-Instruct",
        revision: str | None = "main",
        tokenizer_revision: str | None = "main",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str | None = None,
        seed: int = 42,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool | None = None,
        max_context_len_to_capture: int | None = None,
        max_seq_len_to_capture: int = 8192,
        limit_mm_per_prompt: dict[str, int] | None = None,
        image_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the vLLM engine and load the processor.

        Args:
            pretrained: Model name on the Hugging Face Hub or local path.
            tokenizer: Tokenizer and processor name or path.
            revision: Model revision.
            tokenizer_revision: Tokenizer and processor revision.
            tokenizer_mode: vLLM tokenizer mode.
            trust_remote_code: Allow custom code from the model repository.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            dtype: ``"auto"``, ``"float32"``, ``"float16"``, or ``"bfloat16"``.
            quantization: vLLM quantization method such as ``"awq"``, or ``None``.
            seed: Seed of the vLLM engine.
            gpu_memory_utilization: Fraction of GPU memory vLLM may use.
            swap_space: CPU swap space per GPU in GiB.
            cpu_offload_gb: CPU memory in GiB to offload weights to.
            enforce_eager: Disable CUDA graphs when true.
            max_context_len_to_capture: Deprecated vLLM option, passed through.
            max_seq_len_to_capture: Longest sequence covered by CUDA graphs.
            limit_mm_per_prompt: Maximum images and videos per prompt
                (default: ``{"image": 10, "video": 1}``).
            image_dir: Directory that relative image paths are resolved against.
            **kwargs: Must be empty; present so that unsupported options fail loudly.

        Raises:
            AssertionError: If unsupported keyword arguments are given.
        """
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.pretrained = pretrained
        self.image_dir = image_dir
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = dict(DEFAULT_LIMIT_MM_PER_PROMPT)

        self._model = LLM(
            model=pretrained,
            tokenizer=tokenizer,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=resolve_torch_dtype(dtype),
            quantization=quantization,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            limit_mm_per_prompt=limit_mm_per_prompt,
        )
        self._processor = AutoProcessor.from_pretrained(tokenizer, revision=tokenizer_revision, trust_remote_code=trust_remote_code)
        self._tokenizer = self._processor.tokenizer

    @staticmethod
    def rename_key_for_vllm(gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate ``transformers``-style settings into vLLM sampling parameters.

        ``max_new_tokens`` becomes ``max_tokens``, ``num_beams`` becomes ``n``, and
        ``do_sample`` is dropped. ``gen_kwargs`` is modified in place and returned.

        Raises:
            KeyError: If ``max_new_tokens`` or ``num_beams`` is missing.
        """
        gen_kwargs["max_tokens"] = gen_kwargs.pop("max_new_tokens")
        gen_kwargs.pop("do_sample", None)
        gen_kwargs["n"] = gen_kwargs.pop("num_beams")
        return gen_kwargs

    @property
    def tokenizer(self) -> Any:
        """Tokenizer used for encoding and decoding text."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """The vLLM engine."""
        return self._model

    @property
    def eot_token_id(self) -> int:
        """End-of-text token id of the tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self) -> int:
        """Batch size per GPU; ``batch_size_per_gpu`` is never set, so this raises AttributeError."""
        return self.batch_size_per_gpu  # ty: ignore[unresolved-attribute]

    def tok_encode(self, string: str, left_truncate_len: int | None = None, add_special_tokens: bool | None = None) -> list[int]:
        """Encode ``string`` into token ids.

        Args:
            string: Text to encode.
            left_truncate_len: If set, keep only the last ``left_truncate_len`` ids.
            add_special_tokens: Whether to add special tokens (default: ``False``).

        Returns:
            The token ids.
        """
        add_special_tokens = add_special_tokens if add_special_tokens is not None else False
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode token ids into text."""
        return self.tokenizer.decode(tokens)

    @staticmethod
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """Return the system message for ``system_prompt``."""
        return [content_message("system", system_prompt)]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the user message with one image part per entry of ``zero_shot_data["images"]``."""
        return [content_message("user", zero_shot_data["user"], [_image_part(image) for image in zero_shot_data.get("images") or ()])]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return alternating user and assistant messages for the few-shot examples.

        The image parts of an example reference its ``"images"`` entries as given.
        """
        messages = []
        for example in few_shot_data:
            messages.append(content_message("user", example["user"], [_image_part(image) for image in example.get("images") or ()]))
            messages.append(content_message("assistant", example["assistant"]))
        return messages

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
            few_shot_data: Examples with ``"user"``, ``"assistant"``, and optional
                ``"images"`` entries.
            gen_kwargs: Sampling settings. ``transformers``-style keys are
                translated (see :meth:`rename_key_for_vllm`) and missing keys are
                filled from :data:`DEFAULT_SAMPLING_KWARGS`; generation stops at
                the EOS token or ``<|endoftext|>``.

        Returns:
            The generated text.

        Raises:
            ValueError: If ``text`` has more ``<image>`` placeholders than images.
            FileNotFoundError: If a local image does not exist.
        """
        gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        if gen_kwargs.get("max_new_tokens", None):
            gen_kwargs = Qwen2VLVllm.rename_key_for_vllm(gen_kwargs)

        tokenizer = self._processor.tokenizer
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(END_OF_TEXT_TOKEN)]
        for key, value in DEFAULT_SAMPLING_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        gen_kwargs.setdefault("stop_token_ids", eos_token_ids)
        sampling_params = SamplingParams(**gen_kwargs)

        if images is not None:
            ensure_enough_images(text, images)
            text = remove_image_placeholders(text)
        visuals = load_images(images, self.image_dir)

        message = []
        if system_prompt:
            message += Qwen2VLVllm.create_system_prompt(system_prompt)
        if few_shot_data:
            message += Qwen2VLVllm.create_few_shot_prompt(few_shot_data)
        message += Qwen2VLVllm.create_zero_shot_prompt({"user": text, "images": visuals})

        prompt = self._processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        multi_modal_data = {}
        if image_inputs is not None:
            multi_modal_data["image"] = image_inputs
        if video_inputs is not None:
            multi_modal_data["video"] = video_inputs

        outputs = self.model.generate([{"prompt": prompt, "multi_modal_data": multi_modal_data}], sampling_params=sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text
