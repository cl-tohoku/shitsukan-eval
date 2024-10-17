import os
import pathlib
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from transformers.image_utils import load_image

from tools.utils.data_utils import is_url, set_seed_all

# Set a global random seed
set_seed_all(42)

try:
    import flash_attn

    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"

DEFAULT_IMAGE_TOKEN = "<|image|>"


class MolmoHf:
    """
    A wrapper class for using the LlavaNext models from Hugging Face with conditional generation.

    Args:
        pretrained (str): The name or path of the pretrained model.
        revision (str): The revision of the model to use.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        dtype (Optional[Union[str, torch.dtype]]): The data type to use for the model.
        low_cpu_mem_usage (bool): Whether to use less CPU memory when loading the model.
        batch_size (int): The batch size for generation.
        trust_remote_code (Optional[bool]): Whether to trust remote code when loading the model.
        attn_implementation (Optional[str]): Attention implementation to use ('flash_attention_2' or 'eager').
        device_map (str): The device map for model parallelism.
        use_cache (bool): Whether to use cache during generation.

    Example usage:
    >>> model = MolmoHf(pretrained="allenai/Molmo-7B-D-0924", device_map={"": 0})
    >>> text = "Describe this image: <image>"
    >>> images = ["image1.jpg"]
    >>> response = model.generate_response(text, images)
    >>> print(response)
    "This is an image of a beautiful sunset."
    """

    def __init__(
        self,
        pretrained: str = "allenai/Molmo-7B-D-0924",
        image_dir: str = None,
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        low_cpu_mem_usage: bool = True,
        batch_size: int = 1,
        trust_remote_code: bool | None = True,
        attn_implementation: str | None = best_fit_attn_implementation,
        device_map: str = "auto",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        """
        Initializes the Molmo class by loading the model, processor, and tokenizer.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model to use.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
            dtype (Optional[Union[str, torch.dtype]]): The data type to use for the model.
            low_cpu_mem_usage (bool): Whether to use less CPU memory when loading the model.
            batch_size (int): The batch size for generation.
            trust_remote_code (Optional[bool]): Whether to trust remote code when loading the model.
            attn_implementation (Optional[str]): The attention implementation to use ('flash_attention_2' or 'eager').
            device_map (str): The device map for model parallelism.
            use_cache (bool): Whether to use cache during generation.

        Example:
        >>> model = MolmoHf(pretrained="allenai/Molmo-7B-D-0924", device="cuda")
        """
        super().__init__()
        # Handle unexpected kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self._device = torch.device(device)
        ## multi-gpu での実行 device_map="auto" にした場合に、
        ## RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
        ## ^ のエラーが出る場合があります。その場合、single-gpu での実行 device_map={"": 0} に変更してください。
        self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        # Load the model
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        ).eval()

        # Load processor for tokenization and image handling
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Ensure left-padding for batched generation
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = batch_size
        self.use_cache = use_cache
        self.image_dir = image_dir

    @property
    def config(self):
        """
        Returns the model configuration.
        """
        return self._config

    @property
    def tokenizer(self):
        """
        Returns the tokenizer used for encoding/decoding text.
        """
        return self._tokenizer

    @property
    def model(self):
        """
        Returns the loaded model instance.
        """
        return self._model

    @property
    def eot_token_id(self) -> int:
        """
        Returns the End of Text (EOT) token ID.
        """
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size per GPU.
        """
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        """
        Returns the device the model is running on.
        """
        return self._device

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """
        Tokenizes and encodes a given string into token IDs.

        Args:
            string (str): The input text to encode.
            left_truncate_len (Optional[int]): Truncate the tokenized text from the left if set.
            add_special_tokens (Optional[bool]): Whether to add special tokens during tokenization.

        Returns:
            List[int]: List of token IDs.

        Example:
        >>> model.tok_encode("Hello world", add_special_tokens=True)
        [101, 7592, 2088, 102]
        """
        add_special_tokens = add_special_tokens if add_special_tokens is not None else False
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]  # Left-truncate
        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of token IDs back into text.

        Args:
            tokens (List[int]): List of token IDs to decode.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(tokens)

    @staticmethod
    def create_system_prompt(system_prompt: str) -> str:
        return system_prompt + "\n\n"

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> str:
        role = "user"
        return zero_shot_data[role]

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> str:
        few_shot_prompt = ""
        for shot_data in few_shot_data:
            # User input with optional images
            role = "user"
            few_shot_prompt += shot_data[role] + "\n"

            # Assistant response
            role = "assistant"
            few_shot_prompt += shot_data[role] + "\n\n"

        return few_shot_prompt

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict | None = None,
    ) -> str:
        """
        Generates a response based on the input text and optional images.

        Args:
            text (str): The input text prompt. The text may contain `<image>` tokens to indicate where images should be used.
            images (Optional[List[str]]): List of image file paths (optional).
            system_prompt (Optional[str]): A system-level prompt that can guide the behavior of the assistant (optional).
            few_shot_data (Optional[List[Dict[str, Any]]]): A list of few-shot examples to provide additional context for the model (optional).
            gen_kwargs (Optional[dict]): Optional arguments for controlling the generation behavior (e.g., max tokens, temperature).

        Returns:
            str: The generated text response from the model.

        Raises:
            ValueError: If the number of image tokens (`<image>`) in the text exceeds the number of provided images.

        Example usage:
        >>> model = MolmoHf(pretrained="allenai/Molmo-7B-D-0924")
        >>> response = model.generate_response("Describe this image: <image>", ["image1.jpg"])
        >>> print(response)
        "This is an image of a beautiful sunset."
        """
        # Initialize default generation arguments if not provided
        gen_kwargs = gen_kwargs if gen_kwargs is not None else {}

        # Handle cases where images are provided
        if images is not None:
            # Count the number of <image> tokens in the text
            img_token_num = text.count("<image>")
            if img_token_num > len(images):
                raise ValueError(
                    f"Error! The number of image tokens (img_token_num={img_token_num}) is greater than the number of input images (len(images)={len(images)})."
                )

            # Remove <image> tokens in the text
            if "<image>" in text:
                text = text.replace("<image>", "")

        # Initialize prompt str
        prompt = ""

        # Add system-level prompt if provided
        if system_prompt:
            prompt += MolmoHf.create_system_prompt(system_prompt)

        # Add few-shot examples if provided
        if few_shot_data:
            prompt += MolmoHf.create_few_shot_prompt(few_shot_data)

        # Add the main user input (text and images)
        prompt += MolmoHf.create_zero_shot_prompt({"user": text, "images": images})

        # Load images if provided, otherwise set visuals to None
        if images is not None:
            visuals = list()
            for img_path in images:
                if is_url(img_path):
                    visuals.append(load_image(img_path))
                elif os.path.exists(str(pathlib.Path(self.image_dir).joinpath(img_path))):
                    new_img_path = str(pathlib.Path(self.image_dir).joinpath(img_path))
                    visuals.append(load_image(new_img_path))
                else:
                    raise FileNotFoundError(f"The file path does not exist: {img_path}")
        else:
            visuals = None

        # Process inputs (tokenize and prepare for model inference)
        inputs = self._processor.process(text=prompt, images=visuals, return_tensors="pt")
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        # Set default generation parameters
        gen_kwargs.setdefault("max_new_tokens", 1024)
        gen_kwargs.setdefault("do_sample", True)
        gen_kwargs.setdefault("temperature", 0.05 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("top_p", 1.0 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("num_beams", 1)
        gen_kwargs.setdefault("use_cache", self.use_cache)
        gen_kwargs.setdefault("stop_strings", "<|endoftext|>")
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        # Generate the response using the model
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(inputs, GenerationConfig(**gen_kwargs), tokenizer=self._tokenizer)

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generated_text
