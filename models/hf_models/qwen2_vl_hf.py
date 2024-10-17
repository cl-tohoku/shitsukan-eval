import os
import pathlib
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.image_utils import load_image

from tools.utils.data_utils import is_url, set_seed_all

# Set a global random seed
set_seed_all(42)

try:
    import flash_attn

    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"


class Qwen2VLHf:
    """
    A wrapper class for the Qwen2-VL model for conditional generation with both text and images.

    Args:
        pretrained (str): The name or path of the pretrained model.
        revision (str): The revision of the model to use.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        dtype (Optional[Union[str, torch.dtype]]): The data type to use for the model.
        low_cpu_mem_usage (bool): Whether to use less CPU memory when loading the model.
        batch_size (int): The batch size for generation.
        trust_remote_code (Optional[bool]): Whether to trust remote code when loading the model.
        attn_implementation (Optional[str]): The attention implementation to use.
        device_map (str): Device map for model parallelism.
        use_cache (bool): Whether to cache model results.

    Example usage:
    >>> model = Qwen2VLHf(pretrained="Qwen/Qwen2-VL-7B-Instruct", device_map={"": 0})
    >>> text = "Describe this image."
    >>> images = ["image1.jpg"]
    >>> response = model.generate_response(text, images)
    >>> print(response)
    "This is an image of a beautiful sunset."
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        image_dir: str = None,
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        low_cpu_mem_usage: bool = True,
        batch_size: int = 1,
        trust_remote_code: bool | None = False,
        attn_implementation: str | None = best_fit_attn_implementation,
        device_map: str = "",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Validate unused kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self._device = torch.device(device)
        self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        # Load the model
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        ).eval()

        # Load the processor (tokenizer and image processing tools)
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
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
    def eot_token_id(self):
        """
        Returns the End of Text (EOT) token ID.
        """
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """
        Returns the batch size per GPU.
        """
        return self.batch_size_per_gpu

    @property
    def device(self):
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
            left_truncate_len (Optional[int]): If set, truncates the tokenized text from the left to this length.
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
            encoding = encoding[-left_truncate_len:]  # Truncate from the left
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
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """
        Creates a system-level prompt for the model.

        Args:
            system_prompt (str): The system-level instruction or context.

        Returns:
            List[Dict[str, Any]]: List of system prompt messages.

        Example:
        >>> prompt = Qwen2VLHf.create_system_prompt("You are a helpful assistant.")
        [{"role": "system", "content": [{"type": "text", "text": 'You are a helpful assistant.'}]
        """
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Creates a zero-shot prompt with optional images for the model.

        Args:
            zero_shot_data (Dict[str, Any]): Dictionary containing user text and optional images.

        Returns:
            List[Dict[str, Any]]: List of messages with user input and images.

        Example:
        >>> zero_shot = {"user": "Describe this image.", "images": ["image1.jpg"]}
        >>> prompt = Qwen2VLHf.create_zero_shot_prompt(zero_shot)
        """
        messages = []
        role = "user"
        content = []

        if zero_shot_data.get("images"):
            # Convert images to Base64 and add to content
            for _ in zero_shot_data["images"]:
                content.append({"type": "image"})

        # Add user text
        content.append({"type": "text", "text": zero_shot_data[role]})
        messages.append({"role": role, "content": content})

        return messages

    @staticmethod
    def create_few_shot_prompt(few_shot_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Creates a few-shot prompt for the model with examples.

        Args:
            few_shot_data (List[Dict[str, Any]]): List of few-shot examples with user inputs and assistant responses.

        Returns:
            List[Dict[str, Any]]: List of messages including user inputs and assistant responses.

        Example:
        >>> few_shot = [
                {"user": "What is this?", "assistant": "This is a cat.", "images": ["image1.jpg"]}
            ]
        >>> prompt = Qwen2VLHf.create_few_shot_prompt(few_shot)
        """
        messages = []
        for shot_data in few_shot_data:
            # User input with optional images
            role = "user"
            content = []
            if shot_data.get("images"):
                for _ in shot_data["images"]:
                    content.append({"type": "image"})
            content.append({"type": "text", "text": shot_data[role]})
            messages.append({"role": role, "content": content})

            # Assistant response
            role = "assistant"
            content = [{"type": "text", "text": shot_data[role]}]
            messages.append({"role": role, "content": content})

        return messages

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict | None = None,
    ) -> str:
        """
        Generates a response based on the input text and optionally images.

        Args:
            text (str): The input text prompt. The text may contain `<image>` tokens to indicate where images should be used.
            images (Optional[List[str]]): List of image file paths (optional).
            system_prompt (Optional[str]): A system-level prompt that can guide the behavior of the assistant (optional).
            few_shot_data (Optional[List[Dict[str, Any]]]): A list of few-shot examples to provide additional context for the model (optional).
            gen_kwargs (Optional[dict]): Generation arguments like max tokens, temperature, etc. (optional).

        Returns:
            str: The generated text response from the model.

        Raises:
            ValueError: If the number of image tokens (`<image>`) in the text exceeds the number of provided images.

        Example usage:
        >>> model = Qwen2VLHf()
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

        # Initialize message list
        message = []

        # Add system-level prompt if provided
        if system_prompt:
            message += Qwen2VLHf.create_system_prompt(system_prompt)

        # Add few-shot examples if provided
        if few_shot_data:
            message += Qwen2VLHf.create_few_shot_prompt(few_shot_data)

        # Add the main user input (text and images)
        message += Qwen2VLHf.create_zero_shot_prompt({"user": text, "images": images})

        # Apply chat template (model-specific preprocessing)
        prompt = self._processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

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

        # Process inputs (tokenize and prepare for model)
        inputs = self._processor(text=prompt, images=visuals, padding=True, return_tensors="pt").to(self._device)

        # Set End of Text (EOS) token IDs
        eos_token_id = self._processor.tokenizer.eos_token_id
        end_of_text_token_id = self._processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        eos_token_id_list = [eos_token_id, end_of_text_token_id]

        # Set default generation parameters if not provided
        gen_kwargs.setdefault("max_new_tokens", 1024)
        gen_kwargs.setdefault("do_sample", True)
        gen_kwargs.setdefault("temperature", 0.05 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("top_p", 1.0 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("num_beams", 1)
        gen_kwargs.setdefault("use_cache", self.use_cache)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", eos_token_id_list)

        # Generate the response using the model
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            # Extract the generated tokens (skipping input tokens)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        # Decode the generated token IDs back to text
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return generated_text
