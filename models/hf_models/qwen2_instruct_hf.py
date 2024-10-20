from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.utils.data_utils import set_seed_all

# Set a global random seed
set_seed_all(42)

try:
    import flash_attn

    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"


class Qwen2InstructHf:
    """
    A wrapper class for using the Qwen2 models from Hugging Face with conditional generation.

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
    >>> model = Qwen2InstructHf(pretrained="Qwen/Qwen2-7B-Instruct", device_map={"": 0})
    >>> text = "Where is the capital city of Japan?"
    >>> response = model.generate_response(text)
    >>> print(response)
    "It is Tokyo."
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-7B-Instruct",
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        low_cpu_mem_usage: bool = True,
        batch_size: int = 1,
        trust_remote_code: bool | None = False,
        attn_implementation: str | None = best_fit_attn_implementation,
        device_map: str = "auto",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        """
        Initializes the Qwen2InstructHf class by loading the model, processor, and tokenizer.

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
        >>> model = Qwen2InstructHf(pretrained="Qwen/Qwen2-7B-Instruct", device="cuda")
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

        # Load tokenizer for tokenization and image handling
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        if self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        # Ensure left-padding for batched generation
        self._config = self._model.config
        self.batch_size_per_gpu = batch_size
        self.use_cache = use_cache

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
    def create_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
        """
        Creates a system-level prompt for the model.

        Args:
            system_prompt (str): The system-level instruction or context.

        Returns:
            List[Dict[str, Any]]: List of system prompt messages.

        Example:
        >>> prompt = Qwen2InstructHf.create_system_prompt("You are a helpful assistant.")
        [{"role": "system", "content": 'You are a helpful assistant.'}]
        """
        return [{"role": "system", "content": system_prompt}]

    @staticmethod
    def create_zero_shot_prompt(zero_shot_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Creates a zero-shot prompt with optional images for the model.

        Args:
            zero_shot_data (Dict[str, Any]): Dictionary containing user text and optional images.

        Returns:
            List[Dict[str, Any]]: List of messages with user input and images.

        Example:
        >>> zero_shot = {"user": "Describe this image."}
        >>> prompt = Qwen2InstructHf.create_zero_shot_prompt(zero_shot)
        """
        role = "user"
        return [{"role": role, "content": zero_shot_data[role]}]

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
                {"user": "What is this?", "assistant": "This is a cat.", "images": None}
            ]
        >>> prompt = Qwen2InstructHf.create_few_shot_prompt(few_shot)
        """
        messages = []
        for shot_data in few_shot_data:
            # User input with optional images
            role = "user"
            messages.append({"role": role, "content": shot_data[role]})

            # Assistant response
            role = "assistant"
            messages.append({"role": role, "content": shot_data[role]})

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
        Generates a response based on the input text and optional images.

        Args:
            text (str): The input text prompt. The text may contain `<image>` tokens to indicate where images should be used.
            images (None): image file can't be used this function.
            system_prompt (Optional[str]): A system-level prompt that can guide the behavior of the assistant (optional).
            few_shot_data (Optional[List[Dict[str, Any]]]): A list of few-shot examples to provide additional context for the model (optional).
            gen_kwargs (Optional[dict]): Optional arguments for controlling the generation behavior (e.g., max tokens, temperature).

        Returns:
            str: The generated text response from the model.

        Example usage:
        >>> model = Qwen2InstructHf(pretrained="Qwen/Qwen2-7B-Instruct")
        >>> response = model.generate_response("Where is the capital city of Japan?")
        >>> print(response)
        "It is Tokyo."
        """
        # Initialize default generation arguments if not provided
        gen_kwargs = gen_kwargs if gen_kwargs is not None else {}

        # Initialize message list
        message = []

        # Add system-level prompt if provided
        if system_prompt:
            message += Qwen2InstructHf.create_system_prompt(system_prompt)

        # Add few-shot examples if provided
        if few_shot_data:
            message += Qwen2InstructHf.create_few_shot_prompt(few_shot_data)

        # Add the main user input (text)
        message += Qwen2InstructHf.create_zero_shot_prompt({"user": text})

        # Apply chat template (model-specific preprocessing)
        prompt = self._tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        # Process inputs (tokenize and prepare for model inference)
        inputs = self._tokenizer(text=prompt, return_tensors="pt").to(self._device)

        # Set default generation parameters
        gen_kwargs.setdefault("max_new_tokens", 1024)
        gen_kwargs.setdefault("do_sample", True)
        gen_kwargs.setdefault("temperature", 0.05 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("top_p", 1.0 if gen_kwargs["do_sample"] else None)
        gen_kwargs.setdefault("num_beams", 1)
        gen_kwargs.setdefault("use_cache", self.use_cache)
        gen_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)

        # Generate the response using the model
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            # Extract the generated tokens (skipping input tokens)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        # Decode the generated token IDs back to text
        generated_text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return generated_text
