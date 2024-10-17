import os
import pathlib
from typing import Any, Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.image_utils import load_image
from vllm import LLM, SamplingParams

from tools.utils.data_utils import is_url, set_seed_all

# Set a global random seed
set_seed_all(42)


class Qwen2VLVllm:
    """
    Args:
        pretrained (str): The name or path of a HuggingFace Transformers model.
        tokenizer (str): The name or path of a HuggingFace Transformers tokenizer.
        revision (str): The specific model version to use. It can be a branch name, a tag name, or a commit id.
        tokenizer_revision (str): The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id.
        tokenizer_mode (str): The tokenizer mode. "auto" will use the fast tokenizer if available, and "slow" will always use the slow tokenizer.
        trust_remote_code (bool): Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.
        tensor_parallel_size (int): The number of GPUs to use for distributed execution with tensor parallelism.
        dtype (str): The data type for the model weights and activations. Currently, we support `float32`, `float16`, and `bfloat16`.
                     If `auto`, we use the `torch_dtype` attribute specified in the model config file.
                     However, if the `torch_dtype` in the config is `float32`, we will use `float16` instead.
        quantization (str): The method used to quantize the model weights.
                            Currently, we support "awq", "gptq", "squeezellm", and "fp8" (experimental).
                            If None, we first check the `quantization_config` attribute in the model config file.
                            If that is None, we assume the model weights are not quantized and use `dtype` to determine the data type of the weights.
        seed (int): The seed to initialize the random number generator for sampling.
        gpu_memory_utilization (float): The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
                                        Higher values will increase the KV cache size and thus improve the model's throughput.
                                        However, if the value is too high, it may cause out-of-memory (OOM) errors.
        swap_space (float): The size (GiB) of CPU memory per GPU to use as swap space.
                            This can be used for temporarily storing the states of the requests when their `best_of` sampling parameters are larger than 1.
                            If all requests will have `best_of=1`, you can safely set this to 0.
                            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb (float): The size (GiB) of CPU memory to use for offloading the model weights.
                                This virtually increases the GPU memory space you can use to hold the model weights, at the cost of CPU-GPU data
                                transfer for every forward pass.
        enforce_eager (bool): Whether to enforce eager execution.
                              If True, we will disable CUDA graph and always execute the model in eager mode.
                              If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture (int): Maximum context len covered by CUDA graphs.
                                          When a sequence has context length larger than this, we fall back
                                          to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture (int): Maximum sequence len covered by CUDA graphs.
                                      When a sequence has context length larger than this, we fall back to eager mode.
        limit_mm_per_prompt (dict): For each multimodal plugin, limit how many input instances to allow for each prompt.
                                    Expects a dict, e.g.: image=16,video=2 allows a maximum of 16 images and 2 videos per prompt.
                                    Defaults to 1 for each modality.
        image_dir (str): Directory where input images are stored.

    Example usage:
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
        dtype: str = "auto",  # support "float32", "float16", and "bfloat16"
        quantization: (str | None) = None,  # support "awq", "gptq", "squeezellm", and "fp8"
        seed: int = 42,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool | None = None,
        max_context_len_to_capture: int | None = None,
        max_seq_len_to_capture: int = 8192,
        limit_mm_per_prompt: dict = None,
        image_dir: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Validate unused kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 10, "video": 1}

        # Load the model
        self._model = LLM(
            model=pretrained,
            tokenizer=tokenizer,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
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

        # Load the processor (tokenizer and image processing tools)
        self._processor = AutoProcessor.from_pretrained(
            tokenizer,
            revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )

        self._tokenizer = self._processor.tokenizer
        self.image_dir = image_dir

    @staticmethod
    def rename_key_for_vllm(gen_kwargs: dict) -> dict:
        gen_kwargs["max_tokens"] = gen_kwargs.pop("max_new_tokens")
        gen_kwargs.pop("do_sample", None)
        gen_kwargs["n"] = gen_kwargs.pop("num_beams")

        return gen_kwargs

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
        >>> prompt = Qwen2VLVllm.create_system_prompt("You are a helpful assistant.")
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
        >>> prompt = Qwen2VLVllm.create_zero_shot_prompt(zero_shot)
        """
        messages = []
        role = "user"
        content = []

        if zero_shot_data.get("images"):
            # Convert images to Base64 and add to content
            for image in zero_shot_data["images"]:
                content.append(
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    }
                )

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
        >>> prompt = Qwen2VLVllm.create_few_shot_prompt(few_shot)
        """
        messages = []
        for shot_data in few_shot_data:
            # User input with optional images
            role = "user"
            content = []
            if shot_data.get("images"):
                # Convert images to Base64 and add to content
                for image in shot_data["images"]:
                    content.append(
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        }
                    )
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
        >>> model = Qwen2VLVllm()
        >>> response = model.generate_response("Describe this image: <image>", ["image1.jpg"])
        >>> print(response)
        "This is an image of a beautiful sunset."
        """

        # Initialize default generation arguments if not provided
        gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        # rename key for openai api
        if gen_kwargs.get("max_new_tokens", None):
            gen_kwargs = Qwen2VLVllm.rename_key_for_vllm(gen_kwargs)

        # Set End of Text (EOS) token IDs
        eos_token_id = self._processor.tokenizer.eos_token_id
        end_of_text_token_id = self._processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        eos_token_id_list = [eos_token_id, end_of_text_token_id]

        # Set default generation parameters if not provided
        ## Maximum number of tokens to generate per output sequence.
        gen_kwargs.setdefault("max_tokens", 1024)
        ## Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.
        gen_kwargs.setdefault("temperature", 0)
        ## Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        gen_kwargs.setdefault("top_p", 1.0)
        ## Number of output sequences to return for the given prompt.
        gen_kwargs.setdefault("n", 1)
        ## Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        gen_kwargs.setdefault("presence_penalty", 0)
        ## Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        gen_kwargs.setdefault("frequency_penalty", 0)
        ## Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
        gen_kwargs.setdefault("repetition_penalty", 1.0)
        ## If provided, the engine will construct a logits processor that applies these logit biases. Defaults to None.
        gen_kwargs.setdefault("logit_bias", None)
        ## Number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens. Note that the implementation follows the OpenAI API: The API will always return the log probability of the sampled token, so there may be up to logprobs+1 elements in the response.
        gen_kwargs.setdefault("logprobs", False)
        ## Random seed to use for the generation.
        gen_kwargs.setdefault("seed", 42)
        ## List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens.
        gen_kwargs.setdefault("stop_token_ids", eos_token_id_list)

        sampling_params = SamplingParams(**gen_kwargs)

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

        # Initialize message list
        message = []

        # Add system-level prompt if provided
        if system_prompt:
            message += Qwen2VLVllm.create_system_prompt(system_prompt)

        # Add few-shot examples if provided
        if few_shot_data:
            message += Qwen2VLVllm.create_few_shot_prompt(few_shot_data)

        # Add the main user input (text and images)
        message += Qwen2VLVllm.create_zero_shot_prompt({"user": text, "images": visuals})

        # Apply chat template (model-specific preprocessing)
        prompt = self._processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(message)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        model_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.model.generate([model_inputs], sampling_params=sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text

        return generated_text
