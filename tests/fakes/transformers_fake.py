"""Minimal stand-in for :mod:`transformers` used by the characterization tests.

Model, tokenizer, and processor loaders record the arguments they receive and
return symbolic objects. Chat templates are not rendered: the fake records the
messages together with a digest of the active template and returns a marker that
identifies the prompt, so later calls (tokenization, generation) can be traced back
to the exact messages without reproducing a real tokenizer.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

from tests.fakes import torch_fake
from tests.fakes.module_factory import make_module
from tests.fakes.recorder import RECORDER, RESPONDER, SETTINGS, Snapshottable, prompt_text, rng_summary, snapshot, text_digest

# Token ids returned by ``convert_tokens_to_ids`` for the special tokens the
# wrappers look up; any other token maps to 0.
SPECIAL_TOKEN_IDS = {"<|endoftext|>": 151643, "<end_of_utterance>": 32002}
EOS_TOKEN_ID = 2
PROMPT_LENGTH = 9
GENERATED_LENGTH = 12


def _prompt_marker(messages: Any) -> str:
    """Remember the prompt text for response rules and return its marker string."""
    RESPONDER.remember_prompt(prompt_text(snapshot(messages)))
    return RECORDER.next_prompt_marker()


class FakeImage(Snapshottable):
    """Image returned by the fake ``load_image``; remembers where it came from."""

    def __init__(self, source: str) -> None:
        """Create an image loaded from ``source`` (a path or URL)."""
        self.source = source

    def snapshot(self) -> dict[str, str]:
        """Describe the image by its source."""
        return {"image": self.source}


class FakeBatch(dict, Snapshottable):
    """Dictionary of symbolic tensors standing in for ``BatchFeature``."""

    def to(self, *args: Any, **kwargs: Any) -> FakeBatch:
        """Record the device/dtype move and return the same batch."""
        RECORDER.record("BatchFeature.to", args=args, kwargs=kwargs)
        return self

    def snapshot(self) -> dict[str, Any]:
        """Describe the contained tensors."""
        return {key: snapshot(value) for key, value in self.items()}


def _text_batch(with_pixels: bool) -> FakeBatch:
    """Return the symbolic batch produced by tokenizing one prompt."""
    batch = FakeBatch(
        input_ids=torch_fake.Tensor("input_ids", (1, PROMPT_LENGTH)),
        attention_mask=torch_fake.Tensor("attention_mask", (1, PROMPT_LENGTH)),
    )
    if with_pixels:
        batch["pixel_values"] = torch_fake.Tensor("pixel_values", (1, 3, 336, 336))
    return batch


class FakeTokenizer(Snapshottable):
    """Tokenizer that records template changes, encoding, and decoding."""

    eos_token_id = EOS_TOKEN_ID

    def __init__(self, source: str) -> None:
        """Create a tokenizer loaded from ``source``."""
        self.source = source
        self.pad_token_id = SETTINGS["tokenizer_pad_token_id"]
        self._chat_template: str | None = SETTINGS["tokenizer_chat_template"]
        self._padding_side = "right"

    def snapshot(self) -> dict[str, str]:
        """Describe the tokenizer by its source."""
        return {"tokenizer": self.source}

    @property
    def chat_template(self) -> str | None:
        """Return the active chat template."""
        return self._chat_template

    @chat_template.setter
    def chat_template(self, template: str | None) -> None:
        """Record and replace the chat template."""
        RECORDER.record("Tokenizer.chat_template.set", sha256=text_digest(template))
        self._chat_template = template

    @property
    def padding_side(self) -> str:
        """Return the padding side."""
        return self._padding_side

    @padding_side.setter
    def padding_side(self, side: str) -> None:
        """Record and replace the padding side."""
        RECORDER.record("Tokenizer.padding_side.set", side=side)
        self._padding_side = side

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        """Record the messages and the template digest; return a prompt marker."""
        RECORDER.record(
            "Tokenizer.apply_chat_template",
            conversation=conversation,
            kwargs=kwargs,
            chat_template_sha256=text_digest(self._chat_template),
        )
        return _prompt_marker(conversation)

    def __call__(self, text: Any = None, **kwargs: Any) -> FakeBatch:
        """Record tokenization of ``text`` and return a symbolic batch."""
        RECORDER.record("Tokenizer.__call__", text=text, kwargs=kwargs)
        return _text_batch(with_pixels=False)

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        """Record decoding and return the next scripted response."""
        RECORDER.record("Tokenizer.batch_decode", sequences=sequences, kwargs=kwargs)
        return [RESPONDER.next()]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        """Record decoding and return the next scripted response."""
        RECORDER.record("Tokenizer.decode", token_ids=token_ids, kwargs=kwargs)
        return RESPONDER.next()

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Record encoding and return one id per character."""
        RECORDER.record("Tokenizer.encode", text=text, kwargs=kwargs)
        return [ord(character) for character in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        """Record the lookup and return the id of a known special token (else 0)."""
        RECORDER.record("Tokenizer.convert_tokens_to_ids", token=token)
        return SPECIAL_TOKEN_IDS.get(token, 0)


class FakeProcessor(Snapshottable):
    """Multimodal processor that records prompts, images, and decoding."""

    def __init__(self, source: str) -> None:
        """Create a processor (with its tokenizer) loaded from ``source``."""
        self.source = source
        self.tokenizer = FakeTokenizer(source)

    def snapshot(self) -> dict[str, str]:
        """Describe the processor by its source."""
        return {"processor": self.source}

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        """Record the messages and return a prompt marker."""
        RECORDER.record("Processor.apply_chat_template", conversation=conversation, kwargs=kwargs)
        return _prompt_marker(conversation)

    def __call__(self, text: Any = None, images: Any = None, **kwargs: Any) -> FakeBatch:
        """Record preprocessing of ``text`` and ``images``; return a symbolic batch."""
        RECORDER.record("Processor.__call__", text=text, images=images, kwargs=kwargs)
        return _text_batch(with_pixels=images is not None)

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        """Record decoding and return the next scripted response."""
        RECORDER.record("Processor.batch_decode", sequences=sequences, kwargs=kwargs)
        return [RESPONDER.next()]

    def process(self, text: str | None = None, images: Any = None, **kwargs: Any) -> dict[str, torch_fake.Tensor]:
        """Record Molmo-style preprocessing and return unbatched symbolic tensors."""
        RECORDER.record("Processor.process", text=text, images=images, kwargs=kwargs)
        RESPONDER.remember_prompt(text if text is not None else "")
        inputs = {"input_ids": torch_fake.Tensor("input_ids", (PROMPT_LENGTH,))}
        if images is not None:
            inputs["images"] = torch_fake.Tensor("images", (1, 576, 588))
        return inputs


class FakeConfig(Snapshottable):
    """Model configuration with the vision attributes LLaVA reads."""

    def __init__(self, source: str) -> None:
        """Create a configuration for ``source``.

        When ``SETTINGS["config_has_vision_attributes"]`` is false, the attributes
        are missing so that the wrappers fall back to their defaults.
        """
        self.source = source
        if SETTINGS["config_has_vision_attributes"]:
            self.vision_config = SimpleNamespace(patch_size=16)
            self.vision_feature_select_strategy = "full"
        else:
            self.vision_config = SimpleNamespace()

    def snapshot(self) -> dict[str, str]:
        """Describe the configuration by its source."""
        return {"config": self.source}


class FakeModel(Snapshottable):
    """Loaded model that records generation calls."""

    def __init__(self, loader: str, source: str) -> None:
        """Create the model that ``loader`` loaded from ``source``."""
        self.loader = loader
        self.source = source
        self.config = FakeConfig(source)
        self.device = torch_fake.device("cuda:0")
        self.dtype = torch_fake.bfloat16

    def snapshot(self) -> dict[str, str]:
        """Describe the model by its loader and source."""
        return {"model": f"{self.loader}:{self.source}"}

    def eval(self) -> FakeModel:
        """Record switching to evaluation mode and return the same model."""
        RECORDER.record("Model.eval", model=self.source)
        return self

    def generate(self, *args: Any, **kwargs: Any) -> torch_fake.Tensor:
        """Record ``generate`` and return symbolic generated ids."""
        RECORDER.record("Model.generate", args=args, kwargs=kwargs, rng=rng_summary(torch_fake))
        torch_fake.consume_random_draw()
        return torch_fake.Tensor("generated_ids", (1, GENERATED_LENGTH))

    def generate_from_batch(self, batch: Any, generation_config: Any, **kwargs: Any) -> torch_fake.Tensor:
        """Record Molmo-style generation and return symbolic generated ids."""
        RECORDER.record(
            "Model.generate_from_batch",
            batch=batch,
            generation_config=generation_config,
            kwargs=kwargs,
            rng=rng_summary(torch_fake),
        )
        torch_fake.consume_random_draw()
        return torch_fake.Tensor("generated_ids", (1, GENERATED_LENGTH))


def _model_loader(name: str) -> type:
    """Create a fake model class called ``name`` with a recording ``from_pretrained``."""

    def from_pretrained(cls: type, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> FakeModel:
        RECORDER.record(
            f"{name}.from_pretrained",
            pretrained=pretrained_model_name_or_path,
            args=args,
            kwargs=kwargs,
            rng=rng_summary(torch_fake),
        )
        return FakeModel(name, pretrained_model_name_or_path)

    return type(name, (), {"from_pretrained": classmethod(from_pretrained)})


class AutoConfig:
    """Fake ``AutoConfig``."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> FakeConfig:
        """Record the call and return a :class:`FakeConfig`."""
        RECORDER.record("AutoConfig.from_pretrained", pretrained=pretrained_model_name_or_path, args=args, kwargs=kwargs)
        return FakeConfig(pretrained_model_name_or_path)


class AutoTokenizer:
    """Fake ``AutoTokenizer``."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> FakeTokenizer:
        """Record the call and return a :class:`FakeTokenizer`."""
        RECORDER.record("AutoTokenizer.from_pretrained", pretrained=pretrained_model_name_or_path, args=args, kwargs=kwargs)
        return FakeTokenizer(pretrained_model_name_or_path)


class AutoProcessor:
    """Fake ``AutoProcessor``."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> FakeProcessor:
        """Record the call and return a :class:`FakeProcessor`."""
        RECORDER.record("AutoProcessor.from_pretrained", pretrained=pretrained_model_name_or_path, args=args, kwargs=kwargs)
        return FakeProcessor(pretrained_model_name_or_path)


class GenerationConfig(Snapshottable):
    """Fake ``GenerationConfig`` that keeps its keyword arguments."""

    def __init__(self, **kwargs: Any) -> None:
        """Record the configuration values."""
        RECORDER.record("GenerationConfig", kwargs=kwargs)
        self.kwargs = kwargs

    def snapshot(self) -> dict[str, Any]:
        """Describe the configuration by its values."""
        return {"GenerationConfig": snapshot(self.kwargs)}


def load_image(image: str) -> FakeImage:
    """Record loading ``image`` and return a :class:`FakeImage`."""
    RECORDER.record("load_image", image=image)
    return FakeImage(image)


MODEL_LOADER_NAMES = (
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
    "Idefics2ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
)


def build_modules() -> dict[str, ModuleType]:
    """Return fresh ``transformers`` and ``transformers.image_utils`` modules."""
    image_utils = make_module("transformers.image_utils", load_image=load_image)
    loaders = {name: _model_loader(name) for name in MODEL_LOADER_NAMES}
    helpers = {cls.__name__: cls for cls in (AutoConfig, AutoTokenizer, AutoProcessor, GenerationConfig)}
    transformers = make_module("transformers", image_utils=image_utils, **loaders, **helpers)
    return {"transformers": transformers, "transformers.image_utils": image_utils}
