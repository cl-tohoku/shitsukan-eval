"""Characterization tests (observational records) for the model wrappers.

Real models cannot run in unit tests, so these tests pin down what each wrapper
sends to the model libraries: loader arguments, chat messages, preprocessing
inputs, generation parameters, and decoding options. Together these determine the
generated text, and therefore the evaluation scores, for a given model checkpoint.
The expected values in ``tests/golden/models`` were recorded from the original
implementation.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.fakes.recorder import RECORDER, RESPONDER, SETTINGS
from tests.golden import check_golden, text_lines, to_record

# Values passed by ``shitsukan_eval.evaluator`` (copied from the default YAML configs).
HF_DEFAULT_CONFIG = {
    "revision": "main",
    "device": "cuda",
    "dtype": "bfloat16",
    "low_cpu_mem_usage": True,
    "batch_size": 1,
    "trust_remote_code": False,
    "device_map": {"": 0},
    "use_cache": True,
}
MLLAMA_CONFIG = {**HF_DEFAULT_CONFIG, "attn_implementation": "eager"}
MOLMO_CONFIG = {**HF_DEFAULT_CONFIG, "dtype": "auto", "device_map": "auto", "attn_implementation": "eager", "trust_remote_code": True}
VLLM_DEFAULT_CONFIG = {
    "revision": "main",
    "tokenizer_revision": "main",
    "tokenizer_mode": "auto",
    "trust_remote_code": False,
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "seed": 42,
    "gpu_memory_utilization": 0.9,
    "swap_space": 4,
    "cpu_offload_gb": 0,
    "max_seq_len_to_capture": 8192,
    "limit_mm_per_prompt": {"image": 10, "video": 1},
}
# Generation parameters as written in every task YAML file.
TASK_GENERATION_KWARGS = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.01, "top_p": 1.0, "num_beams": 1}

LOCAL_IMAGE = "images/coco/train2017/000000000001.jpg"
REMOTE_IMAGE = "http://images.example.com/train2017/000000000002.jpg"
FEW_SHOT_IMAGE = "images/few_shot.jpg"
SYSTEM_PROMPT = "Choose the texture word."
TEXT_WITH_IMAGE_TOKEN = "Word: skateboard <image>\nTexture choice: 0:smooth, 1:youthful\nAnswer: "
FEW_SHOT_DATA = [
    {"user": "Word: apple\nTexture choice: 0:smooth, 1:dry\nAnswer: ", "assistant": "0", "images": None},
    {"user": "Word: horn <image>\nTexture choice: 0:rough, 1:prickly\nAnswer: ", "assistant": "1", "images": [FEW_SHOT_IMAGE]},
]
RESPONSE = " 1 "


@dataclass(frozen=True)
class WrapperCase:
    """One model wrapper and the arguments ``shitsukan_eval.evaluator`` passes to it."""

    module: str
    class_name: str
    pretrained: str
    kind: str  # "text", "vision", "api", or "vllm"
    evaluator_config: dict[str, Any] = field(default_factory=dict)

    def evaluator_kwargs(self, image_dir: str) -> dict[str, Any]:
        """Return the keyword arguments used by the evaluator for this wrapper."""
        kwargs: dict[str, Any] = {"pretrained": self.pretrained, **self.evaluator_config}
        if self.kind == "vllm":
            kwargs["tokenizer"] = self.pretrained
        if self.kind != "text":
            kwargs["image_dir"] = image_dir
        return kwargs


CASES = {
    "llama2_instruct": WrapperCase("models.hf_models.llama2_instruct_hf", "Llama2InstructHf", "meta-llama/Llama-2-7b-chat-hf", "text", HF_DEFAULT_CONFIG),
    "llama3_instruct": WrapperCase(
        "models.hf_models.llama3_instruct_hf", "Llama3InstructHf", "meta-llama/Meta-Llama-3.1-8B-Instruct", "text", HF_DEFAULT_CONFIG
    ),
    "qwen2_instruct": WrapperCase("models.hf_models.qwen2_instruct_hf", "Qwen2InstructHf", "Qwen/Qwen2-7B-Instruct", "text", HF_DEFAULT_CONFIG),
    "qwen2_5_instruct": WrapperCase("models.hf_models.qwen2_5_instruct_hf", "Qwen25InstructHf", "Qwen/Qwen2.5-7B-Instruct", "text", HF_DEFAULT_CONFIG),
    "gemma2_instruct": WrapperCase("models.hf_models.gemma2_instruct_hf", "Gemma2InstructHf", "google/gemma-2-9b-it", "text", HF_DEFAULT_CONFIG),
    "llm_jp_3_instruct": WrapperCase("models.hf_models.llm_jp_3_instruct_hf", "LlmJp3InstructHf", "llm-jp/llm-jp-3-13b-instruct", "text", HF_DEFAULT_CONFIG),
    "llava": WrapperCase("models.hf_models.llava_hf", "LlavaHf", "llava-hf/llava-1.5-7b-hf", "vision", HF_DEFAULT_CONFIG),
    "llava_next": WrapperCase("models.hf_models.llava_next_hf", "LlavaNextHf", "llava-hf/llava-v1.6-mistral-7b-hf", "vision", HF_DEFAULT_CONFIG),
    "llava_onevision": WrapperCase(
        "models.hf_models.llava_onevision_hf", "LlavaOnevisionHf", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "vision", HF_DEFAULT_CONFIG
    ),
    "idefics2": WrapperCase("models.hf_models.idefics2_hf", "Idefics2Hf", "HuggingFaceM4/idefics2-8b", "vision", HF_DEFAULT_CONFIG),
    "idefics3": WrapperCase("models.hf_models.idefics3_hf", "Idefics3Hf", "HuggingFaceM4/Idefics3-8B-Llama3", "vision", HF_DEFAULT_CONFIG),
    "mllama": WrapperCase(
        "models.hf_models.llama3_vision_instruct_hf", "MllamaInstructHf", "meta-llama/Llama-3.2-11B-Vision-Instruct", "vision", MLLAMA_CONFIG
    ),
    "qwen2_vl": WrapperCase("models.hf_models.qwen2_vl_hf", "Qwen2VLHf", "Qwen/Qwen2-VL-7B-Instruct", "vision", HF_DEFAULT_CONFIG),
    "molmo": WrapperCase("models.hf_models.molmo_hf", "MolmoHf", "allenai/Molmo-7B-D-0924", "vision", MOLMO_CONFIG),
    "gpt4o": WrapperCase("models.api_models.gpt4o", "GPT4o", "gpt-4o-2024-08-06", "api"),
    "qwen2_vl_vllm": WrapperCase("models.vllm_models.qwen2_vl_vllm", "Qwen2VLVllm", "Qwen/Qwen2-VL-7B-Instruct", "vllm", VLLM_DEFAULT_CONFIG),
}
TEXT_CASES = [name for name, case in CASES.items() if case.kind == "text"]
VISION_CASES = [name for name, case in CASES.items() if case.kind == "vision"]
HF_CASES = TEXT_CASES + VISION_CASES
IMAGE_CASES = [name for name, case in CASES.items() if case.kind != "text"]
LLAVA_CASES = ["llava", "llava_next", "llava_onevision"]
IDEFICS_CASES = ["idefics2", "idefics3"]

# Public attributes that callers (tasks, evaluator, users) may read.
PUBLIC_ATTRIBUTES = (
    "pretrained",
    "image_dir",
    "device_map",
    "use_cache",
    "batch_size_per_gpu",
    "patch_size",
    "vision_feature_select_strategy",
    "image_token",
)
PUBLIC_PROPERTIES = ("config", "tokenizer", "model", "eot_token_id", "batch_size", "device")
_ABSENT = "<absent>"


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Create an image directory with the local images used by the scenarios."""
    root = tmp_path / "image_root"
    for relative, content in ((LOCAL_IMAGE, b"local-image-bytes"), (FEW_SHOT_IMAGE, b"few-shot-image-bytes")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _wrapper_class(case: WrapperCase, import_fresh: Callable[[str], Any]) -> type:
    """Import the wrapper module against the fakes and return the wrapper class."""
    return getattr(import_fresh(case.module), case.class_name)


def _describe_instance(model: Any) -> dict[str, Any]:
    """Describe the wrapper through its public attributes and properties."""
    description: dict[str, Any] = {"class": type(model).__name__}
    for name in PUBLIC_ATTRIBUTES:
        description[name] = getattr(model, name, _ABSENT)
    for name in PUBLIC_PROPERTIES:
        try:
            description[name] = getattr(model, name)
        except AttributeError as error:
            description[name] = {"AttributeError": str(error)}
    return description


def _observe(action: Callable[[], Any], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> dict[str, Any]:
    """Run ``action`` and record events, result or error, and stdout."""
    RECORDER.reset()
    result: Any = None
    error: dict[str, str] | None = None
    try:
        result = action()
    except Exception as exc:  # noqa: BLE001 - errors are part of the observation.
        error = {"type": type(exc).__name__, "message": str(exc)}
    stdout = text_lines(capsys.readouterr().out)
    return to_record({"events": RECORDER.events, "result": result, "error": error, "stdout": stdout}, {str(tmp_path): "<tmp>"})


def _build(case_name: str, import_fresh: Callable[[str], Any], image_dir: Path) -> Any:
    """Construct the wrapper as the evaluator does and forget the construction events."""
    case = CASES[case_name]
    model = _wrapper_class(case, import_fresh)(**case.evaluator_kwargs(str(image_dir)))
    RECORDER.reset()
    return model


# --- construction -----------------------------------------------------------------------
@pytest.mark.parametrize("case_name", list(CASES))
def test_construction_with_evaluator_arguments(case_name, import_fresh, image_dir, capsys, tmp_path):
    case = CASES[case_name]
    wrapper_class = _wrapper_class(case, import_fresh)

    observed = _observe(lambda: _describe_instance(wrapper_class(**case.evaluator_kwargs(str(image_dir)))), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_with_evaluator_arguments", observed)


@pytest.mark.parametrize("fake_libraries", [True], indirect=True)
@pytest.mark.parametrize("case_name", HF_CASES)
def test_construction_when_flash_attn_is_installed(case_name, fake_libraries, import_fresh, image_dir, capsys, tmp_path):
    case = CASES[case_name]
    wrapper_class = _wrapper_class(case, import_fresh)

    observed = _observe(lambda: _describe_instance(wrapper_class(**case.evaluator_kwargs(str(image_dir)))), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_when_flash_attn_is_installed", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_construction_with_default_arguments(case_name, import_fresh, capsys, tmp_path):
    wrapper_class = _wrapper_class(CASES[case_name], import_fresh)

    observed = _observe(lambda: _describe_instance(wrapper_class()), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_with_default_arguments", observed)


@pytest.mark.parametrize("case_name", TEXT_CASES)
def test_construction_keeps_or_replaces_an_existing_chat_template(case_name, import_fresh, image_dir, capsys, tmp_path):
    SETTINGS["tokenizer_chat_template"] = "{{ existing template }}"
    case = CASES[case_name]
    wrapper_class = _wrapper_class(case, import_fresh)

    observed = _observe(lambda: _describe_instance(wrapper_class(**case.evaluator_kwargs(str(image_dir)))), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_with_existing_chat_template", observed)


@pytest.mark.parametrize("case_name", LLAVA_CASES)
def test_construction_falls_back_when_config_lacks_vision_attributes(case_name, import_fresh, image_dir, capsys, tmp_path):
    SETTINGS["config_has_vision_attributes"] = False
    case = CASES[case_name]
    wrapper_class = _wrapper_class(case, import_fresh)

    observed = _observe(lambda: _describe_instance(wrapper_class(**case.evaluator_kwargs(str(image_dir)))), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_without_vision_config_attributes", observed)


@pytest.mark.parametrize("case_name", HF_CASES)
def test_construction_rejects_unexpected_keyword_arguments(case_name, import_fresh, capsys, tmp_path):
    wrapper_class = _wrapper_class(CASES[case_name], import_fresh)

    observed = _observe(lambda: wrapper_class(unexpected_option=1), capsys, tmp_path)

    check_golden(f"models/{case_name}/construction_with_unexpected_kwargs", observed)


# --- generation -------------------------------------------------------------------------
@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_with_system_prompt_few_shots_and_images(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(
        lambda: model.generate_response(
            text=TEXT_WITH_IMAGE_TOKEN,
            images=[LOCAL_IMAGE, REMOTE_IMAGE],
            system_prompt=SYSTEM_PROMPT,
            few_shot_data=FEW_SHOT_DATA,
            gen_kwargs=dict(TASK_GENERATION_KWARGS),
        ),
        capsys,
        tmp_path,
    )

    check_golden(f"models/{case_name}/generate_full_prompt", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_with_text_only_and_default_generation_kwargs(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Word: apple\nAnswer: "), capsys, tmp_path)

    check_golden(f"models/{case_name}/generate_text_only_defaults", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_with_sampling_disabled(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Word: apple\nAnswer: ", gen_kwargs={"do_sample": False}), capsys, tmp_path)

    check_golden(f"models/{case_name}/generate_sampling_disabled", observed)


@pytest.mark.parametrize("case_name", ["gemma2_instruct"])
@pytest.mark.parametrize(
    ("scenario", "system_prompt", "few_shot_data"),
    [("system_prompt_only", SYSTEM_PROMPT, None), ("few_shots_only", None, FEW_SHOT_DATA)],
)
def test_gemma2_merges_the_system_prompt_into_the_first_user_turn(case_name, scenario, system_prompt, few_shot_data, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(
        lambda: model.generate_response(text="Word: apple\nAnswer: ", system_prompt=system_prompt, few_shot_data=few_shot_data),
        capsys,
        tmp_path,
    )

    check_golden(f"models/{case_name}/generate_{scenario}", observed)


@pytest.mark.parametrize("case_name", IDEFICS_CASES)
@pytest.mark.parametrize(("scenario", "text"), [("tokens_match_images", "<image><image>Describe."), ("no_tokens", "Describe.")])
def test_idefics_adds_image_placeholders_for_images_without_tokens(case_name, scenario, text, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text=text, images=[LOCAL_IMAGE, REMOTE_IMAGE]), capsys, tmp_path)

    check_golden(f"models/{case_name}/generate_image_{scenario}", observed)


@pytest.mark.parametrize("case_name", IMAGE_CASES)
def test_generate_response_rejects_more_image_tokens_than_images(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)

    observed = _observe(lambda: model.generate_response(text="<image> <image> <image>", images=[LOCAL_IMAGE]), capsys, tmp_path)

    check_golden(f"models/{case_name}/error_too_many_image_tokens", observed)


@pytest.mark.parametrize("case_name", IMAGE_CASES)
def test_generate_response_rejects_missing_local_images(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Describe.", images=["images/missing.jpg"]), capsys, tmp_path)

    check_golden(f"models/{case_name}/error_missing_image", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_effect_on_the_callers_generation_kwargs(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE, RESPONSE])
    gen_kwargs = dict(TASK_GENERATION_KWARGS)

    def generate_twice() -> dict[str, Any]:
        model.generate_response(text="first", gen_kwargs=gen_kwargs)
        model.generate_response(text="second", gen_kwargs=gen_kwargs)
        return gen_kwargs

    observed = _observe(generate_twice, capsys, tmp_path)

    check_golden(f"models/{case_name}/gen_kwargs_after_two_calls", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_with_empty_images_empty_system_prompt_and_plain_few_shots(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])
    few_shots_without_images_key = [{"user": "Word: apple\nAnswer: ", "assistant": "0"}]

    observed = _observe(
        lambda: model.generate_response(text="Word: horn\nAnswer: ", images=[], system_prompt="", few_shot_data=few_shots_without_images_key),
        capsys,
        tmp_path,
    )

    check_golden(f"models/{case_name}/generate_edge_inputs", observed)


@pytest.mark.parametrize("case_name", list(CASES))
def test_generate_response_leaves_its_inputs_unchanged(case_name, import_fresh, image_dir):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script([RESPONSE])
    images = [LOCAL_IMAGE, REMOTE_IMAGE]
    few_shot_data = copy.deepcopy(FEW_SHOT_DATA)

    model.generate_response(text=TEXT_WITH_IMAGE_TOKEN, images=images, system_prompt=SYSTEM_PROMPT, few_shot_data=few_shot_data)

    assert images == [LOCAL_IMAGE, REMOTE_IMAGE]
    assert few_shot_data == FEW_SHOT_DATA


@pytest.mark.parametrize("case_name", [*HF_CASES, "qwen2_vl_vllm"])
def test_token_helpers(case_name, import_fresh, image_dir, capsys, tmp_path):
    model = _build(case_name, import_fresh, image_dir)
    RESPONDER.script(["decoded text"])

    observed = _observe(
        lambda: {
            "encoded": model.tok_encode("abcd"),
            "encoded_left_truncated": model.tok_encode("abcd", left_truncate_len=2, add_special_tokens=True),
            "decoded": model.tok_decode([1, 2, 3]),
        },
        capsys,
        tmp_path,
    )

    check_golden(f"models/{case_name}/token_helpers", observed)


# --- OpenAI-specific behavior -------------------------------------------------------------
@pytest.fixture
def recorded_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace :func:`asyncio.sleep` so retries are recorded instead of waited for."""

    async def sleep(seconds: float) -> None:
        RECORDER.record("asyncio.sleep", seconds=seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)


@pytest.mark.parametrize(
    ("scenario", "failures"),
    [
        ("retry_then_success", ["RateLimitError", "APIConnectionError"]),
        ("not_found_error", ["NotFoundError"]),
        ("retries_exhausted", ["APIStatusError"] * 5),
    ],
)
def test_gpt4o_retries_failed_requests(scenario, failures, recorded_sleep, import_fresh, image_dir, capsys, tmp_path):
    model = _build("gpt4o", import_fresh, image_dir)
    SETTINGS["openai_failures"] = list(failures)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Word: apple\nAnswer: "), capsys, tmp_path)

    check_golden(f"models/gpt4o/{scenario}", observed)


def test_gpt4o_reports_failed_image_downloads(import_fresh, image_dir, capsys, tmp_path):
    model = _build("gpt4o", import_fresh, image_dir)
    SETTINGS["requests_status_code"] = 404

    observed = _observe(lambda: model.generate_response(text="Describe.", images=[REMOTE_IMAGE]), capsys, tmp_path)

    check_golden("models/gpt4o/error_image_download_failed", observed)


def test_gpt4o_passes_the_api_key_from_the_environment(monkeypatch, import_fresh, image_dir, capsys, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    model = _build("gpt4o", import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Word: apple\nAnswer: "), capsys, tmp_path)

    check_golden("models/gpt4o/api_key_from_environment", observed)


def test_qwen2_vl_vllm_keeps_generation_kwargs_without_max_new_tokens(import_fresh, image_dir, capsys, tmp_path):
    model = _build("qwen2_vl_vllm", import_fresh, image_dir)
    RESPONDER.script([RESPONSE])

    observed = _observe(lambda: model.generate_response(text="Word: apple\nAnswer: ", gen_kwargs={"temperature": 0.5}), capsys, tmp_path)

    check_golden("models/qwen2_vl_vllm/generate_without_max_new_tokens", observed)
