"""Run the Shitsukan evaluation: load a model, run the requested tasks, save results.

Paths to the YAML configurations are relative to the repository root, so run the
evaluation from there (see ``python -m shitsukan_eval --help``).

Every wrapper and task module is imported here eagerly, as in the scripts used
for the paper, so all model libraries are loaded before :func:`evaluate` seeds
the random number generators and loads the model.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from models.api_models.gpt4o import GPT4o
from models.hf_models.gemma2_instruct_hf import Gemma2InstructHf
from models.hf_models.idefics2_hf import Idefics2Hf
from models.hf_models.idefics3_hf import Idefics3Hf
from models.hf_models.llama2_instruct_hf import Llama2InstructHf
from models.hf_models.llama3_instruct_hf import Llama3InstructHf
from models.hf_models.llama3_vision_instruct_hf import MllamaInstructHf
from models.hf_models.llava_hf import LlavaHf
from models.hf_models.llava_next_hf import LlavaNextHf
from models.hf_models.llava_onevision_hf import LlavaOnevisionHf
from models.hf_models.llm_jp_3_instruct_hf import LlmJp3InstructHf
from models.hf_models.molmo_hf import MolmoHf
from models.hf_models.qwen2_5_instruct_hf import Qwen25InstructHf
from models.hf_models.qwen2_instruct_hf import Qwen2InstructHf
from models.hf_models.qwen2_vl_hf import Qwen2VLHf
from models.vllm_models.qwen2_vl_vllm import Qwen2VLVllm
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification_en import commonsense_classification_en
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification_ja import commonsense_classification_ja
from shitsukan_eval.tasks.commonsense.generation.commonsense_generation_ja import commonsense_generation_ja
from shitsukan_eval.tasks.perception.generation.perception_generation_ja import perception_generation_ja
from shitsukan_eval.tasks.perception.selection.perception_selection_en import perception_selection_en
from shitsukan_eval.tasks.perception.selection.perception_selection_ja import perception_selection_ja
from shitsukan_eval.tasks.taxonomic.a_b_classification.taxonomic_a_b_classification_ja import taxonomic_a_b_classification_ja
from shitsukan_eval.tasks.taxonomic.multiple_choice_classification.taxonomic_multiple_choice_classification_ja import (
    taxonomic_multiple_choice_classification_ja,
)
from shitsukan_eval.tasks.taxonomic.yes_no_classification.taxonomic_yes_no_classification_ja import taxonomic_yes_no_classification_ja
from tools.utils.data_utils import load_yaml, save_results_as_jsonl, set_seed_all

SEED = 42
"""Seed of Python's, NumPy's, and PyTorch's generators, set before each model is loaded."""

TaskFunction = Callable[..., list[dict[str, Any]]]

TASK_FUNCTIONS: dict[tuple[str, str, str], TaskFunction] = {
    ("ja", "perception", "generation"): perception_generation_ja,
    ("ja", "perception", "selection"): perception_selection_ja,
    ("ja", "commonsense", "generation"): commonsense_generation_ja,
    ("ja", "commonsense", "classification"): commonsense_classification_ja,
    ("ja", "taxonomic", "a_b_classification"): taxonomic_a_b_classification_ja,
    ("ja", "taxonomic", "yes_no_classification"): taxonomic_yes_no_classification_ja,
    ("ja", "taxonomic", "multiple_choice_classification"): taxonomic_multiple_choice_classification_ja,
    ("en", "perception", "selection"): perception_selection_en,
    ("en", "commonsense", "classification"): commonsense_classification_en,
}
"""Supported ``(lang, task, sub_task)`` combinations and the functions that run them."""

HF_MODEL_DEFAULT_CONFIG = "./models/hf_models/hf_model_default_config.yaml"
VLLM_MODEL_DEFAULT_CONFIG = "./models/vllm_models/vllm_model_default_config.yaml"

# MllamaForConditionalGeneration did not support FlashAttention-2.
MLLAMA_CONFIG_OVERRIDES: dict[str, Any] = {"attn_implementation": "eager"}
# Molmo runs its own code from the Hub and was run with eager attention; make a
# single GPU visible (for example CUDA_VISIBLE_DEVICES=0) when using it.
MOLMO_CONFIG_OVERRIDES: dict[str, Any] = {"dtype": "auto", "device_map": "auto", "attn_implementation": "eager", "trust_remote_code": True}


def _unsupported_combination_error(task: str, sub_task: str, lang: str) -> ValueError:
    """Return the error for a ``(lang, task, sub_task)`` combination without an implementation."""
    return ValueError(f"Error! The selected combination of language ({lang}), task ({task}), and sub-task ({sub_task}) doesn't exist.")


def task_and_lang_to_func(task: str, sub_task: str, lang: str) -> TaskFunction:
    """Return the function that runs ``task``/``sub_task`` in ``lang``.

    Args:
        task: Task name: ``"perception"``, ``"commonsense"``, or ``"taxonomic"``.
        sub_task: Sub-task name, such as ``"generation"`` or ``"selection"``.
        lang: Language code, ``"ja"`` or ``"en"``.

    Returns:
        The task function from :data:`TASK_FUNCTIONS`.

    Raises:
        ValueError: If the combination is not supported.

    Example:
        >>> task_and_lang_to_func("perception", "generation", "ja").__name__
        'perception_generation_ja'
    """
    function = TASK_FUNCTIONS.get((lang, task, sub_task))
    if function is None:
        raise _unsupported_combination_error(task, sub_task, lang)
    return function


def task_config_path(task: str, sub_task: str, lang: str) -> str:
    """Return the YAML path ``./shitsukan_eval/tasks/<task>/<sub_task>/<task>_<sub_task>_<lang>.yaml``."""
    return f"./shitsukan_eval/tasks/{task}/{sub_task}/{task}_{sub_task}_{lang}.yaml"


def task_and_lang_to_config(task: str, sub_task: str, lang: str) -> dict[str, Any]:
    """Load the YAML configuration of ``task``/``sub_task`` in ``lang``.

    Args:
        task: Task name: ``"perception"``, ``"commonsense"``, or ``"taxonomic"``.
        sub_task: Sub-task name, such as ``"generation"`` or ``"selection"``.
        lang: Language code, ``"ja"`` or ``"en"``.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the combination is not supported or its YAML file is missing.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if (lang, task, sub_task) not in TASK_FUNCTIONS:
        raise _unsupported_combination_error(task, sub_task, lang)
    config_path = task_config_path(task, sub_task, lang)
    if not os.path.exists(config_path):
        raise _unsupported_combination_error(task, sub_task, lang)
    return load_yaml(config_path)


def _unknown_model_error(model_name: str) -> ValueError:
    """Return the error for a model name that no wrapper handles."""
    return ValueError(f"Error! The selected model ({model_name}) doesn't exist.")


@dataclass(frozen=True)
class HfWrapperRule:
    """Selects a wrapper when every keyword occurs in the lowercase model name.

    Attributes:
        keywords: Substrings that must all occur in the model name.
        wrapper: Wrapper class to use, or ``None`` to reject the model.
        required_any: If not empty, at least one of these substrings must also
            occur; otherwise the model is rejected.
        takes_images: Whether the wrapper receives ``image_dir``.
        overrides: Values that replace the default model configuration.
    """

    keywords: tuple[str, ...]
    wrapper: type | None
    required_any: tuple[str, ...] = ()
    takes_images: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


HF_WRAPPER_RULES = (
    HfWrapperRule(("llama-3.2", "vision"), MllamaInstructHf, required_any=("instruct",), takes_images=True, overrides=MLLAMA_CONFIG_OVERRIDES),
    HfWrapperRule(("llava", "1.5"), LlavaHf, takes_images=True),
    HfWrapperRule(("llava", "next"), LlavaNextHf, takes_images=True),
    HfWrapperRule(("llava", "1.6"), LlavaNextHf, takes_images=True),
    HfWrapperRule(("llava", "onevision"), LlavaOnevisionHf, takes_images=True),
    HfWrapperRule(("llava",), None),
    HfWrapperRule(("idefics2",), Idefics2Hf, takes_images=True),
    HfWrapperRule(("idefics3",), Idefics3Hf, takes_images=True),
    HfWrapperRule(("qwen2-vl",), Qwen2VLHf, takes_images=True),
    HfWrapperRule(("molmo",), MolmoHf, takes_images=True, overrides=MOLMO_CONFIG_OVERRIDES),
    HfWrapperRule(("llama-3",), Llama3InstructHf, required_any=("instruct",)),
    HfWrapperRule(("llama-2",), Llama2InstructHf, required_any=("chat", "instruct")),
    HfWrapperRule(("gemma-2",), Gemma2InstructHf),
    HfWrapperRule(("qwen2.5",), Qwen25InstructHf, required_any=("instruct",)),
    HfWrapperRule(("qwen2",), Qwen2InstructHf, required_any=("instruct",)),
    HfWrapperRule(("llm-jp-3",), LlmJp3InstructHf, required_any=("instruct",)),
)
"""Wrapper rules in priority order; the first rule whose keywords match decides.

The order matters: Llama 3.2 Vision is matched before other Llama 3 models, and
Qwen2-VL and Qwen2.5 before Qwen2.
"""


def select_hf_wrapper(model_name: str) -> tuple[type, HfWrapperRule]:
    """Return the wrapper class and the first rule of :data:`HF_WRAPPER_RULES` matching ``model_name``.

    Raises:
        ValueError: If no rule matches, or the matching rule rejects the model.
    """
    name = model_name.lower()
    for rule in HF_WRAPPER_RULES:
        if all(keyword in name for keyword in rule.keywords):
            wrapper = rule.wrapper
            if wrapper is None or (rule.required_any and not any(marker in name for marker in rule.required_any)):
                break
            return wrapper, rule
    raise _unknown_model_error(model_name)


def load_hf_models(model_name: str, image_dir: str | None) -> Any:
    """Load a Hugging Face ``transformers`` model with the default configuration.

    Args:
        model_name: Hub name or local path; the wrapper is chosen from it (see
            :data:`HF_WRAPPER_RULES`).
        image_dir: Directory of the images, passed to image-text wrappers.

    Returns:
        The model wrapper.

    Raises:
        ValueError: If no wrapper supports ``model_name``.
    """
    hf_model_config = load_yaml(HF_MODEL_DEFAULT_CONFIG)
    wrapper, rule = select_hf_wrapper(model_name)
    config = {**hf_model_config, **rule.overrides}
    if rule.takes_images:
        return wrapper(pretrained=model_name, image_dir=image_dir, **config)
    return wrapper(pretrained=model_name, **config)


def load_vllm_models(model_name: str, image_dir: str | None) -> Qwen2VLVllm:
    """Load a model with vLLM; only Qwen2-VL is supported.

    Args:
        model_name: Hub name or local path, also used for the tokenizer.
        image_dir: Directory of the images.

    Returns:
        The vLLM wrapper.

    Raises:
        ValueError: If ``model_name`` does not contain ``"qwen2-vl"``.
    """
    vllm_model_config = load_yaml(VLLM_MODEL_DEFAULT_CONFIG)
    # The other wrappers only accept Hugging Face arguments.
    if "qwen2-vl" not in model_name.lower():
        raise ValueError(f"Error! The selected model ({model_name}) doesn't support vllm.")
    return Qwen2VLVllm(pretrained=model_name, tokenizer=model_name, image_dir=image_dir, **vllm_model_config)


def load_api_models(model_name: str, image_dir: str | None) -> GPT4o:
    """Create an OpenAI API model; only GPT-4 family names are supported.

    Raises:
        ValueError: If ``model_name`` does not contain ``"gpt-4"``.
    """
    if "gpt-4" not in model_name.lower():
        raise _unknown_model_error(model_name)
    return GPT4o(pretrained=model_name, image_dir=image_dir)


MODEL_LOADERS: dict[str, Callable[[str, str | None], Any]] = {"hf": load_hf_models, "vllm": load_vllm_models, "api": load_api_models}
"""Model loader for each ``--model-type``."""


def evaluate(
    model_name: str,
    model_type: str,
    tasks: list[str],
    sub_tasks: list[str],
    lang: str,
    image_dir: str | None,
    save_dir: str,
    verbose: bool,
) -> None:
    """Evaluate one model on every combination of ``tasks`` and ``sub_tasks``.

    The random number generators are seeded with :data:`SEED` right before the
    model is loaded, so repeated calls in one process start from the same state.
    The model is loaded once. Each task writes its records to
    ``<save_dir>/<task>/<sub_task>/<task>_<sub_task>_<lang>_<model_type>_<model>.jsonl``
    (where ``<model>`` is the last path component of ``model_name``); tasks with
    subsets and scores write additional files next to it.

    Args:
        model_name: Hub name, local path, or OpenAI model name.
        model_type: ``"hf"``, ``"vllm"``, or ``"api"``.
        tasks: Task names, such as ``["perception"]``.
        sub_tasks: Sub-task names run for every task, such as ``["selection"]``.
        lang: Language code, ``"ja"`` or ``"en"``.
        image_dir: Directory that the relative image paths of the dataset are
            resolved against.
        save_dir: Root directory of the outputs.
        verbose: Print every query with the model's answer.

    Raises:
        ValueError: If the model type, the model, or a task combination is not
            supported. Unsupported tasks are detected after the model is loaded.

    Example:
        >>> evaluate("gpt-4o", "api", ["perception"], ["selection"], "ja", "data", "outputs", verbose=False)
    """
    loader = MODEL_LOADERS.get(model_type)
    if loader is None:
        raise ValueError(f"Error! The selected model_type ({model_type}) isn't available.") from None
    set_seed_all(SEED)
    model = loader(model_name, image_dir)

    simple_model_name = pathlib.Path(model_name).name
    for task in tasks:
        for sub_task in sub_tasks:
            function = task_and_lang_to_func(task=task, sub_task=sub_task, lang=lang)
            task_config = task_and_lang_to_config(task=task, sub_task=sub_task, lang=lang)
            results = function(
                model=model,
                task_config=task_config,
                verbose=verbose,
                save_dir=save_dir,
                task=task,
                sub_task=sub_task,
                sub_subtasks=None,
                lang=lang,
                model_type=model_type,
                model_name=simple_model_name,
            )
            save_results_as_jsonl(
                save_dir=save_dir,
                results=results,
                task=task,
                sub_task=sub_task,
                sub_subtask=None,
                lang=lang,
                model_type=model_type,
                model_name=simple_model_name,
            )
