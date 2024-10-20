import os
import pathlib
from typing import Any

# api models
from models.api_models.gpt4o import GPT4o
from models.hf_models.gemma2_instruct_hf import Gemma2InstructHf

# huggingface models
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

# vllm models
from models.vllm_models.qwen2_vl_vllm import Qwen2VLVllm
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification_en import (
    commonsense_classification_en,
)
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification_ja import (
    commonsense_classification_ja,
)
from shitsukan_eval.tasks.commonsense.generation.commonsense_generation_ja import (
    commonsense_generation_ja,
)

# local function
from shitsukan_eval.tasks.perception.generation.perception_generation_ja import (
    perception_generation_ja,
)
from shitsukan_eval.tasks.perception.selection.perception_selection_en import (
    perception_selection_en,
)
from shitsukan_eval.tasks.perception.selection.perception_selection_ja import (
    perception_selection_ja,
)
from shitsukan_eval.tasks.taxonomic.a_b_classification.taxonomic_a_b_classification_ja import (
    taxonomic_a_b_classification_ja,
)
from shitsukan_eval.tasks.taxonomic.multiple_choice_classification.taxonomic_multiple_choice_classification_ja import (
    taxonomic_multiple_choice_classification_ja,
)
from shitsukan_eval.tasks.taxonomic.yes_no_classification.taxonomic_yes_no_classification_ja import (
    taxonomic_yes_no_classification_ja,
)
from tools.utils.data_utils import load_yaml, save_results_as_jsonl, set_seed_all

# Set a global random seed for reproducibility
set_seed_all(42)


def task_and_lang_to_func(task: str, sub_task: str, lang: str):
    """
    Maps task, sub-task, and language to the corresponding evaluation function.

    Args:
        task (str): The evaluation task (e.g., "perception", "commonsense").
        sub_task (str): The sub-task (e.g., "generation", "selection").
        lang (str): The language code (e.g., "ja" for Japanese, "en" for English).

    Returns:
        Callable: The corresponding function for the task and sub-task.

    Raises:
        ValueError: If the task, sub-task, or language is not supported.

    Example usage:
    >>> func = task_and_lang_to_func("perception", "generation", "ja")
    >>> print(func)  # Outputs the corresponding function for perception generation in Japanese.
    """
    # マッピング辞書の作成
    func_map = {
        ("ja", "perception", "generation"): perception_generation_ja,
        ("ja", "perception", "selection"): perception_selection_ja,
        ("ja", "commonsense", "generation"): commonsense_generation_ja,
        ("ja", "commonsense", "classification"): commonsense_classification_ja,
        ("ja", "taxonomic", "a_b_classification"): taxonomic_a_b_classification_ja,
        (
            "ja",
            "taxonomic",
            "yes_no_classification",
        ): taxonomic_yes_no_classification_ja,
        (
            "ja",
            "taxonomic",
            "multiple_choice_classification",
        ): taxonomic_multiple_choice_classification_ja,
        ("en", "perception", "selection"): perception_selection_en,
        ("en", "commonsense", "classification"): commonsense_classification_en,
    }

    # マッピング辞書から関数を取得
    func = func_map.get((lang, task, sub_task))

    if func is None:
        # サポートされていない組み合わせの場合にエラーを発生させる
        raise ValueError(f"Error! The selected combination of language ({lang}), task ({task}), " f"and sub-task ({sub_task}) doesn't exist.")

    return func


def task_and_lang_to_config(task: str, sub_task: str, lang: str) -> dict[str, Any]:
    """
    Loads the configuration for a specific task, sub-task, and language.

    Args:
        task (str): The evaluation task (e.g., "perception", "commonsense").
        sub_task (str): The sub-task (e.g., "generation", "selection").
        lang (str): The language code (e.g., "ja" for Japanese, "en" for English").

    Returns:
        Dict[str, Any]: The task configuration loaded from a YAML file.

    Raises:
        ValueError: If the task, sub-task, or language is not supported.
    """
    # YAMLファイルのパスをマッピング
    config_map = {
        (
            "ja",
            "perception",
            "generation",
        ): "./shitsukan_eval/tasks/perception/generation/perception_generation_ja.yaml",
        (
            "ja",
            "perception",
            "selection",
        ): "./shitsukan_eval/tasks/perception/selection/perception_selection_ja.yaml",
        (
            "ja",
            "commonsense",
            "generation",
        ): "./shitsukan_eval/tasks/commonsense/generation/commonsense_generation_ja.yaml",
        (
            "ja",
            "commonsense",
            "classification",
        ): "./shitsukan_eval/tasks/commonsense/classification/commonsense_classification_ja.yaml",
        (
            "ja",
            "taxonomic",
            "a_b_classification",
        ): "./shitsukan_eval/tasks/taxonomic/a_b_classification/taxonomic_a_b_classification_ja.yaml",
        (
            "ja",
            "taxonomic",
            "yes_no_classification",
        ): "./shitsukan_eval/tasks/taxonomic/yes_no_classification/taxonomic_yes_no_classification_ja.yaml",
        (
            "ja",
            "taxonomic",
            "multiple_choice_classification",
        ): "./shitsukan_eval/tasks/taxonomic/multiple_choice_classification/taxonomic_multiple_choice_classification_ja.yaml",
        (
            "en",
            "perception",
            "selection",
        ): "./shitsukan_eval/tasks/perception/selection/perception_selection_en.yaml",
        (
            "en",
            "commonsense",
            "classification",
        ): "./shitsukan_eval/tasks/commonsense/classification/commonsense_classification_en.yaml",
    }

    # 辞書からパスを取得
    config_path = config_map.get((lang, task, sub_task))

    if not config_path or not os.path.exists(config_path):
        raise ValueError(f"Error! The selected combination of language ({lang}), task ({task}), " f"and sub-task ({sub_task}) doesn't exist.")

    # YAMLファイルを読み込む
    return load_yaml(config_path)


def load_hf_models(model_name: str, image_dir: str):
    # Load the model's default configuration
    hf_model_config = load_yaml("./models/hf_models/hf_model_default_config.yaml")

    # Normalize the model name
    model_name_lower = model_name.lower()

    # Initialize the model based on the model name
    if "llama-3.2" in model_name_lower and "vision" in model_name_lower:
        if "instruct" in model_name_lower:
            # Force eager attention for Mllama because MllamaForConditionalGeneration does not support Flash Attention 2.0 yet.
            hf_model_config["attn_implementation"] = "eager"
            model = MllamaInstructHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "llava" in model_name_lower:
        if "1.5" in model_name_lower or "v1.5" in model_name_lower:
            model = LlavaHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        elif "next" in model_name_lower or "1.6" in model_name_lower or "v1.6" in model_name_lower:
            model = LlavaNextHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        elif "onevision" in model_name_lower:
            model = LlavaOnevisionHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "idefics2" in model_name_lower:
        model = Idefics2Hf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "idefics3" in model_name_lower:
        model = Idefics3Hf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "qwen2-vl" in model_name_lower:
        model = Qwen2VLHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "molmo" in model_name_lower:
        # [!] When running Molmo, please explicitly specify a single GPU, such as `CUDA_VISIBLE_DEVICES=1`.
        hf_model_config["dtype"] = "auto"
        hf_model_config["device_map"] = "auto"
        # Force eager attention for Molmo because MolmoForCausalLM does not support Flash Attention 2.0 yet.
        hf_model_config["attn_implementation"] = "eager"
        hf_model_config["trust_remote_code"] = True
        model = MolmoHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "llama-3" in model_name_lower:
        if "instruct" in model_name_lower:
            model = Llama3InstructHf(pretrained=model_name, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "llama-2" in model_name_lower:
        if "chat" in model_name_lower or "instruct" in model_name_lower:
            model = Llama2InstructHf(pretrained=model_name, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "gemma-2" in model_name_lower:
        model = Gemma2InstructHf(pretrained=model_name, **hf_model_config)
    elif "qwen2.5" in model_name_lower:
        if "instruct" in model_name_lower:
            model = Qwen25InstructHf(pretrained=model_name, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "qwen2" in model_name_lower:
        if "instruct" in model_name_lower:
            model = Qwen2InstructHf(pretrained=model_name, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    elif "llm-jp-3" in model_name_lower:
        if "instruct" in model_name_lower:
            model = LlmJp3InstructHf(pretrained=model_name, **hf_model_config)
        else:
            raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")
    else:
        raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")

    return model


def load_vllm_models(model_name: str, image_dir: str):
    # Load the model's default configuration
    vllm_model_config = load_yaml("./models/vllm_models/vllm_model_default_config.yaml")

    # Normalize the model name
    model_name_lower = model_name.lower()

    # Initialize the model based on the model name
    if "llama-3.2" in model_name_lower and "vision" in model_name_lower:
        if "instruct" in model_name_lower:
            model = MllamaInstructHf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
    elif "llava" in model_name_lower:
        if "1.5" in model_name_lower or "v1.5" in model_name_lower:
            model = LlavaHf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
        elif "next" in model_name_lower or "1.6" in model_name_lower or "v1.6" in model_name_lower:
            model = LlavaNextHf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
        elif "onevision" in model_name_lower:
            model = LlavaOnevisionHf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
    elif "idefics2" in model_name_lower:
        model = Idefics2Hf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
    elif "idefics3" in model_name_lower:
        model = Idefics3Hf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
    elif "qwen2-vl" in model_name_lower:
        model = Qwen2VLVllm(
            pretrained=model_name,
            tokenizer=model_name,
            image_dir=image_dir,
            **vllm_model_config,
        )
    elif "molmo" in model_name_lower:
        model = MolmoHf(pretrained=model_name, image_dir=image_dir, **vllm_model_config)
    else:
        raise ValueError(f"Error! The selected model ({model_name}) doesn't support vllm.")

    return model


def load_api_models(model_name: str, image_dir: str):
    # Normalize the model name
    model_name_lower = model_name.lower()

    # Initialize the model based on the model name
    if "gpt-4" in model_name_lower:
        model = GPT4o(pretrained=model_name, image_dir=image_dir)
    else:
        raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")

    return model


def evaluate(
    model_name: str,
    model_type: str,
    tasks: list[str],
    sub_tasks: list[str],
    lang: str,
    image_dir: str,
    save_dir: str,
    verbose: bool,
) -> None:
    """
    Evaluates the specified model on a list of tasks and sub-tasks.

    Args:
        model_name (str): The name of the model to evaluate.
        model_type (str): The type of the model (availabel choices: ["hf", "vllm", "api"])
        tasks (List[str]): A list of tasks to evaluate (e.g., "perception").
        sub_tasks (List[str]): A list of sub-tasks to evaluate (e.g., "generation").
        lang (str): The language of evaluation (e.g., "ja" for Japanese, "en" for English).
        image_dir (str): Directory where input images are stored.
        save_dir (str): Directory where evaluation results will be saved.
        verbose (bool): If True, detailed information is printed during evaluation.

    Returns:
        None: The evaluation results are saved as JSONL files in the specified save directory.

    Example usage:
    >>> evaluate(
    >>>     model_name="gpt-4o",
    >>>     model_type="hf",
    >>>     tasks=["perception"],
    >>>     sub_tasks=["generation"],
    >>>     lang="ja",
    >>>     image_dir="data/images",
    >>>     save_dir="outputs",
    >>>     verbose=True
    >>> )
    """
    if model_type == "hf":
        model = load_hf_models(model_name, image_dir)
    elif model_type == "vllm":
        model = load_vllm_models(model_name, image_dir)
    elif model_type == "api":
        model = load_api_models(model_name, image_dir)
    else:
        raise ValueError(f"Error! The selected model_type ({model_type}) isn't available.") from None

    # Evaluate the model on the specified tasks and sub-tasks
    for task in tasks:
        for sub_task in sub_tasks:
            # Get the appropriate evaluation function and task configuration
            function = task_and_lang_to_func(task=task, sub_task=sub_task, lang=lang)
            task_config = task_and_lang_to_config(task=task, sub_task=sub_task, lang=lang)

            # Get the simple model name
            simple_model_name = pathlib.Path(model_name).name

            # Run the evaluation function and get the results
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

            # Save the results in JSONL format
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
