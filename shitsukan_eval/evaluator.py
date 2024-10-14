import pathlib
from typing import Any

from models.gpt4o import GPT4o
from models.idefics2 import Idefics2
from models.idefics3 import Idefics3
from models.llama3_vision_instruct import MllamaInstructHf
from models.llava_hf import LlavaHf
from models.llava_next_hf import LlavaNextHf
from models.llava_onevision_hf import LlavaOnevisionHf
from models.molmo import Molmo
from models.qwen2_vl import Qwen2VL
from shitsukan_eval.tasks.perception.generation.perception_generation_ja import perception_generation_ja
from shitsukan_eval.tasks.perception.selection.perception_selection_en import perception_selection_en
from shitsukan_eval.tasks.perception.selection.perception_selection_ja import perception_selection_ja
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
    if lang == "ja":
        if task == "perception":
            if sub_task == "generation":
                return perception_generation_ja
            elif sub_task == "selection":
                return perception_selection_ja
            else:
                raise ValueError(f"Error! The selected sub-task ({sub_task}) doesn't exist.")
        elif task == "commonsense":
            pass  # Placeholder for future tasks
        elif task == "taxonomic":
            pass
        else:
            raise ValueError(f"Error! The selected task ({task}) doesn't exist.")
    elif lang == "en":
        if task == "perception":
            if sub_task == "selection":
                return perception_selection_en
            else:
                raise ValueError(f"Error! The selected sub-task ({sub_task}) doesn't exist.")
        elif task == "commonsense":
            pass
        elif task == "taxonomic":
            pass
        else:
            raise ValueError(f"Error! The selected task ({task}) doesn't exist.")
    else:
        raise ValueError(f"Error! The selected language ({lang}) doesn't exist.")


def task_and_lang_to_config(task: str, sub_task: str, lang: str) -> dict[str, Any]:
    """
    Loads the configuration for a specific task, sub-task, and language.

    Args:
        task (str): The evaluation task (e.g., "perception", "commonsense").
        sub_task (str): The sub-task (e.g., "generation", "selection").
        lang (str): The language code (e.g., "ja" for Japanese, "en" for English).

    Returns:
        Dict[str, Any]: The task configuration loaded from a YAML file.

    Raises:
        ValueError: If the task, sub-task, or language is not supported.

    Example usage:
    >>> config = task_and_lang_to_config("perception", "generation", "ja")
    >>> print(config)  # Outputs the task configuration for perception generation in Japanese.
    """
    if lang == "ja":
        if task == "perception":
            if sub_task == "generation":
                task_config = load_yaml("./shitsukan_eval/tasks/perception/generation/perception_generation_ja.yaml")
            elif sub_task == "selection":
                task_config = load_yaml("./shitsukan_eval/tasks/perception/selection/perception_selection_ja.yaml")
            else:
                raise ValueError(f"Error! The selected sub-task ({sub_task}) doesn't exist.")
        elif task == "commonsense":
            pass  # Placeholder for future tasks
        elif task == "taxonomic":
            pass
        else:
            raise ValueError(f"Error! The selected task ({task}) doesn't exist.")
    elif lang == "en":
        if task == "perception":
            if sub_task == "selection":
                task_config = load_yaml("./shitsukan_eval/tasks/perception/selection/perception_selection_en.yaml")
            else:
                raise ValueError(f"Error! The selected sub-task ({sub_task}) doesn't exist.")
        elif task == "commonsense":
            pass
        elif task == "taxonomic":
            pass
        else:
            raise ValueError(f"Error! The selected task ({task}) doesn't exist.")
    else:
        raise ValueError(f"Error! The selected language ({lang}) doesn't exist.")

    return task_config


def evaluate(model_name: str, tasks: list[str], sub_tasks: list[str], lang: str, image_dir: str, save_dir: str, verbose: bool) -> None:
    """
    Evaluates the specified model on a list of tasks and sub-tasks.

    Args:
        model_name (str): The name of the model to evaluate.
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
    >>>     tasks=["perception"],
    >>>     sub_tasks=["generation"],
    >>>     lang="ja",
    >>>     image_dir="data/images",
    >>>     save_dir="outputs",
    >>>     verbose=True
    >>> )
    """
    # Load the model's default configuration
    hf_model_config = load_yaml("./models/hf_model_default_config.yaml")

    # Normalize the model name
    model_name_lower = model_name.lower()

    # Initialize the model based on the model name
    if "gpt-4" in model_name_lower:
        model = GPT4o(pretrained=model_name, image_dir=image_dir)
    elif "llama-3.2" in model_name_lower and "vision" in model_name_lower:
        if "instruct" in model_name_lower:
            # Force eager attention for Mllama because MllamaForConditionalGeneration does not support Flash Attention 2.0 yet.
            hf_model_config["attn_implementation"] = "eager"
            model = MllamaInstructHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "llava" in model_name_lower:
        if "1.5" in model_name_lower or "v1.5" in model_name_lower:
            model = LlavaHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        elif "next" in model_name_lower or "1.6" in model_name_lower or "v1.6" in model_name_lower:
            model = LlavaNextHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
        elif "onevision" in model_name_lower:
            model = LlavaOnevisionHf(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "idefics2" in model_name_lower:
        model = Idefics2(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "idefics3" in model_name_lower:
        model = Idefics3(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "qwen2-vl" in model_name_lower:
        model = Qwen2VL(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    elif "molmo" in model_name_lower:
        # [!] When running Molmo, please explicitly specify a single GPU, such as `CUDA_VISIBLE_DEVICES=1`.
        hf_model_config["dtype"] = "auto"
        hf_model_config["device_map"] = "auto"
        # Force eager attention for Molmo because MolmoForCausalLM does not support Flash Attention 2.0 yet.
        hf_model_config["attn_implementation"] = "eager"
        hf_model_config["trust_remote_code"] = True
        model = Molmo(pretrained=model_name, image_dir=image_dir, **hf_model_config)
    else:
        raise ValueError(f"Error! The selected model ({model_name}) doesn't exist.")

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
                model_name=simple_model_name,
            )

            # Save the results in JSONL format
            save_results_as_jsonl(save_dir=save_dir, results=results, task=task, sub_task=sub_task, sub_subtask=None, lang=lang, model_name=simple_model_name)
