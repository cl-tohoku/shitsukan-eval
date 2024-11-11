import pathlib
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.utils.data_utils import (
    load_dataset,
    save_eval_scores_as_jsonl,
    set_seed_all,
    simple_choice_extractors,
)

# Set a global random seed
set_seed_all(42)

# Task description for generating perceptual responses in English
TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_EN = (
    "For this task, you will be asked to describe the shitsukan terms that is evoked by the impression of the object, rather than the shitsukan of the image itself.\n"
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n\n"
    "Are the following combinations of objects and essence natural and common?\n"
    '* If the word indicated does not correspond to a essence, please select "5. not a shitsukan term at all".'
)


def compute_metrics(target: list[int], pred: list[int]) -> dict[str, float]:
    """
    Computes accuracy for predictions against ground truth.

    Args:
        target (List[int]): List of ground truth labels.
        pred (List[int]): List of model predictions.

    Returns:
        Dict[str, float]: A dictionary with accuracy score.

    Example usage:
    >>> compute_metrics([0, 1, 1], [0, 1, 0])
    {'accuracy': 0.6666666666666666}
    """
    accuracy = accuracy_score(target, pred)
    return {"accuracy": accuracy}


def get_eval_scores(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute evaluation scores for the given dataset

    Args:
        data_list (List[Dict[str, Any]]): List of dictionaries containing the dataset.

    Returns:
        Dict[str, Any]: A dict of evaluation scores

    Example usage:
    >>> get_eval_scores(data_list)
    [{'accuracy': 0.94, 'task': 'commonsense', 'sub_task': 'classification', 'lang': 'en', 'model_name': 'gpt-4'}, ...]
    """
    gold_list = []
    pred_list = []
    for data in data_list:
        gold_list.append(int(data["naturalness"]))
        pred_list.append(int(data["model_inference"]))

    eval_score = compute_metrics(gold_list, pred_list)
    eval_score.update(
        {
            "task": data["task"],
            "sub_task": data["sub_task"],
            "lang": data["lang"],
            "model_name": data["model_name"],
        }
    )

    return eval_score


def commonsense_classification_en(
    model,
    task_config: dict[str, Any],
    verbose: bool = False,
    save_dir: str = None,
    task: str = None,
    sub_task: str = None,
    sub_subtasks: str = None,
    lang: str = None,
    model_type: str = None,
    model_name: str = None,
) -> list[dict[str, Any]]:
    """
    Performs a commonsense classification task by prompting the model with object-related questions
    and recording its responses. The model is tasked with generating descriptions of perceived
    textures associated with given words (e.g., objects).

    Args:
        model (Any): The model to use for generating responses.
        task_config (Dict[str, Any]): Configuration for the task. Should include paths to datasets, few-shot examples, and generation parameters.
        verbose (bool, optional): If True, prints the prompt and model responses for debugging. Defaults to False.
        save_dir (Optional[str]): Directory to save the results, if provided.
        task (Optional[str]): Task name for metadata purposes.
        sub_task (Optional[str]): Sub-task name for metadata purposes.
        sub_subtasks (Optional[str]): Sub-subtask name for metadata purposes.
        lang (Optional[str]): Language of the task (e.g., 'en').
        model_type (str): The type of the model (availabel choices: ["hf", "vllm", "api"])
        model_name (Optional[str]): Name of the model being evaluated.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the original data and the model's responses.

    Example usage:
    >>> model = GPT4o(pretrained="gpt-4o")
    >>> task_config = {
    >>>     "dataset_path": "data/dataset.jsonl",
    >>>     "few_shot_path": "data/few_shot.jsonl",
    >>>     "generation_kwargs": {"max_new_tokens": 50}
    >>> }
    >>> commonsense_classification_en(model, task_config, verbose=True)
    [
        {
            'id': 0,
            'object_id': 0,
            'object_shitsukan_id': 0,
            'object': 'icing',
            'shitsukan': 'vivid',
            'naturalness': 2,
            'naturalness_str': 'Possible, but uncommon'
            "zero_shot": "Combinations:\nicing: vivid\n\nChoices:\n1. completely natural\n2. generally possible but not common\n3. possible but not common\n4. unnatural\n5. not a shitsukan term at all\n\nAnswer: ",
            "original_model_inference": "2",
            "model_inference": "2",
            "task": "commonsense",
            "sub_task": "classification",
            "lang": "en",
            "model_name": "Qwen2-VL-7B-Instruct"
        },
        ...
    ]
    """

    # Load dataset and few-shot examples
    data_list = load_dataset(task_config["dataset_path"])
    few_shot_list = load_dataset(task_config["few_shot_path"])

    # Extract model name from the pretrained model's path
    model_name = pathlib.Path(model.pretrained).name

    # Initialize list to store new data with model responses
    new_data_list = []

    # Iterate through the dataset and generate model responses for each item
    for data in tqdm(data_list, desc="Processing dataset", leave=False):
        """(Example)
        data (dict):
        {
            'id': 0,
            'object_id': 0,
            'object_shitsukan_id': 0,
            'object': 'icing',
            'shitsukan': 'vivid',
            'naturalness': 2,
            'naturalness_str': 'Possible, but uncommon'
        }
        """

        # Construct the prompt using the object in the data
        text = f'Combinations:\n{data["object"]}: {data["shitsukan"]}\n\nChoices:\n1. completely natural\n2. generally possible but not common\n3. possible but not common\n4. unnatural\n5. not a shitsukan term at all\n\nAnswer: '

        # Generate the response using the model
        response = model.generate_response(
            text=text,
            images=None,
            system_prompt=TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_EN,  # Task-specific instructions
            few_shot_data=few_shot_list,  # Few-shot examples for better generation
            gen_kwargs=task_config["generation_kwargs"],  # Generation parameters
        )
        """ (Example)
        response (str):
        "2"
        """

        # Print the prompt and response if verbose mode is enabled
        if verbose:
            tqdm.write(f"{text}{simple_choice_extractors(response)}\n")

        # Add response and metadata to data
        data["zero_shot"] = text
        data["original_model_inference"] = response
        data["model_inference"] = simple_choice_extractors(response)
        data["task"] = task
        data["sub_task"] = sub_task
        data["lang"] = lang
        data["model_name"] = model_name

        # Append the updated data to the new data list
        new_data_list.append(data)

    scores = get_eval_scores(new_data_list)
    display_string = "=" * 20 + " " * 2 + f"Model Name: {model_name}" + " " * 4 + f"Language: {lang}" + " " * 2 + "=" * 20 + "\n" + "-" * 40
    print(display_string)
    display_string = f'- Accuracy: {scores["accuracy"]}\n' + "-" * 40
    print(display_string)
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, model_name)

    return new_data_list
