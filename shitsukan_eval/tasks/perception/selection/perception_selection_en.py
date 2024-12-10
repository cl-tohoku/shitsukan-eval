import pathlib
import re
from typing import Any, Dict, List, Optional

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.utils.data_utils import (
    load_dataset,
    save_eval_scores_as_jsonl,
    save_results_as_jsonl,
    set_seed_all,
    simple_choice_extractors,
)

# Set a global random seed
set_seed_all(42)

# Task description for generating perceptual responses in Japanese
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_EN = (
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n"
    "Select and answer one correct word that represents the essence of the object recalled from the following words.\n"
    "Please be sure to answer with the number that corresponds to your choice."
)

TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_EN = (
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n"
    "Select and answer one correct word that represents the essence of the object recalled from the following words.\n"
    "Please be sure to answer with the number that corresponds to your choice.\n"
    'If there is no applicable option, please answer "-1".'
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


def get_eval_scores(data_list: list[dict[str, Any]], sub_subtask_list: list[str]) -> list[dict[str, Any]]:
    """
    Compute evaluation scores for the given dataset and sub-tasks.

    Args:
        data_list (List[Dict[str, Any]]): List of dictionaries containing the dataset.
        sub_subtask_list (List[str]): List of sub-subtasks to compute scores for.

    Returns:
        List[Dict[str, Any]]: A list of evaluation scores for each sub-task.

    Example usage:
    >>> get_eval_scores(data_list, ["choice2_ans1", "choice3_ans1"])
    [{'accuracy': 0.94, 'task': 'perception', 'sub_task': 'selection', 'sub_subtask': 'choice2_ans1', 'lang': 'en', 'model_name': 'gpt-4'}, ...]
    """
    eval_score_list = []
    pattern = re.compile(r"\({0,1}([A-B]|yes|no)[.,:;)]{0,1}")
    for sub_subtask in sub_subtask_list:
        gold_list = []
        pred_list = []
        for data in data_list:
            if sub_subtask == data["sub_subtask"]:
                if len(data["correct_ids"]) > 0:
                    gold_list.append(int(data["correct_ids"][0]))
                else:
                    gold_list.append(-1)
                if pattern.match(data["response"]):
                    pred_list.append(-100)
                else:
                    pred_list.append(int(data["response"]))

        eval_score = compute_metrics(gold_list, pred_list)
        eval_score.update(
            {
                "task": data["task"],
                "sub_task": data["sub_task"],
                "sub_subtask": sub_subtask,
                "lang": data["lang"],
                "model_name": data["model_name"],
            }
        )
        eval_score_list.append(eval_score)

    return eval_score_list


def perception_selection_en(
    model,
    task_config: dict[str, Any],
    verbose: bool,
    save_dir: str,
    task: str,
    sub_task: str,
    sub_subtasks: list[str] | None,
    lang: str,
    model_type: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Performs a perception selection task by prompting the model with object-related questions.
    The model selects the most appropriate word representing the essence of the object from the given choices.

    Args:
        model (Any): The model used to generate responses.
        task_config (Dict[str, Any]): Task configuration including dataset paths, generation parameters, etc.
        verbose (bool): If True, prints detailed information during processing.
        save_dir (str): Directory to save results.
        task (str): The main task name (e.g., 'perception').
        sub_task (str): The sub-task name (e.g., 'selection').
        sub_subtasks (Optional(List[str]): The sub-subtask name list.
        lang (str): The language of the task (e.g., 'en').
        model_type (str): The type of the model (availabel choices: ["hf", "vllm", "api"])
        model_name (str): The name of the model being evaluated.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the model's responses and corresponding data.

    Example usage:
    >>> model = GPT4o(pretrained="gpt-4o")
    >>> task_config = {
    >>>     "subtask": {"choice2_ans1": {"dataset_path": "data.jsonl", "few_shot_path": "few_shot.jsonl"}, "generation_kwargs": {"max_new_tokens": 50}}
    >>> }
    >>> perception_selection_en(model, task_config, verbose=True, save_dir="results", task="perception", sub_task="selection", sub_subtask=None, lang="en", model_name="gpt-4o")
    """
    if sub_subtasks is None:
        sub_subtask_list = (
            "choice2_ans1",
            "choice3_ans1",
            "choice4_ans1",
            "choice5_ans1",
            "random",
        )
    else:
        sub_subtask_list = sub_subtasks
    new_data_list = []

    for sub_subtask in sub_subtask_list:
        data_list = load_dataset(task_config["subtask"][sub_subtask]["dataset_path"])
        few_shot_list = load_dataset(task_config["subtask"][sub_subtask]["few_shot_path"])

        # Initialize list to store new data with model responses
        tmp_data_list = []

        # Iterate through the dataset and generate model responses
        for data in tqdm(data_list, desc=f"Processing sub-task {sub_subtask}", leave=False):
            """(Example)
            data (Dict[str, Any]):
            {
                'image_id': '286376',
                'image': ['images/coco/train2017/000000286376.jpg'],
                'image_url': ['http://images.cocodataset.org/train2017/000000286376.jpg'],
                'object': 'skateboard',
                'correct_ids': [0],
                'uncorrect_ids': [1],
                'correct_shitsukan_list': ['smooth'],
                'correct_ids_with_shitsukan': ['0:stylish'],
                'uncorrect_ids_with_shitsukan': ['1:smooth'],
                'all_ids_with_shitsukan': ['0:stylish', '1:smooth']
            }
            """
            # Construct the choice string from data
            choices = ", ".join(data["all_ids_with_shitsukan"])

            # Construct the prompt
            text = f'Word: {data["object"]}\nTexture choice: {choices}\nAnswer: '

            # Set the appropriate system prompt
            system_prompt = TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_EN if sub_subtask == "random" else TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_EN

            # Generate the response using the model
            response = model.generate_response(
                text=text,
                images=data.get("image"),  # Optionally load image if provided
                system_prompt=system_prompt,
                few_shot_data=few_shot_list,
                gen_kwargs=task_config["generation_kwargs"],
            )
            """ (Example)
            response (str):
            "0"
            """

            # Print the prompt and response if verbose mode is enabled
            if verbose:
                tqdm.write(f"{text}{simple_choice_extractors(response)}\n")

            # Add response and metadata to data
            data["zero_shot"] = text
            data["original_response"] = response
            data["response"] = simple_choice_extractors(response)
            data["task"] = task
            data["sub_task"] = sub_task
            data["sub_subtask"] = sub_subtask
            data["lang"] = lang
            data["model_name"] = model_name

            tmp_data_list.append(data)

        new_data_list.extend(tmp_data_list)
        save_results_as_jsonl(
            save_dir,
            tmp_data_list,
            task,
            sub_task,
            sub_subtask,
            lang,
            model_type,
            model_name,
        )

    scores_list = get_eval_scores(new_data_list, sub_subtask_list)
    display_string = "=" * 20 + " " * 2 + f"Model Name: {model_name}" + " " * 4 + f"Language: {lang}" + " " * 2 + "=" * 20 + "\n" + "-" * 40
    print(display_string)
    for scores in scores_list:
        display_string = f'- Sub-SubTask: {scores["sub_subtask"]}\n' + f'- Accuracy: {scores["accuracy"]}\n' + "-" * 40
        print(display_string)
    save_eval_scores_as_jsonl(save_dir, scores_list, task, sub_task, lang, model_type, model_name)

    return new_data_list
