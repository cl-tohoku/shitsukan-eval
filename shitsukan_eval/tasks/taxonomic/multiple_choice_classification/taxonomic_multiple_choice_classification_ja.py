import pathlib
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
TASK_DESCRIPTION_FOR_TAXONOMIC_MULTIPLE_CHOICE_CLASSIFICATION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の選択肢のうち、質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。"
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
    [{'accuracy': 0.94, 'task': 'taxonomic', 'sub_task': 'multiple_choice_classification', 'sub_subtask': 'choice2_ans1', 'lang': 'ja', 'model_name': 'gpt-4'}, ...]
    """
    eval_score_list = []
    for sub_subtask in sub_subtask_list:
        gold_list = []
        pred_list = []
        for data in data_list:
            if sub_subtask == data["sub_subtask"]:
                if len(data["correct_ids"]) > 0:
                    gold_list.append(int(data["correct_ids"][0]))
                else:
                    gold_list.append(-1)
                pred_list.append(int(data["model_inference"]))

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


def taxonomic_multiple_choice_classification_ja(
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
    Performs a taxonomic multiple-choice classification task by prompting the model with object-related questions.
    The model selects the most appropriate word representing the essence of the object from the given choices.

    Args:
        model (Any): The model used to generate responses.
        task_config (Dict[str, Any]): Task configuration including dataset paths, generation parameters, etc.
        verbose (bool): If True, prints detailed information during processing.
        save_dir (str): Directory to save results.
        task (str): The main task name (e.g., 'taxonomic').
        sub_task (str): The sub-task name (e.g., 'multiple_choice_classification').
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
    >>> taxonomic_multiple_choice_classification_ja(model, task_config, verbose=True, save_dir="results", task="taxonomic", sub_task="multiple_choice_classification", sub_subtask=None, lang="ja", model_name="gpt-4o")
    """
    if sub_subtasks is None:
        sub_subtask_list = (
            "choice2_ans1",
            "choice3_ans1",
            "choice4_ans1",
            "choice5_ans1",
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
                'id': 2775,
                'correct_ids': [3],
                'uncorrect_ids': [0, 1, 2, 4],
                'correct_shitsukan_list': ['睦まじい'],
                'uncorrect_shitsukan_list': ['ピチッと', 'センスがいい', '沈着', '静止感'],
                'correct_ids_with_shitsukan': ['3:睦まじい'],
                'uncorrect_ids_with_shitsukan': ['0:ピチッと', '1:センスがいい', '2:沈着', '4:静止感'],
                'all_ids_with_shitsukan': ['0:ピチッと', '1:センスがいい', '2:沈着', '3:睦まじい', '4:静止感']
            }
            """
            # Construct the choice string from data
            choices = ", ".join(data["all_ids_with_shitsukan"])

            # Construct the text
            text = f"質感語: {choices}\n解答: "

            # Set the appropriate system prompt
            system_prompt = TASK_DESCRIPTION_FOR_TAXONOMIC_MULTIPLE_CHOICE_CLASSIFICATION_JA

            # Generate the response using the model
            response = model.generate_response(
                text=text,
                images=None,
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
            data["original_model_inference"] = response
            data["model_inference"] = simple_choice_extractors(response)
            data["task"] = task
            data["sub_task"] = sub_task
            data["sub_subtask"] = sub_subtask
            data["lang"] = lang
            data["model_name"] = model_name

            tmp_data_list.append(data)

        new_data_list.extend(tmp_data_list)
        # Save the tmp results in JSONL format
        save_results_as_jsonl(
            save_dir=save_dir,
            results=tmp_data_list,
            task=task,
            sub_task=sub_task,
            sub_subtask=sub_subtask,
            lang=lang,
            model_type=model_type,
            model_name=model_name,
        )

    scores_list = get_eval_scores(new_data_list, sub_subtask_list)
    display_string = "=" * 20 + " " * 2 + f"Model Name: {model_name}" + " " * 4 + f"Language: {lang}" + " " * 2 + "=" * 20 + "\n" + "-" * 40
    print(display_string)
    for scores in scores_list:
        display_string = f'- Sub-SubTask: {scores["sub_subtask"]}\n' + f'- Accuracy: {scores["accuracy"]}\n' + "-" * 40
        print(display_string)
    save_eval_scores_as_jsonl(save_dir, scores_list, task, sub_task, lang, model_type, model_name)

    return new_data_list
