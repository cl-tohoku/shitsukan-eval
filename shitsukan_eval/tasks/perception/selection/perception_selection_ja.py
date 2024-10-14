import pathlib
from typing import Any, Dict, List, Optional

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.utils.data_utils import load_dataset, save_eval_scores_as_jsonl, save_results_as_jsonl, set_seed_all, simple_choice_extractors

# Set a global random seed
set_seed_all(42)

# Task description for generating perceptual responses in Japanese
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の単語から想起される質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。"
)

TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の単語から想起される質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。\n"
    'ただし、該当する選択肢がない場合は"-1"と答えてください。'
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
    [{'accuracy': 0.94, 'task': 'perception', 'sub_task': 'selection', 'sub_subtask': 'choice2_ans1', 'lang': 'ja', 'model_name': 'gpt-4'}, ...]
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
                pred_list.append(int(data["response"]))

        eval_score = compute_metrics(gold_list, pred_list)
        eval_score.update(
            {"task": data["task"], "sub_task": data["sub_task"], "sub_subtask": sub_subtask, "lang": data["lang"], "model_name": data["model_name"]}
        )
        eval_score_list.append(eval_score)

    return eval_score_list


def perception_selection_ja(
    model,
    task_config: dict[str, Any],
    verbose: bool,
    save_dir: str,
    task: str,
    sub_task: str,
    sub_subtasks: Optional(list[str]) | None,
    lang: str,
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
        model_name (str): The name of the model being evaluated.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the model's responses and corresponding data.

    Example usage:
    >>> model = GPT4o(pretrained="gpt-4o")
    >>> task_config = {
    >>>     "subtask": {"choice2_ans1": {"dataset_path": "data.jsonl", "few_shot_path": "few_shot.jsonl"}, "generation_kwargs": {"max_new_tokens": 50}}
    >>> }
    >>> perception_selection_en(model, task_config, verbose=True, save_dir="results", task="perception", sub_task="selection", sub_subtask=None, lang="ja", model_name="gpt-4o")
    """
    if sub_subtasks is not None:
        sub_subtask_list = ("choice2_ans1", "choice3_ans1", "choice4_ans1", "choice5_ans1", "random")
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
            """ (Example)
            data (Dict[str, Any]):
            {
                'image_id': '286376',
                'image': ['images/coco/train2017/000000286376.jpg'],
                'image_url': ['http://images.cocodataset.org/train2017/000000286376.jpg'],
                'object': 'スケートボード',
                'correct_ids': [0],
                'uncorrect_ids': [1],
                'correct_shitsukan_list': ['さらさら'],
                'correct_ids_with_shitsukan': ['0:若者のイメージ'],
                'uncorrect_ids_with_shitsukan': ['1:さらさら'],
                'all_ids_with_shitsukan': ['0:若者のイメージ', '1:さらさら']
            }
            """
            # Construct the choice string from data
            choices = ", ".join(data["all_ids_with_shitsukan"])

            # Construct the prompt
            text = f'Word: {data["object"]}\nTexture choice: {choices}\nAnswer: '

            # Set the appropriate system prompt
            system_prompt = TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_JA if sub_subtask == "random" else TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_JA

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
                tqdm.write(f"{text}{response}\n")

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
        save_results_as_jsonl(save_dir, tmp_data_list, task, sub_task, sub_subtask, lang, model_name)

    scores_list = get_eval_scores(new_data_list, sub_subtask_list)
    for scores in scores_list:
        display_string = (
            "=" * 40 + f'Model Name: {scores["model_name"]}' + f'Sub-SubTask: {scores["sub_subtask"]}' + "-" * 20 + f'Accuracy: {scores["accuracy"]}' + "=" * 40
        )
        print(display_string)
    save_eval_scores_as_jsonl(save_dir, scores_list, task, sub_task, lang, model_name)

    return new_data_list
