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
TASK_DESCRIPTION_FOR_TAXONOMIC_YES_NO_CLASSIFICATION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の選択肢のうち、質感語として正しいものを1つ選び解答してください。\n"
    '解答の際には、選択肢に対応する英単語 ("yes" もしくは "no") を使って解答してください。'
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
    label_to_id_dict = {"no": 0, "yes": 1, "-100": -100}

    gold_list = []
    pred_list = []
    for data in data_list:
        gold_list.append(int(data["label"]))
        pred_list.append(label_to_id_dict[data["model_inference"]])

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


def taxonomic_yes_no_classification_ja(
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
    >>> taxonomic_yes_no_classification_ja(model, task_config, verbose=True, save_dir="results", task="perception", sub_task="selection", sub_subtask=None, lang="ja", model_name="gpt-4o")
    """
    new_data_list = []

    data_list = load_dataset(task_config["dataset_path"])
    few_shot_list = load_dataset(task_config["few_shot_path"])

    # Iterate through the dataset and generate model responses
    for data in tqdm(data_list, desc="Processing dataset", leave=False):
        """(Example)
        data (Dict[str, Any]):
        {
            'id': 5,
            'shitsukan': '柔らかく',
            'label': 1,
            'label_str': 'yes'
        }
        """
        # Construct the prompt
        text = f'質感語: {data["shitsukan"]}\n解答: '

        # Set the appropriate system prompt
        system_prompt = TASK_DESCRIPTION_FOR_TAXONOMIC_YES_NO_CLASSIFICATION_JA

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
        "yes"
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

        new_data_list.append(data)

    scores = get_eval_scores(new_data_list)
    display_string = "=" * 20 + " " * 2 + f"Model Name: {model_name}" + " " * 4 + f"Language: {lang}" + " " * 2 + "=" * 20 + "\n" + "-" * 40
    print(display_string)
    display_string = f'- Accuracy: {scores["accuracy"]}\n' + "-" * 40
    print(display_string)
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, model_name)

    return new_data_list
