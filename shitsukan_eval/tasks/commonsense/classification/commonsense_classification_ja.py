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

# Task description for generating perceptual responses in Japanese
TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_JA = (
    "本タスクは「画像」に対する質感ではなく、「オブジェクト」がもつ印象から連想される質感について述べていただきます。\n"
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど）\n\n"
    "以下のモノと質感の組み合わせは、自然で一般的ですか？\n"
    "* 示された単語が質感に当たらない場合は「5.その他：質感ではない」を選択してください。"
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
    [{'accuracy': 0.94, 'average_distance': 0.365, 'task': 'commonsense', 'sub_task': 'classification', 'lang': 'ja', 'model_name': 'gpt-4'}, ...]
    """
    gold_list = []
    pred_list = []
    distance_list = []
    for data in data_list:
        gold_list.append(int(data["naturalness"]))
        pred_list.append(int(data["model_inference"]))
        distance_list.append(data["distribution_distance"])

    eval_score = compute_metrics(gold_list, pred_list)
    eval_score.update(
        {
            "average_distance": np.array(distance_list).mean(),
            "task": data["task"],
            "sub_task": data["sub_task"],
            "lang": data["lang"],
            "model_name": data["model_name"],
        }
    )

    return eval_score


# 予測させたラベルと正解ラベルの分布との距離を測る評価指標を定義し、スコアを計算する関数
def caluculate_distribution_distance(data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_data_list = list()
    for data in data_list:
        gold = int(data["naturalness"])  # 2
        pred = int(data["model_inference"])  # 1

        if pred not in {1, 2, 3, 4, 5}:
            data["distribution_distance"] = None
        else:
            vote_list = data["naturalness_by_worker"]  # [3, 2, 2, 2, 4]

            # 期待値・標準偏差を計算
            mu = np.mean(vote_list)  # 2.6
            sigma = np.std(vote_list, ddof=1)  # 0.8944271909999159

            # 正解ラベルの(期待値・標準偏差から作られる)正規分布での確率密度を計算する (range(0-1))
            # 予測ラベルの(期待値・標準偏差から作られる)正規分布での確率密度を計算する (range(0-1))
            ## 5人のworkerのvoteが全て同じラベルに振れらている場合、標準偏差sigmaが0.00..となり、確率密度を計算時にオーバーフローする可能性があるため、
            ## 5人のworkerのvoteが全て同じラベルに振れらている場合とそうでない場合で場合分けする
            if len(np.unique(vote_list)) == 1:
                f_most_freq = np.array(1.0)
                if gold == pred:
                    f_model_pred = np.array(1.0)
                else:
                    f_model_pred = stats.norm.pdf(x=pred, loc=mu, scale=sigma + 1e-3)
            else:
                f_most_freq = stats.norm.pdf(x=gold, loc=mu, scale=sigma)
                f_model_pred = stats.norm.pdf(x=pred, loc=mu, scale=sigma)

            # 正解ラベルの確率密度と予測ラベルの確率密度の差を取ることで、{object, shitsukan}　のペア１件ごとの距離を算出できる
            data["distribution_distance"] = np.abs(f_most_freq - f_model_pred)

        new_data_list.append(data)
    return new_data_list


def commonsense_classification_ja(
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
        lang (Optional[str]): Language of the task (e.g., 'ja').
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
    >>> commonsense_classification_ja(model, task_config, verbose=True)
    [
        {
            "id": 0,
            "object_id": 0,
            "object_shitsukan_id": 0,
            "object": "アイシング",
            "shitsukan": "鮮やかな",
            "naturalness": 2,
            "naturalness_str": "Possible, but uncommon",
            "naturalness_by_worker": [5, 2, 2, 4, 3],
            "zero_shot": "組み合わせ:\nアイシング: 鮮やかな\n\n選択肢:\n1.完全に自然\n2.一般的だが、他もあり得る\n3.あり得るが一般的ではない\n4.不自然\n5.その他：質感ではない\n\n回答: ",
            "original_model_inference": "2",
            "model_inference": "2",
            "task": "commonsense",
            "sub_task": "classification",
            "lang": "ja",
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
            "id": 0,
            "object_id": 0,
            "object_shitsukan_id": 0,
            "object": "アイシング",
            "shitsukan": "鮮やかな",
            "naturalness": 2,
            "naturalness_str": "Possible, but uncommon",
            "naturalness_by_worker": [5, 2, 2, 4, 3]
        }
        """

        # Construct the prompt using the object in the data
        text = f'組み合わせ:\n{data["object"]}: {data["shitsukan"]}\n\n選択肢:\n1.完全に自然\n2.一般的だが、他もあり得る\n3.あり得るが一般的ではない\n4.不自然\n5.その他：質感ではない\n\n回答: '

        # Generate the response using the model
        response = model.generate_response(
            text=text,
            images=None,
            system_prompt=TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_JA,  # Task-specific instructions
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

    new_data_list = caluculate_distribution_distance(new_data_list)

    scores = get_eval_scores(new_data_list)
    display_string = "=" * 20 + " " * 2 + f"Model Name: {model_name}" + " " * 4 + f"Language: {lang}" + " " * 2 + "=" * 20 + "\n" + "-" * 40
    print(display_string)
    display_string = f'- Accuracy: {scores["accuracy"]}\n' + f'- Average Distance: {scores["average_distance"]}\n' + "-" * 40
    print(display_string)
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, model_name)

    return new_data_list
