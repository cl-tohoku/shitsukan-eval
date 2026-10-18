"""Commonsense classification in Japanese, scored by accuracy and distribution distance.

Besides accuracy, the Japanese data has the five crowd-worker votes per pair,
which give a second metric: the distance between the predicted label and the
vote distribution (see :func:`calculate_distribution_distance`).
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from scipy import stats

from shitsukan_eval.tasks.common import PretrainedChatModel, Record, RunInfo, print_score_report
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification import classify_naturalness, naturalness_accuracy
from tools.utils.data_utils import save_eval_scores_as_jsonl

TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_JA = (
    "本タスクは「画像」に対する質感ではなく、「オブジェクト」がもつ印象から連想される質感について述べていただきます。\n"
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど）\n\n"
    "以下のモノと質感の組み合わせは、自然で一般的ですか？\n"
    "* 示された単語が質感に当たらない場合は「5.その他：質感ではない」を選択してください。"
)

NATURALNESS_LABELS = frozenset({1, 2, 3, 4, 5})
# Added to the standard deviation of unanimous votes (which is 0) so that the
# density of a different label stays finite.
UNANIMOUS_VOTES_SCALE_EPSILON = 1e-3


def build_classification_prompt(record: Record) -> str:
    """Return the query text with the object-texture pair and the five choices."""
    return f"組み合わせ:\n{record['object']}: {record['shitsukan']}\n\n選択肢:\n1.完全に自然\n2.一般的だが、他もあり得る\n3.あり得るが一般的ではない\n4.不自然\n5.その他：質感ではない\n\n回答: "


def _distribution_distance(gold: int, pred: int, votes: list[int]) -> float:
    """Return how far ``pred`` is from ``gold`` under a normal fit of ``votes``.

    The votes define a normal distribution with their mean and sample standard
    deviation; the distance is the absolute difference between the densities at
    the gold and the predicted labels. For unanimous votes the gold density is
    defined as 1, and a different prediction uses a tiny standard deviation.
    """
    mean = np.mean(votes)
    standard_deviation = np.std(votes, ddof=1)
    if len(np.unique(votes)) == 1:
        gold_density = np.array(1.0)
        if gold == pred:
            pred_density = np.array(1.0)
        else:
            pred_density = stats.norm.pdf(x=pred, loc=mean, scale=standard_deviation + UNANIMOUS_VOTES_SCALE_EPSILON)
    else:
        gold_density = stats.norm.pdf(x=gold, loc=mean, scale=standard_deviation)
        pred_density = stats.norm.pdf(x=pred, loc=mean, scale=standard_deviation)
    return np.abs(gold_density - pred_density)


def calculate_distribution_distance(data_list: list[Record]) -> list[Record]:
    """Add ``distribution_distance`` to every record (``None`` for labels outside 1-5).

    Args:
        data_list: Records with ``naturalness``, ``model_inference``, and
            ``naturalness_by_worker`` (the five votes).

    Returns:
        The same records, updated in place.

    Raises:
        ValueError: If a label is not an integer.
    """
    for record in data_list:
        gold = int(record["naturalness"])
        pred = int(record["model_inference"])
        if pred in NATURALNESS_LABELS:
            record["distribution_distance"] = _distribution_distance(gold, pred, record["naturalness_by_worker"])
        else:
            record["distribution_distance"] = None
    return data_list


def get_eval_scores(data_list: list[Record]) -> dict[str, Any]:
    """Return accuracy, the mean distribution distance, and the run metadata.

    Raises:
        TypeError: If a distance is ``None`` (a label outside 1-5), because the
            mean is taken over all records.
    """
    scores = naturalness_accuracy(data_list)
    scores["average_distance"] = np.array([record["distribution_distance"] for record in data_list]).mean()
    metadata = data_list[-1]
    return {
        **scores,
        "task": metadata["task"],
        "sub_task": metadata["sub_task"],
        "lang": metadata["lang"],
        "model_name": metadata["model_name"],
    }


def commonsense_classification_ja(
    model: PretrainedChatModel,
    task_config: dict[str, Any],
    verbose: bool = False,
    save_dir: str | None = None,
    task: str | None = None,
    sub_task: str | None = None,
    sub_subtasks: str | None = None,
    lang: str | None = None,
    model_type: str | None = None,
    model_name: str | None = None,
) -> list[Record]:
    """Run Japanese commonsense classification and save the scores.

    Args:
        model: Model wrapper with a ``pretrained`` attribute; its last path
            component is written as ``model_name``.
        task_config: Parsed ``commonsense_classification_ja.yaml``.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for the score file.
        task: Task name written to the outputs.
        sub_task: Sub-task name written to the outputs.
        sub_subtasks: Unused; accepted for a uniform task interface.
        lang: Language code written to the outputs.
        model_type: Model type used in the score file name.
        model_name: Unused; the name is derived from ``model.pretrained``.

    Returns:
        The records with the prompt, response, label, distance, and metadata added.
    """
    pretrained_name = pathlib.Path(model.pretrained).name
    records = classify_naturalness(
        model,
        task_config,
        system_prompt=TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_JA,
        build_text=build_classification_prompt,
        verbose=verbose,
        run=RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=pretrained_name),
    )
    records = calculate_distribution_distance(records)
    scores = get_eval_scores(records)
    print_score_report(pretrained_name, lang, [[f"- Accuracy: {scores['accuracy']}", f"- Average Distance: {scores['average_distance']}"]])
    # The None defaults only keep the task signature uniform; the evaluator passes every value.
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, pretrained_name)  # ty: ignore[invalid-argument-type]
    return records
