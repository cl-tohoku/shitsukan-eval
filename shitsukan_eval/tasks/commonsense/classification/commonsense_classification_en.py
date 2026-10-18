"""Commonsense classification in English, scored by accuracy."""

from __future__ import annotations

import pathlib
from typing import Any

from shitsukan_eval.tasks.common import PretrainedChatModel, Record, RunInfo, print_score_report
from shitsukan_eval.tasks.commonsense.classification.commonsense_classification import classify_naturalness, naturalness_accuracy
from tools.utils.data_utils import save_eval_scores_as_jsonl

TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_EN = (
    "For this task, you will be asked to describe the shitsukan terms that is evoked by the impression of the object, rather than the shitsukan of the image itself.\n"
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n\n"
    "Are the following combinations of objects and essence natural and common?\n"
    '* If the word indicated does not correspond to a essence, please select "5. not a shitsukan term at all".'
)


def build_classification_prompt(record: Record) -> str:
    """Return the query text with the object-texture pair and the five choices."""
    return f"Combinations:\n{record['object']}: {record['shitsukan']}\n\nChoices:\n1. completely natural\n2. generally possible but not common\n3. possible but not common\n4. unnatural\n5. not a shitsukan term at all\n\nAnswer: "


def get_eval_scores(data_list: list[Record]) -> dict[str, Any]:
    """Return the accuracy and the run metadata."""
    scores = naturalness_accuracy(data_list)
    metadata = data_list[-1]
    return {
        **scores,
        "task": metadata["task"],
        "sub_task": metadata["sub_task"],
        "lang": metadata["lang"],
        "model_name": metadata["model_name"],
    }


def commonsense_classification_en(
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
    """Run English commonsense classification and save the scores.

    Args:
        model: Model wrapper with a ``pretrained`` attribute; its last path
            component is written as ``model_name``.
        task_config: Parsed ``commonsense_classification_en.yaml``.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for the score file.
        task: Task name written to the outputs.
        sub_task: Sub-task name written to the outputs.
        sub_subtasks: Unused; accepted for a uniform task interface.
        lang: Language code written to the outputs.
        model_type: Model type used in the score file name.
        model_name: Unused; the name is derived from ``model.pretrained``.

    Returns:
        The records with the prompt, response, label, and metadata added.
    """
    pretrained_name = pathlib.Path(model.pretrained).name
    records = classify_naturalness(
        model,
        task_config,
        system_prompt=TASK_DESCRIPTION_FOR_COMMONSENSE_CLASSIFICATION_EN,
        build_text=build_classification_prompt,
        verbose=verbose,
        run=RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=pretrained_name),
    )
    scores = get_eval_scores(records)
    print_score_report(pretrained_name, lang, [[f"- Accuracy: {scores['accuracy']}"]])
    # The None defaults only keep the task signature uniform; the evaluator passes every value.
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, pretrained_name)  # ty: ignore[invalid-argument-type]
    return records
