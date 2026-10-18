"""Perception selection in Japanese (see :mod:`perception_selection`)."""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import ChatModel, Record
from shitsukan_eval.tasks.perception.selection.perception_selection import run_perception_selection

# Instructions for the subsets that have exactly one correct choice.
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の単語から想起される質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。"
)

# Instructions for the ``random`` subset, which may have no correct choice.
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の単語から想起される質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。\n"
    'ただし、該当する選択肢がない場合は"-1"と答えてください。'
)


def perception_selection_ja(
    model: ChatModel,
    task_config: dict[str, Any],
    verbose: bool,
    save_dir: str,
    task: str,
    sub_task: str,
    sub_subtasks: list[str] | None,
    lang: str,
    model_type: str,
    model_name: str,
) -> list[Record]:
    """Run Japanese perception selection and save results and scores.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed ``perception_selection_ja.yaml``.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for result and score files.
        task: Task name written to the outputs (``"perception"``).
        sub_task: Sub-task name written to the outputs (``"selection"``).
        sub_subtasks: Subsets to run (default: all five).
        lang: Language code written to the outputs.
        model_type: Model type used in output file names.
        model_name: Model name written to the outputs and file names.

    Returns:
        The records of all subsets with the prompt, response, and metadata added.
    """
    return run_perception_selection(
        model,
        task_config,
        system_prompt=TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_JA,
        random_subset_system_prompt=TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_JA,
        verbose=verbose,
        save_dir=save_dir,
        task=task,
        sub_task=sub_task,
        sub_subtasks=sub_subtasks,
        lang=lang,
        model_type=model_type,
        model_name=model_name,
    )
