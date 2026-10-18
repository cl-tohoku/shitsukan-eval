"""Perception selection in English (see :mod:`perception_selection`)."""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import ChatModel, Record
from shitsukan_eval.tasks.perception.selection.perception_selection import run_perception_selection
from tools.utils.data_utils import set_seed_all

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

# Instructions for the subsets that have exactly one correct choice.
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_EN = (
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n"
    "Select and answer one correct word that represents the essence of the object recalled from the following words.\n"
    "Please be sure to answer with the number that corresponds to your choice."
)

# Instructions for the ``random`` subset, which may have no correct choice.
TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_EN = (
    "The essence of an object refers to the following:\n"
    "- Physical properties (gloss, transparency, etc.)\n"
    "- Condition (dry, frozen, etc.)\n"
    "- Impression (beautiful, ugly, etc.)\n"
    "Select and answer one correct word that represents the essence of the object recalled from the following words.\n"
    "Please be sure to answer with the number that corresponds to your choice.\n"
    'If there is no applicable option, please answer "-1".'
)


def perception_selection_en(
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
    """Run English perception selection and save results and scores.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed ``perception_selection_en.yaml``.
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
        system_prompt=TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_EN,
        random_subset_system_prompt=TASK_DESCRIPTION_FOR_PERCEPTION_SELECTION_RANDOM_EN,
        verbose=verbose,
        save_dir=save_dir,
        task=task,
        sub_task=sub_task,
        sub_subtasks=sub_subtasks,
        lang=lang,
        model_type=model_type,
        model_name=model_name,
    )
