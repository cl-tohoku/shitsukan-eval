"""Perception selection: pick the texture word that a pictured object evokes.

Each record shows an object (with its image) and numbered texture-word choices;
the model answers with the number of the correct choice. Records of the
``random`` subset may have no correct choice, in which case the expected answer
is ``-1``. Results are saved per subset; accuracy is reported per subset.
"""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import (
    INVALID_PREDICTION,
    ChatModel,
    Record,
    RunInfo,
    accuracy_scores,
    collect_labels,
    echo_if_verbose,
    load_records_and_few_shots,
    print_score_report,
    query_records,
)
from tools.utils.data_utils import save_eval_scores_as_jsonl, save_results_as_jsonl, simple_choice_extractors

SUBSETS = ("choice2_ans1", "choice3_ans1", "choice4_ans1", "choice5_ans1", "random")
"""Subsets evaluated when none are requested: 2 to 5 choices with one answer, and ``random``."""

RANDOM_SUBSET = "random"
NO_APPLICABLE_CHOICE = -1
# Letters and yes/no are answers to other tasks; they count as wrong here.
NON_NUMERIC_LABELS = frozenset({"A", "B", "yes", "no"})


def build_selection_prompt(record: Record) -> str:
    """Return the query text listing the object and its numbered texture choices."""
    choices = ", ".join(record["all_ids_with_shitsukan"])
    return f"Word: {record['object']}\nTexture choice: {choices}\nAnswer: "


def gold_choice(record: Record) -> int:
    """Return the first correct choice of ``record``, or ``-1`` if no choice is correct."""
    if record["correct_ids"]:
        return int(record["correct_ids"][0])
    return NO_APPLICABLE_CHOICE


def predicted_choice(label: str) -> int:
    """Convert a parsed answer label into a choice number (``-100`` for letters and yes/no)."""
    if label in NON_NUMERIC_LABELS:
        return INVALID_PREDICTION
    return int(label)


def get_eval_scores(data_list: list[Record], sub_subtask_list: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Return the accuracy of each subset, followed by the run metadata of the records.

    Args:
        data_list: Annotated records of all subsets.
        sub_subtask_list: Subsets to score, in report order.

    Raises:
        ValueError: If a subset has no records.
    """
    scores = []
    for sub_subtask in sub_subtask_list:
        subset = [record for record in data_list if record["sub_subtask"] == sub_subtask]
        gold_labels, predicted_labels = collect_labels(subset, gold_choice, lambda record: predicted_choice(record["response"]))
        accuracy = accuracy_scores(gold_labels, predicted_labels)
        metadata = data_list[-1]
        scores.append(
            {
                **accuracy,
                "task": metadata["task"],
                "sub_task": metadata["sub_task"],
                "sub_subtask": sub_subtask,
                "lang": metadata["lang"],
                "model_name": metadata["model_name"],
            }
        )
    return scores


def run_perception_selection(
    model: ChatModel,
    task_config: dict[str, Any],
    *,
    system_prompt: str,
    random_subset_system_prompt: str,
    verbose: bool,
    save_dir: str,
    task: str,
    sub_task: str,
    sub_subtasks: list[str] | None,
    lang: str,
    model_type: str,
    model_name: str,
) -> list[Record]:
    """Run perception selection for the requested subsets and save results and scores.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed task YAML with ``subtask.<name>.dataset_path``,
            ``subtask.<name>.few_shot_path``, and ``generation_kwargs``.
        system_prompt: Task instructions for subsets with a correct choice.
        random_subset_system_prompt: Task instructions for the ``random`` subset,
            which also explain the ``-1`` answer.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for result and score files.
        task: Task name written to the outputs.
        sub_task: Sub-task name written to the outputs.
        sub_subtasks: Subsets to run (default: :data:`SUBSETS`).
        lang: Language code written to the outputs.
        model_type: Model type used in output file names.
        model_name: Model name written to the outputs and file names.

    Returns:
        The records of all subsets with the prompt, response, and metadata added.
    """
    subsets = SUBSETS if sub_subtasks is None else sub_subtasks
    run = RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=model_name)
    all_records: list[Record] = []
    for subset in subsets:
        records, few_shot_data = load_records_and_few_shots(task_config["subtask"][subset])
        for record, text, response in query_records(
            model,
            records,
            build_text=build_selection_prompt,
            images_for=lambda record: record.get("image"),
            system_prompt=random_subset_system_prompt if subset == RANDOM_SUBSET else system_prompt,
            few_shot_data=few_shot_data,
            gen_kwargs=task_config["generation_kwargs"],
            progress_description=f"Processing sub-task {subset}",
        ):
            label = simple_choice_extractors(response)
            echo_if_verbose(verbose, text, label)
            record.update({"zero_shot": text, "original_response": response, "response": label, **run.metadata(subset)})
        all_records.extend(records)
        save_results_as_jsonl(save_dir, records, task, sub_task, subset, lang, model_type, model_name)

    scores = get_eval_scores(all_records, subsets)
    print_score_report(model_name, lang, ([f"- Sub-SubTask: {score['sub_subtask']}", f"- Accuracy: {score['accuracy']}"] for score in scores))
    save_eval_scores_as_jsonl(save_dir, scores, task, sub_task, lang, model_type, model_name)
    return all_records
