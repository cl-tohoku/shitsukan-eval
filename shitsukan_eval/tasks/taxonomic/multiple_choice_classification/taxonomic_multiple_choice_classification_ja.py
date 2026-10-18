"""Taxonomic multiple-choice classification in Japanese: pick the texture word among 2-5 choices.

Results are saved per subset; accuracy is reported per subset.
"""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import (
    ChatModel,
    Record,
    RunInfo,
    accuracy_scores,
    attach_inference,
    collect_labels,
    echo_if_verbose,
    load_records_and_few_shots,
    print_score_report,
    query_records,
)
from tools.utils.data_utils import save_eval_scores_as_jsonl, save_results_as_jsonl, set_seed_all, simple_choice_extractors

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

TASK_DESCRIPTION_FOR_TAXONOMIC_MULTIPLE_CHOICE_CLASSIFICATION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の選択肢のうち、質感語として正しいものを1つ選び解答してください。\n"
    "解答の際には、選択肢に対応する半角数字を使って解答してください。"
)

SUBSETS = ("choice2_ans1", "choice3_ans1", "choice4_ans1", "choice5_ans1")
"""Subsets evaluated when none are requested: 2 to 5 choices with one correct answer."""

NO_CORRECT_CHOICE = -1


def build_multiple_choice_prompt(record: Record) -> str:
    """Return the query text listing the numbered choices."""
    return f"質感語: {', '.join(record['all_ids_with_shitsukan'])}\n解答: "


def gold_choice(record: Record) -> int:
    """Return the first correct choice of ``record``, or ``-1`` if none is given."""
    if record["correct_ids"]:
        return int(record["correct_ids"][0])
    return NO_CORRECT_CHOICE


def get_eval_scores(data_list: list[Record], sub_subtask_list: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Return the accuracy of each subset, followed by the run metadata of the records.

    Raises:
        ValueError: If a parsed answer is not an integer (for example ``"B"``) or
            a subset has no records.
    """
    scores = []
    for sub_subtask in sub_subtask_list:
        subset = [record for record in data_list if record["sub_subtask"] == sub_subtask]
        gold_labels, predicted_labels = collect_labels(subset, gold_choice, lambda record: int(record["model_inference"]))
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


def taxonomic_multiple_choice_classification_ja(
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
    """Run the multiple-choice task for the requested subsets and save results and scores.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed ``taxonomic_multiple_choice_classification_ja.yaml``.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for result and score files.
        task: Task name written to the outputs.
        sub_task: Sub-task name written to the outputs.
        sub_subtasks: Subsets to run (default: :data:`SUBSETS`).
        lang: Language code written to the outputs.
        model_type: Model type used in output file names.
        model_name: Model name written to the outputs and file names.

    Returns:
        The records of all subsets with the prompt, response, label, and metadata added.
    """
    subsets = SUBSETS if sub_subtasks is None else sub_subtasks
    run = RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=model_name)
    all_records: list[Record] = []
    for subset in subsets:
        records, few_shot_data = load_records_and_few_shots(task_config["subtask"][subset])
        for record, text, response in query_records(
            model,
            records,
            build_text=build_multiple_choice_prompt,
            system_prompt=TASK_DESCRIPTION_FOR_TAXONOMIC_MULTIPLE_CHOICE_CLASSIFICATION_JA,
            few_shot_data=few_shot_data,
            gen_kwargs=task_config["generation_kwargs"],
            progress_description=f"Processing sub-task {subset}",
        ):
            label = simple_choice_extractors(response)
            echo_if_verbose(verbose, text, label)
            attach_inference(record, text=text, response=response, label=label, run=run, sub_subtask=subset)
        all_records.extend(records)
        save_results_as_jsonl(
            save_dir=save_dir,
            results=records,
            task=task,
            sub_task=sub_task,
            sub_subtask=subset,
            lang=lang,
            model_type=model_type,
            model_name=model_name,
        )

    scores = get_eval_scores(all_records, subsets)
    print_score_report(model_name, lang, ([f"- Sub-SubTask: {score['sub_subtask']}", f"- Accuracy: {score['accuracy']}"] for score in scores))
    save_eval_scores_as_jsonl(save_dir, scores, task, sub_task, lang, model_type, model_name)
    return all_records
