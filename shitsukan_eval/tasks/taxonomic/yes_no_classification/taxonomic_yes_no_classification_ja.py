"""Taxonomic yes/no classification in Japanese: decide whether a word is a texture word."""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import (
    INVALID_PREDICTION,
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
from tools.utils.data_utils import save_eval_scores_as_jsonl, set_seed_all, simple_choice_extractors

# Importing this module seeds the global RNGs, as every module of the original scripts did.
set_seed_all(42)

TASK_DESCRIPTION_FOR_TAXONOMIC_YES_NO_CLASSIFICATION_JA = (
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど)\n"
    "以下の選択肢のうち、質感語として正しいものを1つ選び解答してください。\n"
    '解答の際には、選択肢に対応する英単語 ("yes" もしくは "no") を使って解答してください。'
)

# Parsed answer -> class id. Any other parsed answer raises KeyError when scoring.
LABEL_TO_ID = {"no": 0, "yes": 1, "-100": INVALID_PREDICTION}


def build_yes_no_prompt(record: Record) -> str:
    """Return the query text with the word to judge."""
    return f"質感語: {record['shitsukan']}\n解答: "


def get_eval_scores(data_list: list[Record]) -> dict[str, Any]:
    """Return the accuracy and the run metadata.

    Raises:
        KeyError: If a parsed answer is not in :data:`LABEL_TO_ID`.
    """
    gold_labels, predicted_labels = collect_labels(data_list, lambda record: int(record["label"]), lambda record: LABEL_TO_ID[record["model_inference"]])
    scores = accuracy_scores(gold_labels, predicted_labels)
    metadata = data_list[-1]
    return {
        **scores,
        "task": metadata["task"],
        "sub_task": metadata["sub_task"],
        "lang": metadata["lang"],
        "model_name": metadata["model_name"],
    }


def taxonomic_yes_no_classification_ja(
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
    """Run the yes/no classification task and save the scores.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed ``taxonomic_yes_no_classification_ja.yaml``.
        verbose: Print every query with its parsed answer.
        save_dir: Root directory for result and score files.
        task: Task name written to the outputs.
        sub_task: Sub-task name written to the outputs.
        sub_subtasks: Unused; accepted for a uniform task interface.
        lang: Language code written to the outputs.
        model_type: Model type used in output file names.
        model_name: Model name written to the outputs and file names.

    Returns:
        The records with the prompt, response, label, and metadata added.
    """
    run = RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=model_name)
    records, few_shot_data = load_records_and_few_shots(task_config)
    for record, text, response in query_records(
        model,
        records,
        build_text=build_yes_no_prompt,
        system_prompt=TASK_DESCRIPTION_FOR_TAXONOMIC_YES_NO_CLASSIFICATION_JA,
        few_shot_data=few_shot_data,
        gen_kwargs=task_config["generation_kwargs"],
        progress_description="Processing dataset",
    ):
        label = simple_choice_extractors(response)
        echo_if_verbose(verbose, text, label)
        attach_inference(record, text=text, response=response, label=label, run=run)

    scores = get_eval_scores(records)
    print_score_report(model_name, lang, [[f"- Accuracy: {scores['accuracy']}"]])
    save_eval_scores_as_jsonl(save_dir, [scores], task, sub_task, lang, model_type, model_name)
    return records
