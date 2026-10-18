"""Commonsense classification: rate how natural an object-texture pair is.

The model answers with one of five labels: 1 (completely natural) to 4
(unnatural), or 5 (not a texture word). Accuracy compares the parsed answer with
the majority label of the crowd workers.
"""

from __future__ import annotations

from collections.abc import Callable
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
    query_records,
)
from tools.utils.data_utils import simple_choice_extractors


def classify_naturalness(
    model: ChatModel,
    task_config: dict[str, Any],
    *,
    system_prompt: str,
    build_text: Callable[[Record], str],
    verbose: bool,
    run: RunInfo,
) -> list[Record]:
    """Ask the model to rate every object-texture pair and attach the parsed labels.

    Args:
        model: Model wrapper to evaluate.
        task_config: Parsed task YAML with ``dataset_path``, ``few_shot_path``,
            and ``generation_kwargs``.
        system_prompt: Task instructions.
        build_text: Function that builds the query text of a record.
        verbose: Print every query with its parsed answer.
        run: Metadata written to the records.

    Returns:
        The records with the prompt, raw response, parsed label, and metadata added.
    """
    records, few_shot_data = load_records_and_few_shots(task_config)
    for record, text, response in query_records(
        model,
        records,
        build_text=build_text,
        system_prompt=system_prompt,
        few_shot_data=few_shot_data,
        gen_kwargs=task_config["generation_kwargs"],
        progress_description="Processing dataset",
    ):
        label = simple_choice_extractors(response)
        echo_if_verbose(verbose, text, label)
        attach_inference(record, text=text, response=response, label=label, run=run)
    return records


def naturalness_accuracy(records: list[Record]) -> dict[str, Any]:
    """Return the accuracy of the parsed labels against the ``naturalness`` labels.

    Raises:
        ValueError: If a parsed label is not an integer (for example ``"A"``) or
            there are no records.
    """
    gold_labels, predicted_labels = collect_labels(records, lambda record: int(record["naturalness"]), lambda record: int(record["model_inference"]))
    return accuracy_scores(gold_labels, predicted_labels)
