"""Shared implementation of the free-form texture generation tasks.

The model is asked which textures a word evokes and answers in free text. The
responses are attached to the records and not scored here.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

from shitsukan_eval.tasks.common import PretrainedChatModel, Record, RunInfo, echo_if_verbose, load_records_and_few_shots, query_records


def build_generation_prompt(record: Record) -> str:
    """Return the question asking which textures ``record["object"]`` evokes."""
    return f"「{record['object']}」という単語は、どのような質感を持っているように感じますか？\n回答: "


def run_texture_generation(
    model: PretrainedChatModel,
    task_config: dict[str, Any],
    *,
    system_prompt: str,
    images_for: Callable[[Record], list[str] | None],
    verbose: bool,
    task: str | None,
    sub_task: str | None,
    lang: str | None,
) -> list[Record]:
    """Ask the model to describe the textures of every record.

    The ``model_name`` written to the records is the last component of
    ``model.pretrained``.

    Args:
        model: Model wrapper with a ``pretrained`` attribute.
        task_config: Parsed task YAML with ``dataset_path``, ``few_shot_path``,
            and ``generation_kwargs``.
        system_prompt: Task instructions.
        images_for: Function that returns the images shown with a record.
        verbose: Print every query with the raw response.
        task: Task name written to the records.
        sub_task: Sub-task name written to the records.
        lang: Language code written to the records.

    Returns:
        The records with the prompt, the raw response, and metadata added.
    """
    records, few_shot_data = load_records_and_few_shots(task_config)
    run = RunInfo(task=task, sub_task=sub_task, lang=lang, model_name=pathlib.Path(model.pretrained).name)
    for record, text, response in query_records(
        model,
        records,
        build_text=build_generation_prompt,
        images_for=images_for,
        system_prompt=system_prompt,
        few_shot_data=few_shot_data,
        gen_kwargs=task_config["generation_kwargs"],
        progress_description="Processing dataset",
    ):
        echo_if_verbose(verbose, text, response)
        record.update({"zero_shot": text, "original_response": response, **run.metadata()})
    return records
