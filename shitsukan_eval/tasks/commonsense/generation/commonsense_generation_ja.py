"""Commonsense generation in Japanese: describe the textures that the object itself (not a particular image) evokes."""

from __future__ import annotations

from typing import Any

from shitsukan_eval.tasks.common import PretrainedChatModel, Record
from shitsukan_eval.tasks.texture_generation import run_texture_generation

TASK_DESCRIPTION_FOR_COMMONSENSE_GENERATION_JA = (
    "本タスクは「画像」に対する質感ではなく、「オブジェクト」がもつ印象から連想される質感について述べていただきます。\n"
    "質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど）\n\n"
    "提示された「オブジェクト」に対し、できる限り一般的だと考えられる質感を述べて下さい。\n"
    "回答は質感を表す表現を3個「,」で分けて書いてください。\n"
    "その際、3個とも出来るだけ違う観点の表現にしてください。\n"
    "また、似た表現（きらきら・きらんきらんなど）にならないようにしてください。"
)


def commonsense_generation_ja(
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
    """Ask the model for three texture expressions per record, without images.

    Nothing is scored or saved here; the evaluator saves the returned records.

    Args:
        model: Model wrapper with a ``pretrained`` attribute; its last path
            component is written as ``model_name``.
        task_config: Parsed ``commonsense_generation_ja.yaml``.
        verbose: Print every query with the raw response.
        save_dir: Unused; accepted for a uniform task interface.
        task: Task name written to the records.
        sub_task: Sub-task name written to the records.
        sub_subtasks: Unused; accepted for a uniform task interface.
        lang: Language code written to the records.
        model_type: Unused; accepted for a uniform task interface.
        model_name: Unused; the name is derived from ``model.pretrained``.

    Returns:
        The records with the prompt, the raw response, and metadata added.
    """
    return run_texture_generation(
        model,
        task_config,
        system_prompt=TASK_DESCRIPTION_FOR_COMMONSENSE_GENERATION_JA,
        images_for=lambda record: None,
        verbose=verbose,
        task=task,
        sub_task=sub_task,
        lang=lang,
    )
