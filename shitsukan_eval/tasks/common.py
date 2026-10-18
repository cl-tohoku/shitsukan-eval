"""Helpers shared by the evaluation tasks.

Every task follows the same steps: load the records and few-shot examples named
in its YAML configuration, ask the model about each record, attach the prompt,
the response, and run metadata to the record, score the parsed answers, print a
short report, and save the scores. This module implements the steps that do not
depend on the task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.utils.data_utils import load_dataset

INVALID_PREDICTION = -100
"""Score label of an answer that cannot be parsed into one of the choices."""

Record = dict[str, Any]


class ChatModel(Protocol):
    """Interface of the model wrappers used by the tasks."""

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Return the model's answer to ``text`` (and ``images``)."""
        ...


class PretrainedChatModel(ChatModel, Protocol):
    """Model wrapper that also exposes the name or path it was loaded from."""

    pretrained: str


@dataclass(frozen=True)
class RunInfo:
    """Metadata attached to every record and score of one task run.

    Attributes:
        task: Task name, such as ``"perception"``.
        sub_task: Sub-task name, such as ``"selection"``.
        lang: Language code, ``"ja"`` or ``"en"``.
        model_name: Short model name used in output files.
    """

    task: str | None
    sub_task: str | None
    lang: str | None
    model_name: str

    def metadata(self, sub_subtask: str | None = None) -> dict[str, str | None]:
        """Return ``task``, ``sub_task``, optionally ``sub_subtask``, ``lang``, and ``model_name`` in that order."""
        fields: dict[str, str | None] = {"task": self.task, "sub_task": self.sub_task}
        if sub_subtask is not None:
            fields["sub_subtask"] = sub_subtask
        fields["lang"] = self.lang
        fields["model_name"] = self.model_name
        return fields


def load_records_and_few_shots(config: Mapping[str, Any]) -> tuple[list[Record], list[Record]]:
    """Load the records at ``config["dataset_path"]`` and the examples at ``config["few_shot_path"]``."""
    return load_dataset(config["dataset_path"]), load_dataset(config["few_shot_path"])


def query_records(
    model: ChatModel,
    records: Iterable[Record],
    *,
    build_text: Callable[[Record], str],
    system_prompt: str,
    few_shot_data: list[Record],
    gen_kwargs: dict[str, Any],
    progress_description: str,
    images_for: Callable[[Record], list[str] | None] = lambda record: None,
) -> Iterator[tuple[Record, str, str]]:
    """Ask the model about each record, showing a progress bar.

    Args:
        model: Model wrapper to query.
        records: Records to ask about, in order.
        build_text: Function that builds the query text of a record.
        system_prompt: Task instructions.
        few_shot_data: Few-shot examples sent with every query.
        gen_kwargs: Generation settings from the task configuration.
        progress_description: Label of the progress bar.
        images_for: Function that returns the images of a record (default: none).

    Yields:
        ``(record, query text, raw response)`` for each record, one model call at a time.
    """
    for record in tqdm(records, desc=progress_description, leave=False):
        text = build_text(record)
        response = model.generate_response(
            text=text,
            images=images_for(record),
            system_prompt=system_prompt,
            few_shot_data=few_shot_data,
            gen_kwargs=gen_kwargs,
        )
        yield record, text, response


def echo_if_verbose(verbose: bool, text: str, shown_answer: str) -> None:
    """Print the query text followed by the answer (and a blank line) when ``verbose``."""
    if verbose:
        tqdm.write(f"{text}{shown_answer}\n")


def attach_inference(record: Record, *, text: str, response: str, label: str, run: RunInfo, sub_subtask: str | None = None) -> None:
    """Add the query, the raw response, the parsed label, and the run metadata to ``record``."""
    record.update({"zero_shot": text, "original_model_inference": response, "model_inference": label, **run.metadata(sub_subtask)})


def collect_labels(records: Iterable[Record], gold_label: Callable[[Record], Any], predicted_label: Callable[[Record], Any]) -> tuple[list[Any], list[Any]]:
    """Return the gold and the predicted label of every record, converting record by record.

    Raises:
        Exception: Whatever ``gold_label`` or ``predicted_label`` raises for the
            first record that cannot be converted.
    """
    gold_labels = []
    predicted_labels = []
    for record in records:
        gold_labels.append(gold_label(record))
        predicted_labels.append(predicted_label(record))
    return gold_labels, predicted_labels


def accuracy_scores(gold_labels: list[Any], predicted_labels: list[Any]) -> dict[str, Any]:
    """Return ``{"accuracy": ...}`` for the predicted labels.

    Raises:
        ValueError: If there are no labels.
    """
    return {"accuracy": accuracy_score(gold_labels, predicted_labels)}


def print_score_report(model_name: str, lang: str | None, blocks: Iterable[Iterable[str]]) -> None:
    """Print a header line followed by one block of ``"- Name: value"`` lines per score set."""
    print("=" * 20 + "  " + f"Model Name: {model_name}" + "    " + f"Language: {lang}" + "  " + "=" * 20 + "\n" + "-" * 40)
    for lines in blocks:
        print("\n".join(lines) + "\n" + "-" * 40)
