"""Data input/output, random seeding, and answer parsing used by the evaluation.

Results and scores are written as JSON Lines without ASCII escaping, so Japanese
text stays readable in the output files.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import yaml

CHOICE_LABEL_PATTERN = re.compile(r"\({0,1}(-1|[0-5]|[A-B]|yes|no)[.,:;)]{0,1}")
"""Answer label at the start of a response.

The label is ``-1``, a digit from ``0`` to ``5``, ``A``, ``B``, ``yes``, or ``no``,
optionally preceded by ``(`` and followed by one of ``.,:;)``.
"""

UNPARSABLE_LABEL = "-100"
"""Label returned by :func:`simple_choice_extractors` when no label is found."""

IMAGE_CHECK_TIMEOUT_SECONDS = 5
HTTP_OK = 200


# --- Saving -------------------------------------------------------------------------------
def save_json(filename: str, ds: dict[str, Any]) -> None:
    """Write ``ds`` to ``filename`` as indented JSON (ASCII-escaped)."""
    with open(filename, "w") as f:
        json.dump(ds, f, indent=4)


def save_jsonl(filename: str, data_list: list[dict[str, Any]]) -> None:
    """Write one JSON object per line to ``filename``, creating parent directories.

    Args:
        filename: Output path; an existing file is overwritten.
        data_list: Objects to write, in order.
    """
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as fo:
        for record in data_list:
            fo.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_results_as_jsonl(
    save_dir: str,
    results: list[dict[str, Any]],
    task: str,
    sub_task: str,
    sub_subtask: str | None,
    lang: str,
    model_type: str,
    model_name: str,
) -> None:
    """Save evaluation records under the directory of their task.

    The file is
    ``<save_dir>/<task>/<sub_task>/<task>_<sub_task>_<lang>_<model_type>_<model_name>.jsonl``,
    or, with a sub-subtask,
    ``<save_dir>/<task>/<sub_task>/<sub_subtask>/<task>_<sub_task>_<sub_subtask>_<lang>_<model_type>_<model_name>.jsonl``.

    Args:
        save_dir: Root directory of the outputs.
        results: Records to save.
        task: Task name, such as ``"perception"``.
        sub_task: Sub-task name, such as ``"selection"``.
        sub_subtask: Subset name, such as ``"choice2_ans1"``, or ``None``.
        lang: Language code.
        model_type: ``"hf"``, ``"vllm"``, or ``"api"``.
        model_name: Short model name.
    """
    if sub_subtask is not None:
        relative_path = f"{task}/{sub_task}/{sub_subtask}/{task}_{sub_task}_{sub_subtask}_{lang}_{model_type}_{model_name}.jsonl"
    else:
        relative_path = f"{task}/{sub_task}/{task}_{sub_task}_{lang}_{model_type}_{model_name}.jsonl"
    save_jsonl(str(pathlib.Path(save_dir).joinpath(relative_path)), results)


def save_eval_scores_as_jsonl(
    save_dir: str,
    scores: list[dict[str, Any]],
    task: str,
    sub_task: str,
    lang: str,
    model_type: str,
    model_name: str,
) -> None:
    """Save scores to ``<save_dir>/<task>/<sub_task>/<task>_<sub_task>_<lang>_<model_type>_<model_name>_scores.jsonl``.

    Args:
        save_dir: Root directory of the outputs.
        scores: Score dictionaries, one per line.
        task: Task name.
        sub_task: Sub-task name.
        lang: Language code.
        model_type: ``"hf"``, ``"vllm"``, or ``"api"``.
        model_name: Short model name.
    """
    relative_path = f"{task}/{sub_task}/{task}_{sub_task}_{lang}_{model_type}_{model_name}_scores.jsonl"
    save_jsonl(str(pathlib.Path(save_dir).joinpath(relative_path)), scores)


# --- Loading ------------------------------------------------------------------------------
def load_json(filename: str) -> Any:
    """Return the parsed contents of the JSON file ``filename``."""
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filename: str) -> list[dict[str, Any]]:
    """Return the objects of the JSON Lines file ``filename``, one per line."""
    with open(filename, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def convert_json_to_jsonl(filename: str, output_dir: str) -> None:
    """Convert a JSON array of objects into ``<output_dir>/<stem>.jsonl`` with pandas."""
    path = pathlib.Path(filename)
    pd.read_json(path).to_json(pathlib.Path(output_dir).joinpath(path.stem + ".jsonl"), orient="records", force_ascii=False, lines=True)


def convert_csv_to_jsonl(filename: str, output_dir: str) -> None:
    """Convert a CSV file into ``<output_dir>/<name>.jsonl`` (for example ``data.csv.jsonl``) with pandas."""
    path = pathlib.Path(filename)
    pd.read_csv(path).to_json(pathlib.Path(output_dir).joinpath(path.name + ".jsonl"), orient="records", force_ascii=False, lines=True)


def convert_tsv_to_jsonl(filename: str, output_dir: str) -> None:
    """Convert a TSV file into ``<output_dir>/<name>.jsonl`` (for example ``data.tsv.jsonl``) with pandas."""
    path = pathlib.Path(filename)
    pd.read_csv(path, sep="\t").to_json(pathlib.Path(output_dir).joinpath(path.name + ".jsonl"), orient="records", force_ascii=False, lines=True)


_CONVERTERS_TO_JSONL: dict[str, Callable[[str, str], None]] = {
    ".json": convert_json_to_jsonl,
    ".csv": convert_csv_to_jsonl,
    ".tsv": convert_tsv_to_jsonl,
}


def load_dataset(filename: str) -> list[dict[str, Any]]:
    """Load the records of a JSON Lines, JSON, CSV, or TSV file.

    Non-JSONL files are converted with pandas into a temporary ``<name>.jsonl``
    file next to the input, which is read and then deleted. Missing values become
    ``None``.

    Args:
        filename: Path ending in ``.jsonl``, ``.json``, ``.csv``, or ``.tsv``.

    Returns:
        The records in file order.

    Raises:
        ValueError: If the extension is not supported.
    """
    path = pathlib.Path(filename)
    if path.suffix == ".jsonl":
        return load_jsonl(str(path))
    converter = _CONVERTERS_TO_JSONL.get(path.suffix)
    if converter is None:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    # FIXME: convert_json_to_jsonl writes "<stem>.jsonl" while "<name>.jsonl" is read
    # here, so JSON input fails with FileNotFoundError.
    converter(str(path), str(path.parent))
    converted_path = path.parent.joinpath(path.name + ".jsonl")
    records = load_jsonl(str(converted_path))
    os.remove(str(converted_path))
    return records


def load_yaml(filename: str) -> Any:
    """Return the parsed contents of the YAML file ``filename``.

    Returns:
        The YAML document: a dictionary for the configuration files, ``None`` for
        an empty file.

    Raises:
        OSError: If the file cannot be opened (for example FileNotFoundError).
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


# --- Seeding ------------------------------------------------------------------------------
def set_seed_all(seed: int = 42) -> None:
    """Seed Python's, NumPy's, and PyTorch's (CPU and CUDA) global generators with ``seed``.

    Also makes cuDNN choose deterministic algorithms. PyTorch's global
    deterministic mode (``torch.use_deterministic_algorithms``) is not enabled:
    earlier versions of this function assigned ``True`` to that attribute instead
    of calling it, so the published results were produced without it, and
    enabling it can change results or make some CUDA operations fail.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# --- URLs ---------------------------------------------------------------------------------
def is_url(path: str) -> bool:
    """Return whether ``path`` starts with ``http://`` or ``https://``.

    Example:
        >>> is_url("http://images.cocodataset.org/train2017/000000000071.jpg")
        True
        >>> is_url("images/coco/train2017/000000000071.jpg")
        False
    """
    return path.startswith(("http://", "https://"))


def check_image_exists(url: str) -> bool:
    """Return whether a ``HEAD`` request to ``url`` answers with HTTP 200 (``False`` on request errors)."""
    try:
        response = requests.head(url, timeout=IMAGE_CHECK_TIMEOUT_SECONDS)
    except requests.RequestException:
        return False
    return response.status_code == HTTP_OK


# --- Answer parsing -----------------------------------------------------------------------
def simple_choice_extractors(answer: str) -> str:
    """Return the answer label at the start of ``answer`` (after stripping whitespace).

    Only the beginning of the answer is examined (see :data:`CHOICE_LABEL_PATTERN`).

    Returns:
        The label, or ``"-100"`` if the answer does not start with one.

    Example:
        >>> simple_choice_extractors("(1) つやつや")
        '1'
        >>> simple_choice_extractors("10")
        '1'
        >>> simple_choice_extractors("The answer is 1")
        '-100'
    """
    match = CHOICE_LABEL_PATTERN.match(answer.strip())
    if match:
        return match.group(1)
    return UNPARSABLE_LABEL
