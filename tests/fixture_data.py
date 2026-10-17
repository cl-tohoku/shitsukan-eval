"""Synthetic miniature dataset laid out like the Shitsukan dataset.

The records follow the schemas read by the task modules but contain made-up
values; they are not taken from the real dataset. :func:`write_dataset` writes
them to the paths referenced by the task YAML files, relative to a root directory
that a test uses as its working directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SELECTION_SUBTASKS = ("choice2_ans1", "choice3_ans1", "choice4_ans1", "choice5_ans1")
LOCAL_IMAGES = ("images/coco/train2017/000000000071.jpg", "images/coco/val2017/000000000139.jpg")

_TEXTURES = {
    "ja": ["つるつる", "ざらざら", "ふわふわ", "かたい", "つやつや"],
    "en": ["smooth", "rough", "fluffy", "hard", "glossy"],
}
_OBJECTS = {"ja": ["りんご", "スケートボード"], "en": ["apple", "skateboard"]}


def _selection_records(lang: str, choice_count: int, answerable: bool) -> list[dict[str, Any]]:
    """Return perception-selection records with ``choice_count`` options each."""
    records = []
    for index, object_name in enumerate(_OBJECTS[lang]):
        options = [f"{position}:{texture}" for position, texture in enumerate(_TEXTURES[lang][:choice_count])]
        correct = [index % choice_count] if answerable else []
        records.append(
            {
                "id": index,
                "image": [LOCAL_IMAGES[index]],
                "image_url": [f"http://images.example.com/{LOCAL_IMAGES[index]}"],
                "object": object_name,
                "correct_ids": correct,
                "all_ids_with_shitsukan": options,
            }
        )
    return records


def _selection_few_shots(lang: str, choice_count: int) -> list[dict[str, Any]]:
    """Return one few-shot example for perception selection."""
    options = ", ".join(f"{position}:{texture}" for position, texture in enumerate(_TEXTURES[lang][:choice_count]))
    return [{"user": f"Word: {_OBJECTS[lang][0]}\nTexture choice: {options}\nAnswer: ", "assistant": "0", "images": None}]


def _perception_selection_files() -> dict[str, list[dict[str, Any]]]:
    """Return the perception-selection files for both languages and all subsets."""
    files: dict[str, list[dict[str, Any]]] = {}
    for lang in ("ja", "en"):
        for subtask in (*SELECTION_SUBTASKS, "random"):
            choice_count = int(subtask[len("choice")]) if subtask != "random" else 3
            directory = f"data/perception/selection/{subtask}"
            file_stem = subtask.replace("_", "-")
            files[f"{directory}/perception_selection_{file_stem}_{lang}.jsonl"] = _selection_records(lang, choice_count, answerable=subtask != "random")
            files[f"{directory}/few_shot_for_perception-selection-{file_stem}_{lang}.jsonl"] = _selection_few_shots(lang, choice_count)
    return files


def _generation_files() -> dict[str, list[dict[str, Any]]]:
    """Return the perception- and commonsense-generation files (Japanese only)."""
    perception = [
        {"id": index, "image": [image], "object": name, "shitsukan_list": ["かたい", "つめたい"]}
        for index, (image, name) in enumerate(zip(LOCAL_IMAGES, _OBJECTS["ja"], strict=True))
    ]
    commonsense = [{"id": index, "object": name, "shitsukan_list": ["つるつる"]} for index, name in enumerate(_OBJECTS["ja"])]
    few_shot = [
        {"user": "【例】\n「馬」という単語は、どのような質感を持っているように感じますか？", "assistant": "回答: つやつや, 温かい, 柔らかい", "images": None}
    ]
    return {
        "data/perception/generation/perception_generation_ja.jsonl": perception,
        "data/perception/generation/few_shot_for_perception-generation_ja.jsonl": few_shot,
        "data/commonsense/generation/commonsense_generation_ja.jsonl": commonsense,
        "data/commonsense/generation/few_shot_for_commonsense-generation_ja.jsonl": few_shot,
    }


# (votes by five workers, majority label) pairs covering both branches of the
# distribution distance: unanimous votes and disagreeing votes.
CLASSIFICATION_VOTES = (([5, 2, 2, 4, 3], 2), ([1, 1, 1, 1, 1], 1), ([3, 3, 3, 3, 3], 3), ([1, 2, 2, 3, 5], 2))


def _commonsense_classification_files() -> dict[str, list[dict[str, Any]]]:
    """Return the commonsense-classification files for both languages."""
    files: dict[str, list[dict[str, Any]]] = {}
    for lang in ("ja", "en"):
        records = []
        for index, (votes, label) in enumerate(CLASSIFICATION_VOTES):
            record: dict[str, Any] = {
                "id": index,
                "object": _OBJECTS[lang][index % 2],
                "shitsukan": _TEXTURES[lang][index],
                "naturalness": label,
            }
            if lang == "ja":
                record["naturalness_by_worker"] = votes
            records.append(record)
        files[f"data/commonsense/classification/commonsense_classification_{lang}.jsonl"] = records
        prompt = "組み合わせ:\nりんご: つるつる\n\n回答: " if lang == "ja" else "Combinations:\napple: smooth\n\nAnswer: "
        files[f"data/commonsense/classification/few_shot_for_commonsense-classification_{lang}.jsonl"] = [{"user": prompt, "assistant": "1", "images": None}]
    return files


def _taxonomic_files() -> dict[str, list[dict[str, Any]]]:
    """Return the taxonomic-classification files (Japanese only)."""
    a_b = [
        {"id": 0, "correct_ids": [0], "all_ids_with_shitsukan": ["A:つるつる", "B:破壊"]},
        {"id": 1, "correct_ids": [1], "all_ids_with_shitsukan": ["A:高性能", "B:ふわふわ"]},
        {"id": 2, "correct_ids": [0], "all_ids_with_shitsukan": ["A:ざらざら", "B:元気"]},
    ]
    yes_no = [
        {"id": 0, "shitsukan": "つるつる", "label": 1, "label_str": "yes"},
        {"id": 1, "shitsukan": "高性能", "label": 0, "label_str": "no"},
        {"id": 2, "shitsukan": "ふわふわ", "label": 1, "label_str": "yes"},
    ]
    files = {
        "data/taxonomic/a_b_classification/taxonomic_a-b_classification_ja.jsonl": a_b,
        "data/taxonomic/a_b_classification/few_shot_for_taxonomic_a-b_classification_ja.jsonl": [
            {"user": "質感語:\nA:つやつや, B:仲良し\n解答: ", "assistant": "A", "images": None}
        ],
        "data/taxonomic/yes_no_classification/taxonomic_yes-no_classification_ja.jsonl": yes_no,
        "data/taxonomic/yes_no_classification/few_shot_for_taxonomic_yes-no_classification_ja.jsonl": [
            {"user": "質感語: つやつや\n解答: ", "assistant": "yes", "images": None}
        ],
    }
    for subtask in SELECTION_SUBTASKS:
        choice_count = int(subtask[len("choice")])
        options = [f"{position}:{texture}" for position, texture in enumerate(_TEXTURES["ja"][:choice_count])]
        directory = f"data/taxonomic/multiple_choice_classification/{subtask}"
        file_stem = subtask.replace("_", "-")
        files[f"{directory}/taxonomic_multiple-choice_classification_{file_stem}_ja.jsonl"] = [
            {"id": index, "correct_ids": [index % choice_count], "all_ids_with_shitsukan": options} for index in range(2)
        ]
        files[f"{directory}/few_shot_for_taxonomic_multiple-choice_classification_{file_stem}_ja.jsonl"] = [
            {"user": f"質感語: {', '.join(options)}\n解答: ", "assistant": "0", "images": None}
        ]
    return files


def dataset_files() -> dict[str, list[dict[str, Any]]]:
    """Return every synthetic dataset file keyed by its path relative to the root."""
    return {
        **_perception_selection_files(),
        **_generation_files(),
        **_commonsense_classification_files(),
        **_taxonomic_files(),
    }


def write_dataset(root: Path) -> None:
    """Write the synthetic dataset as JSON Lines files below ``root``."""
    for relative_path, records in dataset_files().items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def write_images(image_dir: Path) -> None:
    """Create placeholder files for the local images referenced by the dataset."""
    for relative_path in LOCAL_IMAGES:
        path = image_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder image bytes for " + relative_path.encode())
