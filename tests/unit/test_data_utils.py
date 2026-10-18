"""Unit tests for ``tools.utils.data_utils``.

Tests marked "observational record" pin down current behavior that is not (yet)
an agreed specification, such as surprising edge cases of the answer parser.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.fakes import torch_fake
from tests.fakes.recorder import RECORDER, SETTINGS


@pytest.fixture
def data_utils(import_fresh: Any) -> Any:
    """Import ``tools.utils.data_utils`` against the fake libraries."""
    module = import_fresh("tools.utils.data_utils")
    RECORDER.reset()
    return module


# --- simple_choice_extractors -------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("0", "0", id="digit"),
        pytest.param("5", "5", id="largest digit"),
        pytest.param("-1", "-1", id="no applicable option"),
        pytest.param("(1) つやつや", "1", id="digit in parentheses with text"),
        pytest.param("2.", "2", id="digit with period"),
        pytest.param("  3  ", "3", id="surrounding whitespace"),
        pytest.param("A", "A", id="letter A"),
        pytest.param("B)", "B", id="letter B with parenthesis"),
        pytest.param("yes", "yes", id="yes"),
        pytest.param("no,", "no", id="no with comma"),
    ],
)
def test_simple_choice_extractors_returns_the_leading_label(data_utils, answer, expected):
    assert data_utils.simple_choice_extractors(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("", id="empty"),
        pytest.param("6", id="digit above five"),
        pytest.param("-2", id="negative other than -1"),
        pytest.param("C", id="letter C"),
        pytest.param("Yes", id="capitalized yes"),
        pytest.param("The answer is 1", id="label not at the start"),
    ],
)
def test_simple_choice_extractors_returns_minus_100_without_a_leading_label(data_utils, answer):
    assert data_utils.simple_choice_extractors(answer) == "-100"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("10", "1", id="only the first digit of a number"),
        pytest.param("yesterday", "yes", id="prefix of a longer word"),
        pytest.param("Aは正しい", "A", id="letter followed by text"),
    ],
)
def test_simple_choice_extractors_matches_prefixes(data_utils, answer, expected):
    """Observational record: the parser only looks at the beginning of the answer."""
    assert data_utils.simple_choice_extractors(answer) == expected


# --- is_url / check_image_exists -----------------------------------------------------------
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("http://images.example.com/a.jpg", True, id="http"),
        pytest.param("https://images.example.com/a.jpg", True, id="https"),
        pytest.param("ftp://images.example.com/a.jpg", False, id="other scheme"),
        pytest.param("images/coco/train2017/a.jpg", False, id="relative path"),
    ],
)
def test_is_url(data_utils, path, expected):
    assert data_utils.is_url(path) is expected


@pytest.mark.parametrize(
    ("status_code", "raises", "expected"),
    [
        pytest.param(200, False, True, id="found"),
        pytest.param(404, False, False, id="not found"),
        pytest.param(200, True, False, id="request error"),
    ],
)
def test_check_image_exists(data_utils, status_code, raises, expected):
    SETTINGS["requests_status_code"] = status_code
    SETTINGS["requests_raise"] = raises

    assert data_utils.check_image_exists("http://images.example.com/a.jpg") is expected
    assert RECORDER.events == [{"call": "requests.head", "url": "http://images.example.com/a.jpg", "kwargs": {"timeout": 5}}]


# --- saving ------------------------------------------------------------------------------
def test_save_jsonl_writes_one_unescaped_json_object_per_line(data_utils, tmp_path):
    output = tmp_path / "nested" / "out.jsonl"

    data_utils.save_jsonl(str(output), [{"word": "つるつる"}, {"word": "smooth", "score": 0.5}])

    assert output.read_text(encoding="utf-8") == '{"word": "つるつる"}\n{"word": "smooth", "score": 0.5}\n'


@pytest.mark.parametrize(
    ("sub_subtask", "expected_path"),
    [
        pytest.param(None, "perception/selection/perception_selection_ja_hf_model.jsonl", id="without sub-subtask"),
        pytest.param("random", "perception/selection/random/perception_selection_random_ja_hf_model.jsonl", id="with sub-subtask"),
    ],
)
def test_save_results_as_jsonl_builds_the_output_path(data_utils, tmp_path, sub_subtask, expected_path):
    data_utils.save_results_as_jsonl(str(tmp_path), [{"response": "1"}], "perception", "selection", sub_subtask, "ja", "hf", "model")

    assert (tmp_path / expected_path).read_text(encoding="utf-8") == '{"response": "1"}\n'


def test_save_eval_scores_as_jsonl_builds_the_output_path(data_utils, tmp_path):
    data_utils.save_eval_scores_as_jsonl(str(tmp_path), [{"accuracy": 0.5}], "taxonomic", "a_b_classification", "ja", "api", "gpt-4o")

    output = tmp_path / "taxonomic/a_b_classification/taxonomic_a_b_classification_ja_api_gpt-4o_scores.jsonl"
    assert output.read_text(encoding="utf-8") == '{"accuracy": 0.5}\n'


# --- loading -----------------------------------------------------------------------------
def test_load_dataset_reads_json_lines(data_utils, tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"object": "りんご", "label": 1}\n{"object": "apple", "label": 0}\n', encoding="utf-8")

    assert data_utils.load_dataset(str(path)) == [{"object": "りんご", "label": 1}, {"object": "apple", "label": 0}]


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_load_dataset_reads_delimited_files_without_leaving_files_behind(data_utils, tmp_path, suffix, separator):
    path = tmp_path / f"data{suffix}"
    path.write_text(separator.join(["object", "label", "score"]) + "\n" + separator.join(["りんご", "1", ""]) + "\n", encoding="utf-8")

    records = data_utils.load_dataset(str(path))

    assert records == [{"object": "りんご", "label": 1, "score": None}]
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_load_dataset_reads_json_arrays_without_writing_files(data_utils, tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"object": "りんご", "label": 1}, {"object": "apple", "label": None}]), encoding="utf-8")

    records = data_utils.load_dataset(str(path))

    assert records == [{"object": "りんご", "label": 1.0}, {"object": "apple", "label": None}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.json"]


def test_load_dataset_matches_the_records_of_the_jsonl_converters(data_utils, tmp_path):
    source = tmp_path / "source"
    converted = tmp_path / "converted"
    source.mkdir()
    converted.mkdir()
    path = source / "data.csv"
    path.write_text('object,label,text\nりんご,1,"a, b ""quoted"" [x]"\n,2,\n', encoding="utf-8")

    data_utils.convert_csv_to_jsonl(str(path), str(converted))

    assert data_utils.load_dataset(str(path)) == data_utils.load_jsonl(str(converted / "data.csv.jsonl"))


def test_load_dataset_rejects_unknown_extensions(data_utils, tmp_path):
    with pytest.raises(ValueError, match=r"Unsupported file extension: \.txt"):
        data_utils.load_dataset(str(tmp_path / "data.txt"))


def test_load_yaml_returns_the_parsed_mapping(data_utils, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("generation_kwargs:\n  temperature: 0.01\n  do_sample: true\n", encoding="utf-8")

    assert data_utils.load_yaml(str(path)) == {"generation_kwargs": {"temperature": 0.01, "do_sample": True}}


def test_load_yaml_raises_for_a_missing_file(data_utils, tmp_path):
    with pytest.raises(FileNotFoundError):
        data_utils.load_yaml(str(tmp_path / "missing.yaml"))


def test_load_yaml_raises_for_invalid_yaml(data_utils, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("generation_kwargs: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        data_utils.load_yaml(str(path))


# --- set_seed_all ------------------------------------------------------------------------
def test_set_seed_all_seeds_python_numpy_and_torch(data_utils):
    random.random()
    np.random.random()

    data_utils.set_seed_all(7)

    assert random.random() == random.Random(7).random()
    assert np.random.random() == np.random.RandomState(7).random()
    assert torch_fake.SEED_STATE["torch"] == 7
    assert torch_fake.SEED_STATE["cuda"] == 7
    assert torch_fake.backends.cudnn.deterministic is True


def test_set_seed_all_leaves_torch_deterministic_mode_untouched(data_utils):
    original_function = torch_fake.use_deterministic_algorithms

    data_utils.set_seed_all(42)

    assert torch_fake.use_deterministic_algorithms is original_function
    assert all(event["call"] != "torch.use_deterministic_algorithms" for event in RECORDER.events)
