"""Tests for the command-line interface ``python -m shitsukan_eval``."""

from __future__ import annotations

import sys
from typing import Any

import pytest

FULL_COMMAND_LINE = [
    "shitsukan_eval",
    "--model",
    "gpt-4o",
    "--model-type",
    "api",
    "--tasks",
    "perception",
    "commonsense",
    "--sub-tasks",
    "generation",
    "--lang",
    "ja",
    "--image-dir",
    "data",
    "--save-dir",
    "outputs",
    "--verbose",
]


@pytest.fixture
def cli(import_fresh: Any) -> Any:
    """Import ``shitsukan_eval.__main__`` against the fake libraries."""
    return import_fresh("shitsukan_eval.__main__")


def test_parse_eval_args_reads_every_option(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", FULL_COMMAND_LINE)

    args = cli.parse_eval_args()

    assert vars(args) == {
        "model": "gpt-4o",
        "model_type": "api",
        "tasks": ["perception", "commonsense"],
        "sub_tasks": ["generation"],
        "lang": "ja",
        "image_dir": "data",
        "save_dir": "outputs",
        "verbose": True,
    }


def test_parse_eval_args_defaults_for_optional_options(cli, monkeypatch):
    required_only = [argument for argument in FULL_COMMAND_LINE if argument not in {"--image-dir", "data", "--verbose"}]
    monkeypatch.setattr(sys, "argv", required_only)

    args = cli.parse_eval_args()

    assert args.image_dir is None
    assert args.verbose is False


@pytest.mark.parametrize("option", ["--model", "--model-type", "--tasks", "--sub-tasks", "--lang", "--save-dir"])
def test_parse_eval_args_requires_option(cli, monkeypatch, option):
    position = FULL_COMMAND_LINE.index(option)
    following = FULL_COMMAND_LINE[position + 1 :]
    value_count = next((index for index, argument in enumerate(following) if argument.startswith("-")), len(following))
    monkeypatch.setattr(sys, "argv", FULL_COMMAND_LINE[:position] + following[value_count:])

    with pytest.raises(SystemExit):
        cli.parse_eval_args()


def test_parse_eval_args_rejects_unknown_model_types(cli, monkeypatch):
    command_line = list(FULL_COMMAND_LINE)
    command_line[command_line.index("api")] = "onnx"
    monkeypatch.setattr(sys, "argv", command_line)

    with pytest.raises(SystemExit):
        cli.parse_eval_args()


def test_main_passes_the_arguments_to_evaluate(cli, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "evaluate", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", FULL_COMMAND_LINE)

    cli.main()

    assert calls == [
        {
            "model_name": "gpt-4o",
            "model_type": "api",
            "tasks": ["perception", "commonsense"],
            "sub_tasks": ["generation"],
            "lang": "ja",
            "image_dir": "data",
            "save_dir": "outputs",
            "verbose": True,
        }
    ]
