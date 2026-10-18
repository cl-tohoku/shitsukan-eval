"""Characterization tests (observational records) for ``shitsukan_eval.evaluator``.

These tests run the evaluator end to end with the real model wrappers on top of
the fake libraries and the synthetic dataset. They pin down which wrapper and
arguments each model name selects, which task runs for each request, the RNG
state when the model is loaded and sampled, and every output file.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import torch_fake
from tests.fakes.recorder import RECORDER, RESPONDER, SETTINGS
from tests.fixture_data import write_dataset, write_images
from tests.golden import check_golden, collect_files, text_lines, to_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_GLOBS = ("shitsukan_eval/tasks/**/*.yaml", "models/**/*.yaml")
# Seeding calls are excluded from the compared events: the seeding contract is
# checked through the RNG summaries recorded when models are loaded and sampled.
SEEDING_CALLS = {"torch.manual_seed", "torch.cuda.manual_seed_all", "torch.use_deterministic_algorithms"}
FRESHLY_SEEDED = {
    "python_random_fresh_seed_42": True,
    "numpy_random_fresh_seed_42": True,
    "torch_seed": 42,
    "cuda_seed": 42,
    "draws_since_torch_seed": 0,
    "cudnn_deterministic": True,
}


# (marker found in the prompt, well-formed answer for that task), checked in order:
# the A/B and multiple-choice formats must be recognized before the yes/no format.
ANSWERS_BY_PROMPT_MARKER = (
    ("Texture choice:", "1"),
    ("組み合わせ:", "2"),
    ("Combinations:", "2"),
    ("どのような質感", "かたい, つめたい, ひかる"),
    ("質感語:\nA:", "B"),
    ("質感語: 0:", "1"),
    ("質感語: ", "yes"),
)


def answer_like_a_model(prompt: str | None) -> str:
    """Return a well-formed answer for the task recognized from ``prompt``."""
    text = prompt or ""
    return next((answer for marker, answer in ANSWERS_BY_PROMPT_MARKER if marker in text), "unrecognized prompt")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare a working directory with configs, the synthetic dataset, and images."""
    root = tmp_path / "workspace"
    for pattern in CONFIG_GLOBS:
        for source in REPOSITORY_ROOT.glob(pattern):
            target = root / source.relative_to(REPOSITORY_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    write_dataset(root)
    write_images(root / "image_root")
    monkeypatch.chdir(root)
    return root


def _observe_evaluation(import_fresh: Callable[[str], Any], workspace: Path, capsys: pytest.CaptureFixture[str], **evaluate_kwargs: Any) -> dict[str, Any]:
    """Run ``evaluate`` and record events, errors, stdout, and output files."""
    evaluator = import_fresh("shitsukan_eval.evaluator")
    SETTINGS["record_rng_state"] = True
    RESPONDER.use_rule(answer_like_a_model)
    RECORDER.reset()
    error: dict[str, str] | None = None
    try:
        evaluator.evaluate(save_dir=str(workspace / "outputs"), **evaluate_kwargs)
    except Exception as exc:  # noqa: BLE001 - errors are part of the observation.
        error = {"type": type(exc).__name__, "message": str(exc)}
    observed = {
        "events": [event for event in RECORDER.events if event["call"] not in SEEDING_CALLS],
        "error": error,
        "stdout": text_lines(capsys.readouterr().out),
        "files": collect_files(workspace / "outputs") if (workspace / "outputs").exists() else {},
    }
    return to_record(observed, {str(workspace): "<workspace>"})


EVALUATION_SCENARIOS = {
    "hf_text_model_taxonomic_tasks": {
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "model_type": "hf",
        "tasks": ["taxonomic"],
        "sub_tasks": ["a_b_classification", "yes_no_classification", "multiple_choice_classification"],
        "lang": "ja",
        "image_dir": None,
        "verbose": False,
    },
    "hf_vision_model_perception_tasks": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["selection", "generation"],
        "lang": "ja",
        "image_dir": "image_root",
        "verbose": True,
    },
    "api_model_commonsense_tasks": {
        "model_name": "gpt-4o",
        "model_type": "api",
        "tasks": ["commonsense"],
        "sub_tasks": ["classification", "generation"],
        "lang": "ja",
        "image_dir": "image_root",
        "verbose": False,
    },
    "vllm_model_english_perception_selection": {
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "model_type": "vllm",
        "tasks": ["perception"],
        "sub_tasks": ["selection"],
        "lang": "en",
        "image_dir": "image_root",
        "verbose": False,
    },
    "hf_text_model_english_commonsense_classification": {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_type": "hf",
        "tasks": ["commonsense"],
        "sub_tasks": ["classification"],
        "lang": "en",
        "image_dir": None,
        "verbose": True,
    },
    "hf_gemma2_taxonomic_a_b_classification": {
        "model_name": "google/gemma-2-9b-it",
        "model_type": "hf",
        "tasks": ["taxonomic"],
        "sub_tasks": ["a_b_classification"],
        "lang": "ja",
        "image_dir": None,
        "verbose": False,
    },
    "hf_llm_jp_3_commonsense_classification": {
        "model_name": "llm-jp/llm-jp-3-13b-instruct",
        "model_type": "hf",
        "tasks": ["commonsense"],
        "sub_tasks": ["classification"],
        "lang": "ja",
        "image_dir": None,
        "verbose": False,
    },
    "hf_idefics3_perception_selection": {
        "model_name": "HuggingFaceM4/Idefics3-8B-Llama3",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["selection"],
        "lang": "ja",
        "image_dir": "image_root",
        "verbose": False,
    },
    "hf_mllama_perception_generation": {
        "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["generation"],
        "lang": "ja",
        "image_dir": "image_root",
        "verbose": False,
    },
    "hf_molmo_english_perception_selection": {
        "model_name": "allenai/Molmo-7B-D-0924",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["selection"],
        "lang": "en",
        "image_dir": "image_root",
        "verbose": False,
    },
    "hf_qwen2_vl_perception_generation": {
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["generation"],
        "lang": "ja",
        "image_dir": "image_root",
        "verbose": False,
    },
    "unsupported_language_task_combination": {
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "model_type": "hf",
        "tasks": ["perception"],
        "sub_tasks": ["generation"],
        "lang": "en",
        "image_dir": None,
        "verbose": False,
    },
    "unknown_model_type": {
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "model_type": "onnx",
        "tasks": ["taxonomic"],
        "sub_tasks": ["a_b_classification"],
        "lang": "ja",
        "image_dir": None,
        "verbose": False,
    },
}


@pytest.mark.parametrize("scenario", list(EVALUATION_SCENARIOS))
def test_evaluate_end_to_end(scenario, import_fresh, workspace, capsys):
    observed = _observe_evaluation(import_fresh, workspace, capsys, **EVALUATION_SCENARIOS[scenario])

    check_golden(f"evaluate/{scenario}", observed)


MODEL_SELECTION_CASES = {
    "hf": [
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-3B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "llava-hf/bakLlava-v1-hf",
        "HuggingFaceM4/idefics2-8b",
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "Qwen/Qwen2-VL-7B-Instruct",
        "allenai/Molmo-7B-D-0924",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-7b-hf",
        "google/gemma-2-2b-jpn-it",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B",
        "llm-jp/llm-jp-3-13b-instruct",
        "llm-jp/llm-jp-3-13b",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "vllm": [
        "Qwen/Qwen2-VL-7B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/bakLlava-v1-hf",
        "HuggingFaceM4/idefics2-8b",
        "allenai/Molmo-7B-D-0924",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ],
    "api": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-sonnet"],
}
MODEL_LOADERS = {"hf": "load_hf_models", "vllm": "load_vllm_models", "api": "load_api_models"}


def _slug(model_name: str) -> str:
    """Turn a model name into a file-name-friendly identifier."""
    return model_name.replace("/", "__")


def _observe_model_selection(
    import_fresh: Callable[[str], Any], workspace: Path, capsys: pytest.CaptureFixture[str], model_type: str, model_name: str
) -> dict[str, Any]:
    """Load ``model_name`` with the evaluator's loader for ``model_type`` and observe it."""
    evaluator = import_fresh("shitsukan_eval.evaluator")
    loader = getattr(evaluator, MODEL_LOADERS[model_type])
    RECORDER.reset()
    selected: str | None = None
    error: dict[str, str] | None = None
    try:
        selected = type(loader(model_name, "image_root")).__name__
    except Exception as exc:  # noqa: BLE001 - errors are part of the observation.
        error = {"type": type(exc).__name__, "message": str(exc)}
    return to_record(
        {"selected": selected, "error": error, "events": RECORDER.events, "stdout": text_lines(capsys.readouterr().out)},
        {str(workspace): "<workspace>"},
    )


@pytest.mark.parametrize(
    ("model_type", "model_name"),
    [(model_type, name) for model_type, names in MODEL_SELECTION_CASES.items() for name in names],
)
def test_model_selection(model_type, model_name, import_fresh, workspace, capsys):
    observed = _observe_model_selection(import_fresh, workspace, capsys, model_type, model_name)

    check_golden(f"evaluate/model_selection/{model_type}/{_slug(model_name)}", observed)


# With flash_attn installed, the wrappers default to "flash_attention_2"; this
# exposes the models for which the evaluator forces eager attention.
@pytest.mark.parametrize("fake_libraries", [True], indirect=True)
@pytest.mark.parametrize("model_name", MODEL_SELECTION_CASES["hf"])
def test_model_selection_when_flash_attn_is_installed(model_name, fake_libraries, import_fresh, workspace, capsys):
    observed = _observe_model_selection(import_fresh, workspace, capsys, "hf", model_name)

    check_golden(f"evaluate/model_selection/hf_with_flash_attn/{_slug(model_name)}", observed)


TASK_REQUESTS = [
    (lang, task, sub_task)
    for lang in ("ja", "en")
    for task, sub_task in (
        ("perception", "generation"),
        ("perception", "selection"),
        ("commonsense", "generation"),
        ("commonsense", "classification"),
        ("taxonomic", "a_b_classification"),
        ("taxonomic", "yes_no_classification"),
        ("taxonomic", "multiple_choice_classification"),
        ("taxonomic", "unknown"),
    )
]


@pytest.mark.parametrize(("lang", "task", "sub_task"), TASK_REQUESTS)
def test_task_lookup(lang, task, sub_task, import_fresh, workspace, capsys):
    evaluator = import_fresh("shitsukan_eval.evaluator")

    observed: dict[str, Any] = {}
    for name, lookup in (("function", evaluator.task_and_lang_to_func), ("config", evaluator.task_and_lang_to_config)):
        try:
            found = lookup(task=task, sub_task=sub_task, lang=lang)
            observed[name] = {"value": found.__name__ if callable(found) else found}
        except Exception as exc:  # noqa: BLE001 - errors are part of the observation.
            observed[name] = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    observed["stdout"] = text_lines(capsys.readouterr().out)

    check_golden(f"evaluate/task_lookup/{lang}__{task}__{sub_task}", to_record(observed))


def test_importing_the_evaluator_does_not_seed_the_rngs(import_fresh):
    import_fresh("shitsukan_eval.evaluator")

    assert torch_fake.SEED_STATE["torch"] is None
    assert all(event["call"] not in SEEDING_CALLS for event in RECORDER.events)


def test_evaluate_reseeds_the_rngs_before_loading_the_model_on_every_call(import_fresh, workspace, capsys):
    evaluator = import_fresh("shitsukan_eval.evaluator")
    SETTINGS["record_rng_state"] = True
    RESPONDER.use_rule(answer_like_a_model)
    arguments = EVALUATION_SCENARIOS["hf_text_model_taxonomic_tasks"]

    for run in range(2):
        RECORDER.reset()
        evaluator.evaluate(save_dir=str(workspace / f"outputs_{run}"), **arguments)
        loading = next(event for event in RECORDER.events if event["call"].endswith(".from_pretrained") and event["rng"] is not None)
        first_generation = next(event for event in RECORDER.events if event["call"] == "Model.generate")

        assert loading["rng"] == FRESHLY_SEEDED, f"run {run}: {loading['rng']}"
        assert first_generation["rng"] == FRESHLY_SEEDED, f"run {run}: {first_generation['rng']}"
