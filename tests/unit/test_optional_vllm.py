"""Tests for running without vLLM and qwen-vl-utils.

pyproject.toml installs both packages on Linux x86_64 only, and vLLM 0.6.3 runs
only with torch 2.4.0. Where they are missing, the evaluator must still import
so that the Hugging Face and API models can run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evaluator_without_vllm(fake_libraries: None, import_fresh: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the evaluator as if vLLM and qwen-vl-utils were not installed."""
    # A None entry in sys.modules makes importing that module raise ImportError.
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
    # The loaders read the default model configurations from relative paths.
    monkeypatch.chdir(REPOSITORY_ROOT)
    return import_fresh("shitsukan_eval.evaluator")


def test_hugging_face_models_load_without_vllm(evaluator_without_vllm):
    model = evaluator_without_vllm.load_hf_models("Qwen/Qwen2-VL-7B-Instruct", "data")

    assert type(model).__name__ == "Qwen2VLHf"


def test_loading_a_vllm_model_without_vllm_raises_import_error(evaluator_without_vllm):
    with pytest.raises(ImportError, match="needs vllm and qwen-vl-utils"):
        evaluator_without_vllm.load_vllm_models("Qwen/Qwen2-VL-7B-Instruct", "data")
