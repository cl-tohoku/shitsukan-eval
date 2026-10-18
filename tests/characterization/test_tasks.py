"""Characterization tests (observational records) for the evaluation tasks.

Each task function runs against the synthetic dataset with a scripted model
double. The records pin down the prompts and generation settings sent to the
model, how responses are parsed and scored, the files written, and the printed
summary. Scenarios that crash on unexpected responses record that behavior as it
exists; they do not claim it is the intended specification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fakes.recorder import RECORDER, RESPONDER
from tests.fakes.scripted_model import ScriptedModel
from tests.fixture_data import write_dataset
from tests.golden import check_golden, collect_files, text_lines, to_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPE = "hf"
MODEL_NAME = "scripted-model"


@dataclass(frozen=True)
class TaskCase:
    """A task function together with the arguments the evaluator passes to it."""

    module: str
    function: str
    config: str
    task: str
    sub_task: str
    lang: str


TASKS = {
    "perception_selection_ja": TaskCase(
        "shitsukan_eval.tasks.perception.selection.perception_selection_ja",
        "perception_selection_ja",
        "shitsukan_eval/tasks/perception/selection/perception_selection_ja.yaml",
        "perception",
        "selection",
        "ja",
    ),
    "perception_selection_en": TaskCase(
        "shitsukan_eval.tasks.perception.selection.perception_selection_en",
        "perception_selection_en",
        "shitsukan_eval/tasks/perception/selection/perception_selection_en.yaml",
        "perception",
        "selection",
        "en",
    ),
    "perception_generation_ja": TaskCase(
        "shitsukan_eval.tasks.perception.generation.perception_generation_ja",
        "perception_generation_ja",
        "shitsukan_eval/tasks/perception/generation/perception_generation_ja.yaml",
        "perception",
        "generation",
        "ja",
    ),
    "commonsense_generation_ja": TaskCase(
        "shitsukan_eval.tasks.commonsense.generation.commonsense_generation_ja",
        "commonsense_generation_ja",
        "shitsukan_eval/tasks/commonsense/generation/commonsense_generation_ja.yaml",
        "commonsense",
        "generation",
        "ja",
    ),
    "commonsense_classification_ja": TaskCase(
        "shitsukan_eval.tasks.commonsense.classification.commonsense_classification_ja",
        "commonsense_classification_ja",
        "shitsukan_eval/tasks/commonsense/classification/commonsense_classification_ja.yaml",
        "commonsense",
        "classification",
        "ja",
    ),
    "commonsense_classification_en": TaskCase(
        "shitsukan_eval.tasks.commonsense.classification.commonsense_classification_en",
        "commonsense_classification_en",
        "shitsukan_eval/tasks/commonsense/classification/commonsense_classification_en.yaml",
        "commonsense",
        "classification",
        "en",
    ),
    "taxonomic_a_b_classification_ja": TaskCase(
        "shitsukan_eval.tasks.taxonomic.a_b_classification.taxonomic_a_b_classification_ja",
        "taxonomic_a_b_classification_ja",
        "shitsukan_eval/tasks/taxonomic/a_b_classification/taxonomic_a_b_classification_ja.yaml",
        "taxonomic",
        "a_b_classification",
        "ja",
    ),
    "taxonomic_yes_no_classification_ja": TaskCase(
        "shitsukan_eval.tasks.taxonomic.yes_no_classification.taxonomic_yes_no_classification_ja",
        "taxonomic_yes_no_classification_ja",
        "shitsukan_eval/tasks/taxonomic/yes_no_classification/taxonomic_yes_no_classification_ja.yaml",
        "taxonomic",
        "yes_no_classification",
        "ja",
    ),
    "taxonomic_multiple_choice_classification_ja": TaskCase(
        "shitsukan_eval.tasks.taxonomic.multiple_choice_classification.taxonomic_multiple_choice_classification_ja",
        "taxonomic_multiple_choice_classification_ja",
        "shitsukan_eval/tasks/taxonomic/multiple_choice_classification/taxonomic_multiple_choice_classification_ja.yaml",
        "taxonomic",
        "multiple_choice_classification",
        "ja",
    ),
}

# Responses in dataset order: two items for each of choice2..choice5 and random.
# They cover exact digits, digits with punctuation, letters and yes/no (scored as
# -100), "-1" for "no applicable option", and unparsable text.
SELECTION_RESPONSES = ["0", "1.", "(2) つやつや", "A", "yes", "3:", " 4 ", "no", "-1", "zzz"]
GENERATION_RESPONSES = ["かたい, つめたい, ひかる", "やわらかい"]
# Gold labels of the synthetic items are 2, 1, 3, 2 (see CLASSIFICATION_VOTES).
CLASSIFICATION_RESPONSES = ["2", "1", "4", "5."]
MULTIPLE_CHOICE_RESPONSES = ["0", "0", "1", "1:", "3", "zzz", "4", "1"]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the synthetic dataset and make its root the working directory."""
    root = tmp_path / "workspace"
    write_dataset(root)
    monkeypatch.chdir(root)
    return root


def _load_task_config(case: TaskCase) -> dict[str, Any]:
    """Load the task's YAML configuration from the repository."""
    return yaml.safe_load((REPOSITORY_ROOT / case.config).read_text(encoding="utf-8"))


def _run_task(
    case_name: str,
    responses: list[str],
    import_fresh: Callable[[str], Any],
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    sub_subtasks: list[str] | None = None,
    verbose: bool = True,
    record_error_message: bool = True,
) -> dict[str, Any]:
    """Run one task function as the evaluator does and observe its behavior."""
    case = TASKS[case_name]
    task_function = getattr(import_fresh(case.module), case.function)
    model = ScriptedModel()
    save_dir = workspace / "outputs"
    RECORDER.reset()
    RESPONDER.script(responses)

    result: Any = None
    error: dict[str, str] | None = None
    try:
        result = task_function(
            model=model,
            task_config=_load_task_config(case),
            verbose=verbose,
            save_dir=str(save_dir),
            task=case.task,
            sub_task=case.sub_task,
            sub_subtasks=sub_subtasks,
            lang=case.lang,
            model_type=MODEL_TYPE,
            model_name=MODEL_NAME,
        )
    except Exception as exc:  # noqa: BLE001 - errors are part of the observation.
        error = {"type": type(exc).__name__}
        if record_error_message:
            error["message"] = str(exc)
    observed = {
        "events": RECORDER.events,
        "result": result,
        "error": error,
        "stdout": text_lines(capsys.readouterr().out),
        "files": collect_files(save_dir) if save_dir.exists() else {},
        "contents_by_digest": model.contents_by_digest,
    }
    return to_record(observed, {str(workspace): "<workspace>"})


@pytest.mark.parametrize("case_name", ["perception_selection_ja", "perception_selection_en"])
def test_perception_selection_scores_all_subsets(case_name, import_fresh, workspace, capsys):
    observed = _run_task(case_name, SELECTION_RESPONSES, import_fresh, workspace, capsys)

    check_golden(f"tasks/{case_name}/all_subsets", observed)


def test_perception_selection_runs_only_the_requested_subsets(import_fresh, workspace, capsys):
    observed = _run_task("perception_selection_ja", ["-1", "0"], import_fresh, workspace, capsys, sub_subtasks=["random"])

    check_golden("tasks/perception_selection_ja/random_subset_only", observed)


def test_perception_selection_without_verbose_output(import_fresh, workspace, capsys):
    observed = _run_task("perception_selection_en", SELECTION_RESPONSES, import_fresh, workspace, capsys, verbose=False)

    check_golden("tasks/perception_selection_en/not_verbose", observed)


@pytest.mark.parametrize("case_name", ["perception_generation_ja", "commonsense_generation_ja"])
def test_generation_tasks_attach_responses(case_name, import_fresh, workspace, capsys):
    observed = _run_task(case_name, GENERATION_RESPONSES, import_fresh, workspace, capsys)

    check_golden(f"tasks/{case_name}/responses", observed)


@pytest.mark.parametrize("case_name", ["commonsense_classification_ja", "commonsense_classification_en"])
def test_commonsense_classification_scores_valid_labels(case_name, import_fresh, workspace, capsys):
    observed = _run_task(case_name, CLASSIFICATION_RESPONSES, import_fresh, workspace, capsys)

    check_golden(f"tasks/{case_name}/valid_labels", observed)


def test_commonsense_classification_ja_with_an_out_of_range_label(import_fresh, workspace, capsys):
    # The distance of a label outside 1..5 is None, and averaging it fails. The
    # message comes from NumPy internals, so only the error type is recorded.
    observed = _run_task("commonsense_classification_ja", ["2", "1", "4", "-1"], import_fresh, workspace, capsys, record_error_message=False)

    check_golden("tasks/commonsense_classification_ja/out_of_range_label", observed)


@pytest.mark.parametrize("case_name", ["commonsense_classification_ja", "commonsense_classification_en"])
def test_commonsense_classification_with_a_non_numeric_label(case_name, import_fresh, workspace, capsys):
    observed = _run_task(case_name, ["yes", "1", "1", "1"], import_fresh, workspace, capsys)

    check_golden(f"tasks/{case_name}/non_numeric_label", observed)


@pytest.mark.parametrize(
    ("case_name", "scenario", "responses"),
    [
        ("taxonomic_a_b_classification_ja", "valid_labels", ["A", "A", "C?"]),
        ("taxonomic_a_b_classification_ja", "digit_label", ["1", "A", "B"]),
        ("taxonomic_yes_no_classification_ja", "valid_labels", ["yes", "No", "no."]),
        ("taxonomic_yes_no_classification_ja", "letter_label", ["A", "yes", "no"]),
        ("taxonomic_multiple_choice_classification_ja", "valid_labels", MULTIPLE_CHOICE_RESPONSES),
        ("taxonomic_multiple_choice_classification_ja", "letter_label", ["B", *MULTIPLE_CHOICE_RESPONSES[1:]]),
    ],
)
def test_taxonomic_classification(case_name, scenario, responses, import_fresh, workspace, capsys):
    observed = _run_task(case_name, responses, import_fresh, workspace, capsys)

    check_golden(f"tasks/{case_name}/{scenario}", observed)


def test_taxonomic_multiple_choice_runs_only_the_requested_subsets(import_fresh, workspace, capsys):
    observed = _run_task("taxonomic_multiple_choice_classification_ja", ["2", "1"], import_fresh, workspace, capsys, sub_subtasks=["choice3_ans1"])

    check_golden("tasks/taxonomic_multiple_choice_classification_ja/choice3_subset_only", observed)
