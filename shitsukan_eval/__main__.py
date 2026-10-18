r"""Command-line entry point: ``python -m shitsukan_eval``.

Example:
    Evaluate GPT-4o on Japanese perception generation from the repository root::

        uv run python -m shitsukan_eval \
            --model gpt-4o \
            --model-type api \
            --tasks perception \
            --sub-tasks generation \
            --lang ja \
            --image-dir data \
            --save-dir outputs \
            --verbose
"""

from __future__ import annotations

import argparse

from shitsukan_eval.evaluator import evaluate

MODEL_TYPES = ("hf", "vllm", "api")


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments of the evaluation.

    Args:
        argv: Arguments to parse (default: ``sys.argv[1:]``).

    Returns:
        The parsed arguments: ``model``, ``model_type``, ``tasks``, ``sub_tasks``,
        ``lang``, ``image_dir``, ``save_dir``, and ``verbose``.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, type=str, help="Name or path of the model to evaluate.")
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES, help="The type of the model (available choices: ['hf', 'vllm', 'api']).")
    parser.add_argument("--tasks", required=True, nargs="*", type=str, help="List of tasks to evaluate.")
    parser.add_argument("--sub-tasks", required=True, nargs="*", type=str, help="List of sub-tasks within the tasks.")
    parser.add_argument("--lang", required=True, type=str, help="Language to use for evaluation ('ja' or 'en').")
    parser.add_argument("--image-dir", default=None, type=str, help="Directory where input images are stored (optional).")
    parser.add_argument("-s", "--save-dir", required=True, type=str, help="Directory where evaluation results will be saved.")
    parser.add_argument("-v", "--verbose", action="store_true", help="If set, print detailed information during processing.")
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the evaluation described by ``args`` (parsed from the command line if ``None``)."""
    if not args:
        args = parse_eval_args()
    evaluate(
        model_name=args.model,
        model_type=args.model_type,
        tasks=args.tasks,
        sub_tasks=args.sub_tasks,
        lang=args.lang,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
