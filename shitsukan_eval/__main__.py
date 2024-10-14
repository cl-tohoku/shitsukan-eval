import argparse

from shitsukan_eval.evaluator import evaluate


def parse_eval_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed arguments, including model name, tasks, sub-tasks, language, image directory, save directory, and verbosity.

    Command-line arguments:
        --model (str): The name or path of the model to evaluate.
        --tasks (List[str]): List of tasks to evaluate.
        --sub-tasks (List[str]): List of sub-tasks within the tasks.
        --lang (str): Language to use for the evaluation (default: "ja").
        --image-dir (Optional[str]): Directory where input images are stored (optional).
        --save-dir (str): Directory where evaluation results will be saved.
        -v, --verbose: If set, print detailed information during processing.

    Example usage:
    >>> args = parse_eval_args()
    >>> print(args.model, args.tasks)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Name or path of the model to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        nargs="*",
        type=str,
        help="List of tasks to evaluate.",
    )
    parser.add_argument(
        "--sub-tasks",
        required=True,
        nargs="*",
        type=str,
        help="List of sub-tasks within the tasks.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        default="ja",
        type=str,
        help="Language to use for evaluation (default: 'ja').",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        type=str,
        help="Directory where input images are stored (optional).",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        required=True,
        default="outputs",
        type=str,
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If set, print detailed information during processing.",
    )

    # Parse and return command-line arguments
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace | None = None) -> None:
    """
    Main function to run the evaluation based on command-line arguments.

    Args:
        args (Optional[argparse.Namespace]): Parsed command-line arguments. If None, the arguments will be parsed using parse_eval_args().

    Example usage from command line:
    uv run python shitsukan_eval.py \
        --model "gpt-4o" \
        --tasks "perception" \
        --sub-tasks "generation" \
        --lang "ja" \
        --image-dir "data" \
        --save-dir outputs \
        --verbose

    This will evaluate the 'gpt-4o' model on the 'perception' task with 'generation' sub-task, using Japanese as the language.
    """
    if not args:
        # Parse arguments if not provided
        args = parse_eval_args()

    # Run the evaluation using the provided arguments
    evaluate(
        model_name=args.model,
        tasks=args.tasks,
        sub_tasks=args.sub_tasks,
        lang=args.lang,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    # Run the main function if the script is executed directly
    main()
