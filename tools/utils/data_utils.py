import json
import os
import pathlib
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import yaml


# DATA SAVING
def save_json(filename: str, ds: dict[str, Any]) -> None:
    """
    Save data in JSON format.

    Args:
        filename (str): Path to the output file.
        ds (Dict[str, Any]): Data to be saved in JSON format.

    Example usage:
    >>> save_json("output.json", {"name": "example"})
    """
    with open(filename, "w") as f:
        json.dump(ds, f, indent=4)


def save_jsonl(filename: str, data_list: list[dict[str, Any]]) -> None:
    """
    Save data in JSONL (JSON Lines) format.

    Args:
        filename (str): Path to the output file.
        data_list (List[Dict[str, Any]]): List of data (dict) to be saved.

    Example usage:
    >>> save_jsonl("output.jsonl", [{"name": "example"}])
    """
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as fo:
        for r in data_list:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_results_as_jsonl(save_dir: str, results: list[dict[str, Any]], task: str, sub_task: str, sub_subtask: str | None, lang: str, model_name: str) -> None:
    """
    Save evaluation results as JSONL format based on the task structure.

    Args:
        save_dir (str): Directory where the results will be saved.
        results (List[Dict[str, Any]]): List of results (dict) to be saved.
        task (str): Name of the task.
        sub_task (str): Name of the sub-task.
        sub_subtask (Optional[str]): Name of the sub-subtask (optional).
        lang (str): Language of the evaluation.
        model_name (str): Name of the evaluated model.

    Example usage:
    >>> save_results_as_jsonl("output", results, "perception", "generation", None, "ja", "gpt-4")
    """
    save_dir = pathlib.Path(save_dir)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H:%M:%S")
    if sub_subtask is not None:
        output_file = save_dir.joinpath(f"{task}/{sub_task}/{sub_subtask}/{task}_{sub_task}_{sub_subtask}_{lang}_{model_name}_{timestamp}.jsonl")
    else:
        output_file = save_dir.joinpath(f"{task}/{sub_task}/{task}_{sub_task}_{lang}_{model_name}_{timestamp}.jsonl")

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fo:
        for r in results:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_eval_scores_as_jsonl(save_dir: str, scores: list[dict[str, Any]], task: str, sub_task: str, lang: str, model_name: str) -> None:
    """
    Save evaluation scores as JSONL format.

    Args:
        save_dir (str): Directory where the scores will be saved.
        scores (List[Dict[str, Any]]): List of evaluation scores (dict).
        task (str): Name of the task.
        sub_task (str): Name of the sub-task.
        lang (str): Language of the evaluation.
        model_name (str): Name of the evaluated model.

    Example usage:
    >>> save_eval_scores_as_jsonl("output", scores, "perception", "generation", "ja", "gpt-4")
    """
    save_dir = pathlib.Path(save_dir)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H:%M:%S")
    output_file = save_dir.joinpath(f"{task}/{sub_task}/{task}_{sub_task}_{lang}_{model_name}_scores_{timestamp}.jsonl")

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fo:
        for s in scores:
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")


# DATA LOADING
def load_json(filename: str) -> list[Any]:
    """
    Load data from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        List[Any]: Loaded JSON data.

    Example usage:
    >>> data = load_json("data.json")
    """
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filename: str) -> list[dict[str, Any]]:
    """
    Load data from a JSONL (JSON Lines) file.

    Args:
        filename (str): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: Loaded JSONL data as a list of dictionaries.

    Example usage:
    >>> data = load_jsonl("data.jsonl")
    """
    with open(filename, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# DATA PROCESSING
def convert_json_to_jsonl(filename: str, output_dir: str) -> None:
    """
    Convert a JSON file to JSONL format.

    Args:
        filename (str): Path to the JSON file.
        output_dir (str): Directory where the converted JSONL file will be saved.

    Example usage:
    >>> convert_json_to_jsonl("data.json", "output")
    """
    filename = pathlib.Path(filename)
    output_dir = pathlib.Path(output_dir)
    save_filename = output_dir.joinpath(filename.stem + ".jsonl")
    # load json file
    df = pd.read_json(filename)
    # save as jsonl file
    df.to_json(save_filename, orient="records", force_ascii=False, lines=True)


def convert_csv_to_jsonl(filename: str, output_dir: str) -> None:
    """
    Convert a CSV file to JSONL format.

    Args:
        filename (str): Path to the CSV file.
        output_dir (str): Directory where the converted JSONL file will be saved.

    Example usage:
    >>> convert_csv_to_jsonl("data.csv", "output")
    """
    filename = pathlib.Path(filename)
    output_dir = pathlib.Path(output_dir)
    save_filename = output_dir.joinpath(filename.name + ".jsonl")
    # load json file
    df = pd.read_csv(filename)
    # save as jsonl file
    df.to_json(save_filename, orient="records", force_ascii=False, lines=True)


def load_dataset(filename: str) -> list[dict[str, Any]]:
    """
    Load a dataset from a file (JSON, CSV, or JSONL format).

    Args:
        filename (str): Path to the dataset file.

    Returns:
        List[Dict[str, Any]]: Loaded dataset.

    Raises:
        ValueError: If the file extension is unsupported.

    Example usage:
    >>> data = load_dataset("data.json")
    """
    filename = pathlib.Path(filename)

    if filename.suffix == ".json":
        convert_json_to_jsonl(str(filename), str(filename.parent))
        tmp_save_filename = filename.parent.joinpath(filename.name + ".jsonl")
        data_list = load_jsonl(str(tmp_save_filename))
        os.remove(str(tmp_save_filename))
    elif filename.suffix == ".csv":
        convert_csv_to_jsonl(str(filename), str(filename.parent))
        tmp_save_filename = filename.parent.joinpath(filename.name + ".jsonl")
        data_list = load_jsonl(str(tmp_save_filename))
        os.remove(str(tmp_save_filename))
    elif filename.suffix == ".jsonl":
        data_list = load_jsonl(str(filename))
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")

    return data_list


def load_yaml(filename: str) -> dict[str, Any] | None:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        filename (str): Path to the YAML file.

    Returns:
        Optional[Dict[str, Any]]: Parsed YAML data as a dictionary. Returns None if an error occurs.

    Example usage:
    >>> yaml_data = load_yaml("config.yaml")
    >>> if yaml_data:
    >>>     print(yaml_data)
    """
    try:
        with open(filename, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"An error occurred while reading the YAML file: {e}")
        return None


# Function to set seeds for reproducibility
def set_seed_all(seed: int = 42) -> None:
    """
    Set the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all random generators. Defaults to 42.

    Example usage:
    >>> set_seed_all(123)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


# URL UTILITIES
def is_url(path: str) -> bool:
    """
    Check if a given path is a URL.

    Args:
        path (str): The path or URL to check.

    Returns:
        bool: True if the path is a URL, False otherwise.

    Example usage:
    >>> is_url("http://example.com/image.jpg")
    True
    >>> is_url("/path/to/image.jpg")
    False
    """
    return path.startswith(("http://", "https://"))


def check_image_exists(url: str) -> bool:
    """
    Check if an image exists at the given URL.

    Args:
        url (str): The URL of the image to check.

    Returns:
        bool: True if the image exists, False otherwise.

    Example usage:
    >>> check_image_exists("http://example.com/image.jpg")
    True
    """
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# DATA POSTPROCESSING
def simple_choice_extractors(answer: str) -> str:
    """
    Extract a choice label from the answer string using a regular expression.

    Args:
        answer (str): The answer string to extract the label from.

    Returns:
        str: Extracted label if found, "-100" otherwise.

    Example usage:
    >>> simple_choice_extractors("(1) Some answer")
    '1'
    >>> simple_choice_extractors("Invalid")
    "-100"
    """
    predicted_answer = answer.strip()
    pattern = re.compile(r"\({0,1}(-1|[0-4])[.,:;)]{0,1}")
    extracted_predicted_answer = pattern.match(predicted_answer)

    if extracted_predicted_answer:
        return extracted_predicted_answer.group(1)
    else:
        return "-100"
