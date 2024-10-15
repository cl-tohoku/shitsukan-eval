import pathlib
from typing import Any

from tqdm import tqdm

from tools.utils.data_utils import load_dataset, set_seed_all

# Set a global random seed
set_seed_all(42)

# Task description for generating perceptual responses in Japanese
TASK_DESCRIPTION_FOR_PERCEPTION_GENERATION_JA = (
    "提示された「ことば（単語）」から連想される「質感」を述べていただくタスクです。\n"
    "本タスクは「画像」に対する質感ではなく、「ことば（単語）」がもつ印象から連想される質感について述べていただきます。\n"
    "\u3000\n質感とは、下記のようなものを指します。\n"
    "・物性（光沢感・透明感など）\n"
    "・状態（乾燥・凍結など）\n"
    "・印象（美しい・醜いなど）\n\n"
    "提示された「ことば（単語）」に対し、できる限り一般的だと考えられる質感を述べて下さい。\n"
    "回答は質感を表す表現を3個「,」で分けて書いてください。\n"
    "その際、3個とも出来るだけ違う観点の表現にしてください。\n"
    "また、似た表現（きらきら・きらんきらんなど）にならないようにしてください。"
)


def perception_generation_ja(
    model,
    task_config: dict[str, Any],
    verbose: bool = False,
    save_dir: str = None,
    task: str = None,
    sub_task: str = None,
    sub_subtasks: str = None,
    lang: str = None,
    model_name: str = None,
) -> list[dict[str, Any]]:
    """
    Performs a perception generation task by prompting the model with object-related questions
    and recording its responses. The model is tasked with generating descriptions of perceived
    textures associated with given words (e.g., objects).

    Args:
        model (Any): The model to use for generating responses.
        task_config (Dict[str, Any]): Configuration for the task. Should include paths to datasets, few-shot examples, and generation parameters.
        verbose (bool, optional): If True, prints the prompt and model responses for debugging. Defaults to False.
        save_dir (Optional[str]): Directory to save the results, if provided.
        task (Optional[str]): Task name for metadata purposes.
        sub_task (Optional[str]): Sub-task name for metadata purposes.
        sub_subtasks (Optional[str]): Sub-subtask name for metadata purposes.
        lang (Optional[str]): Language of the task (e.g., 'ja').
        model_name (Optional[str]): Name of the model being evaluated.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the original data and the model's responses.

    Example usage:
    >>> model = GPT4o(pretrained="gpt-4o")
    >>> task_config = {
    >>>     "dataset_path": "data/dataset.jsonl",
    >>>     "few_shot_path": "data/few_shot.jsonl",
    >>>     "generation_kwargs": {"max_new_tokens": 50}
    >>> }
    >>> perception_generation_ja(model, task_config, verbose=True)
    [
        {
            'image_id': '71',
            'image': ['images/coco/train2017/000000000071.jpg'],
            'object': 'レール',
            'shitsukan_list': ['硬い', 'キラキラ', '固い', '古い', '冷たい', 'ザラザラ'],
            'zero_shot': '「レール」という単語は、どのような質感を持っているように感じますか？\n回答: ',
            'original_response': 'スムーズ, 軽快, 風を切る',
            'task': 'perception',
            'sub_task': 'generation',
            'lang': 'ja',
            'model_name': 'gpt-4o'
        },
        ...
    ]
    """

    # Load dataset and few-shot examples
    data_list = load_dataset(task_config["dataset_path"])
    few_shot_list = load_dataset(task_config["few_shot_path"])

    # Extract model name from the pretrained model's path
    model_name = pathlib.Path(model.pretrained).name

    # Initialize list to store new data with model responses
    new_data_list = []

    # Iterate through the dataset and generate model responses for each item
    for data in tqdm(data_list, desc="Processing dataset", leave=False):
        """ (Example)
        data (dict):
        {
            'image_id': '71',
            'image': ['images/coco/train2017/000000000071.jpg'],
            'image_url': ['http://images.cocodataset.org/train2017/000000000071.jpg'],
            'object': 'レール',
            'shitsukan_list': ['硬い', 'キラキラ', '固い', '古い', '冷たい', 'ザラザラ']
        }
        """

        # Construct the prompt using the object in the data
        text = f'写真内にある「{data["object"]}」は、どのような質感を持っているように感じますか？\n回答: '

        # Generate the response using the model
        response = model.generate_response(
            text=text,
            images=data["image"],  # Load the corresponding image for context
            system_prompt=TASK_DESCRIPTION_FOR_PERCEPTION_GENERATION_JA,  # Task-specific instructions
            few_shot_data=few_shot_list,  # Few-shot examples for better generation
            gen_kwargs=task_config["generation_kwargs"],  # Generation parameters
        )
        """ (Example)
        response (str):
        "スムーズ, 軽快, 風を切る"
        """

        # Verbose option for printing the prompt and response
        if verbose:
            tqdm.write(f"{text}{response}\n")

        # Add the prompt and response to the original data
        data["zero_shot"] = text
        data["original_response"] = response
        data["task"] = task
        data["sub_task"] = sub_task
        data["lang"] = lang
        data["model_name"] = model_name

        # Append the updated data to the new data list
        new_data_list.append(data)

    # Return the new data list for further use if needed
    return new_data_list
