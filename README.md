<div align="center">

shitsukan-eval
===========================

<h3>Evaluating Model (LLM / LVLM) Alignment with Human Perception</h3>

[📄 Paper](https://aclanthology.org/2025.coling-main.757/) &nbsp;&nbsp;|&nbsp;&nbsp; [🤗 Dataset](https://huggingface.co/datasets/tohoku-nlp/Shitsukan) &nbsp;&nbsp;|&nbsp;&nbsp; [🇯🇵 日本語](README_ja.md)

<img src="images/shitsukan-eval_overview.png" alt="shitsukan-eval" width="300px">

<div align="left">

<br>

We evaluate the alignment of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) with human perception, focusing on the Japanese concept of *shitsukan*.  
*Shitsukan* represents the sensory experience when perceiving objects, an inherently vague and highly subjective concept.  
We created a new dataset of *shitsukan* terms recalled by individuals in response to images of specified objects. We also designed benchmark tasks to evaluate the *shitsukan* recognition capabilities of LLMs and LVLMs.

<br>

This library is experimental and under active development.
We plan to add some **breaking changes** in the future to improve the usability and performance of the library.

# Table of Contents

- [Supported Models](#supported-models)
- [Usage](#usage)
  - [1. Build Environment](#1-build-environment)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Run Evaluation](#3-run-evaluation)
  - [4. Outputs](#4-outputs)
- [Reproducibility Notes](#reproducibility-notes)
- [Development](#development)
- [Citation](#citation)

## Supported Models

<details>
<summary><b>The currently supported API LLMs/LVLMs are as follows:</b></summary>
<li><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></li>
</details>

<br>

<details>
<summary><b>The currently supported Huggingface LLMs are as follows:</b></summary>
<li><a href="https://arxiv.org/abs/2307.09288">Llama 2</a></li>
<li><a href="https://note.com/elyza/n/na405acaca130">ELYZA-japanese-Llama-2</a></li>
<li><a href="https://arxiv.org/abs/2407.10671">Qwen 2</a></li>
<li><a href="https://arxiv.org/abs/2407.21783">Llama 3</a></li>
<li><a href="https://swallow-llm.github.io/llama3-swallow.ja.html">Llama-3-Swallow</a></li>
<li><a href="https://arxiv.org/abs/2408.00118">Gemma 2</a></li>
<li><a href="https://qwen2.org/qwen2-5/">Qwen 2.5</a></li>
<li><a href="https://llmc.nii.ac.jp/topics/post-707/">LLM-jp-3</a></li>
</details>

<br>

<details>
<summary><b>The currently supported Huggingface LVLMs are as follows:</b></summary>
<li><a href="https://arxiv.org/abs/2310.03744">LLaVA-1.5</a></li>
<li><a href="https://arxiv.org/abs/2405.02246">Idefics2</a></li>
<li><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">LLaVA-NeXT (LLaVA-1.6)</a></li>
<li><a href="https://arxiv.org/abs/2407.21783">Llama-3.2-Vision</a></li>
<li><a href="https://arxiv.org/abs/2408.03326">LLaVA-OneVision</a></li>
<li><a href="https://www.arxiv.org/abs/2408.12637">Idefics3</a></li>
<li><a href="https://arxiv.org/abs/2409.12191">Qwen2-VL</a></li>
<li><a href="https://www.arxiv.org/abs/2409.17146">Molmo</a></li>
</details>

<br>

<details>
<summary><b>The currently supported vLLM LVLMs are as follows:</b></summary>
<li><a href="https://arxiv.org/abs/2409.12191">Qwen2-VL</a></li>
</details>

<br>

<details>
<summary><b>How model names are mapped to model wrappers</b></summary>

The wrapper is chosen from substrings of the lowercase `--model` value; the first matching rule wins.

| `--model-type` | Substrings in the model name | Wrapper |
| --- | --- | --- |
| `hf` | `llama-3.2` and `vision` (and `instruct`) | Llama-3.2-Vision (`MllamaInstructHf`) |
| `hf` | `llava` and `1.5` / `next` or `1.6` / `onevision` | LLaVA-1.5 / LLaVA-NeXT / LLaVA-OneVision |
| `hf` | `idefics2` / `idefics3` / `qwen2-vl` / `molmo` | Idefics2 / Idefics3 / Qwen2-VL / Molmo |
| `hf` | `llama-3` (and `instruct`) | Llama 3 (also Llama-3-Swallow) |
| `hf` | `llama-2` (and `chat` or `instruct`) | Llama 2 (also ELYZA-japanese-Llama-2) |
| `hf` | `gemma-2` | Gemma 2 |
| `hf` | `qwen2.5` / `qwen2` (and `instruct`) | Qwen 2.5 / Qwen 2 |
| `hf` | `llm-jp-3` (and `instruct`) | LLM-jp-3 |
| `vllm` | `qwen2-vl` | Qwen2-VL |
| `api` | `gpt-4` | GPT-4o and other GPT-4 models |

Other names raise an error.
</details>

## Usage

Run all commands from the repository root: the task and model configurations are read from relative paths.

### 1. Build Environment

```bash
cd $HOME
git clone https://github.com/cl-tohoku/shitsukan-eval.git
cd $HOME/shitsukan-eval
uv python install 3.11
uv python pin 3.11
uv sync
```

`uv sync` also installs the development tools. On Linux x86_64 with CPython 3.11, they include `flash-attn`, which is installed from its prebuilt wheel for torch 2.4 and CUDA 12 (see `[tool.uv.sources]` in `pyproject.toml`).
To install only the packages that the evaluation needs, run `uv sync --no-dev` instead; the Hugging Face models then use eager attention (see [Reproducibility Notes](#reproducibility-notes)).

#### GPUs with a compute capability above 9.0 (Blackwell)

The pinned `torch==2.4.0+cu121` has kernels for compute capabilities 5.0 to 9.0 only.
On newer GPUs, for example the RTX PRO 6000 Blackwell (compute capability 12.0), it fails with `CUDA error: no kernel image is available for execution on the device`, and vLLM 0.6.3 and flash-attn 2.6.3 cannot run there either.
To run the Hugging Face models on such GPUs, create a separate environment with torch 2.7.1 for CUDA 12.8 (which needs an NVIDIA driver for CUDA 12.8 or later) and the other packages at the versions in `uv.lock`:

```bash
uv venv .venv-blackwell --python 3.11
uv export --no-dev --no-hashes --prune torch --prune torchvision --prune vllm -o .venv-blackwell/requirements.txt
uv pip install --python .venv-blackwell -r .venv-blackwell/requirements.txt \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --find-links https://download.pytorch.org/whl/cu128/torch \
    --find-links https://download.pytorch.org/whl/cu128/torchvision
```

Then run the evaluation with `.venv-blackwell/bin/python -m shitsukan_eval` instead of `uv run python -m shitsukan_eval`, which uses the pinned environment in `.venv`.
This environment has neither vLLM (`--model-type vllm` is unavailable) nor flash-attn (the Hugging Face models use eager attention), so its outputs can differ from those of the pinned environment.

To evaluate OpenAI models, set your API key:

```bash
export OPENAI_API_KEY="<your_api_key>"
```

### 2. Data Preparation

```bash
# Prepare the COCO 2017 images under <image_dir>/images/coco/{train2017,val2017}
mkdir -p $HOME/data/images/coco
cd $HOME/data/images/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -q train2017.zip
unzip -q val2017.zip

# Prepare the Shitsukan dataset and link its data directory to ./data
cd $HOME
git lfs install
git clone https://huggingface.co/datasets/tohoku-nlp/Shitsukan
ln -s $HOME/Shitsukan/data $HOME/shitsukan-eval/data
```

The task configurations expect the dataset files under `./data` (for example `data/perception/selection/choice2_ans1/perception_selection_choice2-ans1_ja.jsonl`).
The image paths in the dataset (for example `images/coco/train2017/000000000071.jpg`) are resolved relative to `--image-dir`, so with the layout above use `--image-dir $HOME/data`.

### 3. Run Evaluation

The following command evaluates the specified model on the specified tasks in shitsukan-eval.

```bash
export CUDA_VISIBLE_DEVICES=0
uv run python -m shitsukan_eval \
    --model "<model_name_or_path>" \
    --model-type "<model_type>" \
    --tasks "<task_name>" \
    --sub-tasks "<sub-task_name>" \
    --lang "<lang>" \
    --image-dir "<base-image_path>" \
    --save-dir outputs \
    --verbose
```

<details>
<summary><b>Explanation of the available arguments</b></summary>

- `--model` (`str`, required): The name or path of the model to evaluate. (e.g., `"Qwen/Qwen2-VL-7B-Instruct"`)
- `--model-type` (`str`, required): The model type of the specified model.
  - Model type that can be specified: `"api"`, `"hf"`, `"vllm"`
- `--tasks` (`List[str]`, required): The tasks to evaluate.
  - Tasks that can be specified: `"perception"`, `"commonsense"`, `"taxonomic"`
- `--sub-tasks` (`List[str]`, required): The sub-tasks to evaluate. Every sub-task is run for every task, so all combinations must be supported.
  - In case of `--tasks "perception" --lang "ja"`, Sub-tasks that can be specified: `"generation"`, `"selection"`
  - In case of `--tasks "perception" --lang "en"`, Sub-tasks that can be specified: `"selection"`
  - In case of `--tasks "commonsense" --lang "ja"`, Sub-tasks that can be specified: `"generation"`, `"classification"`
  - In case of `--tasks "commonsense" --lang "en"`, Sub-tasks that can be specified: `"classification"`
  - In case of `--tasks "taxonomic" --lang "ja"`, Sub-tasks that can be specified: `"a_b_classification"`, `"yes_no_classification"`, `"multiple_choice_classification"`
  - In case of `--tasks "taxonomic" --lang "en"`, Sub-tasks that can't be specified
- `--lang` (`str`, required): Language to use for the evaluation.
  - Language that can be specified: `"ja"`, `"en"`
- `--image-dir` (`Optional[str]`): Directory that the image paths of the dataset are resolved against (optional).
  - With `--image-dir "$HOME/data"`, the evaluation reads the COCO 2017 images from `$HOME/data/images/coco/train2017/*.jpg` and `$HOME/data/images/coco/val2017/*.jpg`. If you have not prepared the COCO 2017 images, please download them in advance from [here](https://cocodataset.org/#download).
  - Required for LVLMs (including GPT-4o) on the perception tasks.
- `-s`, `--save-dir` (`str`, required): Directory where evaluation results will be saved.
- `-v`, `--verbose` (`bool`): If set, print each prompt with the model's answer during processing.

</details>

<br>

> [!NOTE]
> The configuration files for each task are located at `shitsukan_eval/tasks/{task}/{sub_task}/{task}_{sub_task}_{lang}.yaml`, and the default model settings at `models/hf_models/hf_model_default_config.yaml` and `models/vllm_models/vllm_model_default_config.yaml`.<br>If you want to modify the settings, please change them here.

> [!NOTE]
> When evaluating Molmo, make a single GPU visible (for example `CUDA_VISIBLE_DEVICES=0`).

> [!NOTE]
> `--model-type vllm` needs vLLM and qwen-vl-utils, which `pyproject.toml` installs on Linux x86_64 only (vLLM 0.6.3 also requires the pinned torch 2.4.0). Where they are missing, the other model types still work.

### 4. Outputs

For each task and sub-task, the records with the prompts and the model responses are saved to `<save_dir>/<task>/<sub_task>/<task>_<sub_task>_<lang>_<model_type>_<model>.jsonl`, where `<model>` is the last path component of `--model`.
The classification and selection tasks also save their scores to `<task>_<sub_task>_<lang>_<model_type>_<model>_scores.jsonl` in the same directory and print them.
Tasks with subsets (perception selection and taxonomic multiple-choice classification) additionally save the records of each subset under `<save_dir>/<task>/<sub_task>/<subset>/`.
The generation tasks are not scored by this library.

## Reproducibility Notes

- The random generators of Python, NumPy, and PyTorch are seeded with 42 right before the model is loaded, and the tasks sample with `temperature: 0.01` (see the task configuration files).
- Results can depend on library versions. The packages that affect the model outputs are pinned in `pyproject.toml` to the versions available when the experiments of the paper were run (autumn 2024): `torch==2.4.0`, `transformers==4.46.0`, `qwen-vl-utils==0.0.8`, and `flash-attn==2.6.3`.
  - `transformers` was previously installed from its GitHub main branch. That branch is now transformers 5, which removed `AutoModelForVision2Seq`, so a newly created environment could not even import the evaluator. 4.46.0 is the first release that contains Idefics3, which the wrappers import.
  - The other dependencies are not pinned in `pyproject.toml`. `uv.lock` records the versions of all packages, which `uv sync` installs.
  - uv warns that `transformers==4.46.0` is yanked. The reason given, missing support for Python 3.8, does not affect Python 3.11.
  - If you still have the `uv.lock` of the original experiments, you can reproduce them exactly by pinning `transformers` to the commit recorded there, for example `transformers = { git = "https://github.com/huggingface/transformers.git", rev = "<commit>" }` under `[tool.uv.sources]`.
- When `flash-attn` is installed, the Hugging Face models use `attn_implementation="flash_attention_2"` (except Llama-3.2-Vision and Molmo, which use eager attention); otherwise they use eager attention.

## Development

```bash
# Run the tests (they use fake model libraries, so no GPU or model download is needed)
uv run pytest

# Install the Git hooks: ruff (lint and format), ty (type check), and gitleaks (secret detection)
uv run pre-commit install

# Run all hooks on the whole repository
uv run pre-commit run --all-files
```

The tests in `tests/characterization` compare the prompts, generation settings, outputs, and scores with golden records in `tests/golden`, which were recorded from the original implementation.
Update them with `UPDATE_GOLDEN=1 uv run pytest` only for intentional behavior changes, and review the diff.
`tests/unit/test_environment.py` checks that the installed `transformers` provides every class the wrappers import (it is skipped where `transformers` is not installed).
The ty hook runs `uv check --frozen`, which uses the versions in `uv.lock`; run `uv sync` once before the first commit.

## Citation

```bibtex
@inproceedings{shiono-etal-2025-evaluating,
    title = "Evaluating Model Alignment with Human Perception: A Study on Shitsukan in {LLM}s and {LVLM}s",
    author = "Shiono, Daiki  and
      Brassard, Ana  and
      Ishizuki, Yukiko  and
      Suzuki, Jun",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.757/",
    pages = "11428--11444",
}
```
