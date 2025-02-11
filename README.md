<div align="center">

shitsukan-eval
===========================
<h3>Evaluating Model (LLM / LVLM) Alignment with Human Perception</h3>

[📄 Paper (Coming Soon)]() &nbsp;&nbsp;|&nbsp;&nbsp; [🚀 Project Page]() &nbsp;&nbsp;|&nbsp;&nbsp; [🤗 Dataset]()

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
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Supported Models

<details>
<summary><b>The currently supported API LLMs/LVLMs are as follows:</b></summary>
<li><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></li>
<li>(🚧 Here: Add description for this repo 🚧)</li>
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
<li>(🚧 Here: Add description for this repo 🚧)</li>
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
<li>(🚧 Here: Add description for this repo 🚧)</li>
</details>

<br>

<details>
<summary><b>The currently supported vLLM LLMs are as follows:</b></summary>
<li>(🚧 Here: Add description for this repo 🚧)</li>
</details>

<br>

<details>
<summary><b>The currently supported vLLM LVLMs are as follows:</b></summary>
<li><a href="https://arxiv.org/abs/2409.12191">Qwen2-VL</a></li>
<li>(🚧 Here: Add description for this repo 🚧)</li>
</details>

## Usage

### 1. Build Environment

```bash
cd $HOME
git clone git@github.com:<ANONYMOUS>/shitsukan-eval
cd $HOME/shitsukan-eval
uv python install 3.11
uv python pin 3.11
uv sync --no-dev
uv sync --dev --no-build-isolation

# for developper
# uv run pre-commit install
```

### 2. Data Preparation

```bash
# Prepare COCO 2017 images
mkdir -p $HOME/data/images
cd $HOME/data/images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# Prepare our Shitsukan datasets
mkdir -p  $HOME/shitsukan-eval/data
cd $HOME/shitsukan-eval/data
git lfs install
git clone https://huggingface.co/datasets/<ANONYMOUS>/Shitsukan
```

### 3. Run Evaluation

The following command evaluate the specified model on the specified tasks in shitsukan-eval.

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

- `--model` (`str`): The name or path of the model to evaluate. (e.g., `"Qwen/Qwen2-VL-7B-Instruct"`)
- `--model-type` (`str`): The model type of the specified model.
  - Model type that can be specified: `"api"`, `"hf"`, `"vllm"`
- `--tasks` (`str`): The task name to evaluate.
  - Tasks that can be specified: `"perception"`, `"commonsense"`, `"taxonomic"`
- `--sub-tasks` (`List[str]`): List of sub-tasks within the tasks.
  - In case of `--tasks "perception" --language "ja"`, Sub-tasks that can be specified: `"generation"`, `"selection"`
  - In case of `--tasks "perception" --language "en"`, Sub-tasks that can be specified: `"selection"`
  - In case of `--tasks "commonsense" --language "ja"`, Sub-tasks that can be specified: `"generation"`, `"classification"`
  - In case of `--tasks "commonsense" --language "en"`, Sub-tasks that can't be specified
  - In case of `--tasks "taxonomic" --language "ja"`, Sub-tasks that can be specified: `"a_b_classification"`, `"yes_no_classification"`, `"multiple_choice_classification"`
  - In case of `--tasks "taxonomic" --language "en"`, Sub-tasks that can't be specified
- `--lang` (`str`): Language to use for the evaluation (default: `"ja"`).
  - Language that can be specified: `"ja"`, `"en"`
- `--image-dir` (`Optional[str]`): Directory where input images are stored (optional).
  - If you specify `--image-dir="data"`, the evaluation script will reference the COCO 2017 images located at `data/images/coco2017/train2017/*.png` and `data/images/coco2017/val2017/*.png` during execution. If you have not prepared the COCO 2017 images, please download them in advance from [here](https://cocodataset.org/#download).
- `--save-dir` (`str`): Directory where evaluation results will be saved.
- `-v`, `--verbose` (`bool`): If set, print detailed information during processing.

</details>

<br>

> [!NOTE]
> The configuration files for each task are located at `shitsukan_eval/tasks/{task}/{sub_task}/{task}_{sub_task}_{lang}.yaml`.<br>If you want to modify the settings, please change them here.

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

## Acknowledgement

(🚧 Here: Add description for this repo 🚧)
