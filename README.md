<div align="center">

<h1 align="center"><b>[ACMMM 2025]</b>DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models</h1>

<p align="center">
   <a href="https://github.com/fjc2005">Jiachen Fu</a>, <a href="https://mmcheng.net/clguo/">Chun-Le Guo<sup>*</sup></a>, <a href="https://li-chongyi.github.io/">Chongyi Li<sup>†</sup></a>
</p>

<p align="center">
  *Corresponding Author. <br> †Project Lead.
</p>

<p align="center">
    <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python Version 3.12"></a>
    <a href="https://huggingface.co/spaces/fjc/DetectAnyLLM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Hugging Face Demo"></a>
    <a href="https://fjc2005.github.io/detectanyllm/"><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
</p>

</div>

## 📜 Abstract

**DetectAnyLLM** is an **AI-generated text detection** (i.e., Machine-Generated Text Detection) model based on the [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) framework, optimized using **DDL (Direct Discrepancy Learning)**.

DDL is a novel optimization method specifically designed for AI text detection tasks. It introduces a **task-oriented loss function**, enabling the model to directly learn the intrinsic knowledge of AI text detection during training. We found that DDL largely addresses the overfitting problem commonly seen in previous training-based detectors, significantly improving the generalization performance.

Additionally, considering that existing benchmark datasets lack coverage of proprietary LLMs and do not sufficiently address machine-revised texts, we propose the **MIRAGE** benchmark. MIRAGE collects human-written texts from 10 corpora across 5 domains, and uses 17 powerful LLMs (including 13 proprietary and 4 advanced open-source LLMs) to **re-generate**, **polish**, and **rewrite** these texts, resulting in nearly 100,000 high-quality human-AI text pairs. We hope that the MIRAGE benchmark will contribute to establishing a unified evaluation standard for AI-generated text detection.

<div align="center">
    <img src="./fig/teaser_small.png" alt="teaser" width="98.5%">
</div>

---

## 📖 Table of Contents
- [🔥 News](#-news)
- [🛠️ Setup](#-setup)
- [🚀 Reproduce Results](#-reproduce-results)
- [📝 Citation](#-citation)
- [TODO](#-todo)

---

## 🔥 News
- **[2025-07-17]** 🎉 The code of **DetectAnyLLM** and the data of **MIRAGE** is released!
- **[2025-07-05]** 📝 Our paper **DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models** is accepted by **ACM Multimedia 2025**!

## 🛠️ Setup
1.  **Clone the repository and navigate to the project directory:**
    ```bash
    git clone https://github.com/fjc2005/DetectAnyLLM.git
    cd DetectAnyLLM
    ```
2.  **Create and activate the conda environment:**
    ```bash
    conda create -n DetectAnyLLM python=3.12 -y
    conda activate DetectAnyLLM
    ```
3.  **Install the required dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
4.  **Download necessary models:**
    ```bash
    sh scripts/download_model.sh
    ```
    > If you want to reproduce all experiments reported in our paper, please go to ```./scripts/download_model.sh``` and revise it following the guidance provided by common.

    > If you have trouble downloading, try to set the environment variable before downloading:
    ```bash
    export HF_ENDPOINT="https://hf-mirror.com"
    ```

## 🚀 Reproduce Results

### 🚂 Train DDL
**[GPU memory cost: ~11G]**
```bash
# Login to wandb
wandb login
# or
# export WANDB_MODE=offline
sh scripts/train.sh
```

### 📊 Evaluation
**[GPU memory cost: ~15G]**

**Make sure you have trained DDL or downloaded checkpoints.**
```bash
sh scripts/eval.sh
```
The results will be saved in ```./results```.

### ⚙️ Reproduce Other Methods
**Make sure you have downloaded all models in `download_model.sh`.**
```bash
sh scripts/other_method/eval_${METHOD}.sh
```
`METHOD` is the method you want to reproduce.

For example, to reproduce Fast-DetectGPT, run:
```bash
sh scripts/other_method/eval_fast_det_gpt.sh
```
> **Note:** To reproduce DetectGPT and NPR, you should run the following code first:
```bash
sh scripts/other_method/generate_perturbs.sh
```

---

## 📝 Citation

If you find our work useful, please consider citing:
```bibtex
@inproceedings{fu2025detectanyllm,
  title={DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models},
  author={Fu, Jiachen and Guo, Chun-Le and Li, Chongyi},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025}
}
```

## TODO

- [ ] Jittor implementation of DetectAnyLLM.
- [ ] Code of Local Demo.
- [ ] Release MIRAGE-zh.
- [ ] Online Demo.