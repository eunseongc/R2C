# Reading to Compressing: Exploring the Multi-document Reader for Prompt Compression

This repository contains the official implementation for our EMNLP 2024 Findings paper:

> **From Reading to Compressing: Exploring the Multi-document Reader for Prompt Compression**  
> Eunseong Choi, Sunkyung Lee, Minjin Choi, June Park, and Jongwuk Lee  
> *Findings of the Association for Computational Linguistics: EMNLP 2024*

---

## Overview

Reading To Compression (R2C) aims to compress large input contexts (prompts) by leveraging multi-document reader models (e.g., FiD). It reduces token costs while preserving the key information needed for downstream tasks such as question answering.

In this repository, you will find:

- Scripts to install and configure the environment (Docker-based).
- Instructions to download, prepare, and process data for **LongBench** and **NQ** (Natural Questions).
- Steps to generate compressed prompts and run inference with _LLaMa-2-7b-chat_.
- Links to our provided FiD checkpoints and sample dataset.

---

## 1. Environment Setup

We recommend using a Docker container for reproducibility. Below are the steps to set up the environment:

1. **Export environment variables** (replace placeholders as needed):
   ```bash
   export env_name={your_env_name}
   export home_dir={your_home_dir_path}
   ```
2. **Run Docker** (with GPU support):
   ```bash
   docker run --gpus all --shm-size=8G -it \
       -v ${home_dir}:/workspace \
       --name ${env_name} \
       pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

   # Once the container is running, open a new terminal or use the same one to exec:
   docker exec -it ${env_name} bash
   ```
3. **Install necessary packages**:
   ```bash
   apt-get update
   apt-get install -y git
   ```
4. **Clone this repository** and install Python dependencies:
   ```bash
   git clone https://github.com/eunseongc/R2C
   cd R2C
   pip install -r requirements.txt
   ```

---

## 2. Data and Checkpoints

You can either use our pre-trained FiD checkpoints or train your own. For convenience, we provide:

- **NQ (test data)**: [Download link](https://www.icloud.com/iclouddrive/005ClYRH_zPetsd6heRoN3HXQ#test)  
- **FiD (trained on NQ with 20 passages)**: [Download link](https://www.icloud.com/iclouddrive/0ceIMmpr82jmKTXotT18-G_og#checkpoints_fid)

After downloading, unzip the checkpoint folder and place the data appropriately. Your directory structure should look like this:

```
├── R2C
│   ├── checkpoints_fid/
│   ├── data/
│   │   └── nq/
│   │       └── test.json
```

> **Note**  
> - If you prefer, you can use FiD checkpoints from the original [FiD repository](https://github.com/facebookresearch/FiD).  
> - You can also train FiD yourself using `train_fid.py`.  
> - To run LLama2 models, you **must** include your HuggingFace token in two files:  
>   - `R2C/LLM_inferenc.py`  
>   - `R2C/src_longbench/pred.py`  

---

## 3. Compressing LongBench

### 3.1 Download and Preprocess LongBench Datasets
Execute:
```bash
source scripts/preprocessing_longbench.sh 128 gpt 64
```
This script downloads and preprocesses the LongBench datasets, preparing them for scoring and compression.

### 3.2 Score and Compress LongBench Contexts with R2C
```bash
source scripts/score_and_compress_longbench.sh 0 checkpoints_fid/nq_20_0/checkpoint/best_dev 128_gpt_s_d_a64
```
- `0`: GPU index (change if needed)  
- `checkpoints_fid/nq_20_0/checkpoint/best_dev`: Path to the FiD checkpoint  
- `128_gpt_s_d_a64`: Example label for the configuration (feel free to rename)

### 3.3 Predict with Compressed Prompts on LongBench
```bash
source scripts/pred_and_eval_longbench.sh 128_gpt_s_d_a64
```
This script runs inference using the compressed prompts and evaluates the results.

---

## 4. Compressing NQ

### 4.1 Score and Compress NQ Contexts with R2C
```bash
source scripts/score_and_compress_nq.sh 0 \
    checkpoints_fid/nq_20_0/checkpoint/best_dev \
    data/nq/test.json \
    token_scores/nq_test.pkl
```
- `0`: GPU index  
- `checkpoints_fid/nq_20_0/checkpoint/best_dev`: Path to the FiD checkpoint  
- `data/nq/test.json`: NQ test set  
- `token_scores/nq_test.pkl`: Output file to store token scores

### 4.2 Predict with Compressed Prompts on NQ
We also provide a script for inference on the compressed NQ prompts. **Note**: for using VLLM, you need a specific transformers version:

```bash
pip install transformers==4.40.1
source pred_and_eval_nq.sh
```

---

## Citation

If you find this repository or our work useful, please cite:

```bibtex
@inproceedings{ChoiLCPL24,
  author    = {Eunseong Choi and
               Sunkyung Lee and
               Minjin Choi and
               June Park and
               Jongwuk Lee},
  title     = {From Reading to Compressing: Exploring the Multi-document Reader for Prompt Compression},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages     = {14734--14754},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.findings-emnlp.864},
  year      = {2024},
}
```

---
