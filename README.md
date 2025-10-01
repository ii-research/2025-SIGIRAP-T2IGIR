# A Combination-based Framework for Generative Text–image Retrieval: Dual Identifiers and Hybrid Retrieval Strategies
## 🔍 Overview

This repository provides the official implementation of our paper:
"A Combination-based Framework for Generative Text–image Retrieval: Dual Identifiers and Hybrid Retrieval Strategies" .

We focus on generative cross-modal retrieval (GCMR), especially text-to-image retrieval. Different from classical discriminative approaches, our framework leverages large language models to generate identifiers for candidate images, supporting efficient, scalable, and high-performing retrieval.
###  Framework Overview
Traditional cross-modal retrieval methods rely on joint embedding spaces or cross-attention architectures.
Our approach (ComGTIR) introduces two key innovations for generative retrieval:

- **Dual Identifiers** 
  - Sequential Identifier: Encodes each image as a sequence of tokens, enabling fine-grained, order-sensitive generative retrieval.
  - Order-invariant Identifier: Provides a global, order-free representation to guide decoding, helping avoid local optima during generation.

- **Hybrid Retrieval Strategy**
    - After generative retrieval, we rerank the top-k candidates using dense embedding methods (e.g., CLIP/BLIP). This balances efficiency and effectiveness.
The framework is fully modular and reproducible. All main modules, training, and evaluation scripts are included.

# 📦 Requirements

The code is tested on Python 3.9.18, PyTorch 1.13.1 and CUDA 11.7. 

You can create a conda environment with the required dependencies using the provided `environment.yml` file.

```bash 
conda env create -f environment.yml
conda activate tir
```

## 🧾 Data

1. The dataset used in the paper is the [COCO 2014](http://cocodataset.org/#download) dataset and the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset. The raw images should be downloaded and placed in the `RQ-VAE/data` directory, along with captions. The captions files we used can be found here().

2. Run the following command to preprocess the data to generate the image features and text features:

```bash
cd RQ-VAE
bash scripts/prepare_emb.sh
```

3. You can also use the simple `tools/generate_pseudo_query.py` script to generate pseudo queries to augment the dataset. The pseudo queries we used can be found here().


## 📈 ComGTIR-D Pipeline

### 🔵 Tokenizer (RQ-VAE)

#### Step 1: Train the Tokenizer (RQ-VAE)

```bash
cd RQ-VAE
bash scripts/train_rqvae.sh
```

The trained model will be saved in the `RQ-VAE/output` directory.

#### Step 2: Discrete Image Representations

```bash
bash scripts/generate_codes.sh
```

This script encodes images into discrete token sequences, which will be used in downstream Retriever stages.


### 🟡 Retriever (LLM)

#### Step 3: Prepare Retriever Training Data

Use the previously generated voken codes to construct the training data for the retriever. Make sure you are in the project root directory.

```bash
# from project root
bash scripts/prepare_retriever_dataset.sh
```

#### Step 4: Train the Retriever (Multi-Stage)

This project implements a multi-stage training process to progressively enhance the retriever's performance.



🔹 **Stage 1: Sequential Identifier**

Train an autoregressive decoder on voken sequences.

```bash
# To speed up training, you can set the log_freq parameter to 100. 
# The best model will be saved in the file.
bash scripts/stage1_seqid.sh
# This script calls train_retriever_t5_seqid.py internally.
```
🔹 **Stage 2: Order-invariant Identifier**

Train a global relevance head with InfoNCE loss to learn set-based preferences.

```bash
# This script trains the model for set-based preference.
# Please ensure the script is configured correctly before running.
bash scripts/stage2_setid.sh
```

🔹 **Stage 3: Unified Joint Training（Train the ComGTIR-D）**

Combine set-based and sequential identifier training for guided decoding.

```bash
# Merge the model saved in the first stage with the model in the second stage.
bash scripts/stage3_unified_encdoer.sh

# To speed up training, you can set the log_freq parameter to 100. 
# The best model will be saved in the file.
# For Flickr30k
bash scripts/stage3_tir_flickr.sh

# For COCO
bash scripts/stage3_tir_coco.sh

# Note：For evaluation, recall metrics will be automatically recorded in the log file and wandb.
```



🔹 **Stage 4: Evaluate the ComGTIR-D（inference)**

After Stage 3 has saved its best checkpoint, simply run the evaluation script to  calculate Recall @ {1, 5, 10}
```bash
bash scripts/test_tir.sh
```

If you want to rerank the top-k candidates, using CLIP, you can run the following command:

```bash
bash scripts/test_tir_clip_rerank.sh
```

If you want to rerank the top-k candidates, using BLIP(itm), you can run the following command:

```bash
bash scripts/test_tir_blip_rerank.sh
```




### 📌 Note

- Please update all dataset paths in both training and evaluation scripts to match your local directory structure.You should tune them appropriately for your own dataset and task.
- The default hyperparameters (e.g., learning rate, batch size, number of epochs) are configured for reference datasets (such as Flickr30K or COCO).  
  
