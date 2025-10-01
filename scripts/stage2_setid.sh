#!/bin/bash

T5_MODEL="t5-base"
DATA_ROOT="xxx/data/flickr/pseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse"
CODEBOOK_EMB="${DATA_ROOT}/codebook_embedding.pt"
OUTPUT_DIR="./stage1/ficker_stage1_output/pseudo/"


torchrun --nproc_per_node 4 train_retriever_t5_setid.py \
  --t5_model_name_or_path $T5_MODEL \
  --caption_file_train ${DATA_ROOT}/train.source \
  --image_code_file_train ${DATA_ROOT}/train.target \
  --caption_file_val ${DATA_ROOT}/val.source \
  --image_code_file_val ${DATA_ROOT}/val.target \
  --caption_file_test ${DATA_ROOT}/test.source \
  --image_code_file_test ${DATA_ROOT}/test.target \
  --codebook_embedding_path $CODEBOOK_EMB \
  --pag_code_book_size 1024 \
  --max_seq_length 128 \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 50 \
  --batch_size 256 \
  --learning_rate 1e-3 \
  --warmup_steps 10 \
  --adam_epsilon 1e-8 \
  --seed 42

