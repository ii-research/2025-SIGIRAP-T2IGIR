#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1
python test_tir.py \
  --data_path "data/flickr/nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/" \
  --ckpt      "xxx/output/retriever/flickr_tir/nopseudo/vali_test/nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse_c1024_ep100_lr0.0005_bch128_embadded/checkpoint-3408/" \
  --beam 50 \
  --topk 50 \
  --head_path "xxx/stage1/ficker_stage1_output/nopseudo/best_head_step900.pt" \
  --image_emb_path xxx/RQ-VAE/data/flickr/emb/test_emb_images.npy \
  --caption_emb_path xxx/RQ-VAE/data/flickr/emb/test_emb_captions.npy \
  --rerank
