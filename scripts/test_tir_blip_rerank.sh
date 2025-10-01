#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1
python test_tir.py \
  --data_path "data/flickr/pseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/" \
  --ckpt      "xxx/output/retriever/flickr_tir/pseudo/vali_test/xxx_VT_1024-512_1-c1024_e3000_lr0.0001_mse_c1024_ep100_lr0.0005_bch128_embadded/checkpoint-3809/" \
  --beam 50 \
  --topk 50 \
  --head_path xxx/stage1/ficker_stage1_output/pseudo/best_head_step1000.pt \
  --image_emb_path xxx/RQ-VAE/data/flickr/emb/test_emb_images.npy \
  --caption_emb_path xxx/RQ-VAE/data/flickr/emb/test_emb_captions.npy \
  --blip_rerank \
  --blip_model_path xxx/model_large_retrieval_flickr.pth \
  --blip_image_json xxx/RQ-VAE/data/flickr/emb/test_emb.json \
  --blip_image_dir xxx/RQ-VAE/data/flickr/images/split/test \
  --rerank \
  --blip_caption_path xxx/data/flickr/pseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/test.source
