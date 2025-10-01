python -u train.py \
  --version xxx \
  --epochs 3000 \
  --batch_size 16384 \
  --dropout_prob 0.25 \
  --num_emb_list 2048 \
  --e_dim 768 \
  --layers 2048 1024 \
  --device cuda:0 \
  --dataset flickr \
  --ckpt_dir ./output/rqvae_flickr_2048_12 \
  --code_length 12 \
  --bn \
  --kmeans_init \
  --use_cap \
  --use_pseudo \
  --use_sk



