#  --caption_file data/flickr/mscoco_captions.json \
#python -u prepare_emb.py \
#  --root_dir data/flickr/images/split \
#  --caption_file data/flickr/flickr_split_captions.json \
#  --pseudo_caption_file data/flickr/pseudo_query.json \
#  --batch_size 1024 \
#  --clip_model Salesforce/blip-itm-large-flickr \
#  --save_path data/flickr_blip/emb

#coco
export CUDA_VISIBLE_DEVICES=1
python -u prepare_emb.py \
  --root_dir data/coco/images/split \
  --caption_file data/coco/mscoco_captions.json \
  --pseudo_caption_file data/coco/pseudo_query.json \
  --batch_size 1024 \
  --save_path data/coco/emb


