#!/bin/bash
#flickr-nopseudo
python tools/unified_encdoer.py \
   --base_model_dir "xxx/output/retriever/flickr_base/flickr_no_pseudo_vali/t5-base/xxx_c1024_ep100_lr0.001_bch128_embadded/checkpoint-3976/" \
   --encoder_ckpt "xxx/stage1/ficker_stage1_output/nopseudo/best_enc_step900.pt" \
   --guidance_head_ckpt "xxx/stage1/ficker_stage1_output/nopseudo/best_head_step900.pt" \
   --output_dir "xxx/output/retriever/flickr_merge/nopseudo/vali_test/merged_encoder_guidance_model_flick" \
   --code_book_size 1024

# #coco-pseudo
# python tools/unified_encdoer.py \
#     --base_model_dir "xxx/output/retriever/coco_base/coco_pseudo_vali/t5-base/xxx_c1024_ep100_lr0.001_bch128_embadded/checkpoint-13872/" \
#     --encoder_ckpt "xxx/stage1/coco_stage1_output/pseudo/best_enc_step3000.pt" \
#     --guidance_head_ckpt "xxx/stage1/coco_stage1_output/pseudo/best_head_step3000.pt" \
#     --output_dir "xxx/output/retriever/coco_merge/pseudo/vali_test/merged_encoder_guidance_model_coco" \
#     --code_book_size 1024
