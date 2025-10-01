#filckr-no-pseudo
#python -u tools/prepare_dataset.py \
#    --code_file RQ-VAE/output/rqvae_flickr_no_pseudo/nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/flickr_codes.json \
#    --dataset flickr \
#    --output_dir nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse \


#coco-pseudo
#python -u tools/prepare_dataset.py \
#    --code_file RQ-VAE/output/rqvae_coco/coco_VT_1024-512_1-c1024_e3000_lr0.0001_mse/coco_codes.json \
#    --dataset coco \
#    --output_dir pseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse \
#    --pseudo_file /home/iiserver31/Workbench/likaipeng/avg/RQ-VAE/data/coco/pseudo_query.json

#coco-no-pseudo
python -u tools/prepare_dataset.py \
    --code_file RQ-VAE/output/rqvae_coco_no_pseudo/nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse/coco_codes.json \
    --dataset coco \
    --output_dir nopseudo_VT_1024-512_1-c1024_e3000_lr0.0001_mse



