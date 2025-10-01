# The subprocesses sometimes hang when using deepspeed, this is a trick to avoid that
# export NCCL_P2P_LEVEL=NVL

deepspeed --master_port=29999 train_retriever_t5_tir.py \
    --data_path data/coco/pseudo_coco_VT_1024-512_1-c1024_e3000_lr0.0001_mse/ \
    --output_dir output/retriever/coco_tir/pseudo/vali_test/ \
    --model_name output/retriever/coco_merge/pseudo/vali_test/merged_encoder_guidance_model_coco/ \
    --train_epoch 100 \
    --learning_rate 5e-4 \
    --train_batch_size 128 \
    --code_book_size 1024 \
    --code_book_num 1 \
    --dropout_rate 0.2 \
    --log_freq 100 \
    --source_length 128 \
    --target_length 6 \
    --gen_len 6 \
    --warmup_ratio 0.1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 100 \
    --deepseed_config config/t5_ds_config.json \
    --gradient_accumulation_steps 1 \
    --temperature 0.5 \
    --add_embedding \
    --bf16 \
    --pretrained_head_path  xxx/stage1/coco_stage1_output/pseudo/best_head_step3000.pt \
    --lambda_guidance 0.5
