#!/bin/bash
export MODEL_NAME="realisticVisionV51_v51VAE"
export DATASET_NAME='train_data/train_add_size.csv'

accelerate launch --mixed_precision="fp16" --multi_gpu train_sd15_multi_scale_aspect-ratio-bucketing.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --dataset=${DATASET_NAME} \
  --image_root="./train_data" \
  --dataloader_num_workers=20 \
  --use_ema \
  --resolution=768 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --num_train_epochs=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpoints_total_limit=2 \
  --output_dir="out_model_merge"
  #--logging_step=40 \
