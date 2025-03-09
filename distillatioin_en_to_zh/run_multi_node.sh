#!/bin/bash

set -x
set -e
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_11
export NCCL_ALGO=nvls
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_QPS_PER_CONNECTION=2
export CUDA_DEVICE_MAX_CONNECTIONS=1


is_multinode=true
if [ $is_multinode = true ]; then
    nproc_per_node=8
    nnodes=${WORLD_SIZE}
    node_rank=${RANK}
    master_addr=$(cat /etc/aistudio/master-host)
    master_port=6000
else
    nproc_per_node=8
    nnodes=1
    node_rank=0
    master_addr=localhost
    master_port=23458
fi

OUTPUT_DIR=results_v6_50epoch

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "目录不存在，正在创建..."
    mkdir -p "$OUTPUT_DIR"
else
    echo "目录已存在"
fi
DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --master_addr $master_addr --master_port $master_port"

teacher_model_name_or_path="realisticVisionV51_v51VAE"
student_model_name_or_path="realisticVisionV51_v51VAE"
train_data_file="dataset/translation2019zh/translation2019zh_train_clean.jsonl"
test_data_file="dataset/translation2019zh/translation2019zh_valid_clean.jsonl"
log_file="${OUTPUT_DIR}/training_log_node_${RANK}.txt"
torchrun ${DISTRIBUTED_ARGS} distillatioin_en_to_zh_distributed_v6.py \
    --teacher_model_name_or_path ${teacher_model_name_or_path} \
    --teacher_tokenizer_subfolder "tokenizer" \
    --teacher_text_encoder_subfolder "text_encoder" \
    --student_model_name_or_path ${student_model_name_or_path} \
    --student_tokenizer_subfolder "merged_clip_tokenizer_only_zh_word" \
    --student_text_encoder_subfolder "merged_text_encoder_only_zh_word" \
    --student_tokenizer_type "CLIPTokenizer" \
    --student_fix_clip_embedding_with_teacher True \
    --student_fix_clip_embedding_name "text_encoder.text_model.embeddings.token_embedding.weight" \
    --student_freeze_clip_backbone True \
    --student_add_decoder False \
    --student_add_decoder_layers 2 \
    --student_add_decoder_nhead 8 \
    --distill_intermediate_layers False \
    --distill_layer_index -1 \
    --train_data_file ${train_data_file} \
    --test_data_file ${test_data_file} \
    --learning_rate 1e-3 \
    --weight_decay 0.001 \
    --warmup_ratio 0.001 \
    --lr_scheduler_type "cosine" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1024 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_dir './logs' \
    --logging_steps 100 \
    --load_best_model_at_end False \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --fp16 True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard" \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --save_total_limit 1 \
    --do_train True \
    --save_safetensors False \
    --loss_type "MSELoss" \
    --loss_weight 1000.0 \
    --translate_loss_weight 1.0 \
    2>&1 | tee ${log_file}
    #evaluation_strategy="steps" \
    #eval_steps=1 \
    #save_steps=1 \
