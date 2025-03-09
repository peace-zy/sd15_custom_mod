#!/bin/bash
node_rank=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
nnodes=1
nproc_per_node=4
nproc_per_node=8
master_addr="localhost"
master_port=10
teacher_model_name_or_path="realisticVisionV51_v51VAE"
student_model_name_or_path="realisticVisionV51_v51VAE"
train_data_file="dataset/translation2019zh/translation2019zh_train_clean.jsonl"
test_data_file="dataset/translation2019zh/translation2019zh_valid_clean.jsonl"
log_file="training_log_v6.txt"
echo ${node_rank}
torchrun --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    --node_rank ${node_rank} \
    distillatioin_en_to_zh_distributed_v6.py \
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
    --output_dir './results_v6_50epoch' \
    --num_train_epochs 5000000000000 \
    --per_device_train_batch_size 256 \
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
