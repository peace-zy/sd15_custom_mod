#python infer_sd15_add_cond_lpw.py \
python infer_sd15_add_cond.py \
   --model_path '/mnt/bella/users/zhangyan/MultiModal/text2img/test/realisticVisionV51_v51VAE' \
   --unet_path '../out_model/checkpoint-11000/unet' \
   --width 768 \
   --height 768 \
   --max_embeddings_multiples 3 \
   --test_prompt_file test_prompts.txt \
   --save_dir './infer_result_checkpoint-11000'
   #--save_dir './infer_result'
   #--unet_path '/mnt/bella/users/zhangyan/MultiModal/fc/sd/out_model/checkpoint-45000/unet' \
