import os
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DiffusionPipeline,DPMSolverMultistepScheduler
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
import argparse
import copy

from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

def main():
    # 传参
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='realisticVisionV51_v51VAE')
    parser.add_argument('--unet_path', type=str, default='unet')
    # parser.add_argument('--vae_path', type=str, default='realisticVisionV51_v51VAE')
    parser.add_argument('--width', type=int, default=576)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--test_prompt_file', type=str, default=None)
    parser.add_argument('--max_embeddings_multiples', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='output')
    args = parser.parse_args()

    negative_prompt = 'blurry, raw photo, high saturation, multiple watermark, watermark, (over-bright:1.5)'
    generator = torch.Generator(device="cuda")

    model_path = args.model_path
    unet_path = args.unet_path
    # vae_path = args.unet_path

    # 在time 里增加额外的分辨率condition
    width = args.width
    height = args.height
    try:
        unet = UNet2DConditionModel.from_pretrained(unet_path, use_safetensors=False, torch_dtype=torch.float16,low_cpu_mem_usage=False,ignore_mismatched_sizes=True).to('cuda')
    except:
        unet = UNet2DConditionModel.from_pretrained(unet_path, use_safetensors=True, torch_dtype=torch.float16,low_cpu_mem_usage=False,ignore_mismatched_sizes=True).to('cuda')
    pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
        model_path, unet=unet, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # pipe_dpo.set_progress_bar_config(disable=True)
    seed = 42
    # 测试集

    # df = df.sample(100, random_state=1)
    save_dir = args.save_dir
    print(save_dir)
    # if os.path.exists(save_dir):
    #     os.system('rm -rf '+save_dir)
    os.makedirs(save_dir,exist_ok=True)

    
    test_prompts = []
    if args.test_prompt_file is not None:
        with open(args.test_prompt_file, 'r') as f:
            for line in f:
                test_prompts.append(line.strip())
    else:
        prompt = "A study room in french style. This image showcases an elegantly designed room, possibly a study or a library. The ceiling is wooden with a light finish, and there's a large spherical chandelier hanging from it. The walls are adorned with built-in wooden shelves that hold various decorative items, books, and ornaments. A large window on the left allows natural light to flood the room. The furniture includes a wooden desk with two upholstered chairs, a potted plant, and a decorative mirror. The overall style of the room is classic and sophisticated, with a blend of traditional and contemporary elements."
        test_prompts.append(prompt)
    
    original_addition_embed_type = copy.deepcopy(unet.config.addition_embed_type)
    #pipe.set_progress_bar_config(disable=True)
    for use_size_condition in [True, False]:
        if not use_size_condition:
            pipe.unet.config.addition_embed_type = None
        else:
            pipe.unet.config.addition_embed_type = original_addition_embed_type
        for idx, prompt in enumerate(test_prompts[:20]):
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt, width=width, height=height, num_inference_steps=25, negative_prompt=negative_prompt,
                        use_size_condition=use_size_condition, original_size=(height, width), target_size=(height, width),
                        generator=generator, max_embeddings_multiples=args.max_embeddings_multiples).images[0]

            filename = 'with_condition_' if use_size_condition else 'without_condition_'
            filename += f'{idx}.png'
            save_file = os.path.join(save_dir, filename)
            image.save(save_file)
    print(f"Images saved to {save_dir}.")

if __name__ == '__main__':
    main()
