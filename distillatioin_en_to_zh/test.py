from operator import neg
import torch
import numpy as np
import random
import sys
import os
from diffusers import StableDiffusionPipeline, AutoencoderKL
sys.path.append("../")
from infer_script.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

seed = 2049919398

#seed = 42
"""
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
"""
model_paths = {
    "office": "realisticVisionV51_v51VAE",
    "dist": "realisticVisionV51_v51VAE_dist"
    #"merge": "realisticVisionV51_v51VAE_merge"
}

inputs = [
    {
        "prompt": "Elegant minimalistic photography, 1990s, beautiful 55year old german curvy woman with deep, big blue eyes, long blonde hair, big tits, high cheekbones, heavy eyeliner wearing a crimson jumper, red autumn forest background, natural lighting, chaos, minimalist, close-up portrait photo, shot on portra 400.",
        "negative_prompt": "blurry, low resolution, dull colors, plain, simplistic, bright, monochromatic, static pose, minimal detail, cartoonish, ordinary clothing, neutral expression, incorrect anatomy, unrealistic textures, urban background, daytime, safe, non-surreal",
        "name": "en_valid"
    },
    {
        "prompt": "优雅的极简主义摄影，20世纪90年代，美丽的55岁德国曲线女性，深蓝色大眼睛，金色长发，大奶头，高颧骨，画着厚重的眼线笔，穿着深红色套头衫，红色的秋天森林背景，natural lighting, chaos, minimalist, close-up portrait photo, shot on portra 400.",
        #"negative_prompt": "模糊、低分辨率、暗淡的颜色、朴素、简单、明亮、单色、静态姿势、最小细节、卡通、普通服装、中性表情、不正确的解剖结构、不切实际的纹理、城市背景、白天、安全、非超现实",
        "negative_prompt": "blurry, low resolution, dull colors, plain, simplistic, bright, monochromatic, static pose, minimal detail, cartoonish, ordinary clothing, neutral expression, incorrect anatomy, unrealistic textures, urban background, daytime, safe, non-surreal",
        "name": "zh_valid"
    },
    {
        "prompt": '城堡 大海 夕阳 宫崎骏动画',
        "negative_prompt": "",
        "name": "castle_sea_sunset"
    }
]

#prompt = '城堡 大海 夕阳 宫崎骏动画'
#negative_prompt = ""
#image = pipe(prompt=prompt, width=512, height=512, num_inference_steps=30, guidance_scale=3.5, negative_prompt=negative_prompt).images[0]
#print(prompt, negative_prompt)
width = 512
height = 512
num_inference_steps = 25

def load_ori_model(model_path, use_safetensors=False):
    ori_pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
    ori_pipe.to("cuda")
    return ori_pipe

def load_lpw_model(model_path, use_safetensors=False):
    lpw_pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
    lpw_pipe.to("cuda")
    return lpw_pipe

def run_ori_model(ori_pipe, prompt, negative_prompt, width=512,
        height=512, num_inference_steps=25, save_file="sample_en_ori_dist_valid.png", **kwargs):
    generator = torch.Generator("cuda").manual_seed(seed)

    image = ori_pipe(prompt=prompt, width=width, height=height, generator=generator,
            num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
    image.save(save_file)

def run_lpw_model(lpw_pipe, prompt, negative_prompt, width=512,
        height=512, num_inference_steps=25, save_file="sample_en_lpw_dist_valid.png", **kwargs):
    max_embeddings_multiples = kwargs.get("max_embeddings_multiples", 3)
    generator = torch.Generator("cuda").manual_seed(seed)
    image = lpw_pipe.text2img(prompt=prompt, width=width, height=height, generator=generator,
            num_inference_steps=num_inference_steps, max_embeddings_multiples=max_embeddings_multiples, negative_prompt=negative_prompt).images[0]
    image.save(save_file)

mode = sys.argv[1]
model_types = list(model_paths.keys())
load_types = ["ori", "lpw"]
if mode == "0":
    model_types = ["office"]
elif mode == "1":
    model_types = ["dist"]
elif mode == "2":
    model_types = ["merge"]

out_path = "results"
os.makedirs(out_path, exist_ok=True)

for model_type in model_types:
    for _load_type in load_types:
        pipe = eval(f"load_{_load_type}_model")(model_paths[model_type], use_safetensors=False)
        for input in inputs:
            prompt = input["prompt"]
            #print(f"Running {model_type} {input['name']} {model_type} {_load_type}")
            negative_prompt = input["negative_prompt"]
            save_file = os.path.join(out_path, f"{model_type}_{_load_type}_{input['name']}.png")
            print(f"Running {save_file}")
            eval(f"run_{_load_type}_model")(pipe, prompt=prompt, negative_prompt=negative_prompt, width=width,
                    height=height, num_inference_steps=num_inference_steps, save_file=save_file, max_embeddings_multiples=3)

