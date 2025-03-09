import os
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DiffusionPipeline,DPMSolverMultistepScheduler
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
import argparse

from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

def main():
    # 传参
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet_path', type=str, default='unet')
    # vae
    # parser.add_argument('--vae_path', type=str, default='realisticVisionV51_v51VAE')
    # width
    parser.add_argument('--width', type=int, default=576)
    # height
    parser.add_argument('--height', type=int, default=768)
    # save_dir
    parser.add_argument('--save_dir', type=str, default='output')
    args = parser.parse_args()

    negative_prompt = 'blurry, raw photo, high saturation, multiple watermark, watermark, (over-bright:1.5)'
    generator = torch.Generator(device="cuda")

    model_path = 'realisticVisionV51_v51VAE'
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
    df = pd.read_json('../train_data/v05_all_data_for_test.jsonl',lines=True)
    # df = df.sample(100, random_state=1)
    save_dir = args.save_dir
    print(save_dir)
    # if os.path.exists(save_dir):
    #     os.system('rm -rf '+save_dir)
    os.makedirs(save_dir,exist_ok=True)
    use_size_condition = False
    use_size_condition = True
    for i, row in df.iloc[:].iterrows():
        generator = torch.Generator("cuda").manual_seed(seed)
        print(i)
        prompt = row['text']
        # if row['file_name'] not in os.listdir(save_dir):
        #     image = pipe(prompt,width=width,height=height,num_inference_steps=25,negative_prompt=negative_prompt,size_condition=(height,width)).images[0]
        #     # image.save(os.path.join(save_dir,row['file_name']+'_'+str(width)+'_'+str(height)+'.png'))
        #     image.save(os.path.join(save_dir,row['file_name']))
        # image = pipe(prompt, width=width, height=height, num_inference_steps=25, negative_prompt=negative_prompt,
        #              size_condition=(height, width)).images[0]

        print(f'prompt={prompt}')
        '''
        if use_size_condition:
            image = pipe(prompt, width=width, height=height, num_inference_steps=25, negative_prompt=negative_prompt,
                        size_condition=(height, width), generator=generator).images[0]
        else:
            image = pipe(prompt, width=width, height=height, num_inference_steps=25, negative_prompt=negative_prompt,
                        generator=generator).images[0]
        '''
        if use_size_condition:
            image = pipe(prompt, width=width, height=height, num_inference_steps=25, negative_prompt=negative_prompt,
                        use_size_condition=use_size_condition, original_size=(height, width), target_size=(height, width),
                        generator=generator).images[0]

        # image.save(os.path.join(save_dir,row['file_name']+'_'+str(width)+'_'+str(height)+'.png'))
        filename = 'with_condition_' if use_size_condition else 'without_condition_'
        filename += row['file_name']
        save_file = os.path.join(save_dir, filename)
        image.save(save_file)

        print(f'save_file={save_file}')



if __name__ == '__main__':
    main()
