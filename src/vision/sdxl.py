import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image

from src.utils import cuda_vis_check

n_gpus1 = torch.cuda.device_count() if torch.cuda.is_available() else 0
n_gpus1, gpu_ids = cuda_vis_check(n_gpus1)

def get_device(gpu_id):
    if gpu_id == 'auto':
        device = 'cpu' if n_gpus1 == 0 else 'cuda:0'
    else:
        device = 'cpu' if n_gpus1 == 0 else 'cuda:%d' % gpu_id
    return device


def make_image(prompt, filename=None, gpu_id='auto'):
    #https://huggingface.co/stabilityai/sdxl-turbo

    device = get_device(gpu_id)

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.to("cuda")

    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    if filename:
        image.save(filename)
        return filename
    return image


def change_image(prompt, init_image=None, init_file=None, filename=None, gpu_id='auto'):
    device = get_device(gpu_id)

    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)

    if init_file:
        init_image = load_image(init_file).resize((512, 512))

    image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    if filename:
        image.save(filename)
        return filename
    else:
        return image


