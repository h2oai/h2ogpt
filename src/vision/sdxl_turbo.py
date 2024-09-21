import os

import filelock
import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image

from src.utils import cuda_vis_check, makedirs

n_gpus1 = torch.cuda.device_count() if torch.cuda.is_available() else 0
n_gpus1, gpu_ids = cuda_vis_check(n_gpus1)


def get_device(gpu_id):
    if gpu_id == 'auto':
        device = 'cpu' if n_gpus1 == 0 else 'cuda:0'
    else:
        device = 'cpu' if n_gpus1 == 0 else 'cuda:%s' % gpu_id
    return device


def get_pipe_make_image(gpu_id='auto'):
    # https://huggingface.co/stabilityai/sdxl-turbo
    device = get_device(gpu_id)

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
    return pipe


def make_image(prompt, filename=None, gpu_id='auto', pipe=None,
               image_size="1024x1024", image_quality='standard',
               image_num_inference_steps=1, image_guidance_scale=0.0):
    if pipe is None:
        pipe = get_pipe_make_image(gpu_id=gpu_id)

    if image_quality == 'manual':
        # listen to guidance_scale and num_inference_steps passed in
        pass
    else:
        if image_quality == 'quick':
            image_num_inference_steps = 1
            image_size = "512x512"
        elif image_quality == 'standard':
            image_num_inference_steps = 2
        elif image_quality == 'hd':
            image_num_inference_steps = 3

    lock_type = 'image'
    base_path = os.path.join('locks', 'image_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    with filelock.FileLock(lock_file):
        image = pipe(prompt=prompt,
                     height=int(image_size.lower().split('x')[0]),
                     width=int(image_size.lower().split('x')[1]),
                     num_inference_steps=image_num_inference_steps,  # more than 1 not really helpful
                     guidance_scale=0.0,  # disabled: https://huggingface.co/stabilityai/sdxl-turbo#diffusers
                     ).images[0]
    if filename:
        image.save(filename)
        return filename
    return image


def get_pipe_change_image(gpu_id='auto'):
    device = get_device(gpu_id)

    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
    return pipe


def change_image(prompt, init_image=None, init_file=None, filename=None, gpu_id='auto', pipe=None):
    if pipe is None:
        pipe = get_pipe_change_image(gpu_id)

    if init_file:
        init_image = load_image(init_file).resize((512, 512))

    image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    if filename:
        image.save(filename)
        return filename
    else:
        return image


