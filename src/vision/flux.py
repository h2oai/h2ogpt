import os

import filelock
from diffusers import FluxPipeline
import torch

from src.utils import makedirs
from src.vision.sdxl_turbo import get_device


def get_pipe_make_image(gpu_id):
    device = get_device(gpu_id)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to(device)

    return pipe


def get_pipe_make_image_2(gpu_id):
    device = get_device(gpu_id)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to(device)

    return pipe


def make_image(prompt, filename=None, gpu_id='auto', pipe=None,
               image_guidance_scale=3.0,
               image_size="1024x1024",
               image_quality='standard',
               image_num_inference_steps=50,
               max_sequence_length=512):
    if pipe is None:
        pipe = get_pipe_make_image(gpu_id=gpu_id)

    if image_quality == 'manual':
        # listen to guidance_scale and num_inference_steps passed in
        pass
    else:
        if image_quality == 'quick':
            image_num_inference_steps = 10
            image_size = "512x512"
        elif image_quality == 'standard':
            image_num_inference_steps = 20
        elif image_quality == 'hd':
            image_num_inference_steps = 50

    lock_type = 'image'
    base_path = os.path.join('locks', 'image_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    with filelock.FileLock(lock_file):
        image = pipe(prompt=prompt,
                     height=int(image_size.lower().split('x')[0]),
                     width=int(image_size.lower().split('x')[1]),
                     num_inference_steps=image_num_inference_steps,
                     max_sequence_length=max_sequence_length,
                     guidance_scale=image_guidance_scale).images[0]
    if filename:
        image.save(filename)
        return filename
    return image
