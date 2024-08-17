import os

import filelock
from diffusers import FluxPipeline
import torch

from src.utils import makedirs
from src.vision.sdxl import get_device


def get_pipe_make_image(gpu_id):
    device = get_device(gpu_id)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to(device)

    return pipe


def make_image(prompt, filename=None, gpu_id='auto', pipe=None, guidance_scale=3.0, height=1024, width=1024,
               num_inference_steps=50, max_sequence_length=512):
    if pipe is None:
        pipe = get_pipe_make_image(gpu_id=gpu_id)

    lock_type = 'image'
    base_path = os.path.join('locks', 'image_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    with filelock.FileLock(lock_file):
        image = pipe(prompt=prompt,
                     height=height,
                     width=width,
                     num_inference_steps=num_inference_steps,
                     max_sequence_length=max_sequence_length,
                     guidance_scale=guidance_scale).images[0]
    if filename:
        image.save(filename)
        return filename
    return image
