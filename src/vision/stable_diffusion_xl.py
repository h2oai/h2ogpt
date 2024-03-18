import os

import filelock
from diffusers import DiffusionPipeline
import torch

from src.utils import makedirs
from src.vision.sdxl import get_device


def get_pipe_make_image(gpu_id, refine=True):
    device = get_device(gpu_id)

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    ).to(device)
    if not refine:
        refiner = None
    else:

        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

    return base, refiner


def make_image(prompt, filename=None, gpu_id='auto', pipe=None, guidance_scale=3.0):
    if pipe is None:
        base, refiner = get_pipe_make_image(gpu_id=gpu_id)
    else:
        base, refiner = pipe

    lock_type = 'image'
    base_path = os.path.join('locks', 'image_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    with filelock.FileLock(lock_file):
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

    if filename:
        image.save(filename)
        return filename
    return image
