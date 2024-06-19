import os
import filelock

import torch

from src.utils import makedirs
from src.vision.sdxl import get_device


def get_pipe_make_image(gpu_id, refine=True,
                        base_model="stabilityai/stable-diffusion-xl-base-1.0",
                        refiner_model="stabilityai/stable-diffusion-xl-refiner-1.0",
                        high_noise_frac=0.8):
    if base_model is None:
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    if base_model == "stabilityai/stable-diffusion-xl-base-1.0" and refiner_model is None:
        refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"

    device = get_device(gpu_id)

    if 'diffusion-3' in base_model:
        from diffusers import StableDiffusion3Pipeline
        cls = StableDiffusion3Pipeline
        extra1 = dict()
        extra2 = dict()
    else:
        from diffusers import DiffusionPipeline
        cls = DiffusionPipeline
        # extra1 = dict(denoising_end=high_noise_frac, output_type="latent")
        # extra2 = dict(denoising_end=high_noise_frac)
        extra1 = dict()
        extra2 = dict()

    base = cls.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        # variant="fp16"
    ).to(device)
    if not refine or not refiner_model:
        refiner = None
    else:
        refiner = cls.from_pretrained(
            refiner_model,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # variant="fp16",
        ).to(device)

    return base, refiner, extra1, extra2


def make_image(prompt,
               filename=None,
               gpu_id='auto',
               pipe=None,
               guidance_scale=3.0,
               base_model=None,
               refiner_model=None,
               n_steps=40, high_noise_frac=0.8):
    if pipe is None:
        base, refiner, extra1, extra2 = get_pipe_make_image(gpu_id=gpu_id,
                                                            base_model=base_model,
                                                            refiner_model=refiner_model,
                                                            high_noise_frac=high_noise_frac)
    else:
        base, refiner, extra1, extra2 = pipe

    lock_type = 'image'
    base_path = os.path.join('locks', 'image_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    with filelock.FileLock(lock_file):
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            **extra1,
        ).images
        if refiner:
            image = refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                **extra2,
                image=image,
            ).images[0]

    if filename:
        if isinstance(image, list):
            image = image[-1]
        image.save(filename)
        return filename
    return image
