import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO
import os
import base64
import argparse

class TextToImageModel:
    def __init__(self) -> None:
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        self.pipe.enable_sequential_cpu_offload()

    def generate(self, text: str, width: int, height: int, num_inference_steps: int) -> str:
        out = self.pipe(
            text, 
            guidance_scale = 0,
            height = height,
            width = width,
            num_inference_steps = num_inference_steps,
        ).images[0]

        buffered = BytesIO()
        out.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str

def save_image_to_tempfile(image: str, temp_dir: str, filename:str) -> str:
    # Full path to save the image
    save_path = os.path.join(temp_dir, filename)

    # Ensure the directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Save
    with open(save_path, "wb") as img_file:
        img_file.write(base64.b64decode(image))

    return save_path

def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    # Prompt
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    # Image width
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    # Image height
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    # Num inference steps
    parser.add_argument("--num_inference_steps", type=int, default=3, help="Number of inference steps")
    # File name
    parser.add_argument("--file", type=str, default="output.jpg", help="Name of the output file")
    args = parser.parse_args()

    try:
        # TODO: Store the model in h2ogpt endpoint? New image_generation parameter for generate.py?
        model = TextToImageModel()
        image = model.generate(
            text=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps
        )
        # TODO: Parameterize the temp directory
        temp_dir = "./openai_files/"
        image_path = save_image_to_tempfile(image, temp_dir, args.file)
        print(f"Image generated successfully. Saved to {image_path}")
    except Exception as e:
        print(f"Error generating the image: {e}")  

if __name__ == "__main__":
    main()
