import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO
import os
import base64
import argparse
from openai import OpenAI

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

def main_old():
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

def main():
    try:
        parser = argparse.ArgumentParser(description="Generate images from text prompts")
        # Prompt
        parser.add_argument("--prompt", type=str, required=True, help="User prompt")
        # Model
        parser.add_argument("--model", type=str, default="sdxl_turbo", help="Model name")
        # File name
        parser.add_argument("--file_name", type=str, default="output.jpg", help="Name of the output file")
        args = parser.parse_args()
        ##
        base_url = os.getenv('H2OGPT_OPENAI_BASE_URL')
        assert base_url is not None, "H2OGPT_OPENAI_BASE_URL environment variable is not set"
        server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')

        # TODO: ?
        if not args.model:
            args.model = os.getenv('H2OGPT_OPENAI_IMAGEGEN_MODEL')
        client = OpenAI(base_url=base_url, api_key=server_api_key)
        response = client.images.generate(
        prompt=args.prompt,
        model=args.model,
        )

        # Convert the response to image
        # Extract the base64 encoded data
        image_data_base64 = response.data[0].b64_json  # Correct way to access the base64 image

        # Decode the base64 data
        image_data = base64.b64decode(image_data_base64)

        # TODO: Parameterize the temp directory =?
        temp_dir = "./openai_files/"
        # Create the directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        full_path = os.path.join(temp_dir, args.file_name)

        # Write the image data to a file
        with open(full_path, "wb") as img_file:
            img_file.write(image_data)

        print(f"Image successfully saved to the path: {full_path}")
    except Exception as e:
        print(f"Error generating the image: {e}")

if __name__ == "__main__":
    main()
