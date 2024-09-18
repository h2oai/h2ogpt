import base64
import os
import argparse

from openai import OpenAI

def main():
    try:
        parser = argparse.ArgumentParser(description="Generate images from text prompts")
        # Prompt
        parser.add_argument("--prompt", type=str, required=True, help="User prompt")
        # Model
        parser.add_argument("--model", type=str, required=False, help="Model name")
        # File name
        parser.add_argument("--file_name", type=str, default="output.jpg", help="Name of the output file")
        args = parser.parse_args()
        ##
        imagegen_url = os.getenv("IMAGEGEN_OPENAI_BASE_URL", None)
        assert imagegen_url is not None, "IMAGEGEN_OPENAI_BASE_URL environment variable is not set"
        server_api_key = os.getenv('IMAGEGEN_OPENAI_API_KEY', 'EMPTY')

        if not args.model:
            args.model = os.getenv('IMAGEGEN_OPENAI_MODEL')

        client = OpenAI(base_url=imagegen_url, api_key=server_api_key)
        response = client.images.generate(
        prompt=args.prompt,
        model=args.model,
        )

        # Convert the response to image
        # Extract the base64 encoded data
        image_data_base64 = response.data[0].b64_json  # Correct way to access the base64 image

        # Decode the base64 data
        image_data = base64.b64decode(image_data_base64)

        # Save the image to a file
        base_path = os.getenv("H2OGPT_OPENAI_BASE_FILE_PATH", "./openai_files/")
        # Create the directory if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        full_path = os.path.join(base_path, args.file_name)

        # Write the image data to a file
        with open(full_path, "wb") as img_file:
            img_file.write(image_data)

        print(f"Image successfully saved to the path: {full_path}")
    except Exception as e:
        print(f"Error generating the image: {e}")

if __name__ == "__main__":
    main()
