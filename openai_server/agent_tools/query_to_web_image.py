import matplotlib

matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt

plt.ioff()
import os

os.environ['TERM'] = 'dumb'
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO
import os
import argparse


def download_image(text, file, save_dir='.'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Set up the search parameters
    params = {
        "engine": "google_images",
        "q": text,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }

    # Perform the search
    search = GoogleSearch(params)
    results = search.get_dict()

    # Check if we have image results
    if "images_results" in results and len(results["images_results"]) > 0:
        # Get the first image result
        image_url = results["images_results"][0]["original"]

        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            # Open the image and convert to RGB (in case it's RGBA)
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Generate a filename based on the query
            filepath = os.path.join(save_dir, file)

            # Save the image
            img.save(filepath)
            print(f"Image downloaded and saved as {filepath}")
            return filepath
        else:
            print(f"Failed to download image for text: {text}")
            return None
    else:
        print(f"No image results found for text: {text}")
        return None


def main():
    # check with assert if os.getenv("SERPAPI_API_KEY") is defined, if not, print a message
    assert os.getenv("SERPAPI_API_KEY"), "Please set the SERPAPI_API_KEY environment variable"

    parser = argparse.ArgumentParser(description="Download one image from the web based on a search text")
    parser.add_argument("--text", "--prompt", "--query", type=str, required=True, help="The text to search for")
    parser.add_argument("--output", "--file", type=str, help="The file name to save the image to")
    args = parser.parse_args()
    download_image(text=args.text, file=args.output)


if __name__ == "__main__":
    main()
