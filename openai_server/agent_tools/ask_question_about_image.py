import os
import argparse
import tempfile
import logging
import time


# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# avoid logging that reveals urls
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def convert_svg_to_png(svg_path):
    import cairosvg
    png_path = tempfile.mktemp(suffix='.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return png_path


def convert_pdf_to_images(pdf_path):
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = tempfile.mktemp(suffix=f'_page_{i + 1}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths


def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.svg':
        png_path = convert_svg_to_png(file_path)
        return [png_path] if png_path else []
    elif file_extension.lower() == '.pdf':
        return convert_pdf_to_images(file_path)
    else:
        # For standard image files, just return the original file path
        return [file_path]


def main():
    default_max_time = int(os.getenv('H2OGPT_AGENT_OPENAI_TIMEOUT', "120"))

    parser = argparse.ArgumentParser(description="OpenAI Vision API Script")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for API calls")
    parser.add_argument("--system_prompt", type=str,
                        default="""You are a highly capable AI assistant with advanced vision capabilities.
* Analyze the provided image thoroughly and provide detailed, accurate descriptions or answers based on what you see.
* Consider various aspects such as objects, people, actions, text, colors, composition, and any other relevant details.
* If asked a specific question about the image, focus your response on addressing that question directly.
* Ensure you add a critique of the image, if anything seems wrong, or if anything requires improvement.""",
                        help="System prompt")
    parser.add_argument("--prompt", "--query", type=str, required=True, help="User prompt")
    parser.add_argument("--url", type=str, help="URL of the image")
    parser.add_argument("--file", type=str,
                        help="Path to the image file. Accepts standard image formats (e.g., PNG, JPEG, JPG), SVG, and PDF files.")
    parser.add_argument("--model", type=str, help="OpenAI or Open Source model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for the model")
    parser.add_argument("--stream_output", help="Whether to stream output", default=True, action='store_true')
    parser.add_argument("--max_time", type=float, default=default_max_time, help="Maximum time to wait for response")

    args = parser.parse_args()

    if not args.model:
        args.model = os.getenv('H2OGPT_OPENAI_VISION_MODEL')
    if not args.model:
        raise ValueError("Model name must be provided via --model or H2OGPT_OPENAI_VISION_MODEL environment variable")

    base_url = os.getenv('H2OGPT_OPENAI_BASE_URL')
    assert base_url is not None, "H2OGPT_OPENAI_BASE_URL environment variable is not set"
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=server_api_key, timeout=args.timeout)

    assert args.url or args.file, "Either --url or --file must be provided"
    assert not (args.url and args.file), "--url and --file cannot be used together"

    # if the file is a URL, use it as the URL
    from openai_server.agent_tools.common.utils import filename_is_url
    if filename_is_url(args.file):
        args.url = args.file
        args.file = None

    if args.file:
        from openai_server.openai_client import file_to_base64
        image_paths = process_file(args.file)
        if not image_paths:
            raise ValueError(f"Unsupported file type: {args.file}")
        image_contents = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': file_to_base64(image_path)[image_path],
                    'detail': 'high',
                },
            } for image_path in image_paths
        ]
    else:
        image_paths = []
        image_contents = [{
            'type': 'image_url',
            'image_url': {
                'url': args.url,
                'detail': 'high',
            },
        }]

    messages = [
        {"role": "system", "content": args.system_prompt},
        {
            'role': 'user',
            'content': [
                           {'type': 'text', 'text': args.prompt},
                       ] + image_contents,
        }
    ]

    responses = client.chat.completions.create(
        messages=messages,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=dict(rotate_align_resize_image=True),
        stream=args.stream_output,
    )

    if args.stream_output:
        text = ''
        first_delta = True
        tgen0 = time.time()
        verbose = True
        for chunk in responses:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                text += delta
                if first_delta:
                    first_delta = False
                    print("**Vision Model Response:**\n\n", flush=True)
                print(delta, flush=True, end='')
            if time.time() - tgen0 > args.max_time:
                if verbose:
                    print("Took too long for OpenAI or VLLM Chat: %s" % (time.time() - tgen0),
                          flush=True)
                break
        if not text:
            print("**Vision Model returned an empty response**", flush=True)
    else:
        text = responses.choices[0].message.content if responses.choices else ''
        if text:
            print("**Vision Model Response:**\n\n", text, flush=True)
        else:
            print("**Vision Model returned an empty response**", flush=True)

    # Cleanup temporary files
    for image_path in image_paths:
        if image_path != args.file:  # Don't delete the original file
            try:
                os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {image_path}: {str(e)}")


if __name__ == "__main__":
    main()
