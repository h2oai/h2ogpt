import os
import argparse
import tempfile


def convert_svg_to_pdf(svg_path):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    drawing = svg2rlg(svg_path)
    pdf_path = tempfile.mktemp(suffix='.pdf')
    renderPDF.drawToFile(drawing, pdf_path)
    return pdf_path


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
        pdf_path = convert_svg_to_pdf(file_path)
        return convert_pdf_to_images(pdf_path)
    elif file_extension.lower() == '.pdf':
        return convert_pdf_to_images(file_path)
    else:
        # For standard image files, just return the original file path
        return [file_path]


def main():
    parser = argparse.ArgumentParser(description="OpenAI Vision API Script")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Timeout for API calls")
    parser.add_argument("-s", "--system_prompt", type=str,
                        default="""You are a highly capable AI assistant with advanced vision capabilities.
* Analyze the provided image thoroughly and provide detailed, accurate descriptions or answers based on what you see.
* Consider various aspects such as objects, people, actions, text, colors, composition, and any other relevant details.
* If asked a specific question about the image, focus your response on addressing that question directly.
* Ensure you add a critique of the image, if anything seems wrong, or if anything requires improvement.""",
                        help="System prompt")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("-u", "--url", type=str, help="URL of the image")
    parser.add_argument("-f", "--file", type=str,
                        help="Path to the image file. Accepts standard image formats (e.g., PNG, JPEG, JPG), SVG, and PDF files.")
    parser.add_argument("-m", "--model", type=str, help="OpenAI or Open Source model to use")
    parser.add_argument("-T", "--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for the model")

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

    if args.file:
        from openai_server.openai_client import file_to_base64
        image_paths = process_file(args.file)
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

    response = client.chat.completions.create(
        messages=messages,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # could resize here instead if don't want to trust h2oGPT with doing also rotation and alignment
        extra_body=dict(rotate_align_resize_image=True),
    )

    text = response.choices[0].message.content if response.choices else ''
    if text:
        print("Vision Model Response: ", text)
    else:
        print("Vision Model Failed")


if __name__ == "__main__":
    main()
