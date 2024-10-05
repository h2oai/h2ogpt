import argparse
import os
import subprocess
import tempfile
import datetime
import random
import string
import shlex
import uuid


def generate_unique_filename(format):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"mermaid_{timestamp}_{random_string}.{format}"


def find_chrome_path():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "puppeteer")

    try:
        cmd = f"find {shlex.quote(cache_dir)} -name chrome-headless-shell -type f | sort -V | tail -n 1"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        chrome_path = result.stdout.strip()

        if not chrome_path:
            print("Chrome headless shell not found in the expected location.")
            return None

        return chrome_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to find Chrome: {e}")
        return None


def render_mermaid(mermaid_code, output_file, format='svg'):
    # Find Chrome path
    use_headless = False
    if use_headless:
        chrome_path = find_chrome_path()
        if not chrome_path:
            raise Exception("Chrome headless shell not found. Unable to render Mermaid diagram.")
        # Set PUPPETEER_EXECUTABLE_PATH environment variable
        os.environ["PUPPETEER_EXECUTABLE_PATH"] = chrome_path
    # else let it default to chromium-browser, just still requires no sandbox
    elif os.path.isfile('/usr/bin/chromium-browser'):
        os.environ["PUPPETEER_EXECUTABLE_PATH"] = '/usr/bin/chromium-browser'

    # Create a temporary file for the Mermaid code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp:
        temp.write(mermaid_code)
        temp_path = temp.name

    config_file = f'puppeteer-config{str(uuid.uuid4())}.json'
    try:
        # Construct the mmdc command
        with open(config_file, 'wt') as f:
            f.write('{"args": ["--no-sandbox"]}')
        cmd = ['mmdc', '-i', temp_path, '-o', output_file, '-f', format, '-p', config_file]

        # Run the mmdc command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Check if there was any output (warnings, etc.)
        if result.stdout:
            print("mmdc output:", result.stdout)
        if result.stderr:
            print("mmdc warnings/errors:", result.stderr)

        print(f"Created output file in {format} format: {output_file}")

        # Always make PNG version too, hard for other tools to svg -> png
        if format != 'png':
            # Construct the mmdc command
            base_name = '.'.join(output_file.split('.')[:-1])
            output_file_png = base_name + '.png'
            # FIXME: Would be best to optimize for aspect ratio in choosing -w or -H
            cmd = ['mmdc', '-i', temp_path, '-o', output_file_png, '-f', 'png', '-w', '2048', '-p', config_file]

            # Run the mmdc command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Check if there was any output (warnings, etc.)
            if result.stdout:
                print("mmdc output:", result.stdout)
            if result.stderr:
                print("mmdc warnings/errors:", result.stderr)

            print(
                f"Created mermaid output file in PNG format: {output_file_png} that is a conversion of {output_file}. "
                "Use this for image_query to analyze what SVG looks like, "
                "because other tools do not retain fonts when making PNG."
            )

        # Return the full path of the output file
        return os.path.abspath(output_file)
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
        if os.path.isfile(config_file):
            try:
                os.remove(config_file)
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(description='Render Mermaid diagrams from a file or direct input using mmdc.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', '--input', help='Input file containing Mermaid code')
    input_group.add_argument('-c', '--code', help='Direct Mermaid code input', nargs='+')
    parser.add_argument('-o', '--output', help='Output file name (default: auto-generated unique name)')

    args = parser.parse_args()

    # If no output file is specified, create a unique name
    if args.output is None:
        format = 'svg'
        args.output = generate_unique_filename(format)
    else:
        format = args.output.split('.')[-1]
        assert format in ['svg', 'png', 'pdf'], f"Invalid output filename {args.output} with format: {format}"

    try:
        # Determine the Mermaid code source
        if args.file:
            with open(args.file, 'r') as f:
                mermaid_code = f.read()
        else:
            mermaid_code = ' '.join(args.code)

        # Render the diagram and get the full path of the output file
        output_path = render_mermaid(mermaid_code, args.output, format=format)
        print(f"Mermaid diagram rendered successfully.")
        print(f"Output file: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error rendering Mermaid diagram: {e}")
        print(f"mmdc output: {e.output}")
        print(f"mmdc error: {e.stderr}")


if __name__ == "__main__":
    main()
