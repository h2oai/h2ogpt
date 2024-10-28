import argparse
import os
import subprocess
import sys

try:
    from importlib.metadata import distribution, PackageNotFoundError
    assert distribution('aider-chat') is not None
    have_aider = True
except (PackageNotFoundError, AssertionError):
    have_aider = False


def install_aider():
    if not have_aider:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aider-chat>=0.59.0"])
        print("Successfully installed aider-chat.")


def main():
    # Install aider-chat if not already installed
    try:
        import aider
    except ImportError:
        print("aider-chat not found. Installing...")
        install_aider()

    # Now we can safely import from aider
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput

    default_max_time = int(os.getenv('H2OGPT_AGENT_OPENAI_TIMEOUT', "120"))

    parser = argparse.ArgumentParser(description="Aider Coding Tool")
    parser.add_argument("--model", type=str, help="Model to use for coding assistance")
    parser.add_argument("--files", nargs="+", required=False, help="Files to work on")
    parser.add_argument("--output_dir", type=str, default="aider_output", help="Directory for output files")
    parser.add_argument("--prompt", "--query", type=str, required=True, help="Prompt or query for the coding task")
    parser.add_argument("--max_time", type=int, default=default_max_time, help="Maximum time in seconds for API calls")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up OpenAI-like client
    base_url = os.getenv('H2OGPT_OPENAI_BASE_URL')
    assert base_url is not None, "H2OGPT_OPENAI_BASE_URL environment variable is not set"
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=server_api_key, timeout=args.max_time)

    # Set environment variables for Aider
    os.environ['OPENAI_API_KEY'] = server_api_key
    os.environ['OPENAI_API_BASE'] = base_url

    # Set up InputOutput with streaming enabled
    io = InputOutput(
        yes=True,
        chat_history_file=os.path.join(args.output_dir, "chat_history.txt"),
        pretty=True,
    )

    # Determine which model to use
    if args.model:
        selected_model = args.model
    elif os.getenv('H2OGPT_AGENT_OPENAI_MODEL'):
        selected_model = os.getenv('H2OGPT_AGENT_OPENAI_MODEL')
    else:
        # Only fetch the model list if we need to use the default
        model_list = client.models.list()
        selected_model = model_list.data[0].id

    print(f"Using model: {selected_model}")

    # Set up Model
    main_model = Model(selected_model)

    # Set up Coder with streaming enabled
    coder = Coder.create(
        main_model=main_model,
        fnames=args.files if args.files else [],
        io=io,
        stream=True,
        use_git=False,
        edit_format="diff"
        #edit_format="whole"  # required for weaker models
    )

    # Run the prompt
    output = coder.run(args.prompt)

    # Save the output
    output_file = os.path.join(args.output_dir, "aider_output.txt")
    with open(output_file, "w") as f:
        f.write(output)

    if args.verbose:
        print(f"Task completed. Output saved to {output_file}")


if __name__ == "__main__":
    main()
