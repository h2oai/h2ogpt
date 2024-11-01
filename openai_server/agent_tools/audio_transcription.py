import os
import argparse
import uuid


def check_valid_extension(file):
    """
    OpenAI only allows certain file types
    :param file:
    :return:
    """
    valid_extensions = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']

    # Get the file extension (convert to lowercase for case-insensitive comparison)
    _, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower().lstrip('.')

    if file_extension not in valid_extensions:
        raise ValueError(
            f"Invalid file extension. Expected one of {', '.join(valid_extensions)}, but got '{file_extension}'")

    return True


def main():
    parser = argparse.ArgumentParser(description="Get transcription of an audio (or audio in video) file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio-video file")
    # Model
    parser.add_argument("--model", type=str, required=False,
                        help="Model name (For Azure deployment name must match actual model name, e.g. whisper-1)")
    # File name
    parser.add_argument("--output", "--file", type=str, default='', required=False,
                        help="Path (ensure unique) to output text file")
    args = parser.parse_args()
    ##
    if not args.model:
        args.model = os.getenv('STT_OPENAI_MODEL', 'whisper-1')

    stt_url = os.getenv("STT_OPENAI_BASE_URL", None)
    assert stt_url is not None, "STT_OPENAI_BASE_URL environment variable is not set"

    stt_api_key = os.getenv('STT_OPENAI_API_KEY')
    if stt_url == "https://api.openai.com/v1" or 'openai.azure.com' in stt_url:
        assert stt_api_key, "STT_OPENAI_API_KEY environment variable is not set and is required if using OpenAI or Azure endpoints"

        if 'openai.azure.com' in stt_url:
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/whisper-quickstart?tabs=command-line%2Cpython-new%2Cjavascript&pivots=programming-language-python
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_version="2024-02-01",
                api_key=stt_api_key,
                # like base_url, but Azure endpoint like https://PROJECT.openai.azure.com/
                azure_endpoint=stt_url,
                azure_deployment=args.model,
            )
        else:
            from openai import OpenAI
            client = OpenAI(base_url=stt_url, api_key=stt_api_key)

        check_valid_extension(args.input)
    else:
        from openai import OpenAI
        stt_api_key = os.getenv('STT_OPENAI_API_KEY', 'EMPTY')
        client = OpenAI(base_url=stt_url, api_key=stt_api_key)

    # Read the audio file
    with open(args.input, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=args.model,
            file=f,
            response_format="text",
        )
    if hasattr(transcription, 'text'):
        trans = transcription.text
    else:
        trans = transcription
    # Save the image to a file
    if not args.output:
        args.output = f"transcription_{str(uuid.uuid4())[:6]}.txt"
    # Write the transcription to a file
    with open(args.output, "wt") as f:
        f.write(trans)

    full_path = os.path.abspath(args.output)
    print(f"Transcription successfully saved to the file: {full_path}")
    # generally too much, have agent read if too long for context of LLM
    if len(trans) < 1024:
        print(f"Audio file successfully transcribed as follows:\n\n{trans}")

    print("""\n\nRemember, use ask_question_about_documents.py to ask questions about the transcription.  This is usually preferred over trying to extract information blindly using python regexp etc.""")


if __name__ == "__main__":
    main()
