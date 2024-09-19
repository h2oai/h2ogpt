import os
import argparse
import uuid

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Get transcription of an audio file")
    # Model
    parser.add_argument("--model", type=str, required=False, help="Model name")
    # File name
    parser.add_argument("--output", type=str, required=False, help="Path (ensure unique) to the audio file")
    args = parser.parse_args()
    ##
    stt_url = os.getenv("STT_OPENAI_BASE_URL", None)
    assert stt_url is not None, "STT_OPENAI_BASE_URL environment variable is not set"
    stt_api_key = os.getenv('STT_OPENAI_API_KEY', 'EMPTY')

    if not args.model:
        stt_model = os.getenv('STT_OPENAI_MODEL')
        assert stt_model is not None, "STT_OPENAI_MODEL environment variable is not set"
        args.model = stt_model

    # Read the audio file
    audio_file = open(args.file_path, "rb")
    client = OpenAI(base_url=stt_url, api_key=stt_api_key)
    transcription = client.audio.transcriptions.create(
        model=args.model,
        file=audio_file
    )
    # Save the image to a file
    if not args.output:
        args.output = f"transcription_{uuid.uuid4()}.txt"
    # Write the transcription to a file
    with open(args.output, "wt") as txt_file:
        txt_file.write(transcription.text)

    full_path = os.path.abspath(args.output)
    print(f"Transcription successfully saved to the file: {full_path}")
    # generally too much, have agent read if too long for context of LLM
    if len(transcription.text) < 1024:
        print(f"Audio file successfully transcribed as follows:\n\n{transcription.text}")


if __name__ == "__main__":
    main()
