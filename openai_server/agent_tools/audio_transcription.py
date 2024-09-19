import os
import argparse

from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Get transcription of an audio file")
    parser.add_argument("--model", type=str, default="whisper-1", help="Model name")
    # File name
    parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()
    ##
    stt_url = os.getenv("STT_OPENAI_BASE_URL", None)
    assert stt_url is not None, "STT_OPENAI_BASE_URL environment variable is not set"
    stt_api_key = os.getenv('STT_OPENAI_API_KEY', 'EMPTY')

    # Read the audio file
    audio_file = open(args.file_path, "rb")
    client = OpenAI(base_url=stt_url, api_key=stt_api_key)
    transcription = client.audio.transcriptions.create(
    model=args.model, 
    file=audio_file
    )
    print(f"Audio file successfully transcribed: '{transcription.text}'")


if __name__ == "__main__":
    main()
