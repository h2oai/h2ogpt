import os
import argparse

from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Get transcription of an audio file")
    # Model
    parser.add_argument("--model", type=str, required=False, help="Model name")
    # File name
    parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file")
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
    base_path = os.getenv("H2OGPT_OPENAI_BASE_FILE_PATH", "./openai_files/")
    # Create the directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # Get full path with base_path and audio file name. Note file_path includes the full path and the audio name at the end.
    full_path = os.path.join(base_path, os.path.basename(args.file_path) + ".txt")
    # Write the transcription to a file
    with open(full_path, "w") as txt_file:
        txt_file.write(transcription.text)
    print(f"Transcription successfully saved to the path: {full_path}")
    print(f"Audio file successfully transcribed as: '{transcription.text}'")


if __name__ == "__main__":
    main()
