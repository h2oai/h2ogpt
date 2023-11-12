import base64
from io import BytesIO
import numpy as np
import wavio


def get_speech_model():
    # pip install torchaudio soundfile

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import torch
    from datasets import load_dataset

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker_embeddings


processor, model, vocoder, speaker_embeddings = get_speech_model()


def audio_to_html(audio):
    audio_bytes = BytesIO()
    wavio.write(audio_bytes, audio[1].astype(np.float32), audio[0], sampwidth=4)
    audio_bytes.seek(0)

    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio_base64}" controls autoplay></audio>'

    return audio_player


def text_to_speech(text):
    inputs = processor(text=text, return_tensors="pt")

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    import soundfile as sf
    sf.write("speech.wav", speech.numpy(), samplerate=16000)
