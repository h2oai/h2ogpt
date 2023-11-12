import numpy as np
from transformers import pipeline

p = pipeline("automatic-speech-recognition")

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y) + 1E-7)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]
