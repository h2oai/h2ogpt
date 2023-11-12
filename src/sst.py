
def get_transcriber(model="openai/whisper-base.en"):
    from transformers import pipeline
    transcriber = pipeline("automatic-speech-recognition", model=model)
    return transcriber


def transcribe(stream, new_chunk, transcriber=None):
    import numpy as np
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y) + 1E-7)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]
