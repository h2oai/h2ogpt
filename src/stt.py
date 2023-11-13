from src.utils import get_device


def get_transcriber(model="openai/whisper-base.en", use_gpu=True, gpu_id='auto'):
    if gpu_id == 'auto':
        gpu_id = 0
    device = get_device()
    if device == 'cpu' or not use_gpu:
        device_map = {"", 'cpu'}
    else:
        device_map = {"": gpu_id} if gpu_id >= 0 else {'': 'cuda'}

    from transformers import pipeline
    transcriber = pipeline("automatic-speech-recognition", model=model, device_map=device_map)
    return transcriber


def transcribe(text0, stream, new_chunk, transcriber=None, debug=False):
    import numpy as np
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y) + 1E-7)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    text = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    if debug:
        print("stream.shape: %s %s" % (str(stream.shape), text))
    return stream, text0 + text
