import numpy as np
from src.utils import get_device


def get_transcriber(model="openai/whisper-base.en", use_gpu=True, gpu_id='auto'):
    if gpu_id == 'auto':
        gpu_id = 0
    device = get_device()
    if device == 'cpu' or not use_gpu:
        device_map = 'auto'  # {"", 'cpu'}
    else:
        device_map = {"": gpu_id} if gpu_id >= 0 else {'': 'cuda'}

    from transformers import pipeline
    transcriber = pipeline("automatic-speech-recognition", model=model, device_map=device_map)
    return transcriber


def transcribe(text0, chunks, new_chunk, transcriber=None, max_chunks=None, sst_floor=100.0, reject_no_new_text=True,
               debug=False):
    if max_chunks is not None and len(chunks) > max_chunks:
        # refuse to update
        return chunks, text0
    if chunks is None:
        chunks = []
    # assume sampling rate always same
    # keep chunks so don't normalize on noise periods, which would then saturate noise with non-noise
    sr, y = new_chunk
    avg = np.average(np.abs(y))
    if not np.isfinite(avg):
        avg = 0.0
    if avg > sst_floor:
        if debug or True:
            print("Got possible chunk: %s" % avg, flush=True)
        chunks_new = chunks + [y] if chunks else [y]
    else:
        chunks_new = chunks
        if debug or True:
            print("Rejected quiet chunk: %s" % avg, flush=True)
    if chunks_new:
        stream = np.concatenate(chunks_new)
        stream = stream.astype(np.float32)
        max_stream = np.max(np.abs(stream) + 1E-7)
        stream /= max_stream
        text = transcriber({"sampling_rate": sr, "raw": stream})["text"]

        if chunks:
            stream0 = np.concatenate(chunks)
            stream0 = stream0.astype(np.float32)
            max_stream0 = np.max(np.abs(stream0) + 1E-7)
            stream0 /= max_stream0
            text_y = transcriber({"sampling_rate": sr, "raw": stream0})["text"]
        else:
            text_y = None

        if debug or True:
            print("y.shape: %s stream.shape: %s text0=%s text=%s text_y=%s" % (
            str(y.shape), str(stream.shape), text0, text, text_y))
        if reject_no_new_text and (text == text_y):
            print("Rejected non-textual chunk: %s" % avg, flush=True)
            # if didn't generate text, reject the chunk.  E.g. when typing on keyboard that ends up being loud enough but is definitely not words.
            chunks_new = chunks
    else:
        text = ''

    # work-around race
    # if text0 == text:
    #     text = ''

    return chunks_new, text0 + text
