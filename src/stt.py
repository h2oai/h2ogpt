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


def transcribe(audio_state1, new_chunk, transcriber=None, max_chunks=None, sst_floor=100.0, reject_no_new_text=True,
               debug=False):
    if audio_state1[0] is None:
        audio_state1[0] = ''
    if audio_state1[2] is None:
        audio_state1[2] = []
    if max_chunks is not None and audio_state1[2] is not None and len(audio_state1[2]) > max_chunks:
        # refuse to update
        return audio_state1, audio_state1[1]
    if audio_state1[3] == 'off':
        if debug:
            print("Already ended", flush=True)
        return audio_state1, audio_state1[1]
    # assume sampling rate always same
    # keep chunks so don't normalize on noise periods, which would then saturate noise with non-noise
    sr, y = new_chunk
    if y.shape[0] == 0:
        avg = 0.0
    else:
        # stereo to mono if needed
        if len(y.shape) > 1:
            y = np.mean(y, dim=0)
        avg = np.average(np.abs(y))
    if not np.isfinite(avg):
        avg = 0.0
    if avg > sst_floor:
        if debug:
            print("Got possible chunk: %s" % avg, flush=True)
        chunks_new = audio_state1[2] + [y]
    else:
        chunks_new = audio_state1[2]
        if debug:
            print("Rejected quiet chunk: %s" % avg, flush=True)
    if chunks_new:
        stream = np.concatenate(chunks_new)
        stream = stream.astype(np.float32)
        max_stream = np.max(np.abs(stream) + 1E-7)
        stream /= max_stream
        text = transcriber({"sampling_rate": sr, "raw": stream})["text"]

        if audio_state1[2]:
            try:
                stream0 = np.concatenate(audio_state1[2])
            except:
                raise
            stream0 = stream0.astype(np.float32)
            max_stream0 = np.max(np.abs(stream0) + 1E-7)
            stream0 /= max_stream0
            text_y = transcriber({"sampling_rate": sr, "raw": stream0})["text"]
        else:
            text_y = None

        if debug:
            print("y.shape: %s stream.shape: %s text0=%s text=%s text_y=%s" % (
                str(y.shape), str(stream.shape), audio_state1[0], text, text_y))
        if reject_no_new_text and (text == text_y):
            if debug:
                print("Rejected non-textual chunk: %s" % avg, flush=True)
            # if didn't generate text, reject the chunk.  E.g. when typing on keyboard that ends up being loud enough but is definitely not words.
        else:
            audio_state1[2] = chunks_new
    else:
        text = ''
        # print("H9: %s %s" % (audio_state1[0], text), flush=True)

    # work-around race
    if audio_state1[0] == text:
        # print("H10: %s %s" % (audio_state1[0], text), flush=True)
        text = ''

    if audio_state1[0] is not None:
        # For race, when action hits done while streaming occurs, to know now to use updated result
        audio_state1[1] = audio_state1[0] + text
    return audio_state1, audio_state1[1]
