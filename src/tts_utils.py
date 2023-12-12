import io

import librosa
import numpy as np
import pydub

from src.utils import have_pyrubberband


# Keep non-native package imports out of global space


def get_wave_header(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    import wave
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def prepare_speech(sr=24000):
    # Must set autoplay to True first
    return get_wave_header(sample_rate=sr)


def get_no_audio(return_as_byte=True, return_nonbyte_as_file=False, sr=None):
    if return_as_byte:
        return b""
    else:
        if return_nonbyte_as_file:
            return None
        else:
            assert sr is not None
            return sr, np.array([]).astype(np.int16)


def combine_audios(audios, audio=None, channels=1, sample_width=2, sr=24000, expect_bytes=True):
    no_audio = get_no_audio(sr=sr)
    have_audio = any(x not in [no_audio, None, ''] for x in audios) or audio not in [no_audio, None, '']
    if not have_audio:
        return no_audio

    if audio or audios:
        is_bytes = expect_bytes  # force default as bytes no matter input if know should have been bytes
        if audios:
            is_bytes |= isinstance(audios[0], (bytes, bytearray))
        if audio:
            is_bytes |= isinstance(audio, (bytes, bytearray))
        assert audio is None or isinstance(audio, (bytes, bytearray))
        from pydub import AudioSegment
        combined_wav = AudioSegment.empty()
        for x in audios:
            if x is not None:
                s = io.BytesIO(x) if is_bytes else x
                combined_wav += AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        if audio is not None:
            s = io.BytesIO(audio) if is_bytes else audio
            combined_wav += AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        if is_bytes:
            combined_wav = combined_wav.export(format='raw').read()
        return combined_wav
    # audio just empty stream, but not None, else would nuke audio
    return audio


def chunk_speed_change(chunk, sr, tts_speed=1.0):
    if tts_speed == 1.0:
        return chunk

    if have_pyrubberband:
        import pyrubberband as pyrb
        chunk = pyrb.time_stretch(chunk, sr, tts_speed)
        chunk = (chunk * 32767).astype(np.int16)
        return chunk

    if tts_speed < 1.0:
        # chunk = chunk.astype(np.float32)
        # chunk = 0.5 * chunk / np.max(chunk)
        # chunk = librosa.effects.time_stretch(chunk, rate=tts_speed)
        return chunk

    # speed-up
    from pydub import AudioSegment
    from pydub.effects import speedup

    s = io.BytesIO(chunk)
    channels = 1
    sample_width = 2
    audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
    # chunk = speedup(audio, tts_speed, 150).export(format='raw').read()
    chunk = pydub_to_np(speedup(audio, tts_speed, 150))
    # audio = audio._spawn(audio.raw_data, overrides={
    #    "frame_rate": int(audio.frame_rate * tts_speed)
    # })
    # chunk = np.array(audio.get_array_of_samples())

    return chunk


def pydub_to_np(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.int16 of shape [duration_in_seconds*sample_rate, channels],
    """
    return np.array(audio.get_array_of_samples(), dtype=np.int16).reshape((-1, audio.channels))
