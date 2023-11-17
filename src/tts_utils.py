import io
import numpy as np


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
