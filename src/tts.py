from __future__ import annotations
import base64
import io
import time
from io import BytesIO
import numpy as np
import scipy
import wavio
import soundfile as sf
import torch
import librosa

from src.tts_sentence_parsing import init_sentence_state, get_sentence
from src.tts_utils import prepare_speech, get_no_audio, chunk_speed_change, combine_audios

speaker_embeddings = {
    "BDL": "spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
    "CLB": "spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy",
    "KSP": "spkemb/cmu_us_ksp_arctic-wav-arctic_b0087.npy",
    "RMS": "spkemb/cmu_us_rms_arctic-wav-arctic_b0353.npy",
    "SLT": "spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy",
}


def get_speech_model():
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import torch
    from datasets import load_dataset

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")  # .to("cuda:0")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cuda:0")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cuda:0")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to("cuda:0")
    return processor, model, vocoder, speaker_embedding


def gen_t5(text, processor=None, model=None, speaker_embedding=None, vocoder=None):
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    sf.write("speech.wav", speech.cpu().numpy(), samplerate=16000)


def get_tts_model(t5_model="microsoft/speecht5_tts",
                  t5_gan_model="microsoft/speecht5_hifigan",
                  use_gpu=True,
                  gpu_id='auto'):
    if gpu_id == 'auto':
        gpu_id = 0
    if use_gpu:
        device = 'cuda:%d' % gpu_id
    else:
        device = 'cpu'
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    processor = SpeechT5Processor.from_pretrained(t5_model)
    model = SpeechT5ForTextToSpeech.from_pretrained(t5_model).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(t5_gan_model).to(model.device)

    return processor, model, vocoder


def get_speakers():
    return ["SLT (female)",
            "BDL (male)",
            "CLB (female)",
            "KSP (male)",
            "RMS (male)",
            "Surprise Me!",
            "None",
            ]


def get_speakers_gr(value=None):
    import gradio as gr
    choices = get_speakers()
    if value is None:
        value = choices[0]
    return gr.Dropdown(label="Speech Style",
                       choices=choices,
                       value=value)


def process_audio(sampling_rate, waveform):
    # convert from int16 to floating point
    waveform = waveform / 32678.0

    # convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)

    # resample to 16 kHz if necessary
    if sampling_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)

    # limit to 30 seconds
    waveform = waveform[:16000 * 30]

    # make PyTorch tensor
    waveform = torch.tensor(waveform)
    return waveform


def predict_from_audio(processor, model, speaker_embedding, vocoder, audio, mic_audio=None, sr=16000):
    # audio = tuple (sample_rate, frames) or (sample_rate, (frames, channels))
    if mic_audio is not None:
        sampling_rate, waveform = mic_audio
    elif audio is not None:
        sampling_rate, waveform = audio
    else:
        return sr, np.zeros(0).astype(np.int16)

    waveform = process_audio(sampling_rate, waveform)
    inputs = processor(audio=waveform, sampling_rate=sr, return_tensors="pt")

    speech = model.generate_speech(inputs["input_values"], speaker_embedding, vocoder=vocoder)

    speech = (speech.numpy() * 32767).astype(np.int16)
    return sr, speech


def generate_speech(response, speaker,
                    model=None, processor=None, vocoder=None,
                    speaker_embedding=None,
                    sentence_state=None,
                    sr=16000,
                    tts_speed=1.0,
                    return_as_byte=True, return_gradio=False,
                    is_final=False, verbose=False):
    if response:
        if model is None or processor is None or vocoder is None:
            processor, model, vocoder = get_tts_model()
        if sentence_state is None:
            sentence_state = init_sentence_state()

        sentence, sentence_state, _ = get_sentence(response, sentence_state=sentence_state, is_final=is_final,
                                                   verbose=verbose)
    else:
        sentence = ''
    if sentence:
        if verbose:
            print("begin _predict_from_text")
        audio = _predict_from_text(sentence, speaker, processor=processor, model=model, vocoder=vocoder,
                                   speaker_embedding=speaker_embedding, return_as_byte=return_as_byte, sr=sr,
                                   tts_speed=tts_speed)
        if verbose:
            print("end _predict_from_text")
    else:
        if verbose:
            print("no audio")
        no_audio = get_no_audio(sr=sr, return_as_byte=return_as_byte)
        if return_gradio:
            import gradio as gr
            audio = gr.Audio(value=no_audio, autoplay=False)
        else:
            audio = no_audio
    return audio, sentence, sentence_state


def predict_from_text(text, speaker, tts_speed, processor=None, model=None, vocoder=None, return_as_byte=True,
                      return_prefix_every_yield=False,
                      include_audio0=True,
                      return_dict=False,
                      sr=16000,
                      verbose=False):
    if speaker == "None":
        return
    if return_as_byte:
        audio0 = prepare_speech(sr=16000)
        if not return_prefix_every_yield and include_audio0:
            if not return_dict:
                yield audio0
            else:
                yield dict(audio=audio0, sr=sr)
    else:
        audio0 = None
    sentence_state = init_sentence_state()
    speaker_embedding = get_speaker_embedding(speaker, model.device)

    while True:
        sentence, sentence_state, is_done = get_sentence(text, sentence_state=sentence_state, is_final=False,
                                                         verbose=verbose)
        if sentence is not None:
            audio = _predict_from_text(sentence, speaker, processor=processor, model=model, vocoder=vocoder,
                                       speaker_embedding=speaker_embedding,
                                       return_as_byte=return_as_byte,
                                       tts_speed=tts_speed)
            if return_prefix_every_yield and include_audio0:
                audio_out = combine_audios([audio0], audio=audio, channels=1, sample_width=2, sr=sr,
                                           expect_bytes=return_as_byte)
            else:
                audio_out = audio
            if not return_dict:
                yield audio_out
            else:
                yield dict(audio=audio_out, sr=sr)
        else:
            if is_done:
                break

    sentence, sentence_state, _ = get_sentence(text, sentence_state=sentence_state, is_final=True, verbose=verbose)
    if sentence:
        audio = _predict_from_text(sentence, speaker, processor=processor, model=model, vocoder=vocoder,
                                   speaker_embedding=speaker_embedding,
                                   return_as_byte=return_as_byte)
        if return_prefix_every_yield and include_audio0:
            audio_out = combine_audios([audio0], audio=audio, channels=1, sample_width=2, sr=sr,
                                       expect_bytes=return_as_byte)
        else:
            audio_out = audio
        if not return_dict:
            yield audio_out
        else:
            yield dict(audio=audio_out, sr=sr)


def get_speaker_embedding(speaker, device):
    if speaker == "Surprise Me!":
        # load one of the provided speaker embeddings at random
        idx = np.random.randint(len(speaker_embeddings))
        key = list(speaker_embeddings.keys())[idx]
        speaker_embedding = np.load(speaker_embeddings[key])

        # randomly shuffle the elements
        np.random.shuffle(speaker_embedding)

        # randomly flip half the values
        x = (np.random.rand(512) >= 0.5) * 1.0
        x[x == 0] = -1.0
        speaker_embedding *= x

        # speaker_embedding = np.random.rand(512).astype(np.float32) * 0.3 - 0.15
    else:
        speaker_embedding = np.load(speaker_embeddings[speaker[:3]])

    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).to(device)
    return speaker_embedding


def _predict_from_text(text, speaker, processor=None, model=None, vocoder=None, speaker_embedding=None,
                       return_as_byte=True, sr=16000, tts_speed=1.0):
    if len(text.strip()) == 0:
        return get_no_audio(sr=sr, return_as_byte=return_as_byte)
    if speaker_embedding is None:
        speaker_embedding = get_speaker_embedding(speaker, model.device)

    inputs = processor(text=text, return_tensors="pt")

    # limit input length
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :model.config.max_text_positions].to(model.device)

    chunk = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)
    chunk = chunk.detach().cpu().numpy().squeeze()
    chunk = (chunk * 32767).astype(np.int16)
    chunk = chunk_speed_change(chunk, sr, tts_speed=tts_speed)

    if return_as_byte:
        return chunk.tobytes()
    else:
        return sr, chunk


def audio_to_html(audio):
    audio_bytes = BytesIO()
    wavio.write(audio_bytes, audio[1].astype(np.float32), audio[0], sampwidth=4)
    audio_bytes.seek(0)

    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio_base64}" controls autoplay></audio>'

    return audio_player


def text_to_speech(text, sr=16000):
    processor, model, vocoder, speaker_embedding = get_speech_model()

    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=sr)


def test_bark():
    # Too slow, 20s on GPU
    from transformers import AutoProcessor, AutoModel

    # bark_model = "suno/bark"
    bark_model = "suno/bark-small"

    # processor = AutoProcessor.from_pretrained("suno/bark-small")
    processor = AutoProcessor.from_pretrained(bark_model)
    model = AutoModel.from_pretrained(bark_model).to("cuda")

    inputs = processor(
        text=[
            "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    t0 = time.time()
    speech_values = model.generate(**inputs, do_sample=True)
    print("Duration: %s" % (time.time() - t0), flush=True)

    # sampling_rate = model.config.sample_rate
    sampling_rate = 24 * 1024
    scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
