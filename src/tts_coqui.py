from __future__ import annotations

import functools
import io
import os
import tempfile

import filelock
import numpy as np
import uuid
import subprocess
import time

from src.tts_sentence_parsing import init_sentence_state, get_sentence, clean_sentence, detect_language
from src.tts_utils import prepare_speech, get_no_audio, chunk_speed_change, combine_audios
from src.utils import cuda_vis_check, makedirs

import torch

n_gpus1 = torch.cuda.device_count() if torch.cuda.is_available() else 0
n_gpus1, gpu_ids = cuda_vis_check(n_gpus1)


def list_models():
    from TTS.utils.manage import ModelManager
    return ModelManager().list_tts_models()


def get_xtt(model_name="tts_models/multilingual/multi-dataset/xtts_v2", deepspeed=True, use_gpu=True, gpu_id='auto'):
    if n_gpus1 == 0:
        use_gpu = False

    # By using XTTS you agree to CPML license https://coqui.ai/cpml
    os.environ["COQUI_TOS_AGREED"] = "1"

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.generic_utils import get_user_data_dir

    # This will trigger downloading model
    print("Downloading if not downloaded Coqui XTTS V2")

    from TTS.utils.manage import ModelManager

    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS downloaded")

    print("Loading XTTS")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    # Config will have more correct languages, they may be added before we append here
    ##["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","ja"]
    supported_languages = config.languages

    model = Xtts.init_from_config(config)
    with filelock.FileLock(get_lock_file()):
        model.load_checkpoint(
            config,
            checkpoint_dir=os.path.dirname(os.path.join(model_path, "model.pth")),
            checkpoint_path=os.path.join(model_path, "model.pth"),
            vocab_path=os.path.join(model_path, "vocab.json"),
            eval=True,
            use_deepspeed=deepspeed,
        )
        if use_gpu:
            if gpu_id == 'auto':
                model.cuda()
            else:
                model.cuda(device='cuda:%d' % gpu_id)
    print("Done loading TTS")
    return model, supported_languages


def get_lock_file():
    lock_type = "coqui"
    base_path = os.path.join('locks', 'coqui_locks')
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "%s.lock" % lock_type)
    makedirs(os.path.dirname(lock_file))  # ensure made
    return lock_file


def get_latent(speaker_wav, voice_cleanup=False, model=None, gpt_cond_len=30, max_ref_length=60, sr=24000):
    if model is None:
        model, supported_languages = get_xtt()

    if voice_cleanup:
        speaker_wav = filter_wave_1(speaker_wav)
        # speaker_wav = filter_wave_2(speaker_wav)
    else:
        speaker_wav = speaker_wav

    # create as function as we can populate here with voice cleanup/filtering
    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    # latent = (gpt_cond_latent, speaker_embedding)
    with filelock.FileLock(get_lock_file()):
        latent = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=gpt_cond_len,
                                                max_ref_length=max_ref_length, load_sr=sr)
    return latent


def get_voice_streaming(prompt, language, latent, suffix="0", model=None, sr=24000, tts_speed=1.0):
    if model is None:
        model, supported_languages = get_xtt()

    gpt_cond_latent, speaker_embedding = latent

    try:
        t0 = time.time()
        chunks = model.inference_stream(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=7.0,
            temperature=0.85,
        )

        first_chunk = True
        for i, chunk in enumerate(chunks):
            if first_chunk:
                first_chunk_time = time.time() - t0
                first_chunk = False
            chunk = chunk.detach().cpu().numpy().squeeze()
            chunk = (chunk * 32767).astype(np.int16)

            chunk = chunk_speed_change(chunk, sr, tts_speed=tts_speed)

            yield chunk.tobytes()

    except RuntimeError as e:
        if "device-side assert" in str(e):
            print(f"Restarted required due to exception: %s" % str(e), flush=True)
        else:
            print("Failed to generate wave: %s" % str(e))
    except Exception as e:
        print("Failed to generate wave: %s" % str(e))


def generate_speech(response,
                    model=None,
                    language='autodetect',
                    supported_languages=None,
                    latent=None,
                    sentence_state=None,
                    return_as_byte=True,
                    return_nonbyte_as_file=False,
                    sr=24000,
                    tts_speed=1.0,
                    return_gradio=False,
                    is_final=False,
                    verbose=False,
                    debug=False):
    if model is None or supported_languages is None:
        model, supported_languages = get_xtt()
    if sentence_state is None:
        sentence_state = init_sentence_state()
    if latent is None:
        latent = get_latent("models/female.wav", model=model)

    sentence, sentence_state, _ = get_sentence(response, sentence_state=sentence_state, is_final=is_final,
                                               verbose=verbose)
    if sentence:
        t0 = time.time()
        if verbose:
            print("sentence_to_wave: %s" % sentence)

        audio = sentence_to_wave(sentence,
                                 supported_languages,
                                 tts_speed,
                                 model=model,
                                 latent=latent,
                                 return_as_byte=return_as_byte,
                                 return_nonbyte_as_file=return_nonbyte_as_file,
                                 sr=sr,
                                 language=language,
                                 return_gradio=return_gradio)
        if verbose:
            print("done sentence_to_wave: %s" % (time.time() - t0), flush=True)
    else:
        if verbose and debug:  # too much in general
            print("No audio", flush=True)
        no_audio = get_no_audio(sr=sr, return_as_byte=return_as_byte, return_nonbyte_as_file=return_nonbyte_as_file)
        if return_gradio:
            import gradio as gr
            audio = gr.Audio(value=no_audio, autoplay=False)
        else:
            audio = no_audio
    return audio, sentence, sentence_state


def sentence_to_wave(sentence, supported_languages, tts_speed,
                     latent=None,
                     return_as_byte=False,
                     return_nonbyte_as_file=False,
                     sr=24000, model=None,
                     return_gradio=True, language='autodetect', verbose=False):
    """
    generate speech audio file per sentence
    """
    import noisereduce as nr
    import wave

    sentence = clean_sentence(sentence, verbose=verbose)
    sentence_list = [sentence]

    try:
        wav_bytestream = b""
        for sentence in sentence_list:
            # have to lock entire sentence, model doesn't handle threads,
            # this is ok since usually have many sentences
            with filelock.FileLock(get_lock_file()):

                if any(c.isalnum() for c in sentence):
                    if language == "autodetect":
                        # on first call autodetect, next sentence calls will use same language
                        language = detect_language(sentence, supported_languages, verbose=verbose)

                    # exists at least 1 alphanumeric (utf-8)
                    audio_stream = get_voice_streaming(
                        sentence, language, latent,
                        model=model,
                        tts_speed=tts_speed,
                    )
                else:
                    # likely got a ' or " or some other text without alphanumeric in it
                    audio_stream = None

                if audio_stream is not None:
                    frame_length = 0
                    for chunk in audio_stream:
                        try:
                            wav_bytestream += chunk
                            frame_length += len(chunk)
                        except Exception as e:
                            print("Exception in chunk appending: %s" % str(e), flush=True)
                            continue

            # Filter output for better voice
            filter_output = False
            if filter_output:
                data_s16 = np.frombuffer(wav_bytestream, dtype=np.int16, count=len(wav_bytestream) // 2, offset=0)
                float_data = data_s16 * 0.5 ** 15
                reduced_noise = nr.reduce_noise(y=float_data, sr=sr, prop_decrease=0.8, n_fft=1024)
                wav_bytestream = (reduced_noise * 32767).astype(np.int16)
                if return_as_byte:
                    wav_bytestream = wav_bytestream.tobytes()

            if audio_stream is not None:
                if not return_as_byte:
                    if return_nonbyte_as_file:
                        tmpdir = os.getenv('TMPDDIR', tempfile.mkdtemp())
                        audio_unique_filename = os.path.join(tmpdir, str(uuid.uuid4()) + ".wav")
                        with wave.open(audio_unique_filename, "w") as f:
                            f.setnchannels(1)
                            # 2 bytes per sample.
                            f.setsampwidth(2)
                            f.setframerate(sr)
                            f.writeframes(wav_bytestream)

                        ret_value = audio_unique_filename
                    else:
                        data_s16 = np.frombuffer(wav_bytestream, dtype=np.int16, count=len(wav_bytestream) // 2,
                                                 offset=0)
                        float_data = data_s16 * 0.5 ** 15
                        reduced_noise = nr.reduce_noise(y=float_data, sr=sr, prop_decrease=0.8, n_fft=1024)
                        wav_np = (reduced_noise * 32767).astype(np.int16)
                        ret_value = wav_np
                else:
                    ret_value = wav_bytestream
                if return_gradio:
                    import gradio as gr
                    return gr.Audio(value=ret_value, autoplay=True)
                else:
                    return ret_value
    except RuntimeError as e:
        if "device-side assert" in str(e):
            print(f"Restarted required due to exception: %s" % str(e), flush=True)
        else:
            print("Failed to generate wave: %s" % str(e))
            raise


def get_role_to_wave_map():
    # only for test and initializing state
    roles_map = {}
    roles_map["Female AI Assistant"] = "models/female.wav"
    roles_map["Male AI Assistant"] = "models/male.wav"
    roles_map["AI Beard The Pirate"] = "models/pirate_by_coqui.wav"
    roles_map["None"] = ""
    return roles_map


def allowed_roles():
    return list(get_role_to_wave_map().keys())


def get_roles(choices=None, value=None):
    if choices is None:
        choices = allowed_roles()
    if value is None:
        value = choices[0]
    import gradio as gr
    chatbot_role = gr.Dropdown(
        label="Speech Style",
        choices=choices,
        value=value,
    )
    return chatbot_role


def predict_from_text(response, chatbot_role, language, roles_map, tts_speed,
                      model=None,
                      supported_languages=None,
                      return_as_byte=True, sr=24000,
                      return_prefix_every_yield=False,
                      include_audio0=True,
                      return_dict=False,
                      verbose=False):
    if chatbot_role == "None":
        return
    audio0 = prepare_speech(sr=sr)
    if not return_prefix_every_yield and include_audio0:
        if not return_dict:
            yield audio0
        else:
            yield dict(audio=audio0, sr=sr)
    latent = get_latent(roles_map[chatbot_role], model=model)
    sentence_state = init_sentence_state()
    generate_speech_func = functools.partial(generate_speech,
                                             model=model,
                                             language=language,
                                             supported_languages=supported_languages,
                                             latent=latent,
                                             sentence_state=sentence_state,
                                             return_as_byte=return_as_byte,
                                             sr=sr,
                                             tts_speed=tts_speed,
                                             verbose=verbose)
    while True:
        audio1, sentence, sentence_state = generate_speech_func(response, is_final=False)
        if sentence is not None:
            if return_prefix_every_yield and include_audio0:
                audio_out = combine_audios([audio0], audio=audio1, channels=1, sample_width=2, sr=sr, expect_bytes=return_as_byte)
            else:
                audio_out = audio1
            if not return_dict:
                yield audio_out
            else:
                yield dict(audio=audio_out, sr=sr)
        else:
            break

    audio1, sentence, sentence_state = generate_speech_func(response, is_final=True)
    if return_prefix_every_yield and include_audio0:
        audio_out = combine_audios([audio0], audio=audio1, channels=1, sample_width=2, sr=sr, expect_bytes=return_as_byte)
    else:
        audio_out = audio1
    if not return_dict:
        yield audio_out
    else:
        yield dict(audio=audio_out, sr=sr)


def filter_wave_1(speaker_wav):
    try:
        cleanup_filter = "lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
        resample_filter = "-ac 1 -ar 22050"
        out_filename = speaker_wav + str(uuid.uuid4()) + ".wav"  # ffmpeg to know output format
        # we will use newer ffmpeg as that has afftn denoise filter
        shell_command = f"ffmpeg -y -i {speaker_wav} -af {cleanup_filter} {resample_filter} {out_filename}".split(
            " ")

        command_result = subprocess.run([item for item in shell_command], capture_output=False, text=True,
                                        check=True)
        speaker_wav = out_filename
        print("Filtered microphone input")
    except subprocess.CalledProcessError:
        # There was an error - command exited with non-zero code
        print("Error: failed filtering, use original microphone input")
        return speaker_wav


def filter_wave_2(speaker_wav):
    # Filtering for microphone input, as it has BG noise, maybe silence in beginning and end
    # This is fast filtering not perfect

    # Apply all on demand
    lowpassfilter = denoise = trim = loudness = True

    if lowpassfilter:
        lowpass_highpass = "lowpass=8000,highpass=75,"
    else:
        lowpass_highpass = ""

    if trim:
        # better to remove silence in beginning and end for microphone
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
    else:
        trim_silence = ""

    try:
        out_filename = (
                speaker_wav + str(uuid.uuid4()) + ".wav"
        )  # ffmpeg to know output format

        # we will use newer ffmpeg as that has afftn denoise filter
        shell_command = f"./ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(
            " "
        )

        command_result = subprocess.run(
            [item for item in shell_command],
            capture_output=False,
            text=True,
            check=True,
        )
        speaker_wav = out_filename
        print("Filtered microphone input")
    except subprocess.CalledProcessError:
        # There was an error - command exited with non-zero code
        print("Error: failed filtering, use original microphone input")
    return speaker_wav


def get_languages_gr(visible=True, value=None):
    import gradio as gr
    choices = [
        "autodetect",
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "ja",
        "ko",
        "hu"
    ]
    if value is None:
        value = choices[0]
    language_gr = gr.Dropdown(
        label="Language",
        info="Select an output language for the synthesised speech",
        choices=choices,
        value=value,
        visible=visible,
    )
    return language_gr
