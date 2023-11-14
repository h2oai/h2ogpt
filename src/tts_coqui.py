from __future__ import annotations

import functools

import numpy as np
import uuid
import io
import time

from src.tts_sentence_parsing import init_sentence_state, get_sentence, clean_sentence, detect_language


def get_xxt():
    import os

    # By using XTTS you agree to CPML license https://coqui.ai/cpml
    os.environ["COQUI_TOS_AGREED"] = "1"

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.generic_utils import get_user_data_dir

    # This will trigger downloading model
    print("Downloading if not downloaded Coqui XTTS V2")

    from TTS.utils.manage import ModelManager
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
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
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"),
        eval=True,
        use_deepspeed=True,
    )
    model.cuda()
    print("Done loading TTS")
    return model, supported_languages


def get_latent(speaker_wav, voice_cleanup=False, model=None):
    if model is None:
        model, supported_languages = get_xxt()

    import subprocess

    if voice_cleanup:
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
    else:
        speaker_wav = speaker_wav

    # create as function as we can populate here with voice cleanup/filtering
    (
        gpt_cond_latent,
        speaker_embedding,
    ) = model.get_conditioning_latents(audio_path=speaker_wav)
    return gpt_cond_latent, speaker_embedding


def get_wave_header(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    import wave
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def get_voice_streaming(prompt, language, latent_tuple, suffix="0", model=None):
    if model is None:
        model, supported_languages = get_xxt()

    gpt_cond_latent, speaker_embedding = latent_tuple

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
                metrics_text = f"Latency to first audio chunk: {round(first_chunk_time * 1000)} milliseconds\n"
                first_chunk = False
            # print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

            # In case output is required to be multiple voice files
            # out_file = f'{char}_{i}.wav'
            # write(out_file, 24000, chunk.detach().cpu().numpy().squeeze())
            # audio = AudioSegment.from_file(out_file)
            # audio.export(out_file, format='wav')
            # return out_file
            # directly return chunk as bytes for streaming
            chunk = chunk.detach().cpu().numpy().squeeze()
            chunk = (chunk * 32767).astype(np.int16)

            yield chunk.tobytes()

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(
                f"Exit due to: Unrecoverable exception caused by prompt:{prompt}",
                flush=True,
            )
            print("Cuda device-assert Runtime encountered need restart")
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            # Does not require warning happens on empty chunk and at end
            return None
        return None
    except:
        return None


def prepare_speech():
    # Must set autoplay to True first
    return get_wave_header()


def generate_speech(response, chatbot_role=None, model=None, supported_languages=None, latent_map=None,
                    sentence_state=None,
                    return_as_byte=True, return_gradio=False,
                    is_final=False, verbose=False):
    if model is None or supported_languages is None:
        model, supported_languages = get_xxt()
    if latent_map is None:
        latent_map = get_latent_map(model=model)
    if chatbot_role is None:
        chatbot_role = allowed_roles()[0]
    if sentence_state is None:
        sentence_state = init_sentence_state()

    sentence, sentence_state = get_sentence(response, sentence_state=sentence_state, is_final=is_final, verbose=verbose)
    if sentence:
        if verbose:
            print("BG: inserting sentence to queue")

        audio = sentence_to_wave(chatbot_role, sentence,
                                 supported_languages,
                                 model=model,
                                 latent_map=latent_map,
                                 return_as_byte=return_as_byte,
                                 return_gradio=return_gradio)
    else:
        no_audio = b"" if return_as_byte else None
        if return_gradio:
            import gradio as gr
            audio = gr.Audio(value=no_audio, autoplay=False)
        else:
            audio = no_audio
    return audio, sentence, sentence_state


def sentence_to_wave(chatbot_role, sentence, supported_languages, latent_map={},
                     return_as_byte=False, model=None,
                     return_gradio=True, language='autodetect', verbose=False):
    """
    generate speech audio file per sentence
    """
    import noisereduce as nr
    import wave

    sentence_list = clean_sentence(sentence, verbose=verbose)

    try:
        wav_bytestream = b""
        for sentence in sentence_list:

            if any(c.isalnum() for c in sentence):
                if language == "autodetect":
                    # on first call autodetect, nexts sentence calls will use same language
                    language = detect_language(sentence, supported_languages, verbose=verbose)

                # exists at least 1 alphanumeric (utf-8)
                audio_stream = get_voice_streaming(
                    sentence, language, latent_map[chatbot_role],
                    model=model,
                )
            else:
                # likely got a ' or " or some other text without alphanumeric in it
                audio_stream = None

            # XTTS is actually using streaming response but we are playing audio by sentence
            # If you want direct XTTS voice streaming (send each chunk to voice ) you may set DIRECT_STREAM=1 environment variable
            if audio_stream is not None:
                frame_length = 0
                for chunk in audio_stream:
                    try:
                        wav_bytestream += chunk
                        frame_length += len(chunk)
                    except:
                        # hack to continue on playing. sometimes last chunk is empty , will be fixed on next TTS
                        continue

            # Filter output for better voice
            filter_output = False
            if filter_output:
                data_s16 = np.frombuffer(wav_bytestream, dtype=np.int16, count=len(wav_bytestream) // 2, offset=0)
                float_data = data_s16 * 0.5 ** 15
                reduced_noise = nr.reduce_noise(y=float_data, sr=24000, prop_decrease=0.8, n_fft=1024)
                wav_bytestream = (reduced_noise * 32767).astype(np.int16)
                wav_bytestream = wav_bytestream.tobytes()

            if audio_stream is not None:
                if not return_as_byte:
                    audio_unique_filename = "/tmp/" + str(uuid.uuid4()) + ".wav"
                    with wave.open(audio_unique_filename, "w") as f:
                        f.setnchannels(1)
                        # 2 bytes per sample.
                        f.setsampwidth(2)
                        f.setframerate(24000)
                        f.writeframes(wav_bytestream)

                    ret_value = audio_unique_filename
                else:
                    ret_value = wav_bytestream
                if return_gradio:
                    import gradio as gr
                    return gr.Audio(value=ret_value, autoplay=True)
                else:
                    return ret_value
    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(
                f"Exit due to: Unrecoverable exception caused by prompt:{sentence}",
                flush=True,
            )
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            raise e

    print("All speech ended")
    return


def get_latent_map(model=None):
    latent_map = {}
    latent_map["Female AI Assistant"] = get_latent("models/female.wav", model=model)
    latent_map["Male AI Assistant"] = get_latent("models/male.wav", model=model)
    latent_map["AI Beard The Pirate"] = get_latent("models/pirate_by_coqui.wav", model=model)
    return latent_map


def allowed_roles():
    roles = ["Female AI Assistant", "Male AI Assistant", "AI Beard The Pirate"]
    return roles


def get_roles():
    import gradio as gr
    chatbot_role = gr.Dropdown(
        label="Speech Style",
        choices=allowed_roles(),
        value=allowed_roles()[0],
    )
    return chatbot_role


def predict_from_text(response, chatbot_role, model=None, supported_languages=None, latent_map=None, verbose=False):
    audio0 = prepare_speech()
    yield audio0
    sentence_state = init_sentence_state()
    generate_speech_func = functools.partial(generate_speech,
                                             chatbot_role=chatbot_role,
                                             model=model,
                                             supported_languages=supported_languages,
                                             latent_map=latent_map,
                                             sentence_state=sentence_state,
                                             verbose=verbose)
    while True:
        audio1, sentence, sentence_state = generate_speech_func(response, is_final=False)
        if sentence is not None:
            yield audio1
        else:
            break

    audio1, sentence, sentence_state = generate_speech_func(response, is_final=True)
    yield audio1


def test_sentence_to_wave():
    chatbot_role = "Female AI Assistant"
    sentence = "I am an AI assistant.  I can help you with any tasks."
    # supported_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]
    model, supported_languages = get_xxt()
    latent_map = get_latent_map(model=model)
    generated_speech = sentence_to_wave(chatbot_role, sentence,
                                        supported_languages,
                                        latent_map=latent_map,
                                        model=model,
                                        return_as_byte=False,
                                        return_gradio=False)
    print(generated_speech, flush=True)

    # confirm file is valid wave file
    import wave
    with wave.open(generated_speech, mode='rb') as f:
        pass


def test_generate_speech():
    chatbot_role = "Female AI Assistant"
    model, supported_languages = get_xxt()
    latent_map = get_latent_map(model=model)

    response = 'I am an AI assistant.  What do you want from me?  I am very busy.'
    for char in response:
        generate_speech(char, chatbot_role,
                        model=model, supported_languages=supported_languages, latent_map=latent_map)


def test_full_generate_speech():
    bot = 'I am an AI assistant.  What do you want from me?  I am very busy.'

    def response_gen():
        for word1 in bot.split(' '):
            yield word1

    chatbot_role = "Female AI Assistant"
    model, supported_languages = get_xxt()
    latent_map = get_latent_map(model=model)

    response = ""
    sentence_state = init_sentence_state()

    sentences = []
    audios = []
    sentences_expected = ['I am an AI assistant.', 'What do you want from me?', 'I am very busy.']
    for word in response_gen():
        response += word + ' '
        audio, sentence, sentence_state = \
            generate_speech(response,
                            chatbot_role=chatbot_role,
                            model=model,
                            supported_languages=supported_languages,
                            latent_map=latent_map,
                            sentence_state=sentence_state,
                            return_as_byte=False, return_gradio=False,
                            is_final=False, verbose=True)
        if sentence is not None:
            print(sentence)
            sentences.append(sentence)
        if audio is not None:
            audios.append(audio)
    audio, sentence, sentence_state = \
        generate_speech(response,
                        chatbot_role=chatbot_role,
                        model=model,
                        supported_languages=supported_languages,
                        latent_map=latent_map,
                        sentence_state=sentence_state,
                        return_as_byte=False, return_gradio=False,
                        is_final=True, verbose=True)
    if sentence is not None:
        print(sentence)
        sentences.append(sentence)
    if audio is not None:
        audios.append(audio)
    assert sentences == sentences_expected
    assert len(sentences) == len(audios)
    print(audios)
