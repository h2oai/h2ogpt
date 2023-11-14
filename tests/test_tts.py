import pytest

from src.tts_sentence_parsing import init_sentence_state
from tests.test_sentence_parsing import bot_list


def test_sentence_to_wave():
    from src.tts_coqui import sentence_to_wave, get_xxt, get_latent_map

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
    from src.tts_coqui import generate_speech, get_xxt, get_latent_map

    chatbot_role = "Female AI Assistant"
    model, supported_languages = get_xxt()
    latent_map = get_latent_map(model=model)

    response = 'I am an AI assistant.  What do you want from me?  I am very busy.'
    for char in response:
        generate_speech(char, chatbot_role,
                        model=model, supported_languages=supported_languages, latent_map=latent_map)


def test_full_generate_speech():
    from src.tts_coqui import generate_speech, get_xxt, get_latent_map
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


@pytest.mark.parametrize("bot, sentences_expected", bot_list)
def test_predict_from_text(bot, sentences_expected):
    speeches = []
    from src.tts import get_tts_model, get_speakers
    processor, model, vocoder = get_tts_model()
    speaker = get_speakers()[0]

    from src.tts import predict_from_text
    for sr, speech in predict_from_text(bot, speaker, processor=processor, model=model, vocoder=vocoder, verbose=True):
        speeches.append(speech)
    assert len(speeches) == len(sentences_expected)
