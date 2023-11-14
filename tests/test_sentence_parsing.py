import pytest

from tests.utils import wrap_test_forked

from src.tts_sentence_parsing import init_sentence_state, get_sentence


bot_list = [
    ('I am an AI assistant.  What do you want from me?  I am very busy.',
     ['I am an AI assistant.', 'What do you want from me?', 'I am very busy.']),
    (
    """, I am not capable of having a personal identity or physical existence. I am a computer program designed to assist and provide information to users based on their queries. My primary function is to understand natural language input and generate accurate and helpful responses. I do not have beliefs, values, or feelings, but I strive to provide the best possible service to my users. My knowledge base is constantly expanding as I learn from new data and interactions with users. However, my responses are limited by the accuracy and completeness of the information available to me.""",
    [""", I am not capable of having a personal identity or physical existence.""",
     """I am a computer program designed to assist and provide information to users based on their queries.""",
     """My primary function is to understand natural language input and generate accurate and helpful responses.""",
     """I do not have beliefs, values, or feelings, but I strive to provide the best possible service to my users.""",
     """My knowledge base is constantly expanding as I learn from new data and interactions with users.""",
     """However, my responses are limited by the accuracy and completeness of the information available to me."""]),
    (
    """. I am not a physical being, but rather a program designed to assist and provide information to users. My primary function is to answer questions accurately and efficiently based on the available data. I do not have a personal identity or beliefs, and I do not have the ability to feel emotions or make decisions independently. My responses are generated solely based on the input provided by the user and the knowledge I have been trained on.""",
    [""".I am not a physical being, but rather a program designed to assist and provide information to users.""",
     """My primary function is to answer questions accurately and efficiently based on the available data.""",
     """I do not have a personal identity or beliefs, and I do not have the ability to feel emotions or make decisions independently.""",
     """My responses are generated solely based on the input provided by the user and the knowledge I have been trained on."""])
]


@pytest.mark.parametrize("bot, sentences_expected", bot_list)
def test_get_sentence(bot, sentences_expected):

    def response_gen():
        for word1 in bot.split(' '):
            yield word1

    response = ""
    sentence_state = init_sentence_state()

    sentences = []
    for word in response_gen():
        response += word + ' '
        sentence, sentence_state = get_sentence(response,
                                                sentence_state=sentence_state,
                                                is_final=False, verbose=True)
        if sentence is not None:
            print(sentence)
            sentences.append(sentence)
    sentence, sentence_state = get_sentence(response,
                                            sentence_state=sentence_state,
                                            is_final=True, verbose=True)
    if sentence is not None:
        print(sentence)
        sentences.append(sentence)
    assert sentences == sentences_expected


def test_get_sentence2():
    bot = 'I am an AI assistant.  What do you want from me?  I am very busy.'

    def response_gen():
        yield bot

    response = ""
    sentence_state = init_sentence_state()

    sentences = []
    sentences_expected = ['I am an AI assistant.', 'What do you want from me?', 'I am very busy.']
    for word in response_gen():
        response += word + ' '
        while True:
            sentence, sentence_state = get_sentence(response,
                                                    sentence_state=sentence_state,
                                                    is_final=False, verbose=True)
            if sentence is not None:
                print(sentence)
                sentences.append(sentence)
            else:
                break
    sentence, sentence_state = get_sentence(response,
                                            sentence_state=sentence_state,
                                            is_final=True, verbose=True)
    if sentence is not None:
        print(sentence)
        sentences.append(sentence)
    assert sentences == sentences_expected
