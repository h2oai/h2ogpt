import pytest

from tests.utils import wrap_test_forked

from src.tts_sentence_parsing import init_sentence_state, get_sentence

bot_list = [
    (
    """- NVIDIA's Speech-to-Text (STT) models perform best under low noise conditions but are outperformed by Whisper under high noise levels (SNR < 10 dB).""",
    [
        "- NVIDIA's Speech-to-Text  models perform best under low noise conditions but are outperformed by Whisper under high noise levels ."]),
    ("""Coastal City (Tue) - TC, GF (Wed) - the week’s still young! w/ Elizabeth (AeroTech, Oceanic); LUNA team and champions; Marina Financial Group CIO; GBA corporate bank, Alex Mercer (Jordan dialed in); GEC opening gala (where leaders of Nation A & Nation B meet along with a host of delegates from GEC Gov agencies and select CEOs in Country X.) Coastal City Energy and Water Agency (a pivotal agency for the area; and a mature organization in autoML; fan of LLM Studio and h2oGPTe) - notes below; our meeting excited them to accelerate a transformative partnership - energizing the agency!

"Making Coastal City a leader in AI technology" - very productive meetings with CIO & data & AI teams of the Energy and Water Agency, CIO of Marina Financial in Capital City, Board member of AeroTech, partner XYZ Corp.. The agency will be gateway to all the agencies of the country and a transformative partnership for h2o as well. (Closing imminently.) and Sam & XYZ Corp partnership in the region. The region will be a great area for AI and the people of this region are aspiring for change & seek true partnership and co-creation! They are ready to be makers and joining our movement!""",
     ['Coastal City  - TC, GF  - the week’s still young!',
      'with Elizabeth ; LUNA team and champions; Marina Financial Group CIO; GBA corporate bank, Alex Mercer ; GEC opening gala  where leaders of Nation A & Nation B meet along with a host of delegates from GEC Gov',
      'agencies and select CEOs in Country X.',
      'Coastal City Energy and Water Agency  - notes below; our meeting excited them to accelerate a transformative partnership - energizing the agency!',
      '"Making Coastal City a leader in AI technology" - very productive meetings with CIO & data & AI teams of the Energy and Water Agency, CIO of Marina Financial in Capital City, Board member of AeroTech, partner XYZ Corp..',
      'The agency will be gateway to all the agencies of the country and a transformative partnership for h2o as well.',
      'and Sam & XYZ Corp partnership in the region.',
      'The region will be a great area for AI and the people of this region are aspiring for change & seek true partnership and co-creation!',
      'They are ready to be makers and joining our movement!']),
    ("""Sure, I'd be happy to help! Here are some fun facts about the color purple:

1. Pur""", ["Sure, I'd be happy to help!", 'Here are some fun facts about the color purple:\n\n1.', 'Pur']),
    ('Purple', ['Purple']),
    ('I am an AI assistant.  What do you want from me?  I am very busy.',
     ['I am an AI assistant.', 'What do you want from me?', 'I am very busy.']),
    (
        """, I am not capable of having a personal identity or physical existence. I am a computer program designed to assist and provide information to users based on their queries. My primary function is to understand natural language input and generate accurate and helpful responses. I do not have beliefs, values, or feelings, but I strive to provide the best possible service to my users. My knowledge base is constantly expanding as I learn from new data and interactions with users. However, my responses are limited by the accuracy and completeness of the information available to me.""",
        ["""I am not capable of having a personal identity or physical existence.""",
         """I am a computer program designed to assist and provide information to users based on their queries.""",
         """My primary function is to understand natural language input and generate accurate and helpful responses.""",
         """I do not have beliefs, values, or feelings, but I strive to provide the best possible service to my users.""",
         """My knowledge base is constantly expanding as I learn from new data and interactions with users.""",
         """However, my responses are limited by the accuracy and completeness of the information available to me."""]),
    (
        """. I am not a physical being, but rather a program designed to assist and provide information to users. My primary function is to answer questions accurately and efficiently based on the available data. I do not have a personal identity or beliefs, and I do not have the ability to feel emotions or make decisions independently. My responses are generated solely based on the input provided by the user and the knowledge I have been trained on.""",
        ["""I am not a physical being, but rather a program designed to assist and provide information to users.""",
         """My primary function is to answer questions accurately and efficiently based on the available data.""",
         """I do not have a personal identity or beliefs, and I do not have the ability to feel emotions or make decisions independently.""",
         """My responses are generated solely based on the input provided by the user and the knowledge I have been trained on."""]),

    (""". I'm doing well, thanks for asking! How about you? Feel free to share anything that's been on your mind lately.

. If you have any specific topics or questions you'd like me to help you with, just let me know. I'm here to assist you in any way possible.

. And if you ever need a listening ear or someone to bounce ideas off of, don't hesitate to reach out. I'm always here for you!

. Let's make the most of our time together and see how we can work towards achieving your goals and aspirations.

. Looking forward to connecting with you soon!

. Best regards,

[Your Name]""", ["I'm doing well, thanks for asking!", 'How about you?',
                 "Feel free to share anything that's been on your mind lately.",
                 "If you have any specific topics or questions you'd like me to help you with, just let me know.",
                 "I'm here to assist you in any way possible.",
                 "And if you ever need a listening ear or someone to bounce ideas off of, don't hesitate to reach out.",
                 "I'm always here for you!",
                 "Let's make the most of our time together and see how we can work towards achieving your goals and aspirations.",
                 'Looking forward to connecting with you soon!', 'Best regards,\n\n[Your Name]']),
    (""". I'm doing well, thanks for asking! How about you? Feel free to share anything that's been on your mind lately.

. If you have any specific topics or questions you'd like me to address, just let me know and I'll do my best to provide helpful insights and information.

. Alternatively, if you just want to chat about something random or share some thoughts, that's great too! I'm here to listen and engage in meaningful conversations.

. Whether we're discussing current events, personal experiences, or anything else under the sun, my goal is always to foster a positive and productive dialogue.

. So, what's on your mind today? Let's dive in and explore some ideas together!""",
     ["I'm doing well, thanks for asking!", 'How about you?',
      "Feel free to share anything that's been on your mind lately.",
      "If you have any specific topics or questions you'd like me to address, just let me know and I'll do my best to provide helpful insights and information.",
      "Alternatively, if you just want to chat about something random or share some thoughts, that's great too!",
      "I'm here to listen and engage in meaningful conversations.",
      "Whether we're discussing current events, personal experiences, or anything else under the sun, my goal is always to foster a positive and productive dialogue.",
      "So, what's on your mind today?", "Let's dive in and explore some ideas together!"]),
    (
        """I do not have the ability to feel emotions or do things in the physical world. However, I am programmed to respond to your message and assist you with any queries you may have. So, I'm functioning perfectly fine and ready to help you out! how about you? is there anything I can assist you with today?""",
        ["""I do not have the ability to feel emotions or do things in the physical world.""",
         """However, I am programmed to respond to your message and assist you with any queries you may have.""",
         """So, I'm functioning perfectly fine and ready to help you out!""", """how about you?""",
         """is there anything I can assist you with today?"""])
]


@wrap_test_forked
@pytest.mark.parametrize("bot, sentences_expected", bot_list)
def test_get_sentence_stream(bot, sentences_expected):
    def response_gen():
        for word1 in bot.split(' '):
            yield word1

    response = ""
    sentence_state = init_sentence_state()

    sentences = []
    for word in response_gen():
        response += word + ' '
        sentence, sentence_state, _ = get_sentence(response,
                                                   sentence_state=sentence_state,
                                                   is_final=False, verbose=True)
        if sentence:
            print(sentence)
            sentences.append(sentence)
    sentence, sentence_state, _ = get_sentence(response,
                                               sentence_state=sentence_state,
                                               is_final=True, verbose=True)
    if sentence:
        print(sentence)
        sentences.append(sentence)
    assert sentences == sentences_expected


@wrap_test_forked
@pytest.mark.parametrize("bot, sentences_expected", bot_list)
def test_get_sentence_no_stream(bot, sentences_expected):
    def response_gen():
        yield bot

    response = ""
    sentence_state = init_sentence_state()

    sentences = []
    for word in response_gen():
        response += word + ' '
        while True:
            sentence, sentence_state, is_done = get_sentence(response,
                                                             sentence_state=sentence_state,
                                                             is_final=False, verbose=True)
            if sentence:
                print(sentence)
                sentences.append(sentence)
            else:
                if is_done:
                    break
    sentence, sentence_state, _ = get_sentence(response,
                                               sentence_state=sentence_state,
                                               is_final=True, verbose=True)
    if sentence:
        print(sentence)
        sentences.append(sentence)
    assert sentences == sentences_expected
