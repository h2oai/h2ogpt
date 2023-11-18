import textwrap
import re

from src.utils import flatten_list, have_emoji, have_langid


def setup_nltk():
    import nltk  # we'll use this to split into sentences
    nltk.download("punkt")


# if followed installation, then should already be done, don't break air-gap
# setup_nltk()

sentence_keys = ['sentence_list', 'index']


def init_sentence_state():
    sentence_state = dict(sentence_list=[], index=0)
    return sentence_state


def unpack_state(sentence_state):
    rets = []
    for key in sentence_keys:
        rets.append(sentence_state[key])
    return tuple(rets)


def pack_state(sentence_state, *args):
    # don't change dict reference so parent can reuse.  Ok to lose reference for list
    for keyi, key in enumerate(sentence_keys):
        if isinstance(sentence_state[key], list):
            sentence_state[key] = args[keyi]
        else:
            sentence_state[key] = args[keyi]
    return sentence_state


def split_sentences(sentence, n=250):
    """
    Splits a sentence by spaces into smaller sentences, each with a maximum length of n characters,
    while preserving whitespace characters like new lines.
    # 250 due to [!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio.
    """
    # Splitting on spaces while preserving all whitespace characters in a list
    words = re.split('(\s+)', sentence)
    sentences = []
    current_sentence = []
    current_length = 0

    for word in words:
        # Skip empty strings which can occur due to consecutive whitespace
        if word == '':
            continue

        # Check if the word is a whitespace character
        if word.isspace():
            if word == '\n':
                # If it's a newline, end the current sentence and start a new one
                sentences.append("".join(current_sentence))
                current_sentence = []
                current_length = 0
            else:
                # For other whitespace characters, add them to the current sentence
                current_sentence.append(word)
                current_length += len(word)
        else:
            # Check if adding the next word would exceed the limit
            if current_length + len(word) > n:
                if current_sentence:
                    sentences.append("".join(current_sentence))
                    current_sentence = [word]
                    current_length = len(word)
                else:
                    # If the word itself is longer than n and there's no current sentence
                    sentences.append(word)
                    current_length = 0
            else:
                current_sentence.append(word)
                current_length += len(word)

    # Add the last sentence if it exists
    if current_sentence:
        sentences.append("".join(current_sentence))

    return sentences


def _get_sentences(response, verbose=False, min_start=15, max_length=250):
    # no mutations of characters allowed here, only breaking apart or merging
    import nltk
    # refuse to tokenize first 15 characters into sentence, so language detection works and logic simpler
    sentences = nltk.sent_tokenize(response[min_start:])
    # split any long sentences
    sentences = flatten_list([split_sentences(x, max_length) for x in sentences])
    # drop empty sentences
    sentences = [x for x in sentences if x.strip()]
    # restore first min_start if set
    if sentences and min_start > 0:
        sentences[0] = response[:min_start] + sentences[0]
    elif min_start > 0:
        sentences.append(response[:min_start])

    return sentences


def get_sentence(response, sentence_state, is_final=False, verbose=False):
    # get state items
    sentence_list, index = unpack_state(sentence_state)
    sentences = _get_sentences(response[index:], min_start=15 if index == 0 else 0, verbose=verbose)

    if len(sentences) >= 2:
        # detected new completed sentence
        # find new index
        index_delta = response[index:].index(sentences[0])
        index += index_delta + len(sentences[0])
        sentence_list.append(sentences[0])
        # only clean for result, to avoid mis-handling of sentences index
        cleaned_sentence = clean_sentence(sentences[0], verbose=verbose)
        return cleaned_sentence, pack_state(sentence_state, sentence_list, index), False
    elif is_final:
        # then just return last sentence
        cleaned_sentence = clean_sentence(' '.join(sentences), verbose=verbose)
        sentence_list.append(' '.join(sentences))
        return cleaned_sentence, pack_state(sentence_state, sentence_list, index), True
    else:
        return None, pack_state(sentence_state, sentence_list, index), True


def clean_sentence(sentence, verbose=False):
    if sentence is None or len(sentence) == 0:
        if verbose:
            print("empty sentence")
        return ''

    # Remove code blocks
    sentence = re.sub("```.*?```", "", sentence, flags=re.DOTALL)
    sentence = re.sub("`.*?`", "", sentence, flags=re.DOTALL)
    sentence = re.sub("\(.*?\)", "", sentence, flags=re.DOTALL)

    # remove marks
    sentence = sentence.replace("```", "")
    sentence = sentence.replace("...", " ")
    sentence = sentence.replace("(", " ")
    sentence = sentence.replace(")", " ")

    sentence = sentence.replace("Dr. ", "Doctor ")
    sentence = sentence.replace(" w/ ", " with ")

    sentence = sentence.replace('H2O.ai', "aych two oh ae eye.")
    sentence = sentence.replace('H2O.AI', "aych two oh ae eye.")
    sentence = sentence.replace('h2o.ai', "aych two oh ae eye.")
    sentence = sentence.replace('h2o.ai', "aych two oh ae eye.")

    # filter out emojis
    if have_emoji:
        import emoji
        sentence = ''.join([x for x in sentence if not emoji.is_emoji(x)])

    # fix floating expressions
    sentence = re.sub(r'(\d+)\.(\d+)', r"\1 dot \2", sentence)

    # Fix last bad characters
    sentence = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?|\!)", r"\1\2", sentence)

    sentence = sentence.strip()

    if sentence.startswith('. ') or sentence.startswith('? ') or sentence.startswith('! ') or sentence.startswith(', '):
        sentence = sentence[2:]
    if sentence.startswith('.') or sentence.startswith('?') or sentence.startswith('!') or sentence.startswith(','):
        sentence = sentence[1:]

    if sentence == '1.':
        sentence = 'One'
    if sentence == '2.':
        sentence = 'Two'
    if sentence == '3.':
        sentence = 'Three'
    if sentence == '4.':
        sentence = 'Four'
    if sentence == '5.':
        sentence = 'Five'
    if sentence == '6.':
        sentence = 'Six'
    if sentence == '7.':
        sentence = 'Seven'
    if sentence == '8.':
        sentence = 'Eight'
    if sentence == '9.':
        sentence = 'Nine'
    if sentence == '10.':
        sentence = 'Ten'

    if len(sentence) == 0:
        if verbose:
            print("EMPTY SENTENCE after processing")
        return ''

    if verbose:
        print("Sentence for speech: %s" % sentence)

    return sentence


def detect_language(prompt, supported_languages, verbose=False):
    if not have_langid:
        # if no package, just return english
        return "en"

    import langid
    # Fast language autodetection
    if len(prompt) > 15:
        language_predicted = langid.classify(prompt)[0].strip()  # strip need as there is space at end!
        if language_predicted == "zh":
            # we use zh-cn on xtts
            language_predicted = "zh-cn"

        if language_predicted not in supported_languages:
            print(f"Detected a language not supported by xtts :{language_predicted}, switching to english for now")
            language = "en"
        else:
            language = language_predicted
        if verbose:
            print(f"Language: Predicted sentence language:{language_predicted} , using language for xtts:{language}")
    else:
        # Hard to detect language fast in short sentence, use english default
        language = "en"
        if verbose:
            print(f"Language: Prompt is short or autodetect language disabled using english for xtts")

    return language
