import textwrap
import re

from src.utils import flatten_list


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


def split_sentence(sentence, max_length=350, verbose=False):
    if len(sentence) < max_length:
        # no problem continue on
        sentences = [sentence]
    else:
        # Until now nltk likely split sentences properly but we need additional
        # check for longer sentence and split at last possible position
        # Do whatever necessary, first break at hypens then spaces and then even split very long words
        sentences = textwrap.wrap(sentence, max_length)
        if verbose:
            print("SPLITTED LONG SENTENCE:", sentences)
    return sentences


def split_sentences_preserve_words(sentence, n=350):
    """
    Splits a sentence by spaces into smaller sentences, each with a maximum length of n characters.
    This function ensures that words are not broken up.
    """
    words = sentence.split()
    sentences = []
    current_sentence = ""

    for word in words:
        # Check if adding the next word would exceed the limit
        if len(current_sentence) + len(word) + 1 > n:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = word
            else:
                # If the current word itself is longer than n and there's no current sentence,
                # the word is added as a separate sentence.
                sentences.append(word)
        else:
            if current_sentence:
                # Add a space before the word if it's not the first word in the sentence
                current_sentence += " "
            current_sentence += word

    # Add the last sentence if it exists
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def split_sentences_optimized(sentence, n=350):
    """
    Optimized function to split a sentence by spaces into smaller sentences,
    each with a maximum length of n characters. This function avoids unnecessary string concatenations.
    """
    words = sentence.split()
    sentences = []
    current_length = 0

    # Using a list to build sentences instead of string concatenation
    current_sentence = []

    for word in words:
        word_length = len(word)

        # Check if adding the next word would exceed the limit
        if current_length + word_length + len(current_sentence) > n:
            if current_sentence:
                sentences.append(" ".join(current_sentence))
                current_sentence = [word]
                current_length = word_length
            else:
                # If the current word itself is longer than n and there's no current sentence,
                # the word is added as a separate sentence.
                sentences.append(word)
                current_length = 0
        else:
            current_sentence.append(word)
            current_length += word_length

    # Add the last sentence if it exists
    if current_sentence:
        sentences.append(" ".join(current_sentence))

    return sentences


def _get_sentences(response, verbose=False, min_start=15, max_length=350):
    # no mutations of characters allowed here, only breaking apart or merging
    import nltk
    # refuse to tokenize first 15 characters into sentence, so language detection works and logic simpler
    sentences = nltk.sent_tokenize(response[min_start:])
    sentences = flatten_list([split_sentence(sentence, verbose=verbose) for sentence in sentences])
    sentences = [x for x in sentences if x.strip()]
    if sentences:
        sentences[0] = response[:min_start] + sentences[0]

    # split any long sentences
    sentences = flatten_list([split_sentences_optimized(x, max_length) for x in sentences])

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
        print("empty sentence")
        return []

    # Remove code blocks
    sentence = re.sub("```.*```", "", sentence, flags=re.DOTALL)
    sentence = re.sub("`.*`", "", sentence, flags=re.DOTALL)
    sentence = re.sub("\(.*\)", "", sentence, flags=re.DOTALL)

    # remove marks
    sentence = sentence.replace("```", "")
    sentence = sentence.replace("...", " ")
    sentence = sentence.replace("(", " ")
    sentence = sentence.replace(")", " ")

    sentence = sentence.replace("Dr. ", "Doctor ")

    # filter out emojis
    import emoji
    sentence = ''.join([x for x in sentence if not emoji.is_emoji(x)])

    # fix floating expressions
    sentence = re.sub(r'(\d+)\.(\d+)', r"\1 dot \2", sentence)

    # Fix last bad characters
    sentence = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?|\!)", r"\1\2", sentence)

    sentence = sentence.strip()

    if len(sentence) == 0:
        print("EMPTY SENTENCE after processing")
        return

    if verbose:
        print("Sentence for speech: %s" % sentence)

    return sentence


def detect_language(prompt, supported_languages, verbose=False):
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
