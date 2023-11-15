import textwrap
import re


def setup_nltk():
    import nltk  # we'll use this to split into sentences
    nltk.download("punkt")


setup_nltk()

sentence_keys = ['sentence_list', 'sentence_hash_list', 'stored_sentence', 'stored_sentence_hash']


def init_sentence_state():
    sentence_state = dict(sentence_list=[], sentence_hash_list=[], stored_sentence=None, stored_sentence_hash=None)
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


def get_sentence(response, sentence_state, is_final=False, verbose=False):
    import nltk
    text_to_generate = nltk.sent_tokenize(response.replace("\n", " ").strip())

    # get state items
    sentence_list, sentence_hash_list, stored_sentence, stored_sentence_hash = unpack_state(sentence_state)

    if is_final:
        try:
            last_sentence = nltk.sent_tokenize(response.replace("\n", " ").strip())[-1]
            sentence_hash = hash(last_sentence)
            if sentence_hash not in sentence_hash_list:
                if stored_sentence is not None and stored_sentence_hash is not None:
                    last_sentence = stored_sentence + last_sentence
                    stored_sentence = stored_sentence_hash = None
                    if verbose:
                        print("Last Sentence with stored:", last_sentence)

                sentence_hash_list.append(sentence_hash)
                sentence_list.append(last_sentence)
                if verbose:
                    print("Last Sentence: ", last_sentence)

                return last_sentence, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                                 stored_sentence_hash), True
        except:
            print("ERROR on last sentence history is : %s" % response)

    dif = len(text_to_generate) - len(sentence_list)
    if len(text_to_generate) > 1:

        if dif == 1 and len(sentence_list) != 0:
            # will rely upon is_final=True for last_sentence
            return None, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                    stored_sentence_hash), True

        if dif == 2 and len(sentence_list) != 0 and stored_sentence is not None:
            # will rely upon is_final=True for last_sentence
            return None, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                    stored_sentence_hash), True

        # All this complexity due to trying append first short sentence to next one for proper language auto-detect
        if stored_sentence is not None and stored_sentence_hash is None and dif > 1:
            # means we consumed stored sentence and should look at next sentence to generate
            sentence = text_to_generate[len(sentence_list) + 1]
        elif stored_sentence is not None and len(text_to_generate) > 2 and stored_sentence_hash is not None:
            if verbose:
                print("Appending stored")
            sentence = stored_sentence + text_to_generate[len(sentence_list) + 1]
            stored_sentence_hash = None
        else:
            sentence = text_to_generate[len(sentence_list)]
        # if do below, need to skip empty sentences etc.
        # if len(sentence) > 0 and sentence.strip()[0] in [".", "!", "?"]:
        #     sentence = sentence[1:]

        # too short sentence just append to next one if there is any
        # this is for proper language detection
        if len(sentence) <= 15 and stored_sentence_hash is None and stored_sentence is None:
            if len(sentence) > 0 and sentence[-1] in [".", "!", "?"]:
                if stored_sentence_hash != hash(sentence):
                    stored_sentence = sentence
                    stored_sentence_hash = hash(sentence)
                    print("Storing:", stored_sentence)
                    return None, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                            stored_sentence_hash), dif <= 1

        sentence_hash = hash(sentence)
        if stored_sentence_hash is not None and sentence_hash == stored_sentence_hash:
            return None, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                    stored_sentence_hash), dif <= 1

        if sentence_hash not in sentence_hash_list:
            sentence_hash_list.append(sentence_hash)
            sentence_list.append(sentence)
            if verbose:
                print("New Sentence: ", sentence)
            return sentence, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence,
                                        stored_sentence_hash), dif <= 1

    return None, pack_state(sentence_state, sentence_list, sentence_hash_list, stored_sentence, stored_sentence_hash), dif <= 1


def clean_sentence(sentence, verbose=False, max_length=350):
    if sentence is None or len(sentence) == 0:
        print("empty sentence")
        return

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
    sentence = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?|\!)", r"\1 \2\2", sentence)

    if len(sentence) == 0:
        print("EMPTY SENTENCE after processing")
        return

    if verbose:
        print("Sentence for speech: %s" % sentence)

    if len(sentence) < max_length:
        # no problem continue on
        sentence_list = [sentence]
    else:
        # Until now nltk likely split sentences properly but we need additional
        # check for longer sentence and split at last possible position
        # Do whatever necessary, first break at hypens then spaces and then even split very long words
        sentence_list = textwrap.wrap(sentence, max_length)
        if verbose:
            print("SPLITTED LONG SENTENCE:", sentence_list)
    return sentence_list


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
