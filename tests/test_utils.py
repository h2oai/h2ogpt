import sys

import pytest

from src.utils import get_list_or_str, read_popen_pipes, get_token_count, reverse_ucurve_list, undo_reverse_ucurve_list
from tests.utils import wrap_test_forked
import subprocess as sp


@wrap_test_forked
def test_get_list_or_str():
    assert get_list_or_str(['foo', 'bar']) == ['foo', 'bar']
    assert get_list_or_str('foo') == 'foo'
    assert get_list_or_str("['foo', 'bar']") == ['foo', 'bar']


@wrap_test_forked
def test_stream_popen1():
    cmd_python = sys.executable + " -i -q -u"
    cmd = cmd_python + " -c print('hi')"
    # cmd = cmd.split(' ')

    with sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')

        p.poll()


@wrap_test_forked
def test_stream_popen2():
    script = """for i in 0 1 2 3 4 5
do
    echo "This messages goes to stdout $i"
    sleep 1
    echo This message goes to stderr >&2
    sleep 1
done
"""
    with open('pieces.sh', 'wt') as f:
        f.write(script)
    with sp.Popen(["./pieces.sh"], stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')
        p.poll()


@pytest.mark.parametrize("text_context_list",
                         ['text_context_list1', 'text_context_list2', 'text_context_list3', 'text_context_list4',
                          'text_context_list5', 'text_context_list6'])
@pytest.mark.parametrize("system_prompt", ['auto', ''])
@pytest.mark.parametrize("context", ['context1', 'context2'])
@pytest.mark.parametrize("iinput", ['iinput1', 'iinput2'])
@pytest.mark.parametrize("chat_conversation", ['chat_conversation1', 'chat_conversation2'])
@pytest.mark.parametrize("instruction", ['instruction1', 'instruction2'])
@wrap_test_forked
def test_limited_prompt(instruction, chat_conversation, iinput, context, system_prompt, text_context_list):
    instruction1 = 'Who are you?'
    instruction2 = ' '.join(['foo_%s ' % x for x in range(0, 500)])
    instruction = instruction1 if instruction == 'instruction1' else instruction2

    iinput1 = 'Extra instruction info'
    iinput2 = ' '.join(['iinput_%s ' % x for x in range(0, 500)])
    iinput = iinput1 if iinput == 'iinput1' else iinput2

    context1 = 'context'
    context2 = ' '.join(['context_%s ' % x for x in range(0, 500)])
    context = context1 if context == 'context1' else context2

    chat_conversation1 = []
    chat_conversation2 = [['user_conv_%s ' % x, 'bot_conv_%s ' % x] for x in range(0, 500)]
    chat_conversation = chat_conversation1 if chat_conversation == 'chat_conversation1' else chat_conversation2

    text_context_list1 = []
    text_context_list2 = ['doc_%s ' % x for x in range(0, 500)]
    text_context_list3 = ['doc_%s ' % x for x in range(0, 10)]
    text_context_list4 = ['documentmany_%s ' % x for x in range(0, 10000)]
    import random, string
    text_context_list5 = [
        'documentlong_%s_%s' % (x, ''.join(random.choices(string.ascii_letters + string.digits, k=300))) for x in
        range(0, 20)]
    text_context_list6 = [
        'documentlong_%s_%s' % (x, ''.join(random.choices(string.ascii_letters + string.digits, k=4000))) for x in
        range(0, 1)]
    if text_context_list == 'text_context_list1':
        text_context_list = text_context_list1
    elif text_context_list == 'text_context_list2':
        text_context_list = text_context_list2
    elif text_context_list == 'text_context_list3':
        text_context_list = text_context_list3
    elif text_context_list == 'text_context_list4':
        text_context_list = text_context_list4
    elif text_context_list == 'text_context_list5':
        text_context_list = text_context_list5
    elif text_context_list == 'text_context_list6':
        text_context_list = text_context_list6
    else:
        raise ValueError("No such %s" % text_context_list)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('h2oai/h2ogpt-4096-llama2-7b-chat')

    prompt_type = 'llama2'
    prompt_dict = None
    debug = False
    chat = True
    stream_output = True
    from src.prompter import Prompter
    prompter = Prompter(prompt_type, prompt_dict, debug=debug,
                        stream_output=stream_output,
                        system_prompt=system_prompt)

    min_max_new_tokens = 512  # like in get_limited_prompt()
    max_input_tokens = -1
    max_new_tokens = 1024
    model_max_length = 4096

    from src.gen import get_limited_prompt
    estimated_full_prompt, \
        instruction, iinput, context, \
        num_prompt_tokens, max_new_tokens, \
        num_prompt_tokens0, num_prompt_tokens_actual, \
        history_to_use_final, external_handle_chat_conversation, \
        top_k_docs_trial, one_doc_size, truncation_generation, system_prompt = \
        get_limited_prompt(instruction, iinput, tokenizer,
                           prompter=prompter,
                           max_new_tokens=max_new_tokens,
                           context=context,
                           chat_conversation=chat_conversation,
                           text_context_list=text_context_list,
                           model_max_length=model_max_length,
                           min_max_new_tokens=min_max_new_tokens,
                           max_input_tokens=max_input_tokens,
                           verbose=True)
    print('%s -> %s or %s: len(history_to_use_final): %s top_k_docs_trial=%s one_doc_size: %s' % (num_prompt_tokens0,
                                                                                   num_prompt_tokens,
                                                                                   num_prompt_tokens_actual,
                                                                                   len(history_to_use_final),
                                                                                   top_k_docs_trial,
                                                                                   one_doc_size),
          flush=True, file=sys.stderr)
    assert num_prompt_tokens <= model_max_length + min_max_new_tokens
    # actual might be less due to token merging for characters across parts, but not more
    assert num_prompt_tokens >= num_prompt_tokens_actual
    assert num_prompt_tokens_actual <= model_max_length

    if top_k_docs_trial > 0:
        text_context_list = text_context_list[:top_k_docs_trial]
    elif one_doc_size is not None:
        text_context_list = [text_context_list[0][:one_doc_size]]
    else:
        text_context_list = []
    assert sum([get_token_count(x, tokenizer) for x in text_context_list]) <= model_max_length


@wrap_test_forked
def test_reverse_ucurve():
    ab = []
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 7, 5, 3, 1]
    ab.append([a, b])
    a = [1]
    b = [1]
    ab.append([a, b])
    a = [1, 2]
    b = [2, 1]
    ab.append([a, b])
    a = [1, 2, 3]
    b = [2, 3, 1]
    ab.append([a, b])
    a = [1, 2, 3, 4]
    b = [2, 4, 3, 1]
    ab.append([a, b])

    for a, b in ab:
        assert reverse_ucurve_list(a) == b
        assert undo_reverse_ucurve_list(b) == a
