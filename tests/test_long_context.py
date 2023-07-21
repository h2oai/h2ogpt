import os
import pytest

from tests.utils import wrap_test_forked
from src.enums import LangChainAction
import tiktoken


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_to_num_tokens(string: str, num_tokens) -> str:
    assert num_tokens_from_string(string) >= num_tokens, "too short"
    while num_tokens_from_string(string) > num_tokens:
        # stupid way, could do bisect etc., but should be fast enough
        string = string[:-1]
    assert num_tokens_from_string(string) == num_tokens
    return string


SECRET_KEY = 'BLAH_BOO_FOO_12'  # cannot contain TOP SECRET etc. otherwise safety will kick in
SECRET_VALUE = 'A13B12'

ANSWER_LEN = 100  # give the model some tokens to answer


def get_prompt(before, after):
    return f"{before}. The value of '{SECRET_KEY}' is '{SECRET_VALUE}'. {after}.\n\nWhat is the value of '{SECRET_KEY}'?"


def create_long_prompt_with_secret(prompt_len=None, secret_pos=None):
    import time
    t0 = time.time()
    extra_len = num_tokens_from_string(get_prompt('', ''))
    before = "blah " * secret_pos
    before = truncate_to_num_tokens(before, secret_pos)
    after = "blah " * (prompt_len - secret_pos - extra_len)
    after = truncate_to_num_tokens(after, prompt_len - secret_pos - extra_len - ANSWER_LEN)
    prompt = get_prompt(before, after)
    assert SECRET_VALUE in prompt
    assert num_tokens_from_string(prompt) == prompt_len - ANSWER_LEN
    t1 = time.time()
    print("time to create long prompt: %.4f" % (t1-t0))
    return prompt


@pytest.mark.parametrize("base_model", ['meta-llama/Llama-2-7b-chat-hf'])
@pytest.mark.parametrize("rope_scaling", [
    None,
    "{'type':'linear', 'factor':2}",
    "{'type':'dynamic', 'factor':2}",
    "{'type':'dynamic', 'factor':4}"
])
@pytest.mark.parametrize("prompt_len", [
    1024, 2048, 4096, 8192, 16384
])
@pytest.mark.parametrize("rel_secret_pos", [0.1, 0.5, 0.9])
@wrap_test_forked
def test_gradio_long_context(base_model, rope_scaling, prompt_len, rel_secret_pos):
    import ast
    rope_scaling_factor = 1
    if rope_scaling:
        rope_scaling_factor = ast.literal_eval(rope_scaling).get("factor")
    if prompt_len > 4096 * rope_scaling_factor:
        # FIXME - hardcoded 4K for llama2
        # no chance, speed up tests
        pytest.xfail("no chance")
    secret_pos = int(prompt_len * rel_secret_pos)
    # from transformers import AutoConfig
    # config = AutoConfig.from_pretrained(base_model, use_auth_token=True,
    #                                     trust_remote_code=True)
    main_kwargs = dict(base_model=base_model, chat=True, stream_output=False, gradio=True, num_beams=1,
                       block_gradio_exit=False, rope_scaling=rope_scaling, use_auth_token=True)
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from src.gen import main
    main(**main_kwargs)
    from src.client_test import run_client_chat
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port

    prompt = create_long_prompt_with_secret(prompt_len=prompt_len, secret_pos=secret_pos)

    res_dict, client = run_client_chat(
        prompt=prompt,
        prompt_type="llama2",  # FIXME - shouldn't be needed
        stream_output=False, max_new_tokens=16384,
        langchain_mode='Disabled',
        langchain_action=LangChainAction.QUERY.value,
        langchain_agents=[]
    )
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    print(res_dict['response'])
    assert SECRET_VALUE in res_dict['response']

    print("DONE", flush=True)

