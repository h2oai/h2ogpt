import os
import pytest
from transformers import AutoTokenizer

from tests.utils import wrap_test_forked
from src.enums import LangChainAction

encoding = None


def num_tokens_from_string(string: str, model_name=None) -> int:
    """Returns the number of tokens in a text string."""
    global encoding
    if encoding is None:
        encoding = AutoTokenizer.from_pretrained(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


import uuid
SECRET_KEY = str(uuid.uuid4())
SECRET_VALUE = str(uuid.uuid4())

ANSWER_LEN = 256  # allow space for answer (same as


def get_prompt(before, after):
    return f"{before}'{SECRET_KEY}' = '{SECRET_VALUE}'\n{after}\n\nWhat is the value of the key '{SECRET_KEY}'?"


def create_long_prompt_with_secret(prompt_len=None, secret_pos=None, model_name=None):
    import time
    t0 = time.time()
    before = "## UUID key/value pairs to remember:\n\n"
    while num_tokens_from_string(before, model_name) < secret_pos:
        before += f"'{str(uuid.uuid4())}' = '{str(uuid.uuid4())}'\n"
    after = ""
    while num_tokens_from_string(after, model_name) < (prompt_len - secret_pos - ANSWER_LEN):
        after += f"'{str(uuid.uuid4())}' = '{str(uuid.uuid4())}'\n"
    prompt = get_prompt(before, after)
    assert SECRET_VALUE in prompt
    assert num_tokens_from_string(prompt, model_name) <= prompt_len
    t1 = time.time()
    print("time to create long prompt: %.4f" % (t1-t0))
    return prompt


@pytest.mark.parametrize("base_model", ['meta-llama/Llama-2-13b-chat-hf'])
@pytest.mark.parametrize("rope_scaling", [
    None,
    # "{'type':'linear', 'factor':2}",
    # "{'type':'dynamic', 'factor':2}",
    # "{'type':'dynamic', 'factor':4}"
])
@pytest.mark.parametrize("prompt_len", [
    1000, 2000, 3000,
    4000, 5000, 6000, 10000
])
@pytest.mark.parametrize("rel_secret_pos", [
    0.2,
    0.5,
    0.8
])
@wrap_test_forked
def test_gradio_long_context(base_model, rope_scaling, prompt_len, rel_secret_pos):
    import ast
    rope_scaling_factor = 1
    if rope_scaling:
        rope_scaling_factor = ast.literal_eval(rope_scaling).get("factor")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, use_auth_token=True,
                                        trust_remote_code=True)
    max_len = 4096
    if hasattr(config, 'max_position_embeddings'):
        max_len = config.max_position_embeddings
    if prompt_len > max_len * rope_scaling_factor:
        pytest.xfail("no chance")
    secret_pos = int(prompt_len * rel_secret_pos)
    main_kwargs = dict(base_model=base_model, chat=True, stream_output=False, gradio=True, num_beams=1,
                       block_gradio_exit=False, rope_scaling=rope_scaling, use_auth_token=True, save_dir="long_context")
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from src.gen import main
    main(**main_kwargs)
    from src.client_test import run_client_chat
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port

    prompt = create_long_prompt_with_secret(prompt_len=prompt_len, secret_pos=secret_pos, model_name=base_model)

    res_dict, client = run_client_chat(
        prompt=prompt,
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

