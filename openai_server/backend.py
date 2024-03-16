import ast
import os
import time
import uuid
from collections import deque

from log import logger
from openai_server.backend_utils import convert_messages_to_structure


def decode(x, encoding_name="cl100k_base"):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return encoding.decode(x)
    except ImportError:
        return ''


def encode(x, encoding_name="cl100k_base"):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return encoding.encode(x, disallowed_special=())
    except ImportError:
        return []


def count_tokens(x, encoding_name="cl100k_base"):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(x, disallowed_special=()))
    except ImportError:
        return 0


def get_gradio_client(user=None):
    try:
        from gradio_utils.grclient import GradioClient as Client
        concurrent_client = True
    except ImportError:
        print("Using slower gradio API, for speed ensure gradio_utils/grclient.py exists.")
        from gradio_client import Client
        concurrent_client = False

    gradio_prefix = os.getenv('GRADIO_PREFIX', 'http')
    gradio_host = os.getenv('GRADIO_SERVER_HOST', 'localhost')
    gradio_port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    gradio_url = f'{gradio_prefix}://{gradio_host}:{gradio_port}'

    auth = os.environ.get('GRADIO_AUTH', 'None')
    auth_access = os.environ.get('GRADIO_AUTH_ACCESS', 'open')
    guest_name = os.environ.get('GRADIO_GUEST_NAME', '')
    if auth != 'None':
        if user:
            user_split = user.split(':')
            assert len(user_split) >= 2, "username cannot contain : character and must be in form username:password"
            auth_kwargs = dict(auth=(user_split[0], ':'.join(user_split[1:])))
        elif guest_name:
            auth_kwargs = dict(auth=(guest_name, guest_name))
        elif auth_access == 'open':
            auth_kwargs = dict(auth=(str(uuid.uuid4()), str(uuid.uuid4())))
        else:
            auth_kwargs = None
    else:
        auth_kwargs = dict()
    print("OpenAI user: %s" % auth_kwargs, flush=True)

    if auth_kwargs is not None:
        print("Getting gradio client at %s" % gradio_url, flush=True)
        client = Client(gradio_url, **auth_kwargs)
        if concurrent_client:
            client.setup()
    else:
        print("Can't get gradio client at %s yet, no auth" % gradio_url, flush=True)
        client = None
    return client


gradio_client = get_gradio_client()


def get_client(user=None):
    # concurrent gradio client
    if gradio_client is None or user is not None:
        assert user is not None, "Need user set to username:password"
        client = get_gradio_client(user=user)
    elif hasattr(gradio_client, 'clone'):
        client = gradio_client.clone()
    else:
        print(
            "re-get to ensure concurrency ok, slower if API is large, for speed ensure gradio_utils/grclient.py exists.")
        client = get_gradio_client(user=user)

    # even if not auth, want to login
    if user:
        user_split = user.split(':')
        username = user_split[0]
        password = ':'.join(user_split[1:])
        num_model_lock = client.predict(api_name='/num_model_lock')
        chatbots = [None] * (2 + num_model_lock)
        client.predict(None, username, password, *tuple(chatbots), api_name='/login')

    return client


def get_response(instruction, gen_kwargs, verbose=False, chunk_response=True, stream_output=False):
    import ast
    kwargs = dict(instruction=instruction)
    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY'):
        kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))
    # max_tokens=16 for text completion by default
    gen_kwargs['max_new_tokens'] = gen_kwargs.pop('max_new_tokens', gen_kwargs.pop('max_tokens', 256))
    gen_kwargs['visible_models'] = gen_kwargs.pop('visible_models', gen_kwargs.pop('model', 0))

    if gen_kwargs.get('do_sample') in [False, None]:
        # be more like OpenAI, only temperature, not do_sample, to control
        gen_kwargs['temperature'] = gen_kwargs.pop('temperature', 0.0)  # unlike OpenAI, default to not random
    # https://platform.openai.com/docs/api-reference/chat/create
    if gen_kwargs['temperature'] > 0.0:
        # let temperature control sampling
        gen_kwargs['do_sample'] = True
    elif gen_kwargs['top_p'] != 1.0:
        # let top_p control sampling
        gen_kwargs['do_sample'] = True
        if gen_kwargs.get('top_k') == 1 and gen_kwargs.get('temperature') == 0.0:
            logger.warning("Sampling with top_k=1 has no effect if top_k=1 and temperature=0")
    else:
        # no sampling, make consistent
        gen_kwargs['top_p'] = 1.0
        gen_kwargs['top_k'] = 1
    if gen_kwargs['seed'] is None:
        gen_kwargs['seed'] = 0

    if gen_kwargs.get('repetition_penalty', 1) == 1 and gen_kwargs.get('presence_penalty', 0.0) != 0.0:
        # then user using presence_penalty, convert to repetition_penalty for h2oGPT
        # presence_penalty=(repetition_penalty - 1.0) * 2.0 + 0.0,  # so good default
        gen_kwargs['repetition_penalty'] = 0.5 * (gen_kwargs['presence_penalty'] - 0.0) + 1.0

    kwargs.update(**gen_kwargs)

    # concurrent gradio client
    client = get_client(user=gen_kwargs.get('user'))

    if stream_output:
        job = client.submit(str(dict(kwargs)), api_name='/submit_nochat_api')
        job_outputs_num = 0
        last_response = ''
        while not job.done():
            outputs_list = job.outputs().copy()
            job_outputs_num_new = len(outputs_list[job_outputs_num:])
            for num in range(job_outputs_num_new):
                res = outputs_list[job_outputs_num + num]
                res = ast.literal_eval(res)
                if verbose:
                    logger.info('Stream %d: %s\n\n %s\n\n' % (num, res['response'], res))
                    logger.info('Stream %d' % (job_outputs_num + num))
                response = res['response']
                chunk = response[len(last_response):]
                if chunk_response:
                    if chunk:
                        yield chunk
                else:
                    yield response
                last_response = response
            job_outputs_num += job_outputs_num_new
            time.sleep(0.01)

        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_num:])
        res = {}
        for num in range(job_outputs_num_new):
            res = outputs_list[job_outputs_num + num]
            res = ast.literal_eval(res)
            if verbose:
                logger.info('Final Stream %d: %s\n\n%s\n\n' % (num, res['response'], res))
                logger.info('Final Stream %d' % (job_outputs_num + num))
            response = res['response']
            chunk = response[len(last_response):]
            if chunk_response:
                if chunk:
                    yield chunk
            else:
                yield response
            last_response = response
        job_outputs_num += job_outputs_num_new
        if verbose:
            logger.info("total job_outputs_num=%d" % job_outputs_num)
    else:
        res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        res = ast.literal_eval(res)
        yield res['response']


def chat_completion_action(body: dict, stream_output=False) -> dict:
    messages = body.get('messages', [])
    object_type = 'chat.completions' if not stream_output else 'chat.completions.chunk'
    created_time = int(time.time())
    req_id = "chat_cmpl_id-%s" % str(uuid.uuid4())
    resp_list = 'choices'

    gen_kwargs = body
    instruction, system_message, history = convert_messages_to_structure(messages)
    gen_kwargs.update({
        'system_prompt': system_message,
        'chat_conversation': history,
        'stream_output': stream_output
    })

    def chat_streaming_chunk(content):
        # begin streaming
        chunk = {
            "id": req_id,
            "object": object_type,
            "created": created_time,
            "model": '',
            resp_list: [{
                "index": 0,
                "finish_reason": None,
                "message": {'role': 'assistant', 'content': content},
                "delta": {'role': 'assistant', 'content': content},
            }],
        }
        return chunk

    if stream_output:
        yield chat_streaming_chunk('')

    token_count = count_tokens(instruction)
    generator = get_response(instruction, gen_kwargs, chunk_response=stream_output,
                             stream_output=stream_output)

    answer = ''
    for chunk in generator:
        if stream_output:
            answer += chunk
            chat_chunk = chat_streaming_chunk(chunk)
            yield chat_chunk
        else:
            answer = chunk

    completion_token_count = count_tokens(answer)
    stop_reason = "stop"

    if stream_output:
        chunk = chat_streaming_chunk('')
        chunk[resp_list][0]['finish_reason'] = stop_reason
        chunk['usage'] = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        yield chunk
    else:
        resp = {
            "id": req_id,
            "object": object_type,
            "created": created_time,
            "model": '',
            resp_list: [{
                "index": 0,
                "finish_reason": stop_reason,
                "message": {"role": "assistant", "content": answer}
            }],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": token_count + completion_token_count
            }
        }

        yield resp


def completions_action(body: dict, stream_output=False):
    object_type = 'text_completion.chunk' if stream_output else 'text_completion'
    created_time = int(time.time())
    res_id = "res_id-%s" % str(uuid.uuid4())
    resp_list = 'choices'
    prompt_str = 'prompt'
    assert prompt_str in body, "Missing prompt"

    gen_kwargs = body
    gen_kwargs['stream_output'] = stream_output

    if not stream_output:
        prompt_arg = body[prompt_str]
        if isinstance(prompt_arg, str) or (isinstance(prompt_arg, list) and isinstance(prompt_arg[0], int)):
            prompt_arg = [prompt_arg]

        resp_list_data = []
        total_completion_token_count = 0
        total_prompt_token_count = 0

        for idx, prompt in enumerate(prompt_arg, start=0):
            token_count = count_tokens(prompt)
            total_prompt_token_count += token_count

            response = deque(get_response(prompt, gen_kwargs), maxlen=1).pop()
            completion_token_count = count_tokens(response)
            total_completion_token_count += completion_token_count
            stop_reason = "stop"

            res_idx = {
                "index": idx,
                "finish_reason": stop_reason,
                "text": response,
                "logprobs": None,
            }

            resp_list_data.extend([res_idx])

        res_dict = {
            "id": res_id,
            "object": object_type,
            "created": created_time,
            "model": '',
            resp_list: resp_list_data,
            "usage": {
                "prompt_tokens": total_prompt_token_count,
                "completion_tokens": total_completion_token_count,
                "total_tokens": total_prompt_token_count + total_completion_token_count
            }
        }

        yield res_dict
    else:
        prompt = body[prompt_str]
        token_count = count_tokens(prompt)

        def text_streaming_chunk(content):
            # begin streaming
            chunk = {
                "id": res_id,
                "object": object_type,
                "created": created_time,
                "model": '',
                resp_list: [{
                    "index": 0,
                    "finish_reason": None,
                    "text": content,
                    "logprobs": None,
                }],
            }

            return chunk

        generator = get_response(prompt, gen_kwargs, chunk_response=stream_output,
                                 stream_output=stream_output)
        response = ''
        for chunk in generator:
            response += chunk
            yield_chunk = text_streaming_chunk(chunk)
            yield yield_chunk

        completion_token_count = count_tokens(response)
        stop_reason = "stop"
        chunk = text_streaming_chunk('')
        chunk[resp_list][0]["finish_reason"] = stop_reason
        chunk["usage"] = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }
        yield chunk


def chat_completions(body: dict) -> dict:
    generator = chat_completion_action(body, stream_output=False)
    return deque(generator, maxlen=1).pop()


def stream_chat_completions(body: dict):
    for resp in chat_completion_action(body, stream_output=True):
        yield resp


def completions(body: dict) -> dict:
    generator = completions_action(body, stream_output=False)
    return deque(generator, maxlen=1).pop()


def stream_completions(body: dict):
    for resp in completions_action(body, stream_output=True):
        yield resp


def get_model_info():
    # concurrent gradio client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    return dict(model_names=model_dict[0])


def get_model_list():
    # concurrent gradio client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    base_models = [x['base_model'] for x in model_dict]
    return dict(model_names=base_models)
