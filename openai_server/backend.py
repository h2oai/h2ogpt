import ast
import asyncio
import base64
import functools
import io
import json
import os
import platform
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque

import filelock
import numpy as np

from log import logger
from openai_server.backend_utils import convert_messages_to_structure, convert_gen_kwargs


def start_faulthandler():
    # If hit server or any subprocess with signal SIGUSR1, it'll print out all threads stack trace, but wont't quit or coredump
    # If more than one fork tries to write at same time, then looks corrupted.
    import faulthandler

    # SIGUSR1 in h2oai/__init__.py as well
    faulthandler.enable()
    if hasattr(faulthandler, 'register'):
        # windows/mac
        import signal
        faulthandler.register(signal.SIGUSR1)


start_faulthandler()


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


def get_gradio_auth(user=None, verbose=False):
    if verbose:
        print("GRADIO_SERVER_PORT:", os.getenv('GRADIO_SERVER_PORT'), file=sys.stderr)
        print("GRADIO_GUEST_NAME:", os.getenv('GRADIO_GUEST_NAME'), file=sys.stderr)
        print("GRADIO_AUTH:", os.getenv('GRADIO_AUTH'), file=sys.stderr)
        print("GRADIO_AUTH_ACCESS:", os.getenv('GRADIO_AUTH_ACCESS'), file=sys.stderr)

    gradio_prefix = os.getenv('GRADIO_PREFIX', 'http')
    if platform.system() in ['Darwin', 'Windows']:
        gradio_host = os.getenv('GRADIO_SERVER_HOST', '127.0.0.1')
    else:
        gradio_host = os.getenv('GRADIO_SERVER_HOST', '0.0.0.0')
    gradio_port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    gradio_url = f'{gradio_prefix}://{gradio_host}:{gradio_port}'

    auth = os.environ.get('GRADIO_AUTH', 'None')
    auth_access = os.environ.get('GRADIO_AUTH_ACCESS', 'open')
    guest_name = os.environ.get('GRADIO_GUEST_NAME', '')
    is_guest = False
    if auth != 'None':
        if user:
            user_split = user.split(':')
            assert len(user_split) >= 2, "username cannot contain : character and must be in form username:password"
            username = user_split[0]
            if username == guest_name:
                is_guest = True
            auth_kwargs = dict(auth=(username, ':'.join(user_split[1:])))
        elif guest_name:
            if auth_access == 'closed':
                if os.getenv('H2OGPT_OPENAI_USER'):
                    user = os.getenv('H2OGPT_OPENAI_USER')
                    user_split = user.split(':')
                    assert len(
                        user_split) >= 2, "username cannot contain : character and must be in form username:password"
                    auth_kwargs = dict(auth=(user_split[0], ':'.join(user_split[1:])))
                    is_guest = True
                else:
                    raise ValueError(
                        "If closed access, must set ENV H2OGPT_OPENAI_USER (e.g. as 'user:pass' combination) to login from OpenAI->Gradio with some specific user.")
            else:
                auth_kwargs = dict(auth=(guest_name, guest_name))
                is_guest = True
        elif auth_access == 'open':
            auth_kwargs = dict(auth=(str(uuid.uuid4()), str(uuid.uuid4())))
            is_guest = True
        else:
            auth_kwargs = None
    else:
        auth_kwargs = dict()
    return auth_kwargs, gradio_url, is_guest


def get_gradio_client(user=None, verbose=False):
    auth_kwargs, gradio_url, is_guest = get_gradio_auth(user=user, verbose=verbose)
    print("OpenAI user: %s" % auth_kwargs, flush=True)

    try:
        from gradio_utils.grclient import GradioClient as Client
    except ImportError:
        print("Using slower gradio API, for speed ensure gradio_utils/grclient.py exists.")
        from gradio_client import Client

    if auth_kwargs:
        print("Getting gradio client at %s with auth" % gradio_url, flush=True)
        client = Client(gradio_url, **auth_kwargs)
        if hasattr(client, 'setup'):
            with client_lock:
                client.setup()
    else:
        print("BEGIN: Getting non-user gradio client at %s" % gradio_url, flush=True)
        client = Client(gradio_url)
        if hasattr(client, 'setup'):
            with client_lock:
                client.setup()
        print("END: getting non-user gradio client at %s" % gradio_url, flush=True)
    return client


# Global lock for synchronizing client access
client_lock = threading.Lock()

print("global gradio_client", file=sys.stderr)
gradio_client_list = {}


def sanitize(name):
    bad_chars = ['[', ']', ',', '/', '\\', '\\w', '\\s', '-', '+', '\"', '\'', '>', '<', ' ', '=', ')', '(', ':', '^']
    for char in bad_chars:
        name = name.replace(char, "_")
    return name


def get_client(user=None):
    os.makedirs('locks', exist_ok=True)
    user_lock_file = os.path.join('locks', 'user_%s.lock' % sanitize(str(user)))
    user_lock = filelock.FileLock(user_lock_file)
    # concurrent gradio client
    with user_lock:
        print(list(gradio_client_list.keys()))
        gradio_client = gradio_client_list.get(user)

    if gradio_client is None:
        print("Getting fresh client: %s" % str(user), file=sys.stderr)
        # assert user is not None, "Need user set to username:password"
        gradio_client = get_gradio_client(user=user, verbose=True)
        with user_lock:
            gradio_client_list[user] = gradio_client
        got_fresh_client = True
    else:
        print("re-used gradio_client for user: %s" % user, file=sys.stderr)
        got_fresh_client = False

    if hasattr(gradio_client, 'clone'):
        print("cloning for gradio_client.auth=%s" % str(gradio_client.auth), file=sys.stderr)
        gradio_client0 = gradio_client
        gradio_client = gradio_client0.clone()
        print("client.auth=%s" % str(gradio_client.auth), file=sys.stderr)
        try:
            new_hash = gradio_client.get_server_hash()
            assert new_hash
        except Exception as e:
            ex = traceback.format_exc()
            print(f"re-getting fresh client due to exception: {ex}", file=sys.stderr)
            # just get fresh client if any issues
            print(f"re-getting fresh client due to exception: {str(e)}", file=sys.stderr)
            gradio_client_list[user] = get_gradio_client(user=user, verbose=True)
    if not hasattr(gradio_client, 'clone') and not got_fresh_client:
        print(
            "re-get to ensure concurrency ok, slower if API is large, for speed ensure gradio_utils/grclient.py exists.",
            file=sys.stderr)
        gradio_client = get_gradio_client(user=user)
        gradio_client_list[user] = gradio_client

    # even if not auth, want to login
    auth_kwargs, gradio_url, is_guest = get_gradio_auth(user=user)
    if user and not is_guest and auth_kwargs and 'auth' in auth_kwargs:
        username = auth_kwargs['auth'][0]
        password = auth_kwargs['auth'][1]
        print("start login num lock", flush=True)
        num_model_lock = int(gradio_client.predict(api_name='/num_model_lock'))
        print("finish login num lock", flush=True)
        chatbots = [None] * (2 + num_model_lock)
        h2ogpt_key = ''
        visible_models = []
        side_bar_text = ''
        doc_count_text = ''
        submit_buttons_text = ''
        visible_models_text = ''
        chat_tab_text = ''
        doc_selection_tab_text = ''
        doc_view_tab_text = ''
        chat_history_tab_text = ''
        expert_tab_text = ''
        models_tab_text = ''
        system_tab_text = ''
        tos_tab_text = ''
        login_tab_text = ''
        hosts_tab_text = ''
        print("start login", flush=True)
        t0_login = time.time()
        gradio_client.predict(None,
                              h2ogpt_key, visible_models,

                              side_bar_text, doc_count_text, submit_buttons_text, visible_models_text,
                              chat_tab_text, doc_selection_tab_text, doc_view_tab_text, chat_history_tab_text,
                              expert_tab_text, models_tab_text, system_tab_text, tos_tab_text,
                              login_tab_text, hosts_tab_text,

                              username, password,
                              *tuple(chatbots), api_name='/login')
        print("finish login: %s" % (time.time() - t0_login), flush=True)

    return gradio_client


def get_chunk(outputs_list, job_outputs_num, last_response, num, verbose=False):
    res_str = outputs_list[job_outputs_num + num]
    res_dict = ast.literal_eval(res_str)
    if verbose:
        logger.info('Stream %d: %s\n\n %s\n\n' % (num, res_dict['response'], res_dict))
        logger.info('Stream %d' % (job_outputs_num + num))
    if 'error' in res_dict and res_dict['error']:
        raise RuntimeError(res_dict['error'])
    elif 'error_ex' in res_dict and res_dict['error_ex']:
        raise RuntimeError(res_dict['error_ex'])
    elif 'response' not in res_dict:
        raise RuntimeError("No response in res: %s" % res_dict)
    else:
        response = res_dict['response']
        chunk = response[len(last_response):]
    return chunk, response, res_dict


async def get_response(chunk_response=True, **kwargs):
    assert kwargs['query'] is not None, "query must not be None"
    import ast

    stream_output = kwargs.get('stream_output', True)
    verbose = kwargs.get('verbose', False)

    kwargs = convert_gen_kwargs(kwargs)

    # WIP:
    # if gen_kwargs.get('skip_gradio'):
    #    fun_with_dict_str_plain

    # concurrent gradio client
    client = get_client(user=kwargs.get('user'))

    res_dict = {}

    if stream_output:
        job = client.submit(str(dict(kwargs)), api_name='/submit_nochat_api')
        job_outputs_num = 0
        last_response = ''
        while not job.done():
            outputs_list = job.outputs().copy()
            job_outputs_num_new = len(outputs_list[job_outputs_num:])
            for num in range(job_outputs_num_new):
                chunk, response, res_dict = get_chunk(outputs_list, job_outputs_num, last_response, num,
                                                      verbose=verbose)
                if chunk_response:
                    if chunk:
                        yield chunk
                else:
                    yield response
                last_response = response
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.005)
            job_outputs_num += job_outputs_num_new

        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_num:])
        for num in range(job_outputs_num_new):
            chunk, response, res_dict = get_chunk(outputs_list, job_outputs_num, last_response, num, verbose=verbose)
            if chunk_response:
                if chunk:
                    yield chunk
            else:
                yield response
            last_response = response
            await asyncio.sleep(0.005)
        job_outputs_num += job_outputs_num_new
        if verbose:
            logger.info("total job_outputs_num=%d" % job_outputs_num)
    else:
        res_str = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        res_dict = ast.literal_eval(res_str)
        yield res_dict['response']

    # for usage
    res_dict.pop('audio', None)
    yield res_dict


def split_concatenated_dicts(concatenated_dicts: str):
    # Improved regular expression to handle nested braces
    pattern = r'{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*}'

    try:
        matches = re.findall(pattern, concatenated_dicts)
    except re.error as e:
        print(f"Regular expression error: {e}")
        return []
    except MemoryError:
        print("Memory error: Input might be too large")
        return []

    result = []
    for match in matches:
        try:
            result.append(ast.literal_eval(match))
        except (ValueError, SyntaxError):
            # If parsing fails, add the string as is
            result.append(match)

    return result


def get_generator(instruction, gen_kwargs, use_agent=False, stream_output=False, verbose=False):
    gen_kwargs['stream_output'] = stream_output
    gen_kwargs['query'] = instruction
    if gen_kwargs.get('verbose') is None:
        # for local debugging
        gen_kwargs['verbose'] = verbose

    if use_agent:
        agent_type = gen_kwargs.get('agent_type', 'auto')
        from openai_server.agent_utils import set_dummy_term, run_agent
        set_dummy_term()  # before autogen imported

        if agent_type == 'auto':
            agent_type = 'autogen_2agent'

        if agent_type in ['autogen_2agent']:
            from openai_server.autogen_2agent_backend import run_autogen_2agent
            func = functools.partial(run_agent, run_agent_func=run_autogen_2agent)
            from openai_server.autogen_utils import get_autogen_response
            generator = get_autogen_response(func=func, **gen_kwargs)
        elif agent_type in ['autogen_multi_agent']:
            from openai_server.autogen_multi_agent_backend import run_autogen_multi_agent
            func = functools.partial(run_agent, run_agent_func=run_autogen_multi_agent)
            from openai_server.autogen_utils import get_autogen_response
            generator = get_autogen_response(func=func, **gen_kwargs)
        else:
            raise ValueError("No such agent_type %s" % agent_type)
    else:
        generator = get_response(**gen_kwargs)

    return generator


async def achat_completion_action(body: dict, stream_output=False):
    messages = body.get('messages', [])
    object_type = 'chat.completions' if not stream_output else 'chat.completions.chunk'
    created_time = int(time.time())
    req_id = "chat_cmpl_id-%s" % str(uuid.uuid4())
    resp_list = 'choices'

    gen_kwargs = body
    # Consecutive Autogen messages may have the same role,
    # especially when agent_type involves group chat messages.
    # Therefore, they need to be concatenated.
    agent_type = gen_kwargs.get('agent_type', 'auto')
    if agent_type == "autogen_multi_agent":
        concat_assistant = concat_user = True
    else:
        concat_assistant = concat_user = False

    instruction, system_message, history, image_files = convert_messages_to_structure(
        messages=messages,
        concat_tool=True,  # always concat tool calls
        concat_assistant=concat_assistant,
        concat_user=concat_user,
    )
    # get from messages, unless none, then try to get from gen_kwargs from extra_body
    image_file = image_files if image_files else gen_kwargs.get('image_file', [])
    history = history if history else gen_kwargs.get('chat_conversation', [])
    gen_kwargs.update({
        'system_prompt': system_message,
        'chat_conversation': history,
        'stream_output': stream_output,
        'image_file': image_file,
    })

    use_agent = gen_kwargs.get('use_agent', False)
    if use_agent and os.environ.get('is_agent_server', '0') == '0':
        raise ValueError("Agent is not enabled on this server.")

    model = gen_kwargs.get('model', '')

    def chat_streaming_chunk(content):
        # begin streaming
        msg1 = {'role': 'assistant', 'content': content}
        if gen_kwargs.get('guided_json', {}):
            contents = split_concatenated_dicts(msg1['content'])
            msg1['tool_calls'] = [
                dict(function=dict(name=gen_kwargs['tool_choice'], arguments=json.dumps(x)), id=str(uuid.uuid4())) for x
                in
                contents]
        chunk = {
            "id": req_id,
            "object": object_type,
            "created": created_time,
            "model": model,
            resp_list: [{
                "index": 0,
                "finish_reason": None,
                "message": msg1,
                "delta": msg1,
            }],
        }
        return chunk

    if stream_output:
        yield chat_streaming_chunk('')

    if instruction is None and gen_kwargs.get('langchain_action', '') == 'Query':
        instruction = "Continue your response.  If your prior response was cut short, then continue exactly at end of your last response without any ellipses, else continue your response by starting with new line and proceeding with an additional useful and related response."
    if instruction is None:
        instruction = ''  # allowed by h2oGPT, e.g. for summarize or extract

    generator = get_generator(instruction, gen_kwargs, use_agent=use_agent, stream_output=stream_output)

    answer = ''
    usage = {}
    async for chunk in generator:
        if stream_output:
            if isinstance(chunk, dict):
                usage.update(chunk)
            else:
                chat_chunk = chat_streaming_chunk(chunk)
                answer += chunk
                yield chat_chunk
        else:
            if isinstance(chunk, dict):
                usage.update(chunk)
                if 'response' in chunk:
                    # wil use this if exists
                    answer = chunk['response']
                else:
                    answer = ''
            else:
                # will use this first if exists
                answer = chunk
        await asyncio.sleep(0.005)

    stop_reason = "stop"

    real_prompt_tokens = usage.get('save_dict', {}).get('extra_dict', {}).get('num_prompt_tokens')
    if real_prompt_tokens is not None:
        token_count = real_prompt_tokens
    else:
        token_count = count_tokens(instruction)
    real_completion_tokens = usage.get('save_dict', {}).get('extra_dict', {}).get('ntokens')
    if real_completion_tokens is not None:
        completion_token_count = real_completion_tokens
    else:
        completion_token_count = count_tokens(answer)

    usage.update({
        "prompt_tokens": token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": token_count + completion_token_count,
    })

    if stream_output:
        chunk = chat_streaming_chunk('')
        chunk[resp_list][0]['finish_reason'] = stop_reason
        chunk['usage'] = usage

        yield chunk
    else:
        msg1 = {"role": "assistant", "content": answer}
        if gen_kwargs.get('guided_json', {}):
            contents = split_concatenated_dicts(msg1['content'])
            msg1['tool_calls'] = [
                dict(function=dict(name=gen_kwargs['tool_choice'], arguments=json.dumps(x)), id=str(uuid.uuid4())) for x
                in contents]
        resp = {
            "id": req_id,
            "object": object_type,
            "created": created_time,
            "model": model,
            resp_list: [{
                "index": 0,
                "finish_reason": stop_reason,
                "message": msg1,
            }],
            "usage": usage
        }

        yield resp


async def acompletions_action(body: dict, stream_output=False):
    object_type = 'text_completion.chunk' if stream_output else 'text_completion'
    created_time = int(time.time())
    res_id = "res_id-%s" % str(uuid.uuid4())
    resp_list = 'choices'
    prompt_str = 'prompt'
    assert prompt_str in body, "Missing prompt"

    gen_kwargs = body
    gen_kwargs['stream_output'] = stream_output

    use_agent = gen_kwargs.get('use_agent', False)
    if use_agent and os.environ.get('is_agent_server', '0') == '0':
        raise ValueError("Agents not enabled on this server.")

    usage = {}

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

            generator = get_generator(prompt, gen_kwargs, use_agent=use_agent, stream_output=stream_output)
            ret = {}
            response = ""
            try:
                async for last_value in generator:
                    if isinstance(last_value, dict):
                        ret = last_value
                    else:
                        response = last_value
            except StopIteration:
                pass

            if isinstance(ret, dict):
                usage.update(ret)

            if isinstance(response, str):
                completion_token_count = count_tokens(response)
                total_completion_token_count += completion_token_count
            else:
                # assume image
                total_completion_token_count = 1500
            stop_reason = "stop"

            res_idx = {
                "index": idx,
                "finish_reason": stop_reason,
                "text": response,
                "logprobs": None,
            }

            resp_list_data.extend([res_idx])

        usage.update({
            "prompt_tokens": total_prompt_token_count,
            "completion_tokens": total_completion_token_count,
            "total_tokens": total_prompt_token_count + total_completion_token_count,
        })
        res_dict = {
            "id": res_id,
            "object": object_type,
            "created": created_time,
            "model": '',
            resp_list: resp_list_data,
            "usage": usage
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

        generator = get_generator(prompt, gen_kwargs, use_agent=use_agent, stream_output=stream_output)

        response = ''
        usage = {}
        async for chunk in generator:
            if isinstance(chunk, dict):
                usage.update(chunk)
            else:
                response += chunk
                yield_chunk = text_streaming_chunk(chunk)
                yield yield_chunk
            await asyncio.sleep(0.005)

        completion_token_count = count_tokens(response)
        stop_reason = "stop"
        chunk = text_streaming_chunk('')
        chunk[resp_list][0]["finish_reason"] = stop_reason
        usage.update({
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count,
        })
        chunk["usage"] = usage
        yield chunk


async def astream_chat_completions(body: dict, stream_output=True):
    async for resp in achat_completion_action(body, stream_output=stream_output):
        yield resp


async def astream_completions(body: dict, stream_output=True):
    async for resp in acompletions_action(body, stream_output=stream_output):
        yield resp


def get_model_info():
    # concurrent gradio client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    return dict(model_names=model_dict)


def get_model_list():
    # concurrent gradio client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    base_models = [x['base_model'] for x in model_dict]
    return dict(model_names=base_models)


def split_audio_on_silence(audio_bytes):
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40, keep_silence=200)

    chunk_bytes = []
    for chunk in chunks:
        chunk_buffer = io.BytesIO()
        chunk.export(chunk_buffer, format="wav")
        chunk_bytes.append(chunk_buffer.getvalue())

    return chunk_bytes


def split_audio_fixed_intervals(audio_bytes, interval_ms=10000):
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    chunks = [audio[i:i + interval_ms] for i in range(0, len(audio), interval_ms)]

    chunk_bytes = []
    for chunk in chunks:
        chunk_buffer = io.BytesIO()
        chunk.export(chunk_buffer, format="wav")
        chunk_bytes.append(chunk_buffer.getvalue())

    return chunk_bytes


async def audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
    if chunk != 'none':
        # break-up audio file
        if chunk == 'silence':
            audio_files = split_audio_on_silence(audio_file)
        else:
            audio_files = split_audio_fixed_intervals(audio_file, interval_ms=chunk)

        for audio_file1 in audio_files:
            async for text in _audio_to_text(model, audio_file1, stream, response_format, chunk, **kwargs):
                yield text
    else:
        async for text in _audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
            yield text


async def _audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
    # assumes enable_stt=True set for h2oGPT
    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY') and not kwargs.get('h2ogpt_key'):
        kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))

    client = get_client(kwargs.get('user'))
    h2ogpt_key = kwargs.get('h2ogpt_key', '')

    # string of dict for input
    if not isinstance(audio_file, str):
        audio_file = base64.b64encode(audio_file).decode('utf-8')

    inputs = dict(audio_file=audio_file, stream_output=stream, h2ogpt_key=h2ogpt_key)
    if stream:
        job = client.submit(*tuple(list(inputs.values())), api_name='/transcribe_audio_api')

        # ensure no immediate failure (only required for testing)
        import concurrent.futures
        try:
            e = job.exception(timeout=0.2)
            if e is not None:
                raise RuntimeError(e)
        except concurrent.futures.TimeoutError:
            pass

        n = 0
        for text in job:
            yield dict(text=text.strip())
            n += 1

        # get rest after job done
        outputs = job.outputs().copy()
        for text in outputs[n:]:
            yield dict(text=text.strip())
            n += 1
    else:
        text = client.predict(*tuple(list(inputs.values())), api_name='/transcribe_audio_api')
        yield dict(text=text.strip())


async def text_to_audio(model, voice, input, stream, response_format, **kwargs):
    # tts_model = 'microsoft/speecht5_tts'
    # tts_model = 'tts_models/multilingual/multi-dataset/xtts_v2'
    # assumes enable_tts=True set for h2oGPT

    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY') and not kwargs.get('h2ogpt_key'):
        kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))

    client = get_client(user=kwargs.get('user'))
    h2ogpt_key = kwargs.get('h2ogpt_key')

    if not voice or voice in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
        # ignore OpenAI voices
        speaker = "SLT (female)"
        chatbot_role = "Female AI Assistant"
    else:
        # don't know which model used
        speaker = voice
        chatbot_role = voice

    # string of dict for input
    inputs = dict(chatbot_role=chatbot_role, speaker=speaker, tts_language='autodetect', tts_speed=1.0,
                  prompt=input, stream_output=stream,
                  h2ogpt_key=h2ogpt_key)
    if stream:
        job = client.submit(*tuple(list(inputs.values())), api_name='/speak_text_api')

        # ensure no immediate failure (only required for testing)
        import concurrent.futures
        try:
            e = job.exception(timeout=0.2)
            if e is not None:
                raise RuntimeError(e)
        except concurrent.futures.TimeoutError:
            pass

        n = 0
        for audio_str in job:
            yield audio_str_to_bytes(audio_str, response_format=response_format)
            await asyncio.sleep(0.005)
            n += 1

        # get rest after job done
        outputs = job.outputs().copy()
        for audio_str in outputs[n:]:
            yield audio_str_to_bytes(audio_str, response_format=response_format)
            await asyncio.sleep(0.005)
            n += 1
    else:
        audio_str = client.predict(*tuple(list(inputs.values())), api_name='/speak_text_api')
        yield audio_str_to_bytes(audio_str, response_format=response_format)


def audio_str_to_bytes(audio_str1, response_format='wav'):
    if audio_str1 is None:
        return b''
    # Parse the input string to a dictionary
    audio_dict = ast.literal_eval(audio_str1)

    # Extract the base64 audio data and decode it
    audio = audio_dict['audio']

    # Create a BytesIO stream from the binary data
    s = io.BytesIO(audio)

    # Extract sample rate and define other audio properties
    sr = audio_dict['sr']
    channels = 1  # Assuming mono channel, adjust if necessary
    sample_width = 2  # Assuming 16-bit samples (2 bytes), adjust if necessary

    # Use from_raw to correctly interpret the raw audio data
    from pydub import AudioSegment
    audio_segment = AudioSegment.from_raw(
        s,
        sample_width=sample_width,
        frame_rate=sr,
        channels=channels
    )

    # Export the AudioSegment to a BytesIO object as WAV
    output_stream = io.BytesIO()
    audio_segment.export(output_stream, format=response_format)
    output_bytes = output_stream.getvalue()

    return output_bytes


def list_to_bytes(lst: list) -> str:
    float_array = np.array(lst, dtype="float32")
    bytes_array = float_array.tobytes()
    encoded_bytes = base64.b64encode(bytes_array)
    ascii_string = encoded_bytes.decode('ascii')
    return ascii_string


def text_to_embedding(model, text, encoding_format, **kwargs):
    # assumes enable_stt=True set for h2oGPT
    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY') and not kwargs.get('h2ogpt_key'):
        kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))

    client = get_client(kwargs.get('user'))
    h2ogpt_key = kwargs.get('h2ogpt_key', '')

    inputs = dict(text=text, h2ogpt_key=h2ogpt_key, is_list=str(isinstance(text, list)))
    embeddings = client.predict(*tuple(list(inputs.values())), api_name='/embed_api')
    embeddings = ast.literal_eval(embeddings)

    if encoding_format == "base64":
        data = [{"object": "embedding", "embedding": list_to_bytes(emb), "index": n} for n, emb in
                enumerate(embeddings)]
    elif encoding_format == "float":
        data = [{"object": "embedding", "embedding": emb, "index": n} for n, emb in enumerate(embeddings)]
    else:
        data = [{"object": "embedding", "embedding": emb.tolist(), "index": n} for n, emb in enumerate(embeddings)]

    response = {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }
    return response
