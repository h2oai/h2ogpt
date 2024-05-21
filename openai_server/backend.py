import ast
import base64
import io
import os
import platform
import time
import uuid
from collections import deque

import filelock
import numpy as np

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
    if platform.system() in ['Darwin', 'Windows']:
        gradio_host = os.getenv('GRADIO_SERVER_HOST', '127.0.0.1')
    else:
        gradio_host = os.getenv('GRADIO_SERVER_HOST', '0.0.0.0')
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
            if auth_access == 'closed':
                if os.getenv('H2OGPT_OPENAI_USER'):
                    user = os.getenv('H2OGPT_OPENAI_USER')
                    user_split = user.split(':')
                    assert len(
                        user_split) >= 2, "username cannot contain : character and must be in form username:password"
                    auth_kwargs = dict(auth=(user_split[0], ':'.join(user_split[1:])))
                else:
                    raise ValueError(
                        "If closed access, must set ENV H2OGPT_OPENAI_USER (e.g. as 'user:pass' combination) to login from OpenAI->Gradio with some specific user.")
            else:
                auth_kwargs = dict(auth=(guest_name, guest_name))
        elif auth_access == 'open':
            auth_kwargs = dict(auth=(str(uuid.uuid4()), str(uuid.uuid4())))
        else:
            auth_kwargs = None
    else:
        auth_kwargs = dict()
    print("OpenAI user: %s" % auth_kwargs, flush=True)

    if auth_kwargs:
        print("Getting gradio client at %s with auth" % gradio_url, flush=True)
        client = Client(gradio_url, **auth_kwargs)
        if concurrent_client:
            client.setup()
    else:
        print("Getting non-user gradio client at %s" % gradio_url, flush=True)
        client = Client(gradio_url)
        if concurrent_client:
            client.setup()
    return client


gradio_client = get_gradio_client()


def get_client(user=None):
    # concurrent gradio client
    if gradio_client is None or user is not None:
        # assert user is not None, "Need user set to username:password"
        client = get_gradio_client(user=user)
    elif hasattr(gradio_client, 'clone'):
        client = gradio_client.clone()
        if client.get_server_hash() != gradio_client.server_hash:
            os.makedirs('locks', exist_ok=True)
            with filelock.FileLock(os.path.join('locks', 'openai_gradio_client.lock')):
                gradio_client.refresh_client()
    else:
        print(
            "re-get to ensure concurrency ok, slower if API is large, for speed ensure gradio_utils/grclient.py exists.")
        client = get_gradio_client(user=user)

    # even if not auth, want to login
    if user:
        user_split = user.split(':')
        username = user_split[0]
        password = ':'.join(user_split[1:])
        num_model_lock = int(client.predict(api_name='/num_model_lock'))
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
        client.predict(None,
                       h2ogpt_key, visible_models,

                       side_bar_text, doc_count_text, submit_buttons_text, visible_models_text,
                       chat_tab_text, doc_selection_tab_text, doc_view_tab_text, chat_history_tab_text,
                       expert_tab_text, models_tab_text, system_tab_text, tos_tab_text,
                       login_tab_text, hosts_tab_text,

                       username, password,
                       *tuple(chatbots), api_name='/login')

    return client


def get_response(instruction, gen_kwargs, verbose=False, chunk_response=True, stream_output=False):
    import ast
    kwargs = dict(instruction=instruction)
    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY'):
        kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))
    # max_tokens=16 for text completion by default
    gen_kwargs['max_new_tokens'] = gen_kwargs.pop('max_new_tokens', gen_kwargs.pop('max_tokens', 256))
    gen_kwargs['visible_models'] = gen_kwargs.pop('visible_models', gen_kwargs.pop('model', 0))
    gen_kwargs['top_p'] = gen_kwargs.get('top_p', 1.0)
    gen_kwargs['top_k'] = gen_kwargs.get('top_k', 1)
    gen_kwargs['seed'] = gen_kwargs.get('seed', 0)

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

    if gen_kwargs.get('response_format') and hasattr(gen_kwargs.get('response_format'), 'type'):
        # pydantic ensures type and key
        # transcribe to h2oGPT way of just value
        gen_kwargs['response_format'] = gen_kwargs.get('response_format').type

    kwargs.update(**gen_kwargs)

    # WIP:
    #if gen_kwargs.get('skip_gradio'):
    #    fun_with_dict_str_plain

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
            time.sleep(0.005)

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

    if instruction is None:
        instruction = "Continue your response.  If your prior response was cut short, then continue exactly at end of your last response with any ellipses, else continue your response by starting with new line and proceeding with an additional useful and related response."

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


def audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
    if chunk != 'none':
        # break-up audio file
        if chunk == 'silence':
            audio_files = split_audio_on_silence(audio_file)
        else:
            audio_files = split_audio_fixed_intervals(audio_file, interval_ms=chunk)

        for audio_file1 in audio_files:
            for text in _audio_to_text(model, audio_file1, stream, response_format, chunk, **kwargs):
                yield text
    else:
        for text in _audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
            yield text


def _audio_to_text(model, audio_file, stream, response_format, chunk, **kwargs):
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


def text_to_audio(model, voice, input, stream, response_format, **kwargs):
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
            n += 1

        # get rest after job done
        outputs = job.outputs().copy()
        for audio_str in outputs[n:]:
            yield audio_str_to_bytes(audio_str, response_format=response_format)
            n += 1
    else:
        audio_str = client.predict(*tuple(list(inputs.values())), api_name='/speak_text_api')
        yield audio_str_to_bytes(audio_str, response_format=response_format)


def audio_str_to_bytes(audio_str1, response_format='wav'):
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
