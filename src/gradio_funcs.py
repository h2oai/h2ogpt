import ast
import copy
import functools
import json
import os
import tempfile
import time
import uuid
import filelock

from src.enums import LangChainMode, LangChainAction, no_model_str, LangChainTypes, langchain_modes_intrinsic, \
    DocumentSubset, unknown_prompt_type, my_db_state0, selection_docs_state0, requests_state0, roles_state0, noneset
from src.tts_utils import combine_audios
from src.utils import _save_generate_tokens, clear_torch_cache, remove, save_generate_output, str_to_list, \
    get_accordion_named
from src.db_utils import length_db1
from src.evaluate_params import input_args_list, eval_func_param_names, key_overrides


def evaluate_nochat(*args1, default_kwargs1=None, str_api=False, plain_api=False, verifier=False, kwargs={},
                    my_db_state1=None,
                    selection_docs_state1=None,
                    requests_state1=None,
                    roles_state1=None,
                    model_states=[],
                    is_public=False,
                    verbose=False,
                    **kwargs1):
    if my_db_state1 is None:
        if 'my_db_state0' in kwargs1 and kwargs1['my_db_state0'] is not None:
            my_db_state1 = kwargs1['my_db_state0']
        else:
            my_db_state1 = copy.deepcopy(my_db_state0)
    if selection_docs_state1 is None:
        if 'selection_docs_state0' in kwargs1 and kwargs1['selection_docs_state0'] is not None:
            selection_docs_state1 = kwargs1['selection_docs_state0']
        else:
            selection_docs_state1 = copy.deepcopy(selection_docs_state0)
    if requests_state1 is None:
        if 'requests_state0' in kwargs1 and kwargs1['requests_state0'] is not None:
            requests_state1 = kwargs1['requests_state0']
        else:
            requests_state1 = copy.deepcopy(requests_state0)
    if roles_state1 is None:
        if 'roles_state0' in kwargs1 and kwargs1['roles_state0'] is not None:
            roles_state1 = kwargs1['roles_state0']
        else:
            roles_state1 = copy.deepcopy(roles_state0)
    kwargs_eval_pop_keys = ['selection_docs_state0', 'requests_state0', 'roles_state0']
    for k in kwargs_eval_pop_keys:
        if k in kwargs1:
            kwargs1.pop(k)

    image_files_to_delete = []
    args_list = list(args1)
    if str_api:
        if plain_api:
            if not verifier:
                # i.e. not fresh model, tells evaluate to use model_state0
                args_list.insert(0, kwargs['model_state_none'].copy())
            else:
                args_list.insert(0, kwargs['verifier_model_state0'].copy())
            args_list.insert(1, my_db_state1.copy())
            args_list.insert(2, selection_docs_state1.copy())
            args_list.insert(3, requests_state1.copy())
            args_list.insert(4, roles_state1.copy())
        user_kwargs = args_list[len(input_args_list)]
        assert isinstance(user_kwargs, str)
        user_kwargs = ast.literal_eval(user_kwargs)
    else:
        assert not plain_api
        user_kwargs = {k: v for k, v in zip(eval_func_param_names, args_list[len(input_args_list):])}
    # control kwargs1 for evaluate
    if 'answer_with_sources' not in user_kwargs:
        kwargs1['answer_with_sources'] = -1  # just text chunk, not URL etc.
    if 'sources_show_text_in_accordion' not in user_kwargs:
        kwargs1['sources_show_text_in_accordion'] = False
    if 'append_sources_to_chat' not in user_kwargs:
        kwargs1['append_sources_to_chat'] = False
    if 'append_sources_to_answer' not in user_kwargs:
        kwargs1['append_sources_to_answer'] = False
    if 'show_link_in_sources' not in user_kwargs:
        kwargs1['show_link_in_sources'] = False
    kwargs1['top_k_docs_max_show'] = 30

    if 'image_file' in user_kwargs and user_kwargs['image_file']:
        if isinstance(user_kwargs['image_file'], list):
            image_files = user_kwargs['image_file']
        else:
            image_files = [user_kwargs['image_file']]
        b2imgs = []
        for img_file_one in image_files:
            img_file_b2img = os.path.join(tempfile.gettempdir(), 'image_file_%s' % str(uuid.uuid4()))
            # assume is bytes
            from src.vision.utils_vision import base64_to_img
            img_file_b2img_one = base64_to_img(img_file_one, img_file_b2img)
            b2imgs.append(img_file_b2img_one)
        image_files_to_delete = b2imgs
        user_kwargs['image_file'] = b2imgs  # always just make list

    # only used for submit_nochat_api
    user_kwargs['chat'] = False
    if 'stream_output' not in user_kwargs:
        user_kwargs['stream_output'] = False
    if plain_api:
        user_kwargs['stream_output'] = False
    if 'langchain_mode' not in user_kwargs:
        # if user doesn't specify, then assume disabled, not use default
        if LangChainMode.LLM.value in kwargs['langchain_modes']:
            user_kwargs['langchain_mode'] = LangChainMode.LLM.value
        elif len(kwargs['langchain_modes']) >= 1:
            user_kwargs['langchain_mode'] = kwargs['langchain_modes'][0]
        else:
            # disabled should always be allowed
            user_kwargs['langchain_mode'] = LangChainMode.DISABLED.value
    if 'langchain_action' not in user_kwargs:
        user_kwargs['langchain_action'] = LangChainAction.QUERY.value
    if 'langchain_agents' not in user_kwargs:
        user_kwargs['langchain_agents'] = []
    # be flexible
    if 'instruction' in user_kwargs and 'instruction_nochat' not in user_kwargs:
        user_kwargs['instruction_nochat'] = user_kwargs['instruction']
    if 'iinput' in user_kwargs and 'iinput_nochat' not in user_kwargs:
        user_kwargs['iinput_nochat'] = user_kwargs['iinput']
    if 'visible_models' not in user_kwargs:
        if kwargs['visible_models']:
            if isinstance(kwargs['visible_models'], int):
                user_kwargs['visible_models'] = [kwargs['visible_models']]
            elif isinstance(kwargs['visible_models'], list):
                # only take first one
                user_kwargs['visible_models'] = [kwargs['visible_models'][0]]
            else:
                user_kwargs['visible_models'] = [0]
        else:
            # if no user version or default version, then just take first
            user_kwargs['visible_models'] = [0]

    if 'h2ogpt_key' not in user_kwargs:
        user_kwargs['h2ogpt_key'] = None
    if 'system_prompt' in user_kwargs and user_kwargs['system_prompt'] is None:
        # avoid worrying about below default_kwargs -> args_list that checks if None
        user_kwargs['system_prompt'] = 'None'
    # by default don't do TTS unless specifically requested
    if 'chatbot_role' not in user_kwargs:
        user_kwargs['chatbot_role'] = 'None'
    if 'speaker' not in user_kwargs:
        user_kwargs['speaker'] = 'None'

    set1 = set(list(default_kwargs1.keys()))
    set2 = set(eval_func_param_names)
    assert set1 == set2, "Set diff: %s %s: %s" % (set1, set2, set1.symmetric_difference(set2))

    # correct ordering.  Note some things may not be in default_kwargs, so can't be default of user_kwargs.get()
    model_state1 = args_list[0]
    my_db_state1 = args_list[1]
    selection_docs_state1 = args_list[2]
    requests_state1 = args_list[3]
    roles_state1 = args_list[4]

    args_list = [user_kwargs[k] if k in user_kwargs and user_kwargs[k] is not None else default_kwargs1[k] for k
                 in eval_func_param_names]
    assert len(args_list) == len(eval_func_param_names)
    stream_output1 = args_list[eval_func_param_names.index('stream_output')]
    if len(model_states) >= 1:
        visible_models1 = args_list[eval_func_param_names.index('visible_models')]
        model_active_choice1 = visible_models_to_model_choice(visible_models1, model_states, api=True)
        model_state1 = model_states[model_active_choice1 % len(model_states)]
        for key in key_overrides:
            if user_kwargs.get(key) is None and model_state1.get(key) is not None:
                args_list[eval_func_param_names.index(key)] = model_state1[key]
        if hasattr(model_state1['tokenizer'], 'model_max_length'):
            # ensure listen to limit, with some buffer
            # buffer = 50
            buffer = 0
            args_list[eval_func_param_names.index('max_new_tokens')] = min(
                args_list[eval_func_param_names.index('max_new_tokens')],
                model_state1['tokenizer'].model_max_length - buffer)

    # override overall visible_models and h2ogpt_key if have model_specific one
    # NOTE: only applicable if len(model_states) > 1 at moment
    # else controlled by evaluate()
    if 'visible_models' in model_state1 and model_state1['visible_models'] is not None:
        assert isinstance(model_state1['visible_models'], (int, str, list, tuple))
        which_model = visible_models_to_model_choice(model_state1['visible_models'], model_states)
        args_list[eval_func_param_names.index('visible_models')] = which_model
    if 'h2ogpt_key' in model_state1 and model_state1['h2ogpt_key'] is not None:
        # remote server key if present
        # i.e. may be '' and used to override overall local key
        assert isinstance(model_state1['h2ogpt_key'], str)
        args_list[eval_func_param_names.index('h2ogpt_key')] = model_state1['h2ogpt_key']

    # final full bot() like input for prep_bot etc.
    instruction_nochat1 = args_list[eval_func_param_names.index('instruction_nochat')] or \
                          args_list[eval_func_param_names.index('instruction')]
    args_list[eval_func_param_names.index('instruction_nochat')] = \
        args_list[eval_func_param_names.index('instruction')] = \
        instruction_nochat1
    history = [[instruction_nochat1, None]]
    # NOTE: Set requests_state1 to None, so don't allow UI-like access, in case modify state via API
    requests_state1_bot = None
    args_list_bot = args_list + [model_state1, my_db_state1, selection_docs_state1, requests_state1_bot,
                                 roles_state1] + [history]

    # at this point like bot() as input
    history, fun1, langchain_mode1, db1, requests_state1, \
        valid_key, h2ogpt_key1, \
        max_time1, stream_output1, \
        chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, langchain_action1 = \
        prep_bot(*args_list_bot, kwargs_eval=kwargs1, plain_api=plain_api, kwargs=kwargs, verbose=verbose)

    save_dict = dict()
    ret = {}
    ret_old = ''
    history_str_old = ''
    error_old = ''
    audios = []  # in case not streaming, since audio is always streaming, need to accumulate for when yield
    last_yield = None
    res_dict = {}
    try:
        tgen0 = time.time()
        for res in get_response(fun1, history, chatbot_role1, speaker1, tts_language1, roles_state1,
                                tts_speed1,
                                langchain_action1,
                                kwargs=kwargs,
                                api=True,
                                verbose=verbose):
            history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio1 = res
            res_dict = {}
            res_dict['response'] = history[-1][1] or ''
            res_dict['error'] = error
            res_dict['sources'] = sources
            res_dict['sources_str'] = sources_str
            res_dict['prompt_raw'] = prompt_raw
            res_dict['llm_answers'] = llm_answers
            res_dict['save_dict'] = save_dict
            res_dict['audio'] = audio1

            error = res_dict.get('error', '')
            sources = res_dict.get('sources', [])
            save_dict = res_dict.get('save_dict', {})

            # update save_dict
            save_dict['error'] = error
            save_dict['sources'] = sources
            save_dict['valid_key'] = valid_key
            save_dict['h2ogpt_key'] = h2ogpt_key1

            # below works for both list and string for any reasonable string of image that's been byte encoded with b' to start or as file name
            image_file_check = args_list[eval_func_param_names.index('image_file')]
            save_dict['image_file_present'] = isinstance(image_file_check, (str, list, tuple)) and len(
                image_file_check) >= 1
            text_context_list_check = args_list[eval_func_param_names.index('text_context_list')]
            save_dict['text_context_list_present'] = isinstance(text_context_list_check, (list, tuple)) and len(
                text_context_list_check) >= 1

            if str_api and plain_api:
                save_dict['which_api'] = 'str_plain_api'
            elif str_api:
                save_dict['which_api'] = 'str_api'
            elif plain_api:
                save_dict['which_api'] = 'plain_api'
            else:
                save_dict['which_api'] = 'nochat_api'
            if 'extra_dict' not in save_dict:
                save_dict['extra_dict'] = {}
            if requests_state1:
                save_dict['extra_dict'].update(requests_state1)
            else:
                save_dict['extra_dict'].update(dict(username='NO_REQUEST'))

            if is_public:
                # don't want to share actual endpoints
                if 'save_dict' in res_dict and isinstance(res_dict['save_dict'], dict):
                    res_dict['save_dict'].pop('inference_server', None)
                    if 'extra_dict' in res_dict['save_dict'] and isinstance(res_dict['save_dict']['extra_dict'],
                                                                            dict):
                        res_dict['save_dict']['extra_dict'].pop('inference_server', None)

            # get response
            if str_api:
                # full return of dict, except constant items that can be read-off at end
                res_dict_yield = res_dict.copy()
                # do not stream: ['save_dict', 'prompt_raw', 'sources', 'sources_str', 'response_no_refs']
                only_stream = ['response', 'llm_answers', 'audio']
                for key in res_dict:
                    if key not in only_stream:
                        if isinstance(res_dict[key], str):
                            res_dict_yield[key] = ''
                        elif isinstance(res_dict[key], list):
                            res_dict_yield[key] = []
                        elif isinstance(res_dict[key], dict):
                            res_dict_yield[key] = {}
                        else:
                            print("Unhandled pop: %s" % key)
                            res_dict_yield.pop(key)
                ret = res_dict_yield
            elif kwargs['langchain_mode'] == 'Disabled':
                ret = fix_text_for_gradio(res_dict['response'], fix_latex_dollars=False,
                                          fix_angle_brackets=False)
            else:
                ret = '<br>' + fix_text_for_gradio(res_dict['response'], fix_latex_dollars=False,
                                                   fix_angle_brackets=False)

            do_yield = False
            could_yield = ret != ret_old
            if kwargs['gradio_api_use_same_stream_limits']:
                history_str = str(ret['response'] if isinstance(ret, dict) else str(ret))
                delta_history = abs(len(history_str) - len(str(history_str_old)))
                # even if enough data, don't yield if has been less than min_seconds
                enough_data = delta_history > kwargs['gradio_ui_stream_chunk_size'] or (error != error_old)
                beyond_min_time = last_yield is None or \
                                  last_yield is not None and \
                                  (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_min_seconds']
                do_yield |= enough_data and beyond_min_time
                # yield even if new data not enough if been long enough and have at least something to yield
                enough_time = last_yield is None or \
                              last_yield is not None and \
                              (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_seconds']
                do_yield |= enough_time and could_yield
                # DEBUG: print("do_yield: %s : %s %s %s" % (do_yield, enough_data, beyond_min_time, enough_time), flush=True)
            else:
                do_yield = could_yield

            if stream_output1 and do_yield:
                last_yield = time.time()
                # yield as it goes, else need to wait since predict only returns first yield
                if isinstance(ret, dict):
                    ret_old = ret.copy()  # copy normal one first
                    from src.tts_utils import combine_audios
                    ret['audio'] = combine_audios(audios, audio=audio1, sr=24000 if chatbot_role1 else 16000,
                                                  expect_bytes=kwargs['return_as_byte'], verbose=verbose)
                    audios = []  # reset accumulation
                    yield ret
                else:
                    ret_old = ret
                    yield ret
                # just last response, not actually full history like bot() and all_bot() but that's all that changes
                # we can ignore other dict entries as consequence of changes to main stream in 100% of current cases
                # even if sources added last after full response done, final yield still yields left over
                history_str_old = str(ret_old['response'] if isinstance(ret_old, dict) else str(ret_old))
            else:
                # collect unstreamed audios
                audios.append(res_dict['audio'])
            if time.time() - tgen0 > max_time1 + 10:  # don't use actual, so inner has chance to complete
                if str_api:
                    res_dict['save_dict']['extra_dict']['timeout'] = time.time() - tgen0
                if verbose:
                    print("Took too long evaluate_nochat: %s" % (time.time() - tgen0), flush=True)
                break

        # yield if anything left over as can happen
        # return back last ret
        if str_api:
            res_dict['save_dict']['extra_dict'] = _save_generate_tokens(res_dict.get('response', ''),
                                                                        res_dict.get('save_dict', {}).get(
                                                                            'extra_dict', {}))
            ret = res_dict.copy()
        if isinstance(ret, dict):
            from src.tts_utils import combine_audios
            ret['audio'] = combine_audios(audios, audio=None,
                                          expect_bytes=kwargs['return_as_byte'])
        yield ret

    finally:
        clear_torch_cache(allow_skip=True)
        db1s = my_db_state1
        clear_embeddings(user_kwargs['langchain_mode'], kwargs['db_type'], db1s, kwargs['dbs'])
        for image_file1 in image_files_to_delete:
            if os.path.isfile(image_file1):
                remove(image_file1)
    save_dict['save_dir'] = kwargs['save_dir']
    save_generate_output(**save_dict)


def visible_models_to_model_choice(visible_models1, model_states1, api=False):
    if isinstance(visible_models1, list):
        assert len(
            visible_models1) >= 1, "Invalid visible_models1=%s, can only be single entry" % visible_models1
        # just take first
        model_active_choice1 = visible_models1[0]
    elif isinstance(visible_models1, (str, int)):
        model_active_choice1 = visible_models1
    else:
        assert isinstance(visible_models1, type(None)), "Invalid visible_models1=%s" % visible_models1
        model_active_choice1 = visible_models1
    if model_active_choice1 is not None:
        if isinstance(model_active_choice1, str):
            display_model_list = [x['display_name'] for x in model_states1]
            if model_active_choice1 in display_model_list:
                model_active_choice1 = display_model_list.index(model_active_choice1)
            else:
                # NOTE: Could raise, but sometimes raising in certain places fails too hard and requires UI restart
                if api:
                    raise ValueError(
                        "Invalid model %s, valid models are: %s" % (model_active_choice1, display_model_list))
                model_active_choice1 = 0
    else:
        model_active_choice1 = 0
    return model_active_choice1


def clear_embeddings(langchain_mode1, db_type, db1s, dbs=None):
    # clear any use of embedding that sits on GPU, else keeps accumulating GPU usage even if clear torch cache
    if db_type in ['chroma', 'chroma_old'] and langchain_mode1 not in ['LLM', 'Disabled', None, '']:
        from gpt_langchain import clear_embedding, length_db1
        if dbs is not None:
            db = dbs.get(langchain_mode1)
            if db is not None and not isinstance(db, str):
                clear_embedding(db)
        if db1s is not None and langchain_mode1 in db1s:
            db1 = db1s[langchain_mode1]
            if len(db1) == length_db1():
                clear_embedding(db1[0])


def fix_text_for_gradio(text, fix_new_lines=False, fix_latex_dollars=True, fix_angle_brackets=True):
    if isinstance(text, tuple):
        # images, audio, etc.
        return text

    if not isinstance(text, str):
        # e.g. list for extraction
        text = str(text)

    if fix_latex_dollars:
        ts = text.split('```')
        for parti, part in enumerate(ts):
            inside = parti % 2 == 1
            if not inside:
                ts[parti] = ts[parti].replace('$', '﹩')
        text = '```'.join(ts)

    if fix_new_lines:
        # let Gradio handle code, since got improved recently
        ## FIXME: below conflicts with Gradio, but need to see if can handle multiple \n\n\n etc. properly as is.
        # ensure good visually, else markdown ignores multiple \n
        # handle code blocks
        ts = text.split('```')
        for parti, part in enumerate(ts):
            inside = parti % 2 == 1
            if not inside:
                ts[parti] = ts[parti].replace('\n', '<br>')
        text = '```'.join(ts)
    if fix_angle_brackets:
        # handle code blocks
        ts = text.split('```')
        for parti, part in enumerate(ts):
            inside = parti % 2 == 1
            if not inside:
                if '<a href' not in ts[parti] and \
                        '<img src=' not in ts[parti] and \
                        '<div ' not in ts[parti] and \
                        '</div>' not in ts[parti] and \
                        '<details><summary>' not in ts[parti]:
                    # try to avoid html best one can
                    ts[parti] = ts[parti].replace('<', '\<').replace('>', '\>')
        text = '```'.join(ts)
    return text


def get_response(fun1, history, chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1,
                 langchain_action1, kwargs={}, api=False, verbose=False):
    """
    bot that consumes history for user input
    instruction (from input_list) itself is not consumed by bot
    :return:
    """
    error = ''
    sources = []
    save_dict = dict()
    output_no_refs = ''
    sources_str = ''
    prompt_raw = ''
    llm_answers = {}

    audio0, audio1, no_audio, generate_speech_func_func = \
        prepare_audio(chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, langchain_action1,
                      kwargs=kwargs, verbose=verbose)

    if not fun1:
        yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio1
        return
    try:
        for output_fun in fun1():
            output = output_fun['response']
            output_no_refs = output_fun['response_no_refs']
            sources = output_fun['sources']  # FIXME: can show sources in separate text box etc.
            sources_iter = []  # don't yield full prompt_raw every iteration, just at end
            sources_str = output_fun['sources_str']
            sources_str_iter = ''  # don't yield full prompt_raw every iteration, just at end
            prompt_raw = output_fun['prompt_raw']
            prompt_raw_iter = ''  # don't yield full prompt_raw every iteration, just at end
            llm_answers = output_fun['llm_answers']
            save_dict = output_fun.get('save_dict', {})
            save_dict_iter = {}
            # ensure good visually, else markdown ignores multiple \n
            bot_message = fix_text_for_gradio(output, fix_latex_dollars=not api, fix_angle_brackets=not api)
            history[-1][1] = bot_message

            if generate_speech_func_func is not None:
                while True:
                    audio1, sentence, sentence_state = generate_speech_func_func(output_no_refs, is_final=False)
                    if audio0 is not None:
                        yield history, error, sources_iter, sources_str_iter, prompt_raw_iter, llm_answers, save_dict_iter, audio0
                        audio0 = None
                    yield history, error, sources_iter, sources_str_iter, prompt_raw_iter, llm_answers, save_dict_iter, audio1
                    if not sentence:
                        # while True to handle case when streaming is fast enough that see multiple sentences in single go
                        break
            else:
                yield history, error, sources_iter, sources_str_iter, prompt_raw_iter, llm_answers, save_dict_iter, audio0
        if generate_speech_func_func:
            # print("final %s %s" % (history[-1][1] is None, audio1 is None), flush=True)
            audio1, sentence, sentence_state = generate_speech_func_func(output_no_refs, is_final=True)
            if audio0 is not None:
                yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio0
        else:
            audio1 = None
        # print("final2 %s %s" % (history[-1][1] is None, audio1 is None), flush=True)
        yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio1
    except StopIteration:
        # print("STOP ITERATION", flush=True)
        yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, no_audio
        raise
    except RuntimeError as e:
        if "generator raised StopIteration" in str(e):
            # assume last entry was bad, undo
            history.pop()
            yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, no_audio
        else:
            if history and len(history) > 0 and len(history[0]) > 1 and history[-1][1] is None:
                history[-1][1] = ''
            yield history, str(e), sources, sources_str, prompt_raw, llm_answers, save_dict, no_audio
            raise
    except Exception as e:
        # put error into user input
        ex = "Exception: %s" % str(e)
        if history and len(history) > 0 and len(history[0]) > 1 and history[-1][1] is None:
            history[-1][1] = ''
        yield history, ex, sources, sources_str, prompt_raw, llm_answers, save_dict, no_audio
        raise
    finally:
        # clear_torch_cache()
        # don't clear torch cache here, too early and stalls generation if used for all_bot()
        pass
    return


def prepare_audio(chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, langchain_action1, kwargs={},
                  verbose=False):
    assert kwargs
    from src.tts_sentence_parsing import init_sentence_state
    sentence_state = init_sentence_state()
    if langchain_action1 in [LangChainAction.EXTRACT.value]:
        # don't do audio for extraction in any case
        generate_speech_func_func = None
        audio0 = None
        audio1 = None
        no_audio = None
    elif kwargs['tts_model'].startswith('microsoft') and speaker1 not in [None, "None"]:
        audio1 = None
        from src.tts import get_speaker_embedding
        speaker_embedding = get_speaker_embedding(speaker1, kwargs['model_tts'].device)
        # audio0 = 16000, np.array([]).astype(np.int16)
        from src.tts_utils import prepare_speech, get_no_audio
        sr = 16000
        audio0 = prepare_speech(sr=sr)
        no_audio = get_no_audio(sr=sr)
        generate_speech_func_func = functools.partial(kwargs['generate_speech_func'],
                                                      speaker=speaker1,
                                                      speaker_embedding=speaker_embedding,
                                                      sentence_state=sentence_state,
                                                      return_as_byte=kwargs['return_as_byte'],
                                                      sr=sr,
                                                      tts_speed=tts_speed1,
                                                      verbose=verbose)
    elif kwargs['tts_model'].startswith('tts_models/') and chatbot_role1 not in [None, "None"]:
        audio1 = None
        from src.tts_utils import prepare_speech, get_no_audio
        from src.tts_coqui import get_latent
        sr = 24000
        audio0 = prepare_speech(sr=sr)
        no_audio = get_no_audio(sr=sr)
        latent = get_latent(roles_state1[chatbot_role1], model=kwargs['model_xtt'])
        generate_speech_func_func = functools.partial(kwargs['generate_speech_func'],
                                                      latent=latent,
                                                      language=tts_language1,
                                                      sentence_state=sentence_state,
                                                      return_as_byte=kwargs['return_as_byte'],
                                                      sr=sr,
                                                      tts_speed=tts_speed1,
                                                      verbose=verbose)
    else:
        generate_speech_func_func = None
        audio0 = None
        audio1 = None
        no_audio = None
    return audio0, audio1, no_audio, generate_speech_func_func


def prep_bot(*args, retry=False, which_model=0, kwargs_eval={}, plain_api=False, kwargs={}, verbose=False):
    """

    :param args:
    :param retry:
    :param which_model: identifies which model if doing model_lock
         API only called for which_model=0, default for inputs_list, but rest should ignore inputs_list
    :return: last element is True if should run bot, False if should just yield history
    """
    assert kwargs
    isize = len(input_args_list) + 1  # states + chat history
    # don't deepcopy, can contain model itself
    # NOTE: Update plain_api in evaluate_nochat too
    args_list = list(args).copy()
    model_state1 = args_list[-isize]
    my_db_state1 = args_list[-isize + 1]
    selection_docs_state1 = args_list[-isize + 2]
    requests_state1 = args_list[-isize + 3]
    roles_state1 = args_list[-isize + 4]
    history = args_list[-1]
    if not history:
        history = []
    # NOTE: For these, could check if None, then automatically use CLI values, but too complex behavior
    prompt_type1 = args_list[eval_func_param_names.index('prompt_type')]
    if prompt_type1 == no_model_str:
        # deal with gradio dropdown
        prompt_type1 = args_list[eval_func_param_names.index('prompt_type')] = None
    prompt_dict1 = args_list[eval_func_param_names.index('prompt_dict')]
    max_time1 = args_list[eval_func_param_names.index('max_time')]
    stream_output1 = args_list[eval_func_param_names.index('stream_output')]
    langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]
    langchain_action1 = args_list[eval_func_param_names.index('langchain_action')]
    document_subset1 = args_list[eval_func_param_names.index('document_subset')]
    h2ogpt_key1 = args_list[eval_func_param_names.index('h2ogpt_key')]
    chat_conversation1 = args_list[eval_func_param_names.index('chat_conversation')]
    valid_key = is_valid_key(kwargs['enforce_h2ogpt_api_key'],
                             kwargs['enforce_h2ogpt_ui_key'],
                             kwargs['h2ogpt_api_keys'], h2ogpt_key1,
                             requests_state1=requests_state1)
    chatbot_role1 = args_list[eval_func_param_names.index('chatbot_role')]
    speaker1 = args_list[eval_func_param_names.index('speaker')]
    tts_language1 = args_list[eval_func_param_names.index('tts_language')]
    tts_speed1 = args_list[eval_func_param_names.index('tts_speed')]

    dummy_return = history, None, langchain_mode1, my_db_state1, requests_state1, \
        valid_key, h2ogpt_key1, \
        max_time1, stream_output1, chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, \
        langchain_action1

    if not plain_api and (model_state1['model'] is None or model_state1['model'] == no_model_str):
        # plain_api has no state, let evaluate() handle switch
        return dummy_return

    args_list = args_list[:-isize]  # only keep rest needed for evaluate()
    if not history:
        if verbose:
            print("No history", flush=True)
        return dummy_return
    instruction1 = history[-1][0]
    if retry and history:
        # if retry, pop history and move onto bot stuff
        history = get_llm_history(history)
        instruction1 = history[-1][0] if history and history[-1] and len(history[-1]) == 2 else None
        if history and history[-1]:
            history[-1][1] = None
        if not instruction1:
            return dummy_return
    elif not instruction1:
        if not allow_empty_instruction(langchain_mode1, document_subset1, langchain_action1):
            # if not retrying, then reject empty query
            return dummy_return
    elif len(history) > 0 and history[-1][1] not in [None, '']:
        # reject submit button if already filled and not retrying
        # None when not filling with '' to keep client happy
        return dummy_return

    from src.gen import evaluate, evaluate_fake
    evaluate_local = evaluate if valid_key else evaluate_fake

    # shouldn't have to specify in API prompt_type if CLI launched model, so prefer global CLI one if have it
    prompt_type1, prompt_dict1 = update_prompt(prompt_type1, prompt_dict1, model_state1,
                                               which_model=which_model, **kwargs)
    # apply back to args_list for evaluate()
    args_list[eval_func_param_names.index('prompt_type')] = prompt_type1
    args_list[eval_func_param_names.index('prompt_dict')] = prompt_dict1
    context1 = args_list[eval_func_param_names.index('context')]

    chat_conversation1 = merge_chat_conversation_history(chat_conversation1, history)
    args_list[eval_func_param_names.index('chat_conversation')] = chat_conversation1

    if 'visible_models' in model_state1 and model_state1['visible_models'] is not None:
        assert isinstance(model_state1['visible_models'], (int, str))
        args_list[eval_func_param_names.index('visible_models')] = model_state1['visible_models']
    if 'h2ogpt_key' in model_state1 and model_state1['h2ogpt_key'] is not None:
        # i.e. may be '' and used to override overall local key
        assert isinstance(model_state1['h2ogpt_key'], str)
        args_list[eval_func_param_names.index('h2ogpt_key')] = model_state1['h2ogpt_key']
    elif not args_list[eval_func_param_names.index('h2ogpt_key')]:
        # now that checked if key was valid or not, now can inject default key in case gradio inference server
        # only do if key not already set by user
        args_list[eval_func_param_names.index('h2ogpt_key')] = kwargs['h2ogpt_key']

    args_list[0] = instruction1  # override original instruction with history from user
    args_list[2] = context1

    for k in eval_func_param_names:
        if k in ['prompt_type', 'prompt_dict', 'visible_models', 'h2ogpt_key']:
            # already handled
            continue
        # allow override of expert/user input for other parameters
        if k in model_state1 and model_state1[k] is not None:
            args_list[eval_func_param_names.index(k)] = model_state1[k]

    eval_args = (model_state1, my_db_state1, selection_docs_state1, requests_state1, roles_state1)
    assert len(eval_args) == len(input_args_list)
    fun1 = functools.partial(evaluate_local, *eval_args, *tuple(args_list), **kwargs_eval)

    return history, fun1, langchain_mode1, my_db_state1, requests_state1, \
        valid_key, h2ogpt_key1, \
        max_time1, stream_output1, \
        chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, \
        langchain_action1


def choose_exc(x, is_public=True):
    # don't expose ports etc. to exceptions window
    if is_public:
        return "Endpoint unavailable or failed"
    else:
        return x


def bot(*args, retry=False, kwargs_evaluate={}, kwargs={}, db_type=None, dbs=None, verbose=False):
    history, fun1, langchain_mode1, db1, requests_state1, \
        valid_key, h2ogpt_key1, \
        max_time1, stream_output1, \
        chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, \
        langchain_action1 = prep_bot(*args, retry=retry, kwargs_eval=kwargs_evaluate, kwargs=kwargs, verbose=verbose)
    save_dict = dict()
    error = ''
    error_with_str = ''
    sources = []
    history_str_old = ''
    error_old = ''
    sources_str = None
    from src.tts_utils import get_no_audio
    no_audio = get_no_audio()
    audios = []  # in case not streaming, since audio is always streaming, need to accumulate for when yield
    last_yield = None
    try:
        tgen0 = time.time()
        for res in get_response(fun1, history, chatbot_role1, speaker1, tts_language1, roles_state1,
                                tts_speed1,
                                langchain_action1,
                                kwargs=kwargs,
                                api=False,
                                verbose=verbose,
                                ):
            do_yield = False
            history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio1 = res
            error_with_str = get_accordion_named(choose_exc(error), "Generate Error",
                                                 font_size=2) if error not in ['', None, 'None'] else ''

            # pass back to gradio only these, rest are consumed in this function
            history_str = str(history)
            could_yield = (
                    history_str != history_str_old or
                    error != error_old and
                    (error not in noneset or
                     error_old not in noneset))
            if kwargs['gradio_ui_stream_chunk_size'] <= 0:
                do_yield |= could_yield
            else:
                delta_history = abs(len(history_str) - len(history_str_old))
                # even if enough data, don't yield if has been less than min_seconds
                enough_data = delta_history > kwargs['gradio_ui_stream_chunk_size'] or (error != error_old)
                beyond_min_time = last_yield is None or \
                                  last_yield is not None and \
                                  (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_min_seconds']
                do_yield |= enough_data and beyond_min_time
                # yield even if new data not enough if been long enough and have at least something to yield
                enough_time = last_yield is None or \
                              last_yield is not None and \
                              (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_seconds']
                do_yield |= enough_time and could_yield
                # DEBUG: print("do_yield: %s : %s %s %s %s" % (do_yield, delta_history, enough_data, beyond_min_time, enough_time), flush=True)
            if stream_output1 and do_yield:
                audio1 = combine_audios(audios, audio=audio1, sr=24000 if chatbot_role1 else 16000,
                                        expect_bytes=kwargs['return_as_byte'], verbose=verbose)
                audios = []  # reset accumulation

                yield history, error, audio1
                history_str_old = history_str
                error_old = error
                last_yield = time.time()
            else:
                audios.append(audio1)

            if time.time() - tgen0 > max_time1 + 10:  # don't use actual, so inner has chance to complete
                if verbose:
                    print("Took too long bot: %s" % (time.time() - tgen0), flush=True)
                break

        # yield if anything left over
        final_audio = combine_audios(audios, audio=no_audio,
                                     expect_bytes=kwargs['return_as_byte'], verbose=verbose)
        if error_with_str:
            if history and history[-1] and len(history[-1]) == 2 and error_with_str:
                if not history[-1][1]:
                    history[-1][1] = error_with_str
                else:
                    # separate bot if already text present
                    history.append((None, error_with_str))
        if kwargs['append_sources_to_chat'] and sources_str:
            history.append((None, sources_str))

        yield history, error, final_audio
    except BaseException as e:
        print("evaluate_nochat exception: %s: %s" % (str(e), str(args)), flush=True)
        raise
    finally:
        clear_torch_cache(allow_skip=True)
        clear_embeddings(langchain_mode1, db_type, db1, dbs)

    # save
    if 'extra_dict' not in save_dict:
        save_dict['extra_dict'] = {}
    save_dict['valid_key'] = valid_key
    save_dict['h2ogpt_key'] = h2ogpt_key1
    if requests_state1:
        save_dict['extra_dict'].update(requests_state1)
    else:
        save_dict['extra_dict'].update(dict(username='NO_REQUEST'))
    save_dict['error'] = error
    save_dict['sources'] = sources
    save_dict['which_api'] = 'bot'
    save_dict['save_dir'] = kwargs['save_dir']
    save_generate_output(**save_dict)



def is_from_ui(requests_state1):
    return isinstance(requests_state1, dict) and 'username' in requests_state1 and requests_state1['username']


def is_valid_key(enforce_h2ogpt_api_key, enforce_h2ogpt_ui_key, h2ogpt_api_keys, h2ogpt_key1, requests_state1=None):
    from_ui = is_from_ui(requests_state1)

    if from_ui and not enforce_h2ogpt_ui_key:
        # no token barrier
        return 'not enforced'
    elif not from_ui and not enforce_h2ogpt_api_key:
        # no token barrier
        return 'not enforced'
    else:
        valid_key = False
        if isinstance(h2ogpt_api_keys, list) and h2ogpt_key1 in h2ogpt_api_keys:
            # passed token barrier
            valid_key = True
        elif isinstance(h2ogpt_api_keys, str) and os.path.isfile(h2ogpt_api_keys):
            with filelock.FileLock(h2ogpt_api_keys + '.lock'):
                with open(h2ogpt_api_keys, 'rt') as f:
                    h2ogpt_api_keys = json.load(f)
                if h2ogpt_key1 in h2ogpt_api_keys:
                    valid_key = True
        return valid_key


def get_one_key(h2ogpt_api_keys, enforce_h2ogpt_api_key):
    if not enforce_h2ogpt_api_key:
        # return None so OpenAI server has no keyed access if not enforcing API key on h2oGPT regardless if keys passed
        return None
    if isinstance(h2ogpt_api_keys, list) and h2ogpt_api_keys:
        return h2ogpt_api_keys[0]
    elif isinstance(h2ogpt_api_keys, str) and os.path.isfile(h2ogpt_api_keys):
        with filelock.FileLock(h2ogpt_api_keys + '.lock'):
            with open(h2ogpt_api_keys, 'rt') as f:
                h2ogpt_api_keys = json.load(f)
            if h2ogpt_api_keys:
                return h2ogpt_api_keys[0]


def get_model_max_length(model_state1, model_state0):
    if model_state1 and not isinstance(model_state1["tokenizer"], str):
        tokenizer = model_state1["tokenizer"]
    elif model_state0 and not isinstance(model_state0["tokenizer"], str):
        tokenizer = model_state0["tokenizer"]
    else:
        tokenizer = None
    if tokenizer is not None:
        return int(tokenizer.model_max_length)
    else:
        return 2000


def get_llm_history(history):
    # avoid None users used for sources, errors, etc.
    if history is None:
        history = []
    for ii in range(len(history) - 1, -1, -1):
        if history[ii] and history[ii][0] is not None:
            last_user_ii = ii
            history = history[:last_user_ii + 1]
            break
    return history


def gen1_fake(fun1, history):
    error = ''
    sources = []
    sources_str = ''
    prompt_raw = ''
    llm_answers = {}
    save_dict = dict()
    audio1 = None
    yield history, error, sources, sources_str, prompt_raw, llm_answers, save_dict, audio1
    return


def merge_chat_conversation_history(chat_conversation1, history):
    # chat_conversation and history ordered so largest index of list is most recent
    if chat_conversation1:
        chat_conversation1 = str_to_list(chat_conversation1)
        for conv1 in chat_conversation1:
            assert isinstance(conv1, (list, tuple))
            assert len(conv1) == 2

    if isinstance(history, list):
        # make copy so only local change
        if chat_conversation1:
            # so priority will be newest that comes from actual chat history from UI, then chat_conversation
            history = chat_conversation1 + history.copy()
    elif chat_conversation1:
        history = chat_conversation1
    else:
        history = []
    return history


def update_langchain_mode_paths(selection_docs_state1):
    dup = selection_docs_state1['langchain_mode_paths'].copy()
    for k, v in dup.items():
        if k not in selection_docs_state1['langchain_modes']:
            selection_docs_state1['langchain_mode_paths'].pop(k)
    for k in selection_docs_state1['langchain_modes']:
        if k not in selection_docs_state1['langchain_mode_types']:
            # if didn't specify shared, then assume scratch if didn't login or personal if logged in
            selection_docs_state1['langchain_mode_types'][k] = LangChainTypes.PERSONAL.value
    return selection_docs_state1


# Setup some gradio states for per-user dynamic state
def my_db_state_done(state):
    if isinstance(state, dict):
        for langchain_mode_db, db_state in state.items():
            scratch_data = state[langchain_mode_db]
            if langchain_mode_db in langchain_modes_intrinsic:
                if len(scratch_data) == length_db1() and hasattr(scratch_data[0], 'delete_collection') and \
                        scratch_data[1] == scratch_data[2]:
                    # scratch if not logged in
                    scratch_data[0].delete_collection()
            # try to free from memory
            scratch_data[0] = None
            del scratch_data[0]


def process_audio(file1, t1=0, t2=30):
    # use no more than 30 seconds
    from pydub import AudioSegment
    # in milliseconds
    t1 = t1 * 1000
    t2 = t2 * 1000
    newAudio = AudioSegment.from_wav(file1)[t1:t2]
    new_file = file1 + '.new.wav'
    newAudio.export(new_file, format="wav")
    return new_file


def allow_empty_instruction(langchain_mode1, document_subset1, langchain_action1):
    allow = False
    allow |= langchain_action1 not in [LangChainAction.QUERY.value,
                                       LangChainAction.IMAGE_QUERY.value,
                                       LangChainAction.IMAGE_CHANGE.value,
                                       LangChainAction.IMAGE_GENERATE.value,
                                       LangChainAction.IMAGE_STYLE.value,
                                       ]
    allow |= document_subset1 in [DocumentSubset.TopKSources.name]
    if langchain_mode1 in [LangChainMode.LLM.value]:
        allow = False
    return allow


def update_prompt(prompt_type1, prompt_dict1, model_state1, which_model=0, global_scope=False, **kwargs):
    assert kwargs
    if not prompt_type1 or which_model != 0:
        # keep prompt_type and prompt_dict in sync if possible
        prompt_type1 = kwargs.get('prompt_type', prompt_type1)
        prompt_dict1 = kwargs.get('prompt_dict', prompt_dict1)
        # prefer model specific prompt type instead of global one
        if not global_scope:
            if not prompt_type1 or which_model != 0:
                prompt_type1 = model_state1.get('prompt_type', prompt_type1)
                prompt_dict1 = model_state1.get('prompt_dict', prompt_dict1)

    if not prompt_dict1 or which_model != 0:
        # if still not defined, try to get
        prompt_dict1 = kwargs.get('prompt_dict', prompt_dict1)
        if not global_scope:
            if not prompt_dict1 or which_model != 0:
                prompt_dict1 = model_state1.get('prompt_dict', prompt_dict1)
    if not global_scope and not prompt_type1:
        # if still not defined, use unknown
        prompt_type1 = unknown_prompt_type
    return prompt_type1, prompt_dict1


def get_fun_with_dict_str_plain(default_kwargs, kwargs, **kwargs_evaluate_nochat):
    fun_with_dict_str_plain = functools.partial(evaluate_nochat,
                                                default_kwargs1=default_kwargs,
                                                str_api=True,
                                                plain_api=True,
                                                kwargs=kwargs,
                                                **kwargs_evaluate_nochat,
                                                )
    return fun_with_dict_str_plain
