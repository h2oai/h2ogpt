import os
import sys

# Disable color and advanced terminal features
os.environ['TERM'] = 'dumb'
os.environ['COLORTERM'] = ''
os.environ['CLICOLOR'] = '0'
os.environ['CLICOLOR_FORCE'] = '0'
os.environ['ANSI_COLORS_DISABLED'] = '1'

import os
import shutil
import tempfile
import time
import uuid

import matplotlib as mpl

mpl.use('Agg')

from openai_server.backend_utils import convert_gen_kwargs, get_user_dir, run_upload_api, structure_to_messages
from openai_server.agent_utils import in_pycharm, identify_image_files, set_python_path, merge_group_chat_messages
from openai_server.autogen_streaming import CustomIOStream, iostream_generator
from openai_server.agent_prompting import agent_system_prompt, get_image_query_helper, get_mermaid_renderer_helper, \
    get_chat_doc_context

from autogen.io import IOStream

custom_stream = CustomIOStream()
IOStream.set_global_default(custom_stream)


def terminate_message_func(msg):
    # in conversable agent, roles are flipped relative to actual OpenAI, so can't filter by assistant
    #        isinstance(msg.get('role'), str) and
    #        msg.get('role') == 'assistant' and

    has_message = isinstance(msg, dict) and isinstance(msg.get('content', ''), str)
    has_term = has_message and msg.get('content', '').endswith("TERMINATE") or msg.get('content', '') == ''
    has_execute = has_message and '# execution: true' in msg.get('content', '')

    if has_execute:
        # sometimes model stops without verifying results if it dumped all steps in one turn
        # force it to continue
        return False

    no_stop_if_code = False
    if no_stop_if_code:
        # don't let LLM stop early if it generated code in last message, so it doesn't try to conclude itself
        from autogen.coding import MarkdownCodeExtractor
        code_blocks = MarkdownCodeExtractor().extract_code_blocks(msg.get("content", ''))
        has_code = len(code_blocks) > 0

        # end on TERMINATE or empty message
        if has_code and has_term:
            print("Model tried to terminate with code present: %s" % len(code_blocks), file=sys.stderr)
            # fix
            msg['content'].replace('TERMINATE', '')
            return False
    if has_term:
        return True
    return False


def run_agent(query,
              visible_models=None,
              stream_output=None,
              max_new_tokens=None,
              authorization=None,
              chat_conversation=None,
              text_context_list=None,
              system_prompt=None,
              image_file=None,
              # autogen/agent specific parameters
              agent_type=None,
              autogen_stop_docker_executor=None,
              autogen_run_code_in_docker=None,
              autogen_max_consecutive_auto_reply=None,
              autogen_max_turns=None,
              autogen_timeout=None,
              autogen_cache_seed=None,
              autogen_venv_dir=None,
              agent_code_writer_system_message=None,
              autogen_system_site_packages=None,
              autogen_code_restrictions_level=None,
              autogen_silent_exchange=None,
              agent_verbose=None) -> dict:
    try:
        if agent_type in ['auto', 'autogen']:
            ret_dict = run_autogen(**locals())
        else:
            ret_dict = {}
            raise ValueError("Invalid agent_type: %s" % agent_type)
    except BaseException as e:
        ret_dict = {}
        raise
    finally:
        if autogen_venv_dir is None and 'autogen_venv_dir' in ret_dict and ret_dict['autogen_venv_dir']:
            autogen_venv_dir = ret_dict['autogen_venv_dir']
            if os.path.isdir(autogen_venv_dir):
                if agent_verbose:
                    print("Clean-up: Removing autogen_venv_dir: %s" % autogen_venv_dir)
                shutil.rmtree(autogen_venv_dir)

    return ret_dict


def run_autogen(query=None,
                visible_models=None,
                stream_output=None,
                max_new_tokens=None,
                authorization=None,
                chat_conversation=None,
                text_context_list=None,
                system_prompt=None,
                image_file=None,
                # autogen/agent specific parameters
                agent_type=None,
                autogen_stop_docker_executor=None,
                autogen_run_code_in_docker=None,
                autogen_max_consecutive_auto_reply=None,
                autogen_max_turns=None,
                autogen_timeout=None,
                autogen_cache_seed=None,
                autogen_venv_dir=None,
                agent_code_writer_system_message=None,
                autogen_system_site_packages=None,
                autogen_code_restrictions_level=None,
                autogen_silent_exchange=None,
                agent_verbose=None) -> dict:
    set_python_path()

    assert agent_type in ['autogen', 'auto'], "Invalid agent_type: %s" % agent_type
    # raise openai.BadRequestError("Testing Error Handling")
    # raise ValueError("Testing Error Handling")

    # handle parameters from chatAPI and OpenAI -> h2oGPT transcription versions
    assert visible_models is not None, "No visible_models specified"
    model = visible_models  # transcribe early

    if stream_output is None:
        stream_output = False
    assert max_new_tokens is not None, "No max_new_tokens specified"

    # handle AutoGen specific parameters
    if autogen_stop_docker_executor is None:
        autogen_stop_docker_executor = False
    if autogen_run_code_in_docker is None:
        autogen_run_code_in_docker = False
    if autogen_max_consecutive_auto_reply is None:
        autogen_max_consecutive_auto_reply = 40
    if autogen_max_turns is None:
        autogen_max_turns = 40
    if autogen_timeout is None:
        autogen_timeout = 120
    if autogen_system_site_packages is None:
        autogen_system_site_packages = True
    if autogen_code_restrictions_level is None:
        autogen_code_restrictions_level = 2
    if autogen_silent_exchange is None:
        autogen_silent_exchange = True
    if agent_verbose is None:
        agent_verbose = False
    if agent_verbose:
        print("AutoGen using model=%s." % model, flush=True)

    # TODO: new_agent_flow check can be removed after the new flow is finalized and tested.
    new_agent_flow = True
    if not new_agent_flow:
    ########################################################
        # Create a temporary directory to store the code files.
        # temp_dir = tempfile.TemporaryDirectory().name
        temp_dir = tempfile.mkdtemp()

        # iostream = IOStream.get_default()
        # iostream.print("\033[32m", end="")

        if autogen_run_code_in_docker:
            from autogen.coding import DockerCommandLineCodeExecutor
            # Create a Docker command line code executor.
            executor = DockerCommandLineCodeExecutor(
                image="python:3.10-slim-bullseye",
                timeout=autogen_timeout,  # Timeout for each code execution in seconds.
                work_dir=temp_dir,  # Use the temporary directory to store the code files.
            )
        else:
            from autogen.code_utils import create_virtual_env
            if autogen_venv_dir is None:
                username = str(uuid.uuid4())
                autogen_venv_dir = ".venv_%s" % username
            env_args = dict(system_site_packages=autogen_system_site_packages,
                            with_pip=True,
                            symlinks=True)
            if not in_pycharm():
                virtual_env_context = create_virtual_env(autogen_venv_dir, **env_args)
            else:
                print("in PyCharm, can't use virtualenv, so we use the system python", file=sys.stderr)
                virtual_env_context = None
            # work_dir = ".workdir_%s" % username
            # PythonLoader(name='code', ))

            # Create a local command line code executor.
            if autogen_code_restrictions_level >= 2:
                from autogen_utils import H2OLocalCommandLineCodeExecutor
            else:
                from autogen.coding.local_commandline_code_executor import \
                    LocalCommandLineCodeExecutor as H2OLocalCommandLineCodeExecutor
            executor = H2OLocalCommandLineCodeExecutor(
                timeout=autogen_timeout,  # Timeout for each code execution in seconds.
                virtual_env_context=virtual_env_context,
                work_dir=temp_dir,  # Use the temporary directory to store the code files.
            )

        # Create an agent with code executor configuration.
        from openai_server.autogen_utils import H2OConversableAgent
        code_executor_agent = H2OConversableAgent(
            "code_executor_agent",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
            is_termination_msg=terminate_message_func,
            max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        )

        agent_code_writer_system_message = agent_system_prompt(agent_code_writer_system_message,
                                                               autogen_system_site_packages)

        # FIXME:
        # Auto-pip install
        # Auto-return file list in each turn

        base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
        api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
        if agent_verbose:
            print("base_url: %s" % base_url)
            print("max_tokens: %s" % max_new_tokens)

        image_query_helper = get_image_query_helper(base_url, api_key, model)
        mermaid_renderer_helper = get_mermaid_renderer_helper()

        chat_doc_query, internal_file_names = get_chat_doc_context(text_context_list, image_file,
                                                                   temp_dir,
                                                                   # avoid text version of chat conversation, confuses LLM
                                                                   chat_conversation=None,
                                                                   system_prompt=system_prompt,
                                                                   prompt=query,
                                                                   model=model)

        cwd = os.path.abspath(os.getcwd())
        path_agent_tools = f'{cwd}/openai_server/agent_tools/'

        agent_tools_note = f"\nDo not hallucinate agent_tools tools. The only files in the {path_agent_tools} directory are as follows: {os.listdir('openai_server/agent_tools')}\n"

        system_message = agent_code_writer_system_message + image_query_helper + mermaid_renderer_helper + agent_tools_note + chat_doc_query

        code_writer_agent = H2OConversableAgent(
            "code_writer_agent",
            system_message=system_message,
            llm_config={"config_list": [{"model": model,
                                         "api_key": api_key,
                                         "base_url": base_url,
                                         "stream": stream_output,
                                         "cache_seed": autogen_cache_seed,
                                         'max_tokens': max_new_tokens}]},
            code_execution_config=False,  # Turn off code execution for this agent.
            human_input_mode="NEVER",
            is_termination_msg=terminate_message_func,
            max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        )

        # apply chat history
        if chat_conversation:
            chat_messages = structure_to_messages(None, None, chat_conversation, None)
            for message in chat_messages:
                if message['role'] == 'assistant':
                    code_writer_agent.send(message['content'], code_executor_agent, request_reply=False)
                if message['role'] == 'user':
                    code_executor_agent.send(message['content'], code_writer_agent, request_reply=False)

        chat_kwargs = dict(recipient=code_writer_agent,
                           max_turns=autogen_max_turns,
                           message=query,
                           cache=None,
                           silent=autogen_silent_exchange,
                           clear_history=False,
                           )
        if autogen_cache_seed:
            from autogen import Cache
            # Use DiskCache as cache
            cache_root_path = "./autogen_cache"
            if not os.path.exists(cache_root_path):
                os.makedirs(cache_root_path, exist_ok=True)
            with Cache.disk(cache_seed=autogen_cache_seed, cache_path_root=cache_root_path) as cache:
                chat_kwargs.update(dict(cache=cache))
                chat_result = code_executor_agent.initiate_chat(**chat_kwargs)
        else:
            chat_result = code_executor_agent.initiate_chat(**chat_kwargs)
        #######################################################
    else:
        # New agent flow
        base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
        api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
        temp_dir = tempfile.mkdtemp()
        from openai_server.agents import (
            get_code_executor,
            get_human_proxy_agent,
            get_main_group_chat_manager,
            get_chat_agent,
            get_code_group_chat_manager
            )

        # Create a code executor.
        executor = get_code_executor(
            temp_dir=temp_dir,
            autogen_run_code_in_docker=autogen_run_code_in_docker,
            autogen_timeout=autogen_timeout,
            autogen_system_site_packages=autogen_system_site_packages,
            autogen_code_restrictions_level=autogen_code_restrictions_level,
            autogen_venv_dir=autogen_venv_dir,
        )

        # Prepare the system message for the code writer agent.
        agent_code_writer_system_message = agent_system_prompt(agent_code_writer_system_message,
                                                            autogen_system_site_packages)
        image_query_helper = get_image_query_helper(base_url, api_key, model)
        mermaid_renderer_helper = get_mermaid_renderer_helper()

        chat_doc_query, internal_file_names = get_chat_doc_context(text_context_list, image_file,
                                                                temp_dir,
                                                                # avoid text version of chat conversation, confuses LLM
                                                                chat_conversation=None,
                                                                system_prompt=system_prompt,
                                                                prompt=query,
                                                                model=model)

        cwd = os.path.abspath(os.getcwd())
        path_agent_tools = f'{cwd}/openai_server/agent_tools/'

        agent_tools_note = f"\nDo not hallucinate agent_tools tools. The only files in the {path_agent_tools} directory are as follows: {os.listdir('openai_server/agent_tools')}\n"

        code_writer_system_prompt = agent_code_writer_system_message + image_query_helper + mermaid_renderer_helper + agent_tools_note + chat_doc_query

        # Prepare the LLM config for the agents
        llm_config={"config_list": [{"model": model,
                                        "api_key": api_key,
                                        "base_url": base_url,
                                        "stream": stream_output,
                                        "cache_seed": autogen_cache_seed,
                                        'max_tokens': max_new_tokens}]}
        human_proxy_agent = get_human_proxy_agent(
            llm_config=llm_config,
            autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,

        )
        chat_agent = get_chat_agent(
            llm_config=llm_config,
            autogen_max_consecutive_auto_reply=1, # Always 1 turn for chat agent
        )
        code_group_chat_manager = get_code_group_chat_manager(
            llm_config=llm_config,
            code_writer_system_prompt=code_writer_system_prompt,
            autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
            max_round=40, # TODO: Define variable above
            executor=executor,
        )
        main_group_chat_manager = get_main_group_chat_manager(
            llm_config=llm_config,
            prompt=query,
            agents=[chat_agent, code_group_chat_manager],
            max_round=40,
        )
        # apply chat history to human_proxy_agent and main_group_chat_manager
        if chat_conversation:
            chat_messages = structure_to_messages(None, None, chat_conversation, None)
            for message in chat_messages:
                if message['role'] == 'assistant':
                    main_group_chat_manager.send(message['content'], human_proxy_agent, request_reply=False)
                if message['role'] == 'user':
                    human_proxy_agent.send(message['content'], main_group_chat_manager, request_reply=False)

        chat_result = human_proxy_agent.initiate_chat(
                    main_group_chat_manager,
                    message=query,
                    summary_method="reflection_with_llm", # TODO: is summary really working for group chat? Doesnt include code group messages in it, why?
                    summary_args=dict(summary_role="user"), # System by default, but in chat histort it comes last and drops user message in h2ogpt/convert_messages_to_structure method
                    max_turns=1,
                )
        # It seems chat_result.chat_history doesnt contain code group messages, so I'm manually merging them here. #TODO: research why so?
        merged_group_chat_messages = merge_group_chat_messages(
            code_group_chat_manager.groupchat.messages, main_group_chat_manager.groupchat.messages
        )
        chat_result.chat_history = merged_group_chat_messages
        ### Update summary after including group chats:
        summarize_prompt = (
            "Try to answer first user prompt based on the agents' conversations and outputs so far. "
            "Do not add any introductory phrases. "
            "If you see some code executions done, try to summarize the process. "
            "* In your final summarization, if any key figures or plots were produced, "
            "add inline markdown links to the files so they are rendered as images in the chat history. "
            "Do not include them in code blocks, just directly inlined markdown like ![image](filename.png). "
            "Only use the basename of the file, not the full path, and the user will map the basename to a local copy of the file so rendering works normally. "
            "Do not try to answer the prompt yourself, just answer based on what is provided in the context to you. "
        )
        chat_result.summary = human_proxy_agent._reflection_with_llm(
            prompt=summarize_prompt,
            messages=chat_result.chat_history,
            cache=None,
            role="user"
        )
    # DEBUG
    if agent_verbose:
        print("chat_result:", chat_result)
        print("list_dir:", os.listdir(temp_dir))

    # Get all files in the temp_dir and one level deep subdirectories
    file_list = []
    for root, dirs, files in os.walk(temp_dir):
        # Exclude deeper directories by checking the depth
        if root == temp_dir or os.path.dirname(root) == temp_dir:
            file_list.extend([os.path.join(root, f) for f in files])

    # Filter the list to include only files
    file_list = [f for f in file_list if os.path.isfile(f)]
    internal_file_names_norm_paths = [os.path.normpath(f) for f in internal_file_names]
    # filter out internal files for RAG case
    file_list = [f for f in file_list if os.path.normpath(f) not in internal_file_names_norm_paths]
    if agent_verbose:
        print("file_list:", file_list)

    image_files, non_image_files = identify_image_files(file_list)
    # keep no more than 10 image files:
    image_files = image_files[:10]
    file_list = image_files + non_image_files

    # copy files so user can download
    user_dir = get_user_dir(authorization)
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    file_ids = []
    for file in file_list:
        new_path = os.path.join(user_dir, os.path.basename(file))
        shutil.copy(file, new_path)
        with open(new_path, "rb") as f:
            content = f.read()
        purpose = 'assistants'
        response_dict = run_upload_api(content, new_path, purpose, authorization)
        file_id = response_dict['id']
        file_ids.append(file_id)

    # temp_dir.cleanup()
    if autogen_run_code_in_docker and autogen_stop_docker_executor:
        t0 = time.time()
        executor.stop()  # Stop the docker command line code executor (takes about 10 seconds, so slow)
        if agent_verbose:
            print(f"Executor Stop time taken: {time.time() - t0:.2f} seconds.")

    ret_dict = {}
    if file_list:
        ret_dict.update(dict(files=file_list))
    if file_ids:
        ret_dict.update(dict(file_ids=file_ids))
    if chat_result and hasattr(chat_result, 'chat_history'):
        ret_dict.update(dict(chat_history=chat_result.chat_history))
    if chat_result and hasattr(chat_result, 'cost'):
        ret_dict.update(dict(cost=chat_result.cost))
    if chat_result and hasattr(chat_result, 'summary') and chat_result.summary:
        ret_dict.update(dict(summary=chat_result.summary))
        print("Made summary: %s" % chat_result.summary, file=sys.stderr)
    else:
        if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
            summary = chat_result.chat_history[-1]['content']
            if not summary and len(chat_result.chat_history) >= 2:
                summary = chat_result.chat_history[-2]['content']
            if summary:
                print("Made summary from chat history: %s" % summary, file=sys.stderr)
                ret_dict.update(dict(summary=summary))
            else:
                print("Did NOT make and could not make summary", file=sys.stderr)
                ret_dict.update(dict(summary=''))
        else:
            print("Did NOT make any summary", file=sys.stderr)
            ret_dict.update(dict(summary=''))
    if autogen_venv_dir is not None:
        ret_dict.update(dict(autogen_venv_dir=autogen_venv_dir))
    if agent_code_writer_system_message is not None:
        ret_dict.update(dict(agent_code_writer_system_message=agent_code_writer_system_message))
    if autogen_system_site_packages is not None:
        ret_dict.update(dict(autogen_system_site_packages=autogen_system_site_packages))
    if chat_doc_query:
        ret_dict.update(dict(chat_doc_query=chat_doc_query))
    if image_query_helper:
        ret_dict.update(dict(image_query_helper=image_query_helper))
    ret_dict.update(dict(autogen_code_restrictions_level=autogen_code_restrictions_level))
    ret_dict.update(dict(autogen_silent_exchange=autogen_silent_exchange))
    # can re-use for chat continuation to avoid sending files over
    # FIXME: Maybe just delete files and force send back to agent
    ret_dict.update(dict(temp_dir=temp_dir))

    return ret_dict


def get_agent_response(query, gen_kwargs, chunk_response=True, stream_output=False, use_process=False):
    # raise ValueError("Testing Error Handling 1")  # works

    gen_kwargs = convert_gen_kwargs(gen_kwargs)
    kwargs = gen_kwargs.copy()
    kwargs.update(dict(chunk_response=chunk_response, stream_output=stream_output))
    gen = iostream_generator(run_agent, query, use_process=use_process, **kwargs)

    ret_dict = {}
    try:
        while True:
            res = next(gen)
            yield res
    except StopIteration as e:
        ret_dict = e.value
    return ret_dict
