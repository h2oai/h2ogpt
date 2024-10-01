import os
import tempfile

from openai_server.backend_utils import structure_to_messages
from openai_server.agent_utils import get_ret_dict_and_handle_files
from openai_server.agent_prompting import get_full_system_prompt

from openai_server.autogen_utils import H2OConversableAgent


def run_autogen_2agent(query=None,
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
                       agent_venv_dir=None,
                       agent_code_writer_system_message=None,
                       agent_system_site_packages=None,
                       autogen_code_restrictions_level=None,
                       autogen_silent_exchange=None,
                       agent_verbose=None) -> dict:
    assert agent_type in ['autogen_2agent', 'auto'], "Invalid agent_type: %s" % agent_type
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
    if agent_system_site_packages is None:
        agent_system_site_packages = True
    if autogen_code_restrictions_level is None:
        autogen_code_restrictions_level = 2
    if autogen_silent_exchange is None:
        autogen_silent_exchange = True
    if agent_verbose is None:
        agent_verbose = False
    if agent_verbose:
        print("AutoGen using model=%s." % model, flush=True)

    # Create a temporary directory to store the code files.
    # temp_dir = tempfile.TemporaryDirectory().name
    temp_dir = tempfile.mkdtemp()

    # iostream = IOStream.get_default()
    # iostream.print("\033[32m", end="")

    from openai_server.autogen_agents import get_code_execution_agent
    from openai_server.autogen_utils import get_code_executor
    executor = get_code_executor(
        autogen_run_code_in_docker,
        autogen_timeout,
        agent_system_site_packages,
        autogen_code_restrictions_level,
        agent_venv_dir,
        temp_dir
    )
    code_executor_agent = get_code_execution_agent(executor, autogen_max_consecutive_auto_reply)

    # FIXME:
    # Auto-pip install
    # Auto-return file list in each turn

    base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
    api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
    if agent_verbose:
        print("base_url: %s" % base_url)
        print("max_tokens: %s" % max_new_tokens)

    system_message, internal_file_names, chat_doc_query, ask_question_about_image_helper, mermaid_renderer_helper = \
        get_full_system_prompt(agent_code_writer_system_message,
                               agent_system_site_packages, system_prompt,
                               base_url,
                               api_key, model, text_context_list, image_file,
                               temp_dir, query)

    enable_caching = True
    def code_writer_terminate_func(msg):
        # In case code_writer_agent just passed a chatty answer without <FINISHED_ALL_TASKS> mentioned,
        # then code_executor will return empty string as response (since there was no code block to execute).
        # So at this point, we need to terminate the chat otherwise code_writer_agent will keep on chatting.
        return isinstance(msg, dict) and msg.get('content', '') == ''

    code_writer_agent = H2OConversableAgent(
        "code_writer_agent",
        system_message=system_message,
        llm_config={'timeout': autogen_timeout,
                    'extra_body': dict(enable_caching=enable_caching),
                    "config_list": [{"model": model,
                                     "api_key": api_key,
                                     "base_url": base_url,
                                     "stream": stream_output,
                                     'max_tokens': max_new_tokens,
                                     'cache_seed': autogen_cache_seed,
                                     }]
                    },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        is_termination_msg=code_writer_terminate_func,
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )

    # apply chat history
    if chat_conversation:
        chat_messages = structure_to_messages(None, None, chat_conversation, None)
        for message in chat_messages:
            if message['role'] == 'assistant':
                code_writer_agent.send(message['content'], code_executor_agent, request_reply=False, silent=True)
            if message['role'] == 'user':
                code_executor_agent.send(message['content'], code_writer_agent, request_reply=False, silent=True)

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

    ret_dict = get_ret_dict_and_handle_files(chat_result, temp_dir, agent_verbose, internal_file_names, authorization,
                                             autogen_run_code_in_docker, autogen_stop_docker_executor, executor,
                                             agent_venv_dir, agent_code_writer_system_message,
                                             agent_system_site_packages,
                                             chat_doc_query, ask_question_about_image_helper, mermaid_renderer_helper,
                                             autogen_code_restrictions_level, autogen_silent_exchange)

    return ret_dict
