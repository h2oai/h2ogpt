import os
import tempfile
import uuid

from openai_server.backend_utils import structure_to_messages, run_download_api_all
from openai_server.agent_utils import get_ret_dict_and_handle_files
from openai_server.agent_prompting import get_full_system_prompt, planning_prompt, planning_final_prompt, \
    get_agent_tools

from openai_server.autogen_utils import get_autogen_use_planning_prompt


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
                       agent_accuracy=None,
                       agent_chat_history=None,
                       agent_files=None,
                       agent_work_dir=None,
                       max_stream_length=None,
                       max_memory_usage=None,
                       autogen_use_planning_prompt=None,
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
                       client_metadata=None,
                       agent_verbose=None) -> dict:
    if client_metadata:
        print("BEGIN 2AGENT: client_metadata: %s" % client_metadata, flush=True)
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
    if max_stream_length is None:
        max_stream_length = 4096
    if max_memory_usage is None:
        # per-execution process maximum memory usage
        max_memory_usage = 16 * 1024**3  # 16 GB
    if agent_chat_history is None:
        agent_chat_history = []
    if agent_files is None:
        agent_files = []
    if agent_verbose is None:
        agent_verbose = False
    if agent_verbose:
        print("AutoGen using model=%s." % model, flush=True)

    if agent_work_dir is None:
        # Create a temporary directory to store the code files.
        # temp_dir = tempfile.TemporaryDirectory().name
        agent_work_dir = tempfile.mkdtemp()

    if agent_files:
        # assume list of file_ids for use with File API
        run_download_api_all(agent_files, authorization, agent_work_dir)

    # iostream = IOStream.get_default()
    # iostream.print("\033[32m", end="")

    path_agent_tools, list_dir = get_agent_tools()

    if agent_accuracy is None:
        agent_accuracy = 'standard'
    agent_accuracy_enum = ['quick', 'basic', 'standard', 'maximum']
    assert agent_accuracy in agent_accuracy_enum, "Invalid agent_accuracy: %s" % agent_accuracy

    if agent_accuracy == 'quick':
        agent_tools_usage_hard_limits = {k: 1 for k in list_dir}
        agent_tools_usage_soft_limits = {k: 1 for k in list_dir}
        extra_user_prompt = """Do not verify your response, do not check generated plots or images using the ask_question_about_image tool."""
        initial_confidence_level = 1
        if autogen_use_planning_prompt is None:
            autogen_use_planning_prompt = False
    elif agent_accuracy == 'basic':
        agent_tools_usage_hard_limits = {k: 3 for k in list_dir}
        agent_tools_usage_soft_limits = {k: 2 for k in list_dir}
        extra_user_prompt = """Perform only basic level of verification and basic quality checks on your response.  Files you make and your response can be basic."""
        initial_confidence_level = 1
        if autogen_use_planning_prompt is None:
            autogen_use_planning_prompt = False
    elif agent_accuracy == 'standard':
        agent_tools_usage_hard_limits = dict(ask_question_about_image=5)
        agent_tools_usage_soft_limits = {k: 5 for k in list_dir}
        extra_user_prompt = ""
        initial_confidence_level = 0
        if autogen_use_planning_prompt is None:
            autogen_use_planning_prompt = get_autogen_use_planning_prompt(model)
    elif agent_accuracy == 'maximum':
        agent_tools_usage_hard_limits = dict(ask_question_about_image=10)
        agent_tools_usage_soft_limits = {}
        extra_user_prompt = ""
        initial_confidence_level = 0
        if autogen_use_planning_prompt is None:
            autogen_use_planning_prompt = get_autogen_use_planning_prompt(model)
    else:
        raise ValueError("Invalid agent_accuracy: %s" % agent_accuracy)

    # assume by default that if have agent history, continuing with task, not starting new one
    if agent_chat_history:
        autogen_use_planning_prompt = False

    if extra_user_prompt:
        query = f"""<extra_query_conditions>\n{extra_user_prompt}\n</extra_query_conditions>\n\n""" + query

    from openai_server.autogen_utils import get_code_executor
    if agent_venv_dir is None:
        username = str(uuid.uuid4())
        agent_venv_dir = ".venv_%s" % username

    executor = get_code_executor(
        autogen_run_code_in_docker=autogen_run_code_in_docker,
        autogen_timeout=autogen_timeout,
        agent_system_site_packages=agent_system_site_packages,
        autogen_code_restrictions_level=autogen_code_restrictions_level,
        agent_work_dir=agent_work_dir,
        agent_venv_dir=agent_venv_dir,
        agent_tools_usage_hard_limits=agent_tools_usage_hard_limits,
        agent_tools_usage_soft_limits=agent_tools_usage_soft_limits,
        max_stream_length=max_stream_length,
        max_memory_usage=max_memory_usage,
    )

    code_executor_kwargs = dict(
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": executor},  # Use the local command line code executor.
        human_input_mode="NEVER",  # Always take human input for this agent for safety.
        # NOTE: no termination message, just triggered by executable code blocks present or not
        # is_termination_msg=terminate_message_func,
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        # max_turns is max times allowed executed some code, should be autogen_max_turns in general
        max_turns=autogen_max_turns,
        initial_confidence_level=initial_confidence_level,
    )

    from openai_server.autogen_utils import H2OConversableAgent
    code_executor_agent = H2OConversableAgent("code_executor_agent", **code_executor_kwargs)

    # FIXME:
    # Auto-pip install
    # Auto-return file list in each turn

    base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
    api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
    if agent_verbose:
        print("base_url: %s" % base_url)
        print("max_tokens: %s" % max_new_tokens)

    system_message, internal_file_names, system_message_parts = \
        get_full_system_prompt(agent_code_writer_system_message,
                               agent_system_site_packages, system_prompt,
                               base_url,
                               api_key, model, text_context_list, image_file,
                               agent_work_dir, query, autogen_timeout)

    enable_caching = True

    def code_writer_terminate_func(msg):
        # In case code_writer_agent just passed a chatty answer without <FINISHED_ALL_TASKS> mentioned,
        # then code_executor will return empty string as response (since there was no code block to execute).
        # So at this point, we need to terminate the chat otherwise code_writer_agent will keep on chatting.
        return isinstance(msg, dict) and msg.get('content', '') == ''

    code_writer_kwargs = dict(system_message=system_message,
                              llm_config={'timeout': autogen_timeout,
                                          'extra_body': dict(enable_caching=enable_caching,
                                                             client_metadata=client_metadata,
                                                             ),
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

    code_writer_agent = H2OConversableAgent("code_writer_agent", **code_writer_kwargs)

    planning_messages = []
    chat_result_planning = None
    if autogen_use_planning_prompt:
        # setup planning agents
        code_writer_kwargs_planning = code_writer_kwargs.copy()
        # terminate immediately
        # Note: max_turns and initial_confidence_level not relevant except for code execution agent
        code_writer_kwargs_update = dict(max_consecutive_auto_reply=1)
        # is_termination_msg=lambda x: True
        code_writer_kwargs_planning.update(code_writer_kwargs_update)
        code_writer_agent_planning = H2OConversableAgent("code_writer_agent", **code_writer_kwargs_planning)

        chat_kwargs = dict(recipient=code_writer_agent_planning,
                           max_turns=1,
                           message=planning_prompt(query),
                           cache=None,
                           silent=autogen_silent_exchange,
                           clear_history=False,
                           )
        code_executor_kwargs_planning = code_executor_kwargs.copy()
        code_executor_kwargs_planning.update(dict(
            max_turns=2,
            initial_confidence_level=1,
        ))
        code_executor_agent_planning = H2OConversableAgent("code_executor_agent", **code_executor_kwargs_planning)

        chat_result_planning = code_executor_agent_planning.initiate_chat(**chat_kwargs)

        # transfer planning result to main agents
        if hasattr(chat_result_planning, 'chat_history') and chat_result_planning.chat_history:
            planning_messages = chat_result_planning.chat_history
            for message in planning_messages:
                if 'content' in message:
                    message['content'] = message['content'].replace('<FINISHED_ALL_TASKS>', '').replace('ENDOFTURN', '')
                if 'role' in message and message['role'] == 'assistant':
                    # replace prompt
                    message['content'] = planning_final_prompt(query)

    # apply chat history
    if chat_conversation or planning_messages or agent_chat_history:
        chat_messages = []

        # some high-level chat history
        if chat_conversation:
            chat_messages.extend(structure_to_messages(None, None, chat_conversation, None))

        # pre-append planning
        chat_messages.extend(planning_messages)

        # actual internal agent chat history
        if agent_chat_history:
            chat_messages.extend(agent_chat_history)

        # apply
        for message in chat_messages:
            if message['role'] == 'user':
                code_writer_agent.send(message['content'], code_executor_agent, request_reply=False, silent=True)
            if message['role'] == 'assistant':
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

    if client_metadata:
        print("END 2AGENT: client_metadata: %s" % client_metadata, flush=True)
    ret_dict = get_ret_dict_and_handle_files(chat_result,
                                             chat_result_planning,
                                             model,
                                             agent_work_dir, agent_verbose, internal_file_names, authorization,
                                             autogen_run_code_in_docker, autogen_stop_docker_executor, executor,
                                             agent_venv_dir, agent_code_writer_system_message,
                                             agent_system_site_packages,
                                             system_message_parts,
                                             autogen_code_restrictions_level, autogen_silent_exchange,
                                             agent_accuracy,
                                             client_metadata=client_metadata)
    if client_metadata:
        print("END FILES FOR 2AGENT: client_metadata: %s" % client_metadata, flush=True)

    return ret_dict
