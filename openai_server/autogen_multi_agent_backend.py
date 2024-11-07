import os
import tempfile

from autogen.agentchat import gather_usage_summary

from openai_server.backend_utils import structure_to_messages
from openai_server.agent_utils import get_ret_dict_and_handle_files
from openai_server.agent_prompting import get_full_system_prompt

from openai_server.autogen_utils import merge_group_chat_messages
from openai_server.autogen_utils import get_all_conversable_agents


def run_autogen_multi_agent(query=None,
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
    assert agent_type in ['autogen_multi_agent'], "Invalid agent_type: %s" % agent_type
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

    base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
    api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
    agent_work_dir = tempfile.mkdtemp()
    from openai_server.autogen_utils import get_code_executor
    from openai_server.autogen_agents import (
        get_human_proxy_agent,
        get_main_group_chat_manager,
        get_chat_agent,
        get_code_group_chat_manager
    )

    # Create a code executor.
    executor = get_code_executor(
        autogen_run_code_in_docker=autogen_run_code_in_docker,
        autogen_timeout=autogen_timeout,
        agent_system_site_packages=agent_system_site_packages,
        autogen_code_restrictions_level=autogen_code_restrictions_level,
        agent_work_dir=agent_work_dir,
        agent_venv_dir=agent_venv_dir,
    )

    # Prepare the system message for the code writer agent.
    code_writer_system_prompt, internal_file_names, system_message_parts = \
        get_full_system_prompt(agent_code_writer_system_message,
                               agent_system_site_packages, system_prompt,
                               base_url,
                               api_key, model, text_context_list, image_file,
                               agent_work_dir, query, autogen_timeout)
    # Prepare the LLM config for the agents
    extra_body = {
        "agent_type": agent_type,  # autogen_multi_agent
    }
    llm_config = {"config_list": [{"model": model,
                                   "api_key": api_key,
                                   "base_url": base_url,
                                   "stream": stream_output,
                                   "cache_seed": autogen_cache_seed,
                                   'max_tokens': max_new_tokens,
                                   "extra_body": extra_body,
                                   }]}
    human_proxy_agent = get_human_proxy_agent(
        llm_config=llm_config,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,

    )
    chat_agent = get_chat_agent(
        llm_config=llm_config,
        autogen_max_consecutive_auto_reply=1,  # Always 1 turn for chat agent
    )
    code_group_chat_manager = get_code_group_chat_manager(
        llm_config=llm_config,
        code_writer_system_prompt=code_writer_system_prompt,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        max_round=40,  # TODO: Define variable above
        executor=executor,
    )
    main_group_chat_manager = get_main_group_chat_manager(
        llm_config=llm_config,
        prompt=query,
        agents=[chat_agent, code_group_chat_manager],
        max_round=40,
    )
    # apply chat history to human_proxy_agent and main_group_chat_manager
    # TODO: check if working
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
        # summary_method="last_msg", # TODO: is summary really working for group chat? Doesnt include code group messages in it, why?
        # summary_args=dict(summary_role="user"), # System by default, but in chat histort it comes last and drops user message in h2ogpt/convert_messages_to_structure method
        max_turns=1,
    )
    # It seems chat_result.chat_history doesnt contain code group messages, so I'm manually merging them here. #TODO: research why so?
    merged_group_chat_messages = merge_group_chat_messages(
        code_group_chat_manager.groupchat.messages, main_group_chat_manager.groupchat.messages
    )
    chat_result.chat_history = merged_group_chat_messages
    # Update summary after including group chats:
    used_agents = list(set([msg['name'] for msg in chat_result.chat_history]))
    # besides human_proxy_agent, check if there is only chat_agent and human_proxy_agent in the used_agents
    if len(used_agents) == 2 and 'chat_agent' in used_agents:
        # If it's only chat_agent and human_proxy_agent, then use last message as summary
        summary = chat_result.chat_history[-1]['content']
    else:
        summarize_prompt = (
            "* Given all the conversation and findings so far, try to answer first user instruction. "
            "* Do not add any introductory phrases. "
            "* After answering user instruction, now you can try to summarize the process. "
            "* In your final summarization, if any key figures or plots were produced, "
            "add inline markdown links to the files so they are rendered as images in the chat history. "
            "Do not include them in code blocks, just directly inlined markdown like ![image](filename.png). "
            "Only use the basename of the file, not the full path, "
            "and the user will map the basename to a local copy of the file so rendering works normally. "
            "* If you have already displayed some images in your answer to the user, you don't need to add them again in the summary. "
            "* Do not try to answer the instruction yourself, just answer based on what is in chat history. "
        )
        summary_chat_history = [msg for msg in chat_result.chat_history]
        for msg in summary_chat_history:
            if msg['name'] == 'human_proxy_agent':
                msg['role'] = 'user'
            else:
                msg['role'] = 'assistant'

        summary = human_proxy_agent._reflection_with_llm(
            prompt=summarize_prompt,
            messages=chat_result.chat_history,
            cache=None,
            role="user"
        )

    # A little sumamry clean-up
    summary = summary.replace("ENDOFTURN", " ").replace("<FINISHED_ALL_TASKS>", " ")
    # Update chat_result with summary
    chat_result.summary = summary
    # Update final usage cost
    all_conversable_agents = [human_proxy_agent] + get_all_conversable_agents(main_group_chat_manager)
    chat_result.cost = gather_usage_summary(all_conversable_agents)
    #### end
    ret_dict = get_ret_dict_and_handle_files(chat_result,
                                             None,
                                             model,
                                             agent_work_dir, agent_verbose, internal_file_names, authorization,
                                             autogen_run_code_in_docker, autogen_stop_docker_executor, executor,
                                             agent_venv_dir, agent_code_writer_system_message,
                                             agent_system_site_packages,
                                             system_message_parts,
                                             autogen_code_restrictions_level, autogen_silent_exchange,
                                             agent_accuracy)

    return ret_dict
