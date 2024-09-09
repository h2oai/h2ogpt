
from autogen import GroupChat
from autogen.coding import DockerCommandLineCodeExecutor, CodeExecutor
import uuid
from openai_server.agent_utils import in_pycharm
from openai_server.autogen_utils import H2OConversableAgent, H2OGroupChatManager

def get_code_executor(
    temp_dir,
    autogen_run_code_in_docker: bool,
    autogen_timeout: int,
    autogen_system_site_packages: bool,
    autogen_code_restrictions_level: int,
    autogen_venv_dir,
) -> CodeExecutor:
    if autogen_run_code_in_docker:
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
    return executor

def get_code_executor_agent(
        executor,
        autogen_max_consecutive_auto_reply: int,
        ) -> H2OConversableAgent:
    code_executor_agent = H2OConversableAgent(
        name="code_executor_agent",
        llm_config=False, 
        code_execution_config={"executor": executor},
        human_input_mode="NEVER", 
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    # TODO: improve the description
    code_executor_agent.description = "This agent is able to execute the python or shell codes."
    return code_executor_agent

def get_code_writer_agent(
        code_writer_system_prompt:str,
        llm_config:dict,
        autogen_max_consecutive_auto_reply:int,
        ) -> H2OConversableAgent:
    from openai_server.autogen_utils import H2OConversableAgent
    code_writer_agent = H2OConversableAgent(
        "code_writer_agent",
        system_message=code_writer_system_prompt,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    # TODO: improve the description
    code_writer_agent.description = "This agent is an AI assistant designed to solve complex tasks using coding (Python and shell scripting) and language skills. Don't pick this agent for easy non-code tasks"
    return code_writer_agent

def get_general_knowledge_agent(
    llm_config:dict,
    prompt:str,
    autogen_max_consecutive_auto_reply:int,
) -> H2OConversableAgent:
    gk_system_message = "You answer the question or request provided with natural language only. You can not generate or execute codes. You can not talk to web. You are good at chatting. "
    # TODO: Think about the Terminate procedure
    # gk_system_message += (
    #     f"Add 'TERMINATE' at the end of your response if you think you have enough finding or results to answer user request: {prompt}"
    # )
    general_knowledge_agent = H2OConversableAgent(
        name="general_knowledge_agent",
        system_message=gk_system_message,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    # TODO: improve the description
    general_knowledge_agent.description = "This agent is able to answer general knowledge questions based on its own memory or past conversation context. Only answers with natural language. It can not execute codes. It can not generate code examples. It's only good at chatting."
    return general_knowledge_agent

def get_human_proxy_agent(
    llm_config:dict,
    autogen_max_consecutive_auto_reply:int,
) -> H2OConversableAgent:
    # Human Proxy 
    human_proxy_agent = H2OConversableAgent(
        name="human_proxy_agent",
        system_message="You should act like the user who has the request. You are interested in to see if your request or message is answered or delivered by other agents.",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    return human_proxy_agent

def get_code_group_chat_manager(
        llm_config:dict,
        code_writer_system_prompt:str,
        autogen_max_consecutive_auto_reply:int,
        group_chat_manager_max_round:int,
        executor,
) -> H2OGroupChatManager:
    code_writer_agent = get_code_writer_agent(
        code_writer_system_prompt=code_writer_system_prompt,
        llm_config=llm_config,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    code_executor_agent = get_code_executor_agent(
        executor=executor,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    def group_terminate_flow(msg):
        # Terminate the chat if the message contains 'TERMINATE' or is empty.
        return 'TERMINATE' in msg['content'] or msg['content']==""

    # Group Chats
    code_group_chat = GroupChat(
    agents=[code_writer_agent, code_executor_agent],
    messages=[],
    max_round=group_chat_manager_max_round,
    speaker_selection_method="round_robin" # call in order as defined in agents
    )
    code_group_chat_manager = H2OGroupChatManager(
        groupchat=code_group_chat,
        llm_config=llm_config,
        is_termination_msg=group_terminate_flow,
        name="code_group_chat_manager",
        system_message="You are able to generate and execute codes. You can talk to web. You can solve complex tasks using coding (Python and shell scripting) and language skills. "
    )
    # TODO: improve the description
    code_group_chat_manager.description = "Completes simple or complex tasks via python or sh coding. Complex tasks can involve many coding operations and web search. It can both generate and execute the code. This agent has to be picked for any coding related task. "
    return code_group_chat_manager

def get_main_group_chat_manager(
        llm_config:dict,
        prompt:str,
        autogen_max_consecutive_auto_reply:int,
        group_chat_manager_max_round:int,
        code_writer_system_prompt:str,
        executor,
) -> H2OGroupChatManager:
    general_knowledge_agent = get_general_knowledge_agent(
        llm_config=llm_config,
        prompt=prompt,
        autogen_max_consecutive_auto_reply=1, # Always 1 turn for general knowledge agent
    )
    code_group_chat_manager = get_code_group_chat_manager(
        llm_config=llm_config,
        code_writer_system_prompt=code_writer_system_prompt,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        group_chat_manager_max_round=group_chat_manager_max_round,
        executor=executor,
    )
    # todo: it seems main_group_chat always picks the first agent in the list?
    main_group_agents = [general_knowledge_agent, code_group_chat_manager]
    select_speaker_message_template = (
               "You are in a role play game. The following roles are available:"
                "{roles}."
                "Read the following conversation."
                "Then select the next role from {agentlist} to play. Only return the role name."
                f"Important: This is the user prompt: {prompt}"
                "If you think that the user request is answered, return empty string as the role name."
    )
    
    main_group_chat = GroupChat(
        agents=main_group_agents,
        messages=[],
        max_round=group_chat_manager_max_round,
        allow_repeat_speaker=True, # Allow the same agent to speak in consecutive rounds.
        send_introductions=True, # Make agents aware of each other.
        speaker_selection_method="auto", # LLM decides which agent to call next.
        select_speaker_prompt_template=None, # This was adding new system prompt at the end, and was causing instruction to be dropped in h2ogpt/convert_messages_to_structure method
        select_speaker_message_template=select_speaker_message_template,
    )

    def main_terminate_flow(msg):
        # Terminate the chat if the message contains 'TERMINATE' or is empty.
        return 'TERMINATE' in msg['content'] or msg['content']==""

    print("in get_main_group_chat_manager, llm_config:", llm_config)
    main_group_chat_manager = H2OGroupChatManager(
        groupchat=main_group_chat,
        llm_config=llm_config,
        is_termination_msg=main_terminate_flow,
        name="main_group_chat_manager",
        # system_message="You are responsible for completing or answering user request or task. You select which agent to call next based on the user request. "
    )
    return main_group_chat_manager

