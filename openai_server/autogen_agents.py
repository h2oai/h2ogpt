import sys
import uuid

from openai_server.agent_utils import in_pycharm, set_python_path
from openai_server.autogen_utils import terminate_message_func


def get_execution_agent(
        autogen_run_code_in_docker,
        autogen_timeout,
        agent_system_site_packages,
        autogen_max_consecutive_auto_reply,
        autogen_code_restrictions_level,
        agent_venv_dir,
        temp_dir
        ):
    if autogen_run_code_in_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
        # Create a Docker command line code executor.
        executor = DockerCommandLineCodeExecutor(
            image="python:3.10-slim-bullseye",
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
            work_dir=temp_dir,  # Use the temporary directory to store the code files.
        )
    else:
        set_python_path()
        from autogen.code_utils import create_virtual_env
        if agent_venv_dir is None:
            username = str(uuid.uuid4())
            agent_venv_dir = ".venv_%s" % username
        env_args = dict(system_site_packages=agent_system_site_packages,
                        with_pip=True,
                        symlinks=True)
        if not in_pycharm():
            virtual_env_context = create_virtual_env(agent_venv_dir, **env_args)
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
    return code_executor_agent, executor


def get_code_executor(
    temp_dir,
    agent_system_site_packages,
    agent_venv_dir: str,
    autogen_run_code_in_docker: bool = False,
    autogen_timeout: int = 60,
    autogen_code_restrictions_level: int = 2,
):
    from autogen.coding import DockerCommandLineCodeExecutor
    if autogen_run_code_in_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
        # Create a Docker command line code executor.
        executor = DockerCommandLineCodeExecutor(
            image="python:3.10-slim-bullseye",
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
            work_dir=temp_dir,  # Use the temporary directory to store the code files.
        )
    else:
        set_python_path()
        from autogen.code_utils import create_virtual_env
        if agent_venv_dir is None:
            username = str(uuid.uuid4())
            agent_venv_dir = ".venv_%s" % username
        env_args = dict(system_site_packages=agent_system_site_packages,
                        with_pip=True,
                        symlinks=True)
        if not in_pycharm():
            virtual_env_context = create_virtual_env(agent_venv_dir, **env_args)
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

# TODO: Reuse get_execution_agent ?
def get_code_executor_agent(
        executor,
        autogen_max_consecutive_auto_reply: int = 1,
        ):
    from openai_server.autogen_utils import H2OConversableAgent
    code_executor_agent = H2OConversableAgent(
        name="code_executor_agent",
        llm_config=False, 
        code_execution_config={"executor": executor},
        human_input_mode="NEVER", 
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    return code_executor_agent

def get_code_writer_agent(
        llm_config:dict,
        code_writer_system_prompt:str | None = None,
        autogen_max_consecutive_auto_reply:int = 1,
        ):
    from openai_server.autogen_utils import H2OConversableAgent
    code_writer_agent = H2OConversableAgent(
        "code_writer_agent",
        system_message=code_writer_system_prompt,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    return code_writer_agent

def get_chat_agent(
    llm_config:dict,
    autogen_max_consecutive_auto_reply:int = 1,
):
    from openai_server.autogen_utils import H2OConversableAgent
    system_message = (
        "You answer the question or request provided with natural language only. "
        "You can not generate or execute codes. "
        "You can not talk to web. "
        "You can not do any math or calculations, "
        "even simple ones like adding numbers. "
        "You are good at chatting. "
        "You are good at answering general knowledge questions "
        "based on your own memory or past conversation context. "
        )

    chat_agent = H2OConversableAgent(
        name="chat_agent",
        system_message=system_message,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    chat_agent.description = (
        "This agent is able to answer general knowledge questions "
        "based on its own memory or past conversation context. "
        "Only answers with natural language. "
        "It can not execute codes. "
        "It can not generate code examples. "
        "It can not access the web. "
        "It can not do any math or calculations, "
        "even simple ones like adding numbers. "
        "It's only good at chatting and answering simple questions. "
        )
    return chat_agent

def get_human_proxy_agent(
    llm_config:dict,
    autogen_max_consecutive_auto_reply:int = 1,
):
    # Human Proxy 
    from openai_server.autogen_utils import H2OConversableAgent
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
        executor,
        code_writer_system_prompt:str | None = None,
        autogen_max_consecutive_auto_reply:int = 1,
        max_round:int = 10,
):
    """
    Returns a group chat manager for code writing and execution.
    The group chat manager contains two agents: code_writer_agent and code_executor_agent.
    Each time group chat manager is called, it will call code_writer_agent first and then code_executor_agent in order.
    """
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
    from autogen import GroupChat
    code_group_chat = GroupChat(
    agents=[code_writer_agent, code_executor_agent],
    messages=[],
    max_round=max_round,
    speaker_selection_method="round_robin" # call in order as defined in agents
    )
    from openai_server.autogen_utils import H2OGroupChatManager
    code_group_chat_manager = H2OGroupChatManager(
        groupchat=code_group_chat,
        llm_config=llm_config,
        is_termination_msg=group_terminate_flow,
        name="code_group_chat_manager",
        system_message=(
            "You are able to generate and execute codes. "
            "You can talk to web. "
            "You can solve complex tasks using coding (Python and shell scripting) and language skills. "
            ),
    )
    code_group_chat_manager.description = (
        "This agent excels at solving tasks through code generation and execution, "
        "using both Python and shell scripts. "
        "It can handle anything from complex computations and data processing to "
        "generating and running executable code. "
        "Additionally, it can access the web to fetch real-time data, "
        "making it ideal for tasks that require automation, coding, or retrieving up-to-date information. "
        "This agent has to be picked for any coding related task or tasks that are "
        "more complex than just chatting or simple question answering. "
        "It can also do math and calculations, from simple arithmetic to complex equations. "
        )
    return code_group_chat_manager

def get_main_group_chat_manager(
        llm_config:dict,
        prompt:str,
        agents= None,
        max_round:int = 10,
):
    """
    Returns Main Group Chat Manager to distribute the roles among the agents.
    The main group chat manager can contain multiple agents.
    Uses LLMs to select the next agent to play the role.
    """
    if agents is None:
        agents = []
    select_speaker_message_template = (
               "You are in a role play game. The following roles are available:"
                "{roles}."
                "Read the following conversation."
                "Then select the next role from {agentlist} to play. Only return the role name."
                f"Important: This is the user prompt: {prompt}"
                "If you think that the user request is answered, return empty string as the role name."
    )
    from autogen import GroupChat
    main_group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        allow_repeat_speaker=True, # Allow the same agent to speak in consecutive rounds.
        send_introductions=True, # Make agents aware of each other.
        speaker_selection_method="auto", # LLM decides which agent to call next.
        select_speaker_prompt_template=None, # This was adding new system prompt at the end, and was causing instruction to be dropped in h2ogpt/convert_messages_to_structure method
        select_speaker_message_template=select_speaker_message_template,
    )

    def main_terminate_flow(msg):
        # Terminate the chat if the message contains 'TERMINATE' or is empty.
        return 'TERMINATE' in msg['content'] or msg['content']==""
    
    from openai_server.autogen_utils import H2OGroupChatManager
    main_group_chat_manager = H2OGroupChatManager(
        groupchat=main_group_chat,
        llm_config=llm_config,
        is_termination_msg=main_terminate_flow,
        name="main_group_chat_manager",
    )
    return main_group_chat_manager
