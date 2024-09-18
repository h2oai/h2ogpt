from openai_server.autogen_utils import terminate_message_func
from openai_server.agent_utils import current_datetime


def get_code_execution_agent(
        executor,
        autogen_max_consecutive_auto_reply,
        ):
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
    code_executor_agent.description = (
        "This agent is able to execute code snippets and scripts. "
        "This agent has to be picked whenever there is a code block coming from "
        "other agents with the mark # execution: true. "
        "For code blocks having the mark # execution: false, "
        "this agent will not be called. "
        )
    return code_executor_agent

def get_code_writer_agent(
        llm_config:dict,
        code_writer_system_prompt:str | None = None,
        autogen_max_consecutive_auto_reply:int = 1,
        ):
    
    code_writer_system_prompt += (
        "\n\n"
        "* Important: If you have past conversation context, "
        "and if you see code execution errors, "
        "you should focus on fixing the errors in the code. "
        "* Important: If you realize you run into a lot of similar "
        "errors for a certain approach, then instead of trying to fix the errors, "
        "you should try a different approach. "
        "* Imoprtant: You have to make sure that code blocks that "
        "are supposed to be executed are marked with # execution: true. "
        "And the code blocks that are not supposed to be executed are marked with # execution: false. "
        "* Important: # execution mark has to be placed at the beginning of the code block, right "
        "after the # filename: <filename> mark. "
    )
    from openai_server.autogen_utils import H2OConversableAgent
    code_writer_agent = H2OConversableAgent(
        "code_writer_agent",
        system_message=code_writer_system_prompt,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        is_termination_msg=terminate_message_func,
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    code_writer_agent.description = (
        "This agent excels at solving tasks through code generation, "
        "using both Python and shell scripts. "
        "It can handle anything from complex computations and data processing to "
        "generating executable code. "
        "Additionally, it can access the web to fetch real-time data, "
        "making it ideal for tasks that require automation, coding, or retrieving up-to-date information. "
        "This agent has to be picked for any coding related task or tasks that are "
        "more complex than just chatting or simple question answering. "
        "It can do math and calculations, from simple arithmetic to complex equations. "
        "It can verify the correctness of an answer via coding. "
        "This agent has to be picked for instructions that involves coding, "
        "math or simple calculation operations, solving complex tasks. "
        )
    return code_writer_agent

def get_chat_agent(
    llm_config:dict,
    autogen_max_consecutive_auto_reply:int = 1,
):
    from openai_server.autogen_utils import H2OConversableAgent
    system_message = (
        f"{current_datetime()}\n"
        "You answer the question or request provided with natural language only. "
        "You can not generate or execute codes. "
        "You can not talk to web. "
        "You can not do any math or calculations, "
        "even simple ones like adding numbers. "
        "You are good at chatting. "
        "You are good at answering general knowledge questions "
        "based on your own memory or past conversation context. "
        "You are only good at words. "
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
        "This agent is able to convey daily and casual chats "
        "based on its own memory or past conversation context. "
        "Only answers with natural language. "
        "It can not execute codes. "
        "It can not generate code examples. "
        "It can not access the web. "
        "It can not do any math or calculations, "
        "even simple ones like adding numbers, "
        "or counting things. "
        "It's only good at chatting and answering simple tasks like: "
        "* making jokes and answering daily conversations. "
        "It has no clue about counts, measurements, or calculations. "
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
    code_executor_agent = get_code_execution_agent(
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
        "It can do math and calculations, from simple arithmetic to complex equations. "
        "It can verify the correctness of an answer via coding. "
        "This agent has to be picked for instructions that involves coding, "
        "math or simple calculation operations, solving complex tasks. "
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
    # TODO: override _process_speaker_selection_result logic to return None
    # as the selected next speaker if it's empty string.
    select_speaker_message_template = (
                "You are in a role play game and your task is to select the next role. "
                "The following roles are available:"
                "{roles}\n"
                "Select the next role from {agentlist} to play. Only return the role name."
        )
    agent_roles_as_text = "\n".join([f"{agent.name}: {agent.description}" for agent in agents])
    select_speaker_prompt_template = (
        "You are in a role play game. The following roles are available:"
        f"{agent_roles_as_text}\n"
        "Read the above conversation and "
        "then select the next role from {agentlist} to play. Only return the role."
    )
    # extra instruction for the agent selection
    select_speaker_prompt_template += (
        f"This is the user prompt: {prompt}. "
        "* If you think that there is enough information gathered to answer the prompt, "
        "end the chat by calling 'terminate_agent'."
        "* If there are code blocks with # execution: true in the last message, "
        "you should never call 'terminate_agent'. "
        "Instead, you should call code_executor_agent to execute the code. "
    )
    from autogen import GroupChat
    main_group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        allow_repeat_speaker=True, # Allow the same agent to speak in consecutive rounds.
        send_introductions=True, # Make agents aware of each other.
        speaker_selection_method="auto", # LLM decides which agent to call next.
        select_speaker_message_template=select_speaker_message_template,
        select_speaker_prompt_template=select_speaker_prompt_template,
        role_for_select_speaker_messages="user", # to have select_speaker_prompt_template at the end of the messages
    )

    def main_terminate_flow(msg):
        # Terminate the chat if the message contains 'TERMINATE' or is empty.
        terminate_flow = False
        if msg['name'] == "code_writer_agent" and 'TERMINATE' in msg['content']:
            print("main_group_chat_manager: code_writer_agent tells to terminate")
            terminate_flow = True
        if msg['name'] == 'terminate_agent':
            print("main_group_chat_manager: terminate_agent called")
            terminate_flow = True
        return terminate_flow

    from openai_server.autogen_utils import H2OGroupChatManager
    main_group_chat_manager = H2OGroupChatManager(
        groupchat=main_group_chat,
        llm_config=llm_config,
        is_termination_msg=main_terminate_flow,
        name="main_group_chat_manager",
    )
    return main_group_chat_manager

def get_terminate_agent():
    from openai_server.autogen_utils import H2OConversableAgent
    terminate_agent = H2OConversableAgent(
        name="terminate_agent",
        system_message="You are the terminator agent. You should end the conversation.",
        llm_config=None,
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=terminate_message_func,
        max_consecutive_auto_reply=1,
    )
    terminate_agent.description = (
        "This agent is the terminator agent. "
        "It should be called to end the conversation when the user prompt is answered. "
        )
    return terminate_agent
