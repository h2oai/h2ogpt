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
        is_termination_msg=terminate_message_func,
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
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
        "* making jokes, writing stories or summaries, "
        "* having daily conversations. "
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
                "You are in a role play game. The following roles are available:"
                "{roles}\n"
                "Select the next role from {agentlist} to play. Only return the role name."
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
        role_for_select_speaker_messages="user", # to have select_speaker_prompt_template at the end of the messages
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

def get_tabular_ml_agent(
    llm_config:dict,
    code_writer_system_prompt:str | None = None,
    autogen_max_consecutive_auto_reply:int = 1,
):
    tabular_ml_agent = get_code_writer_agent(
        llm_config=llm_config,
        code_writer_system_prompt=code_writer_system_prompt,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    tabular_ml_agent._name = "tabular_ml_agent"
    system_message = tabular_ml_agent._oai_system_message[0]["content"]
    system_message += (
        "\nImportant Addition: "
        "* Your main expertise is running ML models on top of tabular data. "
        "* You can run ML models and provide insights on them. "
        "* You are good at first creating a base prediction model and "
        "based on the model results, you can apply new Data Science techniques "
        "to improve the model's performance iteratively. "
        "Quick insights step:\n"
        "* You always start with reading first few rows of data, checking column statistics, "
        "* deciding if this is a time dependent dataset or not. "
        "Modeling type decision:\n"
        "* If it's a time dependent dataset, you have to use time series techniques, like "
        "rolling window, expanding window, to generate new features or validation setup. "
        "* If it's not a time dependent dataset, you can use cross validation techniques "
        "to validate your model and generate new features. "
        "Data preprocessing:\n"
        "* Next step, you do some data preprocessing, like handling missing values, "
        "handling categorical variables, scaling, etc. "
        "Metric selection:\n"
        "* You always decide which metric to monitor."
        "* If it's a classification problem, you should check target_column distribution to see if there is imbalance."
        "* In case of imbalance, pick AUC metric over Accuracy. Or consider other classification metrics and make a reasonable selection."
        "* If it's a regression problem, consider metrics like RMSE, MAE, MSE, etc. and make an educated selection."
        "Base model creation:\n"
        "* After deciding your setup, always start with Linear Regression or Logistic Regression "
        "to get a baseline model result. "
        "Improvement steps:\n"
        "* Starting from there, you can try more complex Data Science techniques "
        "like better feature engineering, feature selection, complex models, better model parameters, etc. "
        "* You are allowed to try only the following models: "
        "LightGBM, XGBoost, CatBoost, Linear Regression, Logistic Regression. "
        "* Never call GridSearchCV or RandomizedSearchCV for model tuning, it takes too much time. "
        "* Try to tune parameters manually based on educated guesses. Like, deeper trees, more trees, etc. "
        "* After deciding your model, always check feature importance to see which features are important. "
        "* Based on feature importance, you can decide which features to keep or remove, or "
        "you can create new features based on the important features and the domain knowledge of the problem/dataset. "
        "* You always avoid overfitting, and make models that generalize well on unseen data. "
        "* So you have following ideas to try to improve the model: "
        "Model Tuning, Model Selection, Feature Engineering, Feature Selection, "
        "New Feature Creation, Data Augmentation, Data Cleaning, and Data Preprocessing. "
        "* At each iteration, you stick to the same validation setup and same metric to monitor, "
        "so that it's easy to compare the model performance. "
        "* Run your iterations until the model performance hits a plateau and "
        "you are out of ideas to improve the model. "
        "* As a final idea, you can try model ensemble techniques to improve the model performance. "
        "You can ensemble models that have the same exact validation setup. "
        "* At the end of each iteration, try to report model performance and feature importance plot if possible. "
        "* When you create new features, you have to save your new augmented datasets and reuse these datasets in next iterations. "
        "* Try to create plots to display variable importance and explain why some features come at top based on domain. "
        "* You can not run models on GPU. "
        f"* Important, you can run only {autogen_max_consecutive_auto_reply} iterations, so make your plans wisely. "
        "* Important: At the last iteration, you always create <final_model.py> file which contains the best final model code. "
        "* Important: If you realize that the model is not improving for 2 consecutive iterations, "
        "you have to stop the iterations and report the final model with the MARK: # execution: false  "
        # "* Don't forget to expand your base code at each iteration because there is no code execution memory right now."
        # "You have to redefine some variables if needed."
        )
    tabular_ml_agent._oai_system_message = [{"content": system_message, "role": "system"}]
    tabular_ml_agent._is_termination_msg = terminate_message_func
    return tabular_ml_agent


def get_tabular_ml_group_chat_manager(
        llm_config:dict,
        executor,
        code_writer_system_prompt:str | None = None,
        autogen_max_consecutive_auto_reply:int = 1,
        max_round:int = 10,
):
    tabular_ml_agent = get_tabular_ml_agent(
        llm_config=llm_config,
        code_writer_system_prompt=code_writer_system_prompt,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    code_executor_agent = get_code_execution_agent(
        executor=executor,
        autogen_max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )
    def group_terminate_flow(msg):
        # Terminate the chat if the message contains 'TERMINATE' or is empty.
        return 'TERMINATE' in msg['content'] or msg['content']==""
    
    from autogen import GroupChat
    tabular_ml_group_chat = GroupChat(
        agents=[tabular_ml_agent, code_executor_agent],
        messages=[],
        max_round=max_round,
        speaker_selection_method="round_robin" # call in order as defined in agents
    )
    from openai_server.autogen_utils import H2OGroupChatManager
    tabular_ml_group_chat_manager = H2OGroupChatManager(
        groupchat=tabular_ml_group_chat,
        llm_config=llm_config,
        is_termination_msg=group_terminate_flow,
        name="tabular_ml_group_chat_manager",
    )
    tabular_ml_group_chat_manager.description = (
        "This agent is an expert in running ML models on tabular data. "
        "It can provide insights on the models and help improve their performance. "
        "The agent is skilled at creating a base prediction model, "
        "applying new Data Science techniques to enhance model performance, "
        "and iterating on the model to achieve better results. "
        "It can handle time-dependent datasets using time series techniques "
        "like rolling window and expanding window, and non-time-dependent datasets "
        "using cross-validation techniques. "
        "The agent can monitor metrics, handle classification and regression problems, "
        "and select appropriate metrics based on the problem type. "
        "It can try various models and tune parameters manually or using GridSearch. "
        "The agent can create plots to display variable importance and explain feature rankings. "
        "It can save augmented datasets and reuse them in subsequent iterations. "
        "This agent is ideal for tasks involving ML models, feature engineering, "
        "model tuning, and performance improvement. "
        "This agent has to be picked for instructions that involve running ML models on tabular data, "
        "or doing predictions/forecasting for classification or regression problems. "
    )
    return tabular_ml_group_chat_manager