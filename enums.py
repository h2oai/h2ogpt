from enum import Enum


class PromptType(Enum):
    custom = -1
    plain = 0
    instruct = 1
    quality = 2
    human_bot = 3
    dai_faq = 4
    summarize = 5
    simple_instruct = 6
    instruct_vicuna = 7
    instruct_with_end = 8
    human_bot_orig = 9
    prompt_answer = 10
    open_assistant = 11
    wizard_lm = 12
    wizard_mega = 13
    instruct_vicuna2 = 14
    instruct_vicuna3 = 15
    wizard2 = 16
    wizard3 = 17
    instruct_simple = 18


class DocumentChoices(Enum):
    All_Relevant = 0
    All_Relevant_Only_Sources = 1
    Only_All_Sources = 2
    Just_LLM = 3


class LangChainMode(Enum):
    """LangChain mode"""

    DISABLED = "Disabled"
    CHAT_LLM = "ChatLLM"
    LLM = "LLM"
    ALL = "All"
    WIKI = "wiki"
    WIKI_FULL = "wiki_full"
    USER_DATA = "UserData"
    MY_DATA = "MyData"
    GITHUB_H2OGPT = "github h2oGPT"
    H2O_DAI_DOCS = "DriverlessAI docs"
