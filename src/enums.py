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
    wizard_vicuna = 19
    openai = 20
    openai_chat = 21
    gptj = 22
    prompt_answer_openllama = 23
    vicuna11 = 24
    mptinstruct = 25
    mptchat = 26
    falcon = 27
    guanaco = 28
    llama2 = 29
    beluga = 30
    wizard3nospace = 31
    one_shot = 32
    falcon_chat = 33
    mistral = 34
    zephyr = 35
    xwin = 36
    mistrallite = 37


class DocumentSubset(Enum):
    Relevant = 0
    RelSources = 1
    TopKSources = 2


non_query_commands = [
    DocumentSubset.RelSources.name,
    DocumentSubset.TopKSources.name
]


class DocumentChoice(Enum):
    ALL = 'All'


class LangChainMode(Enum):
    """LangChain mode"""

    DISABLED = "Disabled"
    LLM = "LLM"
    WIKI = "wiki"
    WIKI_FULL = "wiki_full"
    USER_DATA = "UserData"
    MY_DATA = "MyData"
    GITHUB_H2OGPT = "github h2oGPT"
    H2O_DAI_DOCS = "DriverlessAI docs"


class LangChainTypes(Enum):
    SHARED = 'shared'
    PERSONAL = 'personal'
    EITHER = 'either'  # used when user did not pass which one, so need to try both


# modes should not be removed from visible list or added by name
langchain_modes_intrinsic = [LangChainMode.DISABLED.value,
                             LangChainMode.LLM.value,
                             LangChainMode.MY_DATA.value]

langchain_modes_non_db = [LangChainMode.DISABLED.value,
                          LangChainMode.LLM.value]


class LangChainAction(Enum):
    """LangChain action"""

    QUERY = "Query"
    # WIP:
    # SUMMARIZE_MAP = "Summarize_map_reduce"
    SUMMARIZE_MAP = "Summarize"
    SUMMARIZE_ALL = "Summarize_all"
    SUMMARIZE_REFINE = "Summarize_refine"
    EXTRACT = "Extract"


class LangChainAgent(Enum):
    """LangChain agents"""

    SEARCH = "Search"
    COLLECTION = "Collection"
    PYTHON = "Python"
    CSV = "CSV"
    PANDAS = "Pandas"
    JSON = 'JSON'
    SMART = 'SMART'


no_server_str = no_lora_str = no_model_str = '[None/Remove]'

# from site-packages/langchain/llms/openai.py
# but needed since ChatOpenAI doesn't have this information
model_token_mapping = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,  # supports function tools
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,  # supports function tools
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,  # supports function tools
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,  # supports function tools
    "gpt-3.5-turbo-instruct": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}

openai_supports_functiontools = ["gpt-4-0613", "gpt-4-32k-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"]


def does_support_functiontools(inference_server, model_name):
    if any([inference_server.startswith(x) for x in ['openai_azure', 'openai_azure_chat']]):
        return model_name.lower() in openai_supports_functiontools
    elif any([inference_server.startswith(x) for x in ['openai', 'openai_chat']]):
        # assume OpenAI serves updated models
        return True
    else:
        return False


font_size = 2
head_acc = 40  # 40 for 6-way
source_prefix = "Sources [Score | Link]:"
source_postfix = "End Sources<p>"

super_source_prefix = f"""<details><summary><font size="{font_size}">Sources</font></summary><font size="{font_size}"><font size="{font_size}">Sources [Score | Link]:"""
super_source_postfix = f"""End Sources<p></font></font></details>"""


def t5_type(model_name):
    return 't5' == model_name.lower() or \
        't5-' in model_name.lower() or \
        'flan-' in model_name.lower() or \
        'fastchat-t5' in model_name.lower()


def get_langchain_prompts(pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary,
                          model_name, inference_server, model_path_llama):
    if model_name and ('falcon' in model_name or
                       'Llama-2'.lower() in model_name.lower() or
                       model_path_llama and 'llama-2' in model_path_llama.lower()) or \
            model_name in [None, '']:
        # use when no model, like no --base_model
        pre_prompt_query1 = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends.\n"
        prompt_query1 = "According to only the information in the document sources provided within the context above, "
    elif inference_server and inference_server.startswith('openai'):
        pre_prompt_query1 = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends.  If the answer cannot be primarily obtained from information within the context, then respond that the answer does not appear in the context of the documents.\n"
        prompt_query1 = "According to (primarily) the information in the document sources provided within context above, "
    else:
        pre_prompt_query1 = ""
        prompt_query1 = ""

    pre_prompt_summary1 = """In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text\n"""
    prompt_summary1 = "Using only the information in the document sources above, write a condensed and concise summary of key results (preferably as bullet points):\n"

    if pre_prompt_query is None:
        pre_prompt_query = pre_prompt_query1
    if prompt_query is None:
        prompt_query = prompt_query1
    if pre_prompt_summary is None:
        pre_prompt_summary = pre_prompt_summary1
    if prompt_summary is None:
        prompt_summary = prompt_summary1

    return pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary


def gr_to_lg(image_loaders,
             pdf_loaders,
             url_loaders,
             **kwargs,
             ):
    if image_loaders is None:
        image_loaders = kwargs['image_loaders_options0']
    if pdf_loaders is None:
        pdf_loaders = kwargs['pdf_loaders_options0']
    if url_loaders is None:
        url_loaders = kwargs['url_loaders_options0']
    # translate:
    # 'auto' wouldn't be used here
    ret = dict(
        # urls
        use_unstructured='Unstructured' in url_loaders,
        use_playwright='PlayWright' in url_loaders,
        use_selenium='Selenium' in url_loaders,

        # pdfs
        use_pymupdf='on' if 'PyMuPDF' in pdf_loaders else 'off',
        use_unstructured_pdf='on' if 'Unstructured' in pdf_loaders else 'off',
        use_pypdf='on' if 'PyPDF' in pdf_loaders else 'off',
        enable_pdf_ocr='on' if 'OCR' in pdf_loaders else 'off',
        enable_pdf_doctr='on' if 'DocTR' in pdf_loaders else 'off',
        try_pdf_as_html='on' if 'TryHTML' in pdf_loaders else 'off',

        # images
        enable_ocr='OCR' in image_loaders,
        enable_doctr='DocTR' in image_loaders,
        enable_pix2struct='Pix2Struct' in image_loaders,
        enable_captions='Caption' in image_loaders or 'CaptionBlip2' in image_loaders,
    )
    if 'CaptionBlip2' in image_loaders:
        # just override, don't actually do both even if user chose both
        captions_model = "Salesforce/blip2-flan-t5-xl"
    else:
        captions_model = kwargs['captions_model']
    return ret, captions_model


invalid_key_msg = 'Invalid Access Key, request access key from sales@h2o.ai or jon.mckinney@h2o.ai'

docs_ordering_types = ['best_first', 'best_near_prompt', 'reverse_ucurve_sort']

docs_token_handlings = ['chunk', 'split_or_merge']

docs_ordering_types_default = 'reverse_ucurve_sort'
docs_token_handling_default = 'split_or_merge'
docs_joiner_default = '\n\n'

db_types = ['chroma', 'weaviate']
db_types_full = ['chroma', 'weaviate', 'faiss']
