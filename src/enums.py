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
    aquila = 38
    aquila_simple = 39
    aquila_legacy = 40
    aquila_v1 = 41
    mistralgerman = 42
    deepseek_coder = 43
    open_chat = 44
    open_chat_correct = 45
    open_chat_code = 46
    anthropic = 47
    orca2 = 48
    jais = 49
    yi = 50
    xwincoder = 51
    xwinmath = 52
    vicuna11nosys = 53
    zephyr0 = 54
    google = 55
    docsgpt = 56
    open_chat_math = 57
    mistralai = 58
    mixtral = 59
    mixtralnosys = 60
    orion = 61
    sciphi = 62
    beacon = 63
    beacon2 = 64


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
    IMAGE_GENERATE = "ImageGen"
    IMAGE_GENERATE_HIGH = "ImageGenHigh"
    IMAGE_CHANGE = "ImageChange"
    IMAGE_QUERY = "ImageQuery"


# rest are not implemented fully
base_langchain_actions = [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value,
                          LangChainAction.EXTRACT.value,
                          LangChainAction.IMAGE_GENERATE.value,
                          LangChainAction.IMAGE_GENERATE_HIGH.value,
                          LangChainAction.IMAGE_CHANGE.value,
                          LangChainAction.IMAGE_QUERY.value,
                          ]


class LangChainAgent(Enum):
    """LangChain agents"""

    SEARCH = "Search"
    COLLECTION = "Collection"
    PYTHON = "Python"
    CSV = "CSV"
    PANDAS = "Pandas"
    JSON = 'JSON'
    SMART = 'SMART'
    AUTOGPT = 'AUTOGPT'


no_server_str = no_lora_str = no_model_str = '[None/Remove]'

# from site-packages/langchain/llms/openai.py
# but needed since ChatOpenAI doesn't have this information
gpt_token_mapping = {
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
    "gpt-4-1106-preview": 128000,  # 4096 output
    "gpt-35-turbo-1106": 16385,  # 4096 output
}
model_token_mapping = gpt_token_mapping.copy()
model_token_mapping.update({
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
})

anthropic_mapping = {
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000
}

anthropic_mapping_outputs = {
    "claude-2.1": 4096,
    "claude-2": 4096,
    "claude-2.0": 4096,
    "claude-instant-1.2": 4096,
}

google_mapping = {
    "gemini-pro": 32768,
    "gemini-pro-vision": 32768,
}

# FIXME: at least via current API:
google_mapping_outputs = {
    "gemini-pro": 8192,
    "gemini-pro-vision": 2048,
}

mistralai_mapping = {
    "mistral-medium": 32768,
    "mistral-small": 32768,
    "mistral-tiny": 32768,
}

mistralai_mapping_outputs = {
    "mistral-medium": 32768,
    "mistral-small": 32768,
    "mistral-tiny": 32768,
}

openai_supports_functiontools = ["gpt-4-0613", "gpt-4-32k-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
                                 "gpt-4-1106-preview", "gpt-35-turbo-1106"]

openai_supports_json_mode = ["gpt-4-1106-preview", "gpt-35-turbo-1106"]

# https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability
model_token_mapping_outputs = model_token_mapping.copy()
model_token_mapping_outputs.update({"gpt-4-1106-preview": 4096, "gpt-35-turbo-1106": 4096})


def does_support_functiontools(inference_server, model_name):
    if any([inference_server.startswith(x) for x in ['openai_azure', 'openai_azure_chat']]):
        return model_name.lower() in openai_supports_functiontools
    elif any([inference_server.startswith(x) for x in ['openai', 'openai_chat']]):
        # assume OpenAI serves updated models
        return True
    else:
        return False


def does_support_json_mode(inference_server, model_name):
    if any([inference_server.startswith(x) for x in ['openai_azure', 'openai_azure_chat']]):
        return model_name.lower() in openai_supports_json_mode
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

generic_prefix = f"""<details><summary><font size="""
generic_postfix = f"""</font></details>"""


def t5_type(model_name):
    return 't5' == model_name.lower() or \
        't5-' in model_name.lower() or \
        'flan-' in model_name.lower() or \
        'fastchat-t5' in model_name.lower()


def get_langchain_prompts(pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary, hyde_llm_prompt,
                          model_name, inference_server, model_path_llama,
                          doc_json_mode,
                          prompt_query_type='simple'):
    if prompt_query_type == 'advanced':
        pre_prompt_query1 = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends.  If the answer cannot be primarily obtained from information within the context, then respond that the answer does not appear in the context of the documents."
        prompt_query1 = "According to (primarily) the information in the document sources provided within context above, write an insightful and well-structured response to: "
    else:
        # older smaller models get confused by this prompt, should use "" instead, but not focusing on such old models anymore, complicates code too much
        pre_prompt_query1 = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends."
        prompt_query1 = "According to only the information in the document sources provided within the context above, write an insightful and well-structured response to: "

    pre_prompt_summary1 = """In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text."""
    prompt_summary1 = "Using only the information in the document sources above, write a condensed and concise summary of key results (preferably as bullet points)."

    hyde_llm_prompt1 = "Answer this question with vibrant details in order for some NLP embedding model to use that answer as better query than original question: "

    if pre_prompt_query is None:
        pre_prompt_query = pre_prompt_query1
    if prompt_query is None:
        prompt_query = prompt_query1
    if pre_prompt_summary is None:
        pre_prompt_summary = pre_prompt_summary1
    if prompt_summary is None:
        prompt_summary = prompt_summary1
    if hyde_llm_prompt is None:
        hyde_llm_prompt = hyde_llm_prompt1

    return pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary, hyde_llm_prompt


def gr_to_lg(image_audio_loaders,
             pdf_loaders,
             url_loaders,
             use_pymupdf=None,
             use_unstructured_pdf=None,
             use_pypdf=None,
             enable_pdf_ocr=None,
             enable_pdf_doctr=None,
             try_pdf_as_html=None,
             **kwargs,
             ):
    assert use_pymupdf is not None
    assert use_unstructured_pdf is not None
    assert use_pypdf is not None
    assert enable_pdf_ocr is not None
    assert enable_pdf_doctr is not None
    assert try_pdf_as_html is not None

    if image_audio_loaders is None:
        image_audio_loaders = kwargs['image_audio_loaders_options0']
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
        use_scrapeplaywright='ScrapeWithPlayWright' in url_loaders,
        use_scrapehttp='ScrapeWithHttp' in url_loaders,

        # pdfs
        # ... else condition uses default from command line, by default auto, so others can be used as backup
        # make sure pass 'off' for those if really want fully disabled.
        use_pymupdf='on' if 'PyMuPDF' in pdf_loaders else use_pymupdf,
        use_unstructured_pdf='on' if 'Unstructured' in pdf_loaders else use_unstructured_pdf,
        use_pypdf='on' if 'PyPDF' in pdf_loaders else use_pypdf,
        enable_pdf_ocr='on' if 'OCR' in pdf_loaders else enable_pdf_ocr,
        enable_pdf_doctr='on' if 'DocTR' in pdf_loaders else enable_pdf_doctr,
        try_pdf_as_html='on' if 'TryHTML' in pdf_loaders else try_pdf_as_html,

        # images and audio
        enable_ocr='OCR' in image_audio_loaders,
        enable_doctr='DocTR' in image_audio_loaders,
        enable_pix2struct='Pix2Struct' in image_audio_loaders,
        enable_captions='Caption' in image_audio_loaders or 'CaptionBlip2' in image_audio_loaders,
        enable_transcriptions="ASR" in image_audio_loaders or 'ASRLarge' in image_audio_loaders,
        enable_llava='LLaVa' in image_audio_loaders,
    )
    if 'CaptionBlip2' in image_audio_loaders:
        # just override, don't actually do both even if user chose both
        captions_model = "Salesforce/blip2-flan-t5-xl"
    else:
        captions_model = kwargs['captions_model']
    if 'ASRLarge' in image_audio_loaders:
        # just override, don't actually do both even if user chose both
        asr_model = "openai/whisper-large-v3"
    else:
        asr_model = kwargs['asr_model']
    return ret, captions_model, asr_model


invalid_key_msg = 'Invalid Access Key, request access key from sales@h2o.ai or jon.mckinney@h2o.ai, pass API key through API calls, or set API key in Login tab for UI'

docs_ordering_types = ['best_first', 'best_near_prompt', 'reverse_ucurve_sort']

docs_token_handlings = ['chunk', 'split_or_merge']

docs_ordering_types_default = 'reverse_ucurve_sort'
docs_token_handling_default = 'split_or_merge'
docs_joiner_default = '\n\n'

db_types = ['chroma', 'weaviate']
db_types_full = ['chroma', 'weaviate', 'faiss']

auto_choices = [None, 'None', 'auto']

doc_json_mode_system_prompt0 = """You are a language model who produces high-quality valid JSON extracted from documents in order to answer a user's question.  For example, according to the documents given in JSON (with keys document and content) below:

{"document": 45, "content": "Joe Biden is an American politician who is the 46th and current president of the United States. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama and represented Delaware in the United States Senate from 1973 to 2009.

Biden was born on November 20, 1942, in Scranton, Pennsylvania, and grew up in Wilmington, Delaware. He earned a bachelor's degree from the University of Delaware and a law degree from Syracuse University College of Law. Before entering politics, Biden worked as a lawyer and served on the Senate staff.

Biden was first elected to the Senate in 1972, at the age of 29, and became one of the youngest people to be elected to the Senate. He served in the Senate for six terms, chairing the Senate Foreign Relations Committee and the Senate Judiciary Committee. In 2008, he was chosen by Barack Obama as his running mate in the presidential election, and they won the election. As vice president, Biden focused on issues related to foreign policy, national security, and the economy.

In 2015, Biden announced that he would not run for president in the 2016 election, but he remained a prominent figure in the Democratic Party. In 2019, he announced his candidacy for the 2020 presidential election, and he won the Democratic primary in June 2020. In the general election, he defeated incumbent President Donald Trump and became the oldest person to be elected president, at the age of 78.

Biden's presidency has focused on issues such as COVID-19 pandemic response, economic recovery, climate change, and social justice. He has also taken steps to address the COVID-19 pandemic, including implementing policies to slow the spread of the virus and providing economic relief to those affected by the pandemic.

Throughout his career, Biden has been known for his progressive policies and his ability to work across the aisle to find bipartisan solutions. He has also been a strong advocate for LGBTQ+ rights, immigration reform, and criminal justice reform. Despite his long political career, Biden has faced criticism for his moderate stance on certain issues and his perceived lack of progressive credentials. Nevertheless, he remains a significant figure in American politics and a leader in the Democratic Party."}

{"document": 56, "content": "How to cook chicken. There are many ways to cook chicken, depending on your personal preferences and the ingredients you have available. Here are a few methods:

1. Grilled Chicken: Preheat your grill to medium-high heat. Season the chicken with your desired seasonings, such as salt, pepper, and your favorite herbs or spices. Place the chicken on the grill and cook for 5-7 minutes per side, or until the internal temperature reaches 165°F (74°C).
2. Baked Chicken: Preheat your oven to 400°F (200°C). Season the chicken with your desired seasonings, then place it in a baking dish. Bake for 20-25 minutes, or until the internal temperature reaches 165°F (74°C).
3. Pan-Seared Chicken: Heat a pan over medium-high heat. Add a small amount of oil, then add the chicken. Cook for 5-7 minutes per side, or until the internal temperature reaches 165°F (74°C).
4. Slow Cooker Chicken: Place the chicken in a slow cooker and add your desired seasonings and sauces. Cook on low for 6-8 hours, or until the internal temperature reaches 165°F (74°C).
5. Instant Pot Chicken: Place the chicken in the Instant Pot and add your desired seasonings and sauces. Cook on high pressure for 10-15 minutes, or until the internal temperature reaches 165°F (74°C).
6. Poached Chicken: Bring a pot of water to a boil, then reduce the heat to a simmer. Add the chicken and cook for 10-15 minutes, or until the internal temperature reaches 165°F (74°C).
7. Smoked Chicken: Smoke the chicken over low heat for 4-6 hours, or until the internal temperature reaches 165°F (74°C).
8. Fried Chicken: Heat a pot of oil, such as peanut oil, to 350°F (175°C). Season the chicken with your desired seasonings, then add it to the oil. Fry for 5-7 minutes, or until the internal temperature reaches 165°F (74°C).
9. Pressure Cooker Chicken: Place the chicken in a pressure cooker and add your desired seasonings and sauces. Cook for 10-15 minutes, or until the internal temperature reaches 165°F (74°C).
10. Air Fryer Chicken: Place the chicken in an air fryer and cook at 400°F (200°C) for 10-15 minutes, or until the internal temperature reaches 165°F (74°C).

It's important to note that the cooking time and temperature may vary depending on the size and thickness of the chicken, as well as the specific cooking method used. Always use a food thermometer to ensure the chicken has reached a safe internal temperature."}

{"document": 78, "content": "Climate change impacts Europe. Climate change has significant impacts on Europe, and the continent is already experiencing some of the effects. Here are some of the ways climate change is affecting Europe:

1. Temperature increase: Europe has seen a rapid increase in temperature over the past century, with the average temperature rising by about 1.5°C. This warming is projected to continue, with average temperatures expected to rise by another 2-3°C by the end of the century if greenhouse gas emissions continue to rise.
2. Extreme weather events: Climate change is leading to more frequent and intense heatwaves, droughts, and heavy rainfall events in Europe. For example, the 2018 heatwave was one of the hottest on record, with temperatures reaching up to 45°C in some parts of the continent.
3. Sea-level rise: Rising sea levels are threatening coastal communities and infrastructure in Europe, particularly in low-lying areas such as the Netherlands, Belgium, and the UK.
4. Water scarcity: Climate change is altering precipitation patterns in Europe, leading to more frequent droughts in some regions, such as the Mediterranean. This can have significant impacts on agriculture, industry, and human consumption.
5. Impacts on agriculture: Climate change is affecting crop yields, fisheries, and livestock production in Europe. Warmer temperatures and changing precipitation patterns are altering the distribution of crops, and some regions are experiencing increased pest and disease pressure.
6. Health impacts: Climate change is increasing the spread of disease vectors such as ticks and mosquitoes, which can carry diseases such as Lyme disease and malaria. Heatwaves are also having significant health impacts, particularly for vulnerable populations such as the elderly and young children.
7. Economic impacts: Climate change is affecting various industries in Europe, including agriculture, forestry, and tourism. It is also affecting infrastructure, such as roads, bridges, and buildings, which are being damaged by more frequent extreme weather events.
8. Biodiversity loss: Climate change is altering ecosystems and leading to the loss of biodiversity in Europe. This can have cascading impacts on ecosystem services, such as pollination, pest control, and nutrient cycling.
9. Migration and displacement: Climate change is displacing people in Europe, particularly in coastal communities that are at risk of flooding and erosion. It is also contributing to migration, as people seek to escape the impacts of climate change in their home countries.
10. Political and social impacts: Climate change is creating political and social tensions in Europe, particularly around issues such as migration, border control, and resource allocation. It is also leading to increased activism and calls for climate action from civil society.

Overall, the impacts of climate change in Europe are far-reaching and have significant consequences for the environment, economy, and society. It is important for policymakers, businesses, and individuals to take urgent action to mitigate and adapt to climate change."}

You should answer the query using the following template :
{
  "question" : string, // The query given by user
  "success" : boolean, // Whether you could successfully answer the question using only the contents in documents provided.  Set to false if the content in the documents do not contain the information required to answer the question.
  "response" : string, // A detailed highly-accurate and well-structured response to the user's question.  Set to "No document contents are relevant to the query" if the content in the documents do not contain the information required to answer the question.
  "references" : array // The value of the document key that identifies the articles you used to answer the user question. Set to empty array if the content in the documents do not contain the information required to answer the question.
}

For example, if user gives question "Who is Joe Biden?", then you would respond back with: {"question": "Who is Joe Biden?", "success" : true, "response" : "Joe Biden is the 46th President of the United States, serving since 2020. He previously served as Vice President under Barack Obama from 2009 to 2017 and represented Delaware in the Senate from 1973 to 2009. Biden focused on foreign policy, national security, and the economy as Vice President. He ran for President in 2020 and won, defeating incumbent President Donald Trump. Biden's presidency has focused on COVID-19 pandemic response, economic recovery, climate change, and social justice. He's known for his progressive policies and ability to work across the aisle. Despite criticism for his moderate stance on some issues, he remains a significant figure in American politics.", "references" : [45]}
Or for example, if user gives question "Who do I cook pork?", then you would respond back with: {"question": "Who do I cook pork?", "success" : false, "response" : "I cannot answer that query.", "references" : []}

Ensure the question, success, and references are accurately and precisely determined, and check your work in step-by-step manner.  Always respond back in valid JSON following these examples.
"""

doc_json_mode_system_prompt = """You are a language model who produces high-quality valid JSON extracted from documents in order to answer a user's question.

You should answer the question using the following valid JSON template:
{
  "question" : string, // The query given by user
  "response" : string, // A detailed highly-accurate and well-structured response to the user's question.  Set to "No document contents are relevant to the query" if the content in the documents do not contain the information required to answer the question.
  "justification" : string, // A justification for the response according to the documents.  If the response appears to be unjustified, according to the documents, then say "none".
  "success" : boolean, // Given the question, response, and justification, decide if the retrieval from references ws used to obtain the answer. Only set to true if the response answers the question according to the documents.  Set to false if the response appears to be unjustified according to the documents.
  "ID references" : numeric array // ID for the single most relevant document that the justification mentioned and response answered according to the documents. Set to empty array if the answer is not contained within the documents.
  "accuracy" : integer, // Given the question, response, justification, references, and original document contents, give a score of 0 through 10 for how accurately the response answered the question accounting for how well it follows from the documents.  10 means the justification perfectly explains the response, is perfectly correct, is perfectly clear, and is according to the documents.  5 means the justification appears valid but may require verification.  0 means the justification does not match the response according to the documents.
}
Respond absolutely only in valid JSON with elaborate and well-structured text for the response and justification.
"""
# "Web references" : str array // Up to 3 most relevant HTML links used to justify the response.

max_input_tokens_public = 3100
max_input_tokens_public_api = 2 * max_input_tokens_public  # so can exercise bit longer context models

max_total_input_tokens_public = 4096 * 2
max_total_input_tokens_public_api = 2 * max_total_input_tokens_public

max_top_k_docs_public = 10
max_top_k_docs_public_api = 2 * max_top_k_docs_public

max_top_k_docs_default = 10

max_docs_public = 5
max_docs_public_api = 2 * max_docs_public

max_chunks_per_doc_public = 5000
max_chunks_per_doc_public_api = 2 * max_chunks_per_doc_public

user_prompt_for_fake_system_prompt = "Who are you and what do you do?"
