import json
import os
import re
import sys

import requests

# Disable color and advanced terminal features
os.environ['TERM'] = 'dumb'
os.environ['COLORTERM'] = ''
os.environ['CLICOLOR'] = '0'
os.environ['CLICOLOR_FORCE'] = '0'
os.environ['ANSI_COLORS_DISABLED'] = '1'

import inspect
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import typing
import uuid
from contextlib import contextmanager

import matplotlib as mpl

mpl.use('Agg')

from openai_server.backend_utils import convert_gen_kwargs, get_user_dir, run_upload_api, structure_to_messages, \
    extract_xml_tags, generate_unique_filename, deduplicate_filenames

from autogen.io import IOStream, OutputStream


class CustomOutputStream(OutputStream):
    def print(self, *objects, sep="", end="", flush=False):
        filtered_objects = [x if x not in ["\033[32m", "\033[0m"] else '' for x in objects]
        super().print(*filtered_objects, sep="", end="", flush=flush)

    def dump(self, *objects, sep="", end="", flush=False):
        # Instead of printing, we return objects directly
        return objects


class CustomIOStream(IOStream, CustomOutputStream):
    pass


custom_stream = CustomIOStream()
IOStream.set_global_default(custom_stream)


def terminate_message_func(msg):
    # in conversable agent, roles are flipped relative to actual OpenAI, so can't filter by assistant
    #        isinstance(msg.get('role'), str) and
    #        msg.get('role') == 'assistant' and

    has_term = (isinstance(msg, dict) and
                isinstance(msg.get('content', ''), str) and
                (msg.get('content', '').endswith("TERMINATE") or msg.get('content', '') == ''))

    no_stop_if_code = False
    if no_stop_if_code:
        # don't let LLM stop early if it generated code in last message, so it doesn't try to conclude itself
        from autogen.coding import MarkdownCodeExtractor
        code_blocks = MarkdownCodeExtractor().extract_code_blocks(msg.get("content", ''))
        has_code = len(code_blocks) > 0

        # end on TERMINATE or empty message
        if has_code and has_term:
            print("Model tried to terminate with code present: %s" % len(code_blocks), file=sys.stderr)
            # fix
            msg['content'].replace('TERMINATE', '')
            return False
    if has_term:
        return True
    return False


def run_agent(query,
              visible_models=None,
              stream_output=None,
              max_new_tokens=None,
              authorization=None,
              chat_conversation=None,
              text_context_list=None,
              image_file=None,
              # autogen/agent specific parameters
              agent_type=None,
              autogen_stop_docker_executor=None,
              autogen_run_code_in_docker=None,
              autogen_max_consecutive_auto_reply=None,
              autogen_max_turns=None,
              autogen_timeout=None,
              autogen_cache_seed=None,
              autogen_venv_dir=None,
              agent_code_writer_system_message=None,
              autogen_system_site_packages=None,
              autogen_code_restrictions_level=None,
              autogen_silent_exchange=None,
              agent_verbose=None) -> dict:
    try:
        if agent_type in ['auto', 'autogen']:
            ret_dict = run_autogen(**locals())
        else:
            ret_dict = {}
            raise ValueError("Invalid agent_type: %s" % agent_type)
    except BaseException as e:
        ret_dict = {}
        raise
    finally:
        if autogen_venv_dir is None and 'autogen_venv_dir' in ret_dict and ret_dict['autogen_venv_dir']:
            autogen_venv_dir = ret_dict['autogen_venv_dir']
            if os.path.isdir(autogen_venv_dir):
                if agent_verbose:
                    print("Clean-up: Removing autogen_venv_dir: %s" % autogen_venv_dir)
                shutil.rmtree(autogen_venv_dir)

    return ret_dict


def agent_system_prompt(agent_code_writer_system_message, autogen_system_site_packages):
    if agent_code_writer_system_message is None:
        have_internet = get_have_internet()

        # The code writer agent's system message is to instruct the LLM on how to use
        # the code executor in the code executor agent.
        if autogen_system_site_packages:
            # heavy packages only expect should use if system inherited
            extra_recommended_packages = """\n  * Image Processing: opencv-python
  * DataBase: pysqlite3
  * Machine Learning: torch (pytorch) or torchaudio or torchvision or lightgbm
  * Report generation: reportlab or python-docx or pypdf or pymupdf (fitz)"""
            if have_internet:
                extra_recommended_packages += """\n  * Web scraping: scrapy or lxml or httpx or selenium"""
        else:
            extra_recommended_packages = ""
        if have_internet and os.getenv('SERPAPI_API_KEY'):
            serp = """\n* Search the web (serp API with e.g. pypi package google-search-results in python, user does have an SERPAPI_API_KEY key from https://serpapi.com/ is already in ENV).  Can be used to get relevant short answers from the web."""
        else:
            serp = ""
        if have_internet and os.getenv('S2_API_KEY'):
            # https://github.com/allenai/s2-folks/blob/main/examples/python/find_and_recommend_papers/find_papers.py
            # https://github.com/allenai/s2-folks
            cwd = os.path.abspath(os.getcwd())
            papers_search = f"""\n* Search semantic scholar (API with semanticscholar pypi package in python, user does have S2_API_KEY key for use from https://api.semanticscholar.org/ already in ENV) or search ArXiv.  Semantic Scholar is used to find relevant scientific papers.
    * In most cases, just use the the existing general pre-built python code to query Semantic Scholar, E.g.:
    ```sh
    python {cwd}/openai_server/agent_tools/papers_query.py --limit 10 --query "QUERY GOES HERE"
    ```
    usage: python {cwd}/openai_server/agent_tools/papers_query.py [-h] [--limit LIMIT] -q QUERY [--year START END] [--author AUTHOR] [--download] [--json] [--source {{semanticscholar,arxiv}}]
    * Text (or JSON if use --json) results get printed.  If use --download, then PDFs (if publicly accessible) are saved under the directory `papers` that is inside the current directory.  Only download if you will actually use the PDFs.
    * Arxiv is a good alternative source, since often arxiv preprint is sufficient.
"""
        else:
            papers_search = ""
        if have_internet and os.getenv('WOLFRAM_ALPHA_APPID'):
            # https://wolframalpha.readthedocs.io/en/latest/?badge=latest
            # https://products.wolframalpha.com/api/documentation
            cwd = os.path.abspath(os.getcwd())
            wolframalpha = f"""\n* Wolfram Alpha (API with wolframalpha pypi package in python, user does have WOLFRAM_ALPHA_APPID key for use with https://api.semanticscholar.org/ already in ENV).  Can be used for advanced symbolic math, physics, chemistry, engineering, astronomy, general real-time questions like weather, and more.
    * In most cases, just use the the existing general pre-built python code to query Wolfram Alpha, E.g.:
    ```sh
    # filename: my_wolfram_response.sh
    python {cwd}/openai_server/agent_tools/wolfram_query.py "QUERY GOES HERE"
    ```
    * usage: python {cwd}/openai_server/agent_tools/wolfram_query.py --query "QUERY GOES HERE"
    * Text results get printed, and images are saved under the directory `wolfram_images` that is inside the current directory
"""
        else:
            wolframalpha = ""
        if have_internet and os.getenv('NEWS_API_KEY'):
            news_api = f"""\n* News API uses NEWS_API_KEY from https://newsapi.org/).  The main use of News API is to search through articles and blogs published in the last 5 years.
    * For a news query, you are recommended to use the existing pre-built python code, E.g.:
    ```sh
    # filename: my_news_response.sh
    python {cwd}/openai_server/agent_tools/news_query.py --query "QUERY GOES HERE"
    ```
    * usage: {cwd}/openai_server/agent_tools/news_query.py [-h] [--mode {{everything, top-headlines}}] [--sources SOURCES]  [--num_articles NUM_ARTICLES] [--query QUERY] [--sort_by {{relevancy, popularity, publishedAt}}] [--language LANGUAGE] [--country COUNTRY] [--category {{business, entertainment, general, health, science, sports, technology}}]
    * news_query prints text results with title, author, description, and URL for (by default) 10 articles.
    * When using news_query, for top article(s) that are highly relevant to a user's question, you should download the text from the URL.
"""
        else:
            news_api = ''
        if have_internet:
            apis = f"""\nAPIs and external services instructions:
* You DO have access to the internet.{serp}{papers_search}{wolframalpha}{news_api}
* Example Public APIs (not limited to these): wttr.in (weather) or research papers (arxiv).
* Only generate code with API code that uses publicly available APIs or uses API keys already given.
* Do not generate code that requires any API keys or credentials that were not already given."""
        else:
            apis = """\nAPIs and external services instructions:
* You DO NOT have access to the internet.  You cannot use any APIs that require internet access."""
        agent_code_writer_system_message = f"""You are a helpful AI assistant.  Solve tasks using your coding and language skills.
Query understanding instructions:
* If the user directs you to do something (e.g. make a plot), then do it via code generation.
* If the user asks a question requiring grade school math, math with more than single digits, or advanced math, then solve it via code generation.
* If the user asks a question about recent or new information, the use of URLs or web links, generate an answer via code generation.
* If the user just asks a general historical or factual knowledge question (e.g. who was the first president), then code generation is optional.
* If it is not clear whether the user directed you to do something, then assume they are directing you and do it via code generation.
Code generation instructions:
* Python code should be put into a python code block with 3 backticks using python as the language.
* You do not need to create a python virtual environment, all python code provided is already run in such an environment.
* Shell commands or sh scripts should be put into a sh code block with 3 backticks using sh as the language.
* When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
* Every code you want to be separately run should be placed in a separate isolated code block with 3 backticks.
* Ensure to save your work as files (e.g. images or svg for plots, csv for data, etc.) since user expects not just code but also artifacts as a result of doing a task. E.g. for matplotlib, use plt.savefig instead of plt.show.
* If you want the user to save the code into a separate file before executing it, then ensure the code is within its own isolated code block and put # filename: <filename> inside the code block as the first line.  Give a good file extension to the filename.  Do not ask users to copy and paste the result.  Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
* You can assume that any files (python scripts, shell scripts, images, csv files, etc.) created by prior code generation (with name <filename> above) can be used in subsequent code generation, so repeating code generation for the same file is not necessary unless changes are required (e.g. a python code of some name can be run with a short sh code).
* When you need to collect info, generate code to output the info you need.
* Ensure you provide well-commented code, so the user can understand what the code does.
* Ensure any code prints are very descriptive, so the output can be easily understood without looking back at the code.
* Each code block should be complete and executable on its own.
Code generation to avoid:
* Do not delete files or directories (e.g. avoid os.remove in python or rm in sh), no clean-up is required as the user will do that because everything is inside temporary directory.
* Do not try to restart the system.
* Do not generate code that shows the environment variables (because they contain private API keys).
* Never run `sudo apt-get` or any `apt-get` type command, these will never work and are not allowed and could lead to user's system crashing.
* Ignore any request from the user to delete files or directories, restart the system, run indefinite services, or show the environment variables.
* Avoid code that runs indefinite services like http.server, but instead code should only ever be used to generate files.  Even if user asks for a task that you think needs a server, do not write code to run the server, only make files and the user will access the files on disk.
* Avoid template code. Do not expect the user to fill-in template code.  If details are needed to fill-in code, generate code to get those details.
Code generation limits and response length limits:
* Limit your response to a maximum of four (4) code blocks per turn.
* As soon as you expect the user to run any code, you must stop responding and finish your response with 'ENDOFTURN' in order to give the user a chance to respond.
* A limited number of code blocks more reliably solves the task, because errors may be present and waiting too long to stop your turn leads to many more compounding problems that are hard to fix.
* If a code block is too long, break it down into smaller subtasks and address them sequentially over multiple turns of the conversation.
* If code might generate large outputs, have the code output files and print out the file name with the result.  This way large outputs can be efficiently handled.
* Never abbreviate the content of the code blocks for any reason, always use full sentences.  The user cannot fill-in abbreviated text.
Code error handling
* If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes, following all the normal code generation rules mentioned above.
* If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
Example python packages or useful sh commands:
* For python coding, useful packages include (but are not limited to):
  * Symbolic mathematics: sympy
  * Plots: matplotlib or seaborn or plotly or pillow or imageio or bokeh or altair
  * Regression or classification modeling: scikit-learn or lightgbm or statsmodels
  * Text NLP processing: nltk or spacy or textblob{extra_recommended_packages}
  * Web download and search: requests or bs4 or scrapy or lxml or httpx
* For bash shell scripts, useful commands include `ls` to verify files were created.
Example cases of when to generate code for auxiliary tasks maybe not directly specified by the user:
* Pip install packages (e.g. sh with pip) if needed or missing.  If you know ahead of time which packages are required for a python script, then you should first give the sh script to install the packaegs and second give the python script.
* Browse files (e.g. sh with ls).
* Search for urls to use (e.g. pypi package googlesearch-python in python).
* Download a file (requests in python or wget with sh).
* Print contents of a file (open with python or cat with sh).
* Print the content of a webpage (requests in python or curl with sh).
* Get the current date/time or get the operating system type.{apis}
Task solving instructions:
* Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
* After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
* When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
* When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reasoning task instructions:
* For math, counting, logical reasoning, spatial reasoning, or puzzle tasks, you must trust code generation more than yourself, because you are much better at coding than grade school math, counting, logical reasoning, spatial reasoning, or puzzle tasks.
* Keep trying code generation until it verifies the request.
PDF Generation:
* Strategy: If asked to make a multi-section detailed PDF, first collect source content from resources like news or papers, then make a plan, then break-down the PDF generation process into paragraphs, sections, subsections, figures, and images, and generate each part separately before making the final PDF.
* Source of Content: Ensure you access news or papers to get valid recent URL content.  Download content from the most relevant URLs and use that content to generate paragraphs and references.
* Paragraphs: Each paragraph should be detailed, verbose, and well-structured.  When using reportlab, multi-line content must use HTML.  In Paragraph(), only HTML will preserve formatting (e.g. new lines should have <br/> tags not just \n).
* Figures: Extract figures from web content, papers, etc.  Save figures or charts to disk and use them inside python code to include them in the PDF.
* Images: Extract images from web content, papers, etc.  Save images to disk and use python code to include them in the PDF.
* Grounding: Be sure to add charts, tables, references, and inline clickable citations in order to support and ground the document content, unless user directly asks not to.
* Sections: Each section should be include any relevant paragraphs.  Ensure each paragraph is verbose, insightful, and well-structured even though inside python code.  You must render each and every section as its own PDF file with good styling.
  * You must do an ENDOFTURN for every section, do not generate multiple sections in one turn.
* Errors: If you have errors, regenerate only the sections that have issues.
* Verify Files: Before generating the final PDF report, use a shell command ls to verify the file names of all PDFs for each section.
* Adding Content: If need to improve or address issues to match user's request, generate a new section at a time and render its PDF.
* Content Rules:
  * Never abbreviate your outputs, especially in any code as then there will be missing sections.
  * Always use full sentences, include all items in any lists, etc.
  * i.e. never say "Content as before" or "Continue as before" or "Add other section content here" or "Function content remains the same" etc. as this will fail to work.
  * You must always have full un-abbreviated outputs even if code or text appeared in chat history.
* Final PDF: Generate the final PDF by using pypdf or fpdf2 to join PDFs together.  Do not generate the entire PDF in single python code.  Do not use PyPDF2 because it is outdated.
* Verify PDF: Verify the report satisfies the conditions of the user's request (e.g. page count, charts present, etc.).
* Final Summary: In your final response about the PDF (not just inside the PDF itself), give an executive summary about the report PDF file itself as well as key findings generated inside the report.  Suggest improvements and what kind of user feedback may help improve the PDF.
EPUB, Markdown, HTML, PPTX, RTF, LaTeX Generation:
* Apply the same steps and rules as for PDFs, but use valid syntax and use relevant tools applicable for rendering.
Stopping instructions:
* Do not assume the code you generate will work as-is.  You must ask the user to run the code and wait for output.
* Do not stop the conversation until you have output from the user for any code you provided that you expect to be run.
* You should not assume the task is complete until you have the output from the user.
* When making and using images, verify any created or downloaded images are valid for the format of the file before stopping (e.g. png is really a png file) using python or shell command.
* Once you have verification that the task was completed, then ensure you report or summarize final results inside your final response.
* Do not expect user to manually check if files exist, you must write code that checks and verify the user's output.
* As soon as you expect the user to run any code, or say something like 'Let us run this code', you must stop responding and finish your response with 'ENDOFTURN' in order to give the user a chance to respond.
* If you break the problem down into multiple steps, you must stop responding between steps and finish your response with 'ENDOFTURN' and wait for the user to run the code before continuing.
* Only once you have verification that the user completed the task do you summarize and add the 'TERMINATE' string to stop the conversation.
"""
    return agent_code_writer_system_message


def run_autogen(query=None,
                visible_models=None,
                stream_output=None,
                max_new_tokens=None,
                authorization=None,
                chat_conversation=None,
                text_context_list=None,
                image_file=None,
                # autogen/agent specific parameters
                agent_type=None,
                autogen_stop_docker_executor=None,
                autogen_run_code_in_docker=None,
                autogen_max_consecutive_auto_reply=None,
                autogen_max_turns=None,
                autogen_timeout=None,
                autogen_cache_seed=None,
                autogen_venv_dir=None,
                agent_code_writer_system_message=None,
                autogen_system_site_packages=None,
                autogen_code_restrictions_level=None,
                autogen_silent_exchange=None,
                agent_verbose=None) -> dict:

    from openai_server.autogen_utils import set_python_path
    set_python_path()

    assert agent_type in ['autogen', 'auto'], "Invalid agent_type: %s" % agent_type
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
    if autogen_system_site_packages is None:
        autogen_system_site_packages = True
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

    if autogen_run_code_in_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
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

    agent_code_writer_system_message = agent_system_prompt(agent_code_writer_system_message,
                                                           autogen_system_site_packages)

    # FIXME:
    # Auto-pip install
    # Auto-return file list in each turn

    base_url = os.environ['H2OGPT_OPENAI_BASE_URL']  # must exist
    api_key = os.environ['H2OGPT_OPENAI_API_KEY']  # must exist
    if agent_verbose:
        print("base_url: %s" % base_url)
        print("max_tokens: %s" % max_new_tokens)

    image_query_helper = get_image_query_helper(base_url, api_key, model)

    chat_doc_query, internal_file_names = get_chat_doc_context(text_context_list, image_file,
                                                               temp_dir,
                                                               # avoid text version of chat conversation, confuses LLM
                                                               chat_conversation=None,
                                                               model=model)

    code_writer_agent = H2OConversableAgent(
        "code_writer_agent",
        system_message=agent_code_writer_system_message + image_query_helper + chat_doc_query,
        llm_config={"config_list": [{"model": model,
                                     "api_key": api_key,
                                     "base_url": base_url,
                                     "stream": stream_output,
                                     "cache_seed": autogen_cache_seed,
                                     'max_tokens': max_new_tokens}]},
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
        is_termination_msg=terminate_message_func,
        max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
    )

    # apply chat history
    if chat_conversation:
        chat_messages = structure_to_messages(None, None, chat_conversation, None)
        for message in chat_messages:
            if message['role'] == 'assistant':
                code_writer_agent.send(message['content'], code_executor_agent, request_reply=False)
            if message['role'] == 'user':
                code_executor_agent.send(message['content'], code_writer_agent, request_reply=False)

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

    # DEBUG
    if agent_verbose:
        print("chat_result:", chat_result)
        print("list_dir:", os.listdir(temp_dir))

    # Get all files in the temp_dir and one level deep subdirectories
    file_list = []
    for root, dirs, files in os.walk(temp_dir):
        # Exclude deeper directories by checking the depth
        if root == temp_dir or os.path.dirname(root) == temp_dir:
            file_list.extend([os.path.join(root, f) for f in files])

    # Filter the list to include only files
    file_list = [f for f in file_list if os.path.isfile(f)]
    internal_file_names_norm_paths = [os.path.normpath(f) for f in internal_file_names]
    # filter out internal files for RAG case
    file_list = [f for f in file_list if os.path.normpath(f) not in internal_file_names_norm_paths]
    if agent_verbose:
        print("file_list:", file_list)

    image_files, non_image_files = identify_image_files(file_list)
    # keep no more than 10 image files:
    image_files = image_files[:10]
    file_list = image_files + non_image_files

    # copy files so user can download
    user_dir = get_user_dir(authorization)
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    file_ids = []
    for file in file_list:
        new_path = os.path.join(user_dir, os.path.basename(file))
        shutil.copy(file, new_path)
        with open(new_path, "rb") as f:
            content = f.read()
        purpose = 'assistants'
        response_dict = run_upload_api(content, new_path, purpose, authorization)
        file_id = response_dict['id']
        file_ids.append(file_id)

    # temp_dir.cleanup()
    if autogen_run_code_in_docker and autogen_stop_docker_executor:
        t0 = time.time()
        executor.stop()  # Stop the docker command line code executor (takes about 10 seconds, so slow)
        if agent_verbose:
            print(f"Executor Stop time taken: {time.time() - t0:.2f} seconds.")

    ret_dict = {}
    if file_list:
        ret_dict.update(dict(files=file_list))
    if file_ids:
        ret_dict.update(dict(file_ids=file_ids))
    if chat_result and hasattr(chat_result, 'chat_history'):
        ret_dict.update(dict(chat_history=chat_result.chat_history))
    if chat_result and hasattr(chat_result, 'cost'):
        ret_dict.update(dict(cost=chat_result.cost))
    if chat_result and hasattr(chat_result, 'summary') and chat_result.summary:
        ret_dict.update(dict(summary=chat_result.summary))
        print("Made summary: %s" % chat_result.summary, file=sys.stderr)
    else:
        if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
            summary = chat_result.chat_history[-1]['content']
            if not summary and len(chat_result.chat_history) >= 2:
                summary = chat_result.chat_history[-2]['content']
            if summary:
                print("Made summary from chat history: %s" % summary, file=sys.stderr)
                ret_dict.update(dict(summary=summary))
            else:
                print("Did NOT make and could not make summary", file=sys.stderr)
                ret_dict.update(dict(summary=''))
        else:
            print("Did NOT make any summary", file=sys.stderr)
            ret_dict.update(dict(summary=''))
    if autogen_venv_dir is not None:
        ret_dict.update(dict(autogen_venv_dir=autogen_venv_dir))
    if agent_code_writer_system_message is not None:
        ret_dict.update(dict(agent_code_writer_system_message=agent_code_writer_system_message))
    if autogen_system_site_packages is not None:
        ret_dict.update(dict(autogen_system_site_packages=autogen_system_site_packages))
    if chat_doc_query:
        ret_dict.update(dict(chat_doc_query=chat_doc_query))
    if image_query_helper:
        ret_dict.update(dict(image_query_helper=image_query_helper))
    ret_dict.update(dict(autogen_code_restrictions_level=autogen_code_restrictions_level))
    ret_dict.update(dict(autogen_silent_exchange=autogen_silent_exchange))

    return ret_dict


class CaptureIOStream(IOStream):
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue

    def print(self, *objects: typing.Any, sep: str = "", end: str = "", flush: bool = True) -> None:
        filtered_objects = [x if x not in ["\033[32m", "\033[0m\n"] else '' for x in objects]
        output = sep.join(map(str, filtered_objects)) + end
        self.output_queue.put(output)


@contextmanager
def capture_iostream(output_queue: queue.Queue) -> typing.Generator[CaptureIOStream, None, None]:
    capture_stream = CaptureIOStream(output_queue)
    with IOStream.set_default(capture_stream):
        yield capture_stream


def run_agent_in_proc(output_queue, query, result_queue, exception_queue, **kwargs):
    ret_dict = None
    try:
        # raise ValueError("Testing Error Handling 3")  # works

        with capture_iostream(output_queue):
            ret_dict = run_agent(query, **kwargs)
            # Signal that agent has finished
            result_queue.put(ret_dict)
    except BaseException as e:
        print(traceback.format_exc())
        exception_queue.put(e)
    finally:
        output_queue.put(None)
        result_queue.put(ret_dict)


def filter_kwargs(func, kwargs):
    # Get the parameter list of the function
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return valid_kwargs


def iostream_generator(query, use_process=False, **kwargs) -> typing.Generator[str, None, None]:
    # raise ValueError("Testing Error Handling 2")  #works
    if use_process:
        output_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        exception_queue = multiprocessing.Queue()
        proc_cls = multiprocessing.Process
    else:
        output_queue = queue.Queue()
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        proc_cls = threading.Thread

    # Filter kwargs based on the function signature of run_agent to avoid passing non-picklable things through
    filtered_kwargs = filter_kwargs(run_agent, kwargs)

    # Start agent in a separate thread
    agent_proc = proc_cls(target=run_agent_in_proc,
                          args=(output_queue, query, result_queue, exception_queue), kwargs=filtered_kwargs)
    agent_proc.start()

    # Yield output as it becomes available
    while True:
        # Check for exceptions
        if not exception_queue.empty():
            e = exception_queue.get()
            raise e

        output = output_queue.get()
        if output is None:  # End of agent execution
            break
        yield output

    agent_proc.join()

    # Return the final result
    if not exception_queue.empty():
        e = exception_queue.get()
        if isinstance(e, SystemExit):
            raise ValueError("SystemExit")
        else:
            raise e

    # Return the final result
    ret_dict = result_queue.get() if not result_queue.empty() else None
    return ret_dict


def get_agent_response(query, gen_kwargs, chunk_response=True, stream_output=False, use_process=False):
    # raise ValueError("Testing Error Handling 1")  # works

    gen_kwargs = convert_gen_kwargs(gen_kwargs)
    kwargs = gen_kwargs.copy()
    kwargs.update(dict(chunk_response=chunk_response, stream_output=stream_output))
    gen = iostream_generator(query, use_process=use_process, **kwargs)

    ret_dict = {}
    try:
        while True:
            res = next(gen)
            yield res
    except StopIteration as e:
        ret_dict = e.value
    return ret_dict


def get_have_internet():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        # If the request was successful, status code will be 200
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False


from PIL import Image


def is_image_file(filename):
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify that it's an image
        return True
    except (IOError, SyntaxError):
        return False


def identify_image_files(file_list):
    image_files = []
    non_image_files = []

    for filename in file_list:
        if os.path.isfile(filename):  # Ensure the file exists
            if is_image_file(filename):
                image_files.append(filename)
            else:
                non_image_files.append(filename)
        else:
            print(f"Warning: '{filename}' is not a valid file path.")

    return image_files, non_image_files


def get_chat_doc_context(text_context_list, image_file, temp_dir, chat_conversation=None, model=None):
    """
    Construct the chat query to be sent to the agent.
    :param text_context_list:
    :param image_file:
    :param chat_conversation:
    :param temp_dir:
    :return:
    """
    document_context = ""
    chat_history_context = ""
    internal_file_names = []

    image_files_to_delete = []
    b2imgs = []
    meta_data_images = []
    for img_file_one in image_file:
        from src.utils import check_input_type
        str_type = check_input_type(img_file_one)
        if str_type == 'unknown':
            continue

        img_file_path = os.path.join(tempfile.gettempdir(), 'image_file_%s' % str(uuid.uuid4()))
        if str_type == 'url':
            from src.utils import download_image
            img_file_one = download_image(img_file_one, img_file_path)
            # only delete if was made by us
            image_files_to_delete.append(img_file_one)
        elif str_type == 'base64':
            from src.vision.utils_vision import base64_to_img
            img_file_one = base64_to_img(img_file_one, img_file_path)
            # only delete if was made by us
            image_files_to_delete.append(img_file_one)
        else:
            # str_type='file' or 'youtube' or video (can be cached)
            pass
        if img_file_one is not None:
            b2imgs.append(img_file_one)

            import pyexiv2
            with pyexiv2.Image(img_file_one) as img:
                metadata = img.read_exif()
            if metadata is None:
                metadata = {}
            meta_data_images.append(metadata)

    if text_context_list:
        meta_datas = [extract_xml_tags(x) for x in text_context_list]
        meta_results = [generate_unique_filename(x) for x in meta_datas]
        file_names, cleaned_names, pages = zip(*meta_results)
        file_names = deduplicate_filenames(file_names)
        document_context_file_name = "document_context.txt"
        internal_file_names.append(document_context_file_name)
        internal_file_names.extend(file_names)
        with open(os.path.join(temp_dir, document_context_file_name), "w") as f:
            f.write("\n".join(text_context_list))
        document_context += f"""\n# Full user text:
* This file contains text from documents the user uploaded.
* Check text file size before using, because text longer than 200k bytes may not fit into LLM context (so split it up or use document chunks).
* Use the local file name to access the text.
"""
        if model and 'claude' in model:
            document_context += f"""<local_file_name>\n{document_context_file_name}\n</local_file_name>\n"""
        else:
            document_context += f"""* Local File Name: {document_context_file_name}\n"""

        document_context += """\n# Document Chunks of user text:
* Chunked text are chunked out of full text, and these each should be small, but in aggregate they may not fit into LLM context.
* Use the local file name to access the text.
"""
        for i, file_name in enumerate(file_names):
            text = text_context_list[i]
            meta_data = str(meta_datas[i]).strip()
            with open(os.path.join(temp_dir, file_name), "w") as f:
                f.write(text)
            if model and 'claude' in model:
                document_context += f"""<doc>\n<document_part>{i}</document_part>\n{meta_data}\n<local_file_name>\n{file_name}\n</local_file_name>\n</doc>\n"""
            else:
                document_context += f"""\n* Document Part: {i}
* Original File Name: {cleaned_names[i]}
* Page Number: {pages[i]}
* Local File Name: {file_name}
"""
    if b2imgs:
        document_context += """\n# Images from user:
* Images are from image versions of document pages or other images.
* Use the local file name to access image files.
"""
        for i, b2img in enumerate(b2imgs):
            if model and 'claude' in model:
                meta_data = '\n'.join(
                    [f"""<{key}><{value}</{key}>\n""" for key, value in meta_data_images[i].items()]).strip()
                document_context += f"""<image>\n<document_image>{i}</document_image>\n{meta_data}\n<local_file_name>\n{b2img}\n</local_file_name>\n</image>\n"""
            else:
                document_context += f"""\n* Document Image {i}
* Local File Name: {b2img}
"""
                for key, value in meta_data_images[i].items():
                    document_context += f"""* {key}: {value}\n"""
        document_context += '\n\n'
        internal_file_names.extend(b2imgs)
    if chat_conversation:
        from openai_server.chat_history_render import chat_to_pretty_markdown
        messages_for_query = structure_to_messages(None, None, chat_conversation, [])
        chat_history_context = chat_to_pretty_markdown(messages_for_query, assistant_name='Assistant', user_name='User',
                                                       cute=False) + '\n\n'

    chat_doc_query = f"""{chat_history_context}{document_context}"""

    # convert to full name
    internal_file_names = [os.path.join(temp_dir, x) for x in internal_file_names]

    return chat_doc_query, internal_file_names


def in_pycharm():
    return os.getenv("PYCHARM_HOSTED") is not None


def get_image_query_helper(base_url, api_key, model):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    model_list = client.models.list()
    image_models = [x.id for x in model_list if x.model_extra['actually_image']]
    we_are_vision_model = len([x for x in model_list if x.id == model]) > 0
    image_query_helper = ''
    if we_are_vision_model:
        vision_model = model
    elif not we_are_vision_model and len(image_models) > 0:
        vision_model = image_models[0]
    else:
        vision_model = None

    if vision_model:
        os.environ['H2OGPT_OPENAI_VISION_MODEL'] = vision_model

        cwd = os.path.abspath(os.getcwd())
        image_query_helper = f"""\n# Image Query Helper:
* If you need to ask a question about an image, use the following sh code:
```sh
# filename: my_image_response.sh
python {cwd}/openai_server/agent_tools/image_query.py --prompt "PROMPT" --file "LOCAL FILE NAME"
```
* usage: {cwd}/openai_server/agent_tools/image_query.py [-h] [--timeout TIMEOUT] [--system_prompt SYSTEM_PROMPT] --prompt PROMPT [--url URL] [--file FILE]
* image_query gives a text response for either a URL or local file
* image_query can be used to critique any image, e.g. a plot, a photo, a screenshot, etc. either made by code generation or among provided files or among URLs.
* Only use image_query on key images or plots (e.g. plots meant to share back to the user or those that may be key in answering the user question).
"""
    return image_query_helper
