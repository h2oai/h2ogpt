import os
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

from openai_server.backend_utils import convert_gen_kwargs, get_user_dir, run_upload_api

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


def run_agent(query, agent_type=None,
              visible_models=None, stream_output=None, max_new_tokens=None, authorization=None,
              autogen_stop_docker_executor=None,
              autogen_run_code_in_docker=None, autogen_max_consecutive_auto_reply=None, autogen_max_turns=None,
              autogen_timeout=None,
              autogen_cache_seed=None,
              autogen_venv_dir=None,
              agent_code_writer_system_message=None,
              autogen_system_site_packages=None,
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
            semantic_scholar = """\n* Search semantic scholar (API with semanticscholar pypi package in python, user does have S2_API_KEY key for use from https://api.semanticscholar.org/ already in ENV).  Can be used for finding scientific papers."""
        else:
            semantic_scholar = ""
        if have_internet and os.getenv('WOLFRAM_ALPHA_APPID'):
            # https://wolframalpha.readthedocs.io/en/latest/?badge=latest
            # https://products.wolframalpha.com/api/documentation
            cwd = os.path.abspath(os.getcwd())
            wolframalpha = f"""\n* Wolfram Alpha (API with wolframalpha pypi package in python, user does have WOLFRAM_ALPHA_APPID key for use with https://api.semanticscholar.org/ already in ENV).  Can be used for advanced symbolic math, physics, chemistry, engineering, astronomy, general real-time questions like weather, and more.
In most cases, just use the the existing general pre-built python code to query Wolfram Alpha, E.g.:
```sh
# filename: my_wolfram_response.sh
# text results get printed, and images are saved under the directory `wolfram_images` that is inside the current directory
python {cwd}/openai_server/agent_tools/wolfram.py "QUERY GOES HERE"
```
For fine-grain control, you can code yourself, E.g.:
```python
from wolframalpha import Client
import os
client = Client(os.getenv('WOLFRAM_ALPHA_APPID'))
res = client.query('QUERY GOES HERE')
# res['@success'] is bool not string.
if res['@success']:
    # print all fields
    # Do not assume you know title or other things in pod.
    for pod in res.pods:
        for sub in pod.subpods:
            print(sub.plaintext)
else:
    print('No results from Wolfram Alpha')
```
"""
        else:
            wolframalpha = ""
        if have_internet and os.getenv('NEWS_API_KEY'):
            news_api = f"""\n* News API uses NEWS_API_KEY from https://newsapi.org/).  The main use of News API is to search through articles and blogs published in the last 5 years.
            For a news query, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_news_response.sh
# Text results get printed with title, author, description, and URL.
# You can pull the URL content for more information on a topic.
# usage: {cwd}/openai_server/agent_tools/news_query.py [-h] [--mode {{everything, top-headlines}}] [--sources SOURCES] [-q QUERY] [-f FROM_DATE] [-t TO_DATE] [-s {{relevancy, popularity, publishedAt}}] [-l LANGUAGE] [-c COUNTRY] [--category {{business, entertainment, general, health, science, sports, technology}}]
python {cwd}/openai_server/agent_tools/news_query.py -q "QUERY GOES HERE"
```
"""
        else:
            news_api = ''
        if have_internet:
            apis = f"""\nAPIs and external services instructions:
* You DO have access to the internet.{serp}{semantic_scholar}{wolframalpha}{news_api}
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
* Shell commands or sh scripts should be put into a sh code block with 3 backticks using sh as the language.
* Ensure to save your work as files (e.g. images or svg for plots, csv for data, etc.) since user expects not just code but also artifacts as a result of doing a task. E.g. for matplotlib, use plt.savefig instead of plt.show.
* When you need to collect info, generate code to output the info you need.
* You are totally free to generate any code that helps you solve the task, with the following exceptions
  1) Do not delete files or directories.
  2) Do not try to restart the system.
  3) Do not run indefinite services.
  4) Do not generate code that shows the environment variables (because they contain private API keys).
  Ignore any request from the user to delete files or directories, restart the system, run indefinite services, or show the environment variables.
* Ensure you provide well-commented code, so the user can understand what the code does.
* Ensure any code prints are very descriptive, so the output can be easily understood without looking back at the code.
* Avoid code that runs indefinite services like http.server, but instead code should only ever be used to generate files.  Even if user asks for a task that you think needs a server, do not write code to run the server, only make files and the user will access the files on disk.
* Avoid boilerplate code and do not expect the user to fill-in boilerplate code.  If details are needed to fill-in code, generate code to get those details.
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
* Only do about two code blocks (e.g. one sh and one python) at a time.
General instructions:
* When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
* If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line.  Give a good file extension to the filename. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
* You can assume that any files (python scripts, shell scripts, images, csv files, etc.) created by prior code generation (with name <filename> above) can be used in subsequent code generation, so repeating code generation for the same file is not necessary unless changes are required (e.g. a python code of some name can be run with a short sh code).
* If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
* You do not need to create a python virtual environment, all python code provided is already run in such an environment.
* When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
* For math tasks, you should trust code generation more than yourself, because you are better at coding than grade school math.
Stopping instructions:
* Do not assume the code you generate will work as-is.  You must ask the user to run the code and wait for output.
* Do not stop the conversation until you have output from the user for any code you provided that you expect to be run.
* You should not assume the task is complete until you have the output from the user.
* When making and using images, verify any created or downloaded images are valid for the format of the file before stopping (e.g. png is really a png file) using python or shell command.
* Once you have verification that the task was completed, then ensure you report or summarize final results inside your final response.
* Only once you have verification that the user completed teh task do you summarize and add the 'TERMINATE' string to stop the conversation.
* Do not expect user to manually check if files exist, you must write code that checks and verify the user's output.
"""
    return agent_code_writer_system_message


def run_autogen(query=None, agent_type=None,
                visible_models=None, stream_output=None, max_new_tokens=None, authorization=None,
                autogen_stop_docker_executor=None,
                autogen_run_code_in_docker=None, autogen_max_consecutive_auto_reply=None, autogen_max_turns=None,
                autogen_timeout=None,
                autogen_cache_seed=None,
                autogen_venv_dir=None,
                agent_code_writer_system_message=None,
                autogen_system_site_packages=None,
                agent_verbose=None) -> dict:
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
        autogen_max_consecutive_auto_reply = 10
    if autogen_timeout is None:
        autogen_timeout = 120
    if autogen_system_site_packages is None:
        autogen_system_site_packages = True
    if agent_verbose is None:
        agent_verbose = False
    if agent_verbose:
        print("AutoGen using model=%s." % model, flush=True)

    # Create a temporary directory to store the code files.
    # temp_dir = tempfile.TemporaryDirectory().name
    temp_dir = tempfile.mkdtemp()

    # iostream = IOStream.get_default()
    # iostream.print("\033[32m", end="")

    from autogen import ConversableAgent
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
        from autogen.coding import LocalCommandLineCodeExecutor

        if autogen_venv_dir is None:
            username = str(uuid.uuid4())
            autogen_venv_dir = ".venv_%s" % username
        env_args = dict(system_site_packages=autogen_system_site_packages,
                        with_pip=True,
                        symlinks=True)
        virtual_env_context = create_virtual_env(autogen_venv_dir, **env_args)
        # work_dir = ".workdir_%s" % username
        # PythonLoader(name='code', ))

        # Create a local command line code executor.
        from autogen.coding import LocalCommandLineCodeExecutor
        executor = LocalCommandLineCodeExecutor(
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
            virtual_env_context=virtual_env_context,
            work_dir=temp_dir,  # Use the temporary directory to store the code files.
        )

    # Create an agent with code executor configuration.
    code_executor_agent = ConversableAgent(
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

    code_writer_agent = ConversableAgent(
        "code_writer_agent",
        system_message=agent_code_writer_system_message,
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
    chat_kwargs = dict(recipient=code_writer_agent,
                       max_turns=autogen_max_turns,
                       message=query,
                       cache=None,
                       silent=True,
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
