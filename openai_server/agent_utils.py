import functools
import inspect
import os
import re
import shutil
import sys
import time

import requests
from PIL import Image

from openai_server.backend_utils import get_user_dir, run_upload_api, extract_xml_tags


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


def in_pycharm():
    return os.getenv("PYCHARM_HOSTED") is not None


def get_inner_function_signature(func):
    # Check if the function is a functools.partial object
    if isinstance(func, functools.partial):
        # Get the original function
        assert func.keywords is not None and func.keywords, "The function must have keyword arguments."
        func = func.keywords['run_agent_func']
        return inspect.signature(func)
    else:
        return inspect.signature(func)


def filter_kwargs(func, kwargs):
    # Get the parameter list of the function
    sig = get_inner_function_signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return valid_kwargs


def set_python_path():
    # Get the current working directory
    current_dir = os.getcwd()
    current_dir = os.path.abspath(current_dir)

    # Retrieve the existing PYTHONPATH, if it exists, and append the current directory
    pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = current_dir if not pythonpath else pythonpath + os.pathsep + current_dir

    # Update the PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = new_pythonpath

    # Also, ensure sys.path is updated
    if current_dir not in sys.path:
        sys.path.append(current_dir)


def current_datetime():
    from datetime import datetime
    import tzlocal

    # Get the local time zone
    local_timezone = tzlocal.get_localzone()

    # Get the current time in the local time zone
    now = datetime.now(local_timezone)

    # Format the date, time, and time zone
    formatted_date_time = now.strftime("%A, %B %d, %Y - %I:%M %p %Z")

    # Print the formatted date, time, and time zone
    return "For current user query: Current Date, Time, and Local Time Zone: %s. Note some APIs may have data from different time zones, so may reflect a different date." % formatted_date_time


def run_agent(run_agent_func=None,
              **kwargs,
              ) -> dict:
    ret_dict = {}
    try:
        assert run_agent_func is not None, "run_agent_func must be provided."
        ret_dict = run_agent_func(**kwargs)
    finally:
        if kwargs.get('agent_venv_dir') is None and 'agent_venv_dir' in ret_dict and ret_dict['agent_venv_dir']:
            agent_venv_dir = ret_dict['agent_venv_dir']
            if os.path.isdir(agent_venv_dir):
                if kwargs.get('agent_verbose'):
                    print("Clean-up: Removing agent_venv_dir: %s" % agent_venv_dir)
                shutil.rmtree(agent_venv_dir)

    return ret_dict


def set_dummy_term():
    # Disable color and advanced terminal features
    os.environ['TERM'] = 'dumb'
    os.environ['COLORTERM'] = ''
    os.environ['CLICOLOR'] = '0'
    os.environ['CLICOLOR_FORCE'] = '0'
    os.environ['ANSI_COLORS_DISABLED'] = '1'

    # force matplotlib to use terminal friendly backend
    import matplotlib as mpl
    mpl.use('Agg')

    # Turn off interactive mode
    import matplotlib.pyplot as plt
    plt.ioff()


def fix_markdown_image_paths(text):
    def replace_path(match):
        alt_text = match.group(1)
        full_path = match.group(2)
        base_name = os.path.basename(full_path)
        return f"![{alt_text}]({base_name})"

    # Pattern for inline images: ![alt text](path/to/image.jpg)
    inline_pattern = r'!\[(.*?)\]\s*\((.*?)\)'
    text = re.sub(inline_pattern, replace_path, text)

    # Pattern for reference-style images: ![alt text][ref]
    ref_pattern = r'!\[(.*?)\]\s*\[(.*?)\]'

    def collect_references(text):
        ref_dict = {}
        ref_def_pattern = r'^\s*\[(.*?)\]:\s*(.*?)$'
        for match in re.finditer(ref_def_pattern, text, re.MULTILINE):
            ref_dict[match.group(1)] = match.group(2)
        return ref_dict

    ref_dict = collect_references(text)

    def replace_ref_image(match):
        alt_text = match.group(1)
        ref = match.group(2)
        if ref in ref_dict:
            full_path = ref_dict[ref]
            base_name = os.path.basename(full_path)
            ref_dict[ref] = base_name  # Update reference
            return f"![{alt_text}][{ref}]"
        return match.group(0)  # If reference not found, leave unchanged

    text = re.sub(ref_pattern, replace_ref_image, text)

    # Update reference definitions
    def replace_ref_def(match):
        ref = match.group(1)
        if ref in ref_dict:
            return f"[{ref}]: {ref_dict[ref]}"
        return match.group(0)

    text = re.sub(r'^\s*\[(.*?)\]:\s*(.*?)$', replace_ref_def, text, flags=re.MULTILINE)

    return text


def get_ret_dict_and_handle_files(chat_result, chat_result_planning,
                                  model,
                                  agent_work_dir, agent_verbose, internal_file_names, authorization,
                                  autogen_run_code_in_docker, autogen_stop_docker_executor, executor,
                                  agent_venv_dir, agent_code_writer_system_message, agent_system_site_packages,
                                  system_message_parts,
                                  autogen_code_restrictions_level, autogen_silent_exchange,
                                  agent_accuracy,
                                  client_metadata=''):
    # DEBUG
    if agent_verbose:
        print("chat_result:", chat_result_planning)
        print("chat_result:", chat_result)
        print("list_dir:", os.listdir(agent_work_dir))

    # Get all files in the temp_dir and one level deep subdirectories
    file_list = []
    for root, dirs, files in os.walk(agent_work_dir):
        # Exclude deeper directories by checking the depth
        if root == agent_work_dir or os.path.dirname(root) == agent_work_dir:
            file_list.extend([os.path.join(root, f) for f in files])

    # ensure files are sorted by creation time so newest are last in list
    file_list.sort(key=lambda x: os.path.getctime(x), reverse=True)

    # 10MB limit to avoid long conversions
    file_size_bytes_limit = int(os.getenv('H2OGPT_AGENT_FILE_SIZE_LIMIT', 10 * 1024 * 1024))
    file_list = [
        f for f in file_list if os.path.getsize(f) <= file_size_bytes_limit
    ]

    # Filter the list to include only files
    file_list = [f for f in file_list if os.path.isfile(f)]
    internal_file_names_norm_paths = [os.path.normpath(f) for f in internal_file_names]
    # filter out internal files for RAG case
    file_list = [f for f in file_list if os.path.normpath(f) not in internal_file_names_norm_paths]
    if agent_verbose or client_metadata:
        print(f"FILE LIST: client_metadata: {client_metadata} file_list: {file_list}", flush=True)

    image_files, non_image_files = identify_image_files(file_list)
    # keep no more than 10 image files among latest files created
    if agent_accuracy == 'maximum':
        pass
    elif agent_accuracy == 'standard':
        image_files = image_files[-20:]
    elif agent_accuracy == 'basic':
        image_files = image_files[-10:]
    else:
        image_files = image_files[-5:]
    file_list = image_files + non_image_files

    # guardrail artifacts even if LLM never saw them, shouldn't show user either
    file_list = guardrail_files(file_list)

    # copy files so user can download
    user_dir = get_user_dir(authorization)
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    file_ids = []
    for file in file_list:
        file_stat = os.stat(file)
        created_at_orig = int(file_stat.st_ctime)

        new_path = os.path.join(user_dir, os.path.basename(file))
        shutil.copy(file, new_path)
        with open(new_path, "rb") as f:
            content = f.read()
        purpose = 'assistants'
        response_dict = run_upload_api(content, new_path, purpose, authorization, created_at_orig=created_at_orig)
        file_id = response_dict['id']
        file_ids.append(file_id)

    # temp_dir.cleanup()
    if autogen_run_code_in_docker and autogen_stop_docker_executor:
        t0 = time.time()
        executor.stop()  # Stop the docker command line code executor (takes about 10 seconds, so slow)
        if agent_verbose:
            print(f"Executor Stop time taken: {time.time() - t0:.2f} seconds.")

    def cleanup_response(x):
        return x.replace('ENDOFTURN', '').replace('<FINISHED_ALL_TASKS>', '').strip()

    ret_dict = {}
    if file_list:
        ret_dict.update(dict(files=file_list))
    if file_ids:
        ret_dict.update(dict(file_ids=file_ids))
    if chat_result and hasattr(chat_result, 'chat_history'):
        print(f"CHAT HISTORY: client_metadata: {client_metadata}: chat history: {len(chat_result.chat_history)}", flush=True)
        ret_dict.update(dict(chat_history=chat_result.chat_history))
    if chat_result and hasattr(chat_result, 'cost'):
        if hasattr(chat_result_planning, 'cost'):
            usage_no_caching = chat_result.cost["usage_excluding_cached_inference"]
            usage_no_caching_planning = chat_result_planning.cost["usage_excluding_cached_inference"]
            usage_no_caching[model]["prompt_tokens"] += usage_no_caching_planning[model]["prompt_tokens"]
            usage_no_caching[model]["completion_tokens"] += usage_no_caching_planning[model]["completion_tokens"]

        ret_dict.update(dict(cost=chat_result.cost))
    if chat_result and hasattr(chat_result, 'summary') and chat_result.summary:
        print("Existing summary: %s" % chat_result.summary, file=sys.stderr)

        if '<constrained_output>' in chat_result.summary and '</constrained_output>' in chat_result.summary:
            extracted_summary = extract_xml_tags(chat_result.summary, tags=['constrained_output'])['constrained_output']
            if extracted_summary:
                chat_result.summary = extracted_summary
        chat_result.summary = cleanup_response(chat_result.summary)
        # above may lead to no summary, we'll fix that below
    elif chat_result:
        chat_result.summary = ''

    if chat_result and not chat_result.summary:
        # construct alternative summary if none found or no-op one
        if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
            summary = cleanup_response(chat_result.chat_history[-1]['content'])
            if not summary and len(chat_result.chat_history) >= 3:
                summary = cleanup_response(chat_result.chat_history[-3]['content'])
            if summary:
                print(f"Made summary from chat history: {summary} : {client_metadata}", file=sys.stderr)
                chat_result.summary = summary
            else:
                print(f"Did NOT make and could not make summary {client_metadata}", file=sys.stderr)
                chat_result.summary = 'No summary or chat history available'
        else:
            print(f"Did NOT make any summary {client_metadata}", file=sys.stderr)
            chat_result.summary = 'No summary available'

    if chat_result:
        if '![image](' not in chat_result.summary:
            latest_image_file = image_files[-1] if image_files else None
            if latest_image_file:
                chat_result.summary += f'\n![image]({os.path.basename(latest_image_file)})'
        else:
            try:
                chat_result.summary = fix_markdown_image_paths(chat_result.summary)
            except:
                print("Failed to fix markdown image paths", file=sys.stderr)
    if chat_result:
        ret_dict.update(dict(summary=chat_result.summary))
    ret_dict.update(dict(agent_venv_dir=agent_venv_dir))
    if agent_code_writer_system_message is not None:
        ret_dict.update(dict(agent_code_writer_system_message=agent_code_writer_system_message))
    if agent_system_site_packages is not None:
        ret_dict.update(dict(agent_system_site_packages=agent_system_site_packages))
    if system_message_parts:
        ret_dict.update(dict(helpers=system_message_parts))
    ret_dict.update(dict(autogen_code_restrictions_level=autogen_code_restrictions_level))
    ret_dict.update(dict(autogen_silent_exchange=autogen_silent_exchange))
    # can re-use for chat continuation to avoid sending files over
    # FIXME: Maybe just delete files and force send back to agent
    ret_dict.update(dict(agent_work_dir=agent_work_dir))

    return ret_dict


def guardrail_files(file_list, hard_fail=False):
    from openai_server.autogen_utils import H2OLocalCommandLineCodeExecutor

    file_list_new = []
    for file in file_list:
        try:
            # Determine if the file is binary or text
            is_binary = is_binary_file(file)

            if is_binary:
                # For binary files, read in binary mode and process in chunks
                with open(file, "rb") as f:
                    chunk_size = 1024 * 1024  # 1 MB chunks
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        # Convert binary chunk to string for guardrail check
                        text = chunk.decode('utf-8', errors='ignore')
                        H2OLocalCommandLineCodeExecutor.text_guardrail(text)
            else:
                # For text files, read as text
                with open(file, "rt", encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                H2OLocalCommandLineCodeExecutor.text_guardrail(text, any_fail=True, max_bad_lines=1)

            file_list_new.append(file)
        except Exception as e:
            print(f"Guardrail failed for file: {file}, {e}", flush=True)
            if hard_fail:
                raise e

    return file_list_new


def is_binary_file(file_path, sample_size=1024):
    """
    Check if a file is binary by reading a sample of its contents.
    """
    with open(file_path, 'rb') as f:
        sample = f.read(sample_size)

    text_characters = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
    return bool(sample.translate(None, text_characters))


def extract_agent_tool(input_string):
    """
    Extracts and returns the agent_tool filename from the input string.
    Can be used to detect the agent_tool usages in chat history.
    """
    # FIXME: This missing if agent_tool is imported into python code, but usually that fails to work by LLM
    # Regular expression pattern to match Python file paths
    pattern = r'openai_server/agent_tools/([a-zA-Z_]+\.py)'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    if match:
        # Return the filename if found
        return match.group(1)
    else:
        # Return None if no match is found
        return None


def get_openai_client(max_time=120):
    # Set up OpenAI-like client
    base_url = os.getenv('H2OGPT_OPENAI_BASE_URL')
    assert base_url is not None, "H2OGPT_OPENAI_BASE_URL environment variable is not set"
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=server_api_key, timeout=max_time)
    return client
