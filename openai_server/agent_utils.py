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


def get_ret_dict_and_handle_files(chat_result, temp_dir, agent_verbose, internal_file_names, authorization,
                                  autogen_run_code_in_docker, autogen_stop_docker_executor, executor,
                                  agent_venv_dir, agent_code_writer_system_message, agent_system_site_packages,
                                  chat_doc_query, image_query_helper, mermaid_renderer_helper,
                                  autogen_code_restrictions_level, autogen_silent_exchange):
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

    # ensure files are sorted by creation time so newest are last in list
    file_list.sort(key=lambda x: os.path.getctime(x), reverse=True)

    # Filter the list to include only files
    file_list = [f for f in file_list if os.path.isfile(f)]
    internal_file_names_norm_paths = [os.path.normpath(f) for f in internal_file_names]
    # filter out internal files for RAG case
    file_list = [f for f in file_list if os.path.normpath(f) not in internal_file_names_norm_paths]
    if agent_verbose:
        print("file_list:", file_list)

    image_files, non_image_files = identify_image_files(file_list)
    # keep no more than 10 image files among latest files created
    image_files = image_files[-10:]
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
        print("Existing summary: %s" % chat_result.summary, file=sys.stderr)
    else:
        if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
            summary = chat_result.chat_history[-1]['content']
            if not summary and len(chat_result.chat_history) >= 2:
                summary = chat_result.chat_history[-2]['content']
            if summary:
                print("Made summary from chat history: %s" % summary, file=sys.stderr)
                chat_result.summary = summary
            else:
                print("Did NOT make and could not make summary", file=sys.stderr)
                chat_result.summary = 'No summary or chat history available'
        else:
            print("Did NOT make any summary", file=sys.stderr)
            chat_result.summary = 'No summary available'
    if chat_result and hasattr(chat_result, 'summary') and chat_result.summary:
        if '<constrained_output>' in chat_result.summary and '</constrained_output>' in chat_result.summary:
            extracted_summary = extract_xml_tags(chat_result.summary, tags=['constrained_output'])['constrained_output']
            if extracted_summary:
                chat_result.summary = extracted_summary
        chat_result.summary = chat_result.summary.replace('ENDOFTURN', '').replace('TERMINATE', '')

        if '![image](' not in chat_result.summary:
            latest_image_file = image_files[-1] if image_files else None
            if latest_image_file:
                chat_result.summary += f'\n![image]({os.path.basename(latest_image_file)})'
        else:
            try:
                chat_result.summary = fix_markdown_image_paths(chat_result.summary)
            except:
                print("Failed to fix markdown image paths", file=sys.stderr)

        ret_dict.update(dict(summary=chat_result.summary))
    if agent_venv_dir is not None:
        ret_dict.update(dict(agent_venv_dir=agent_venv_dir))
    if agent_code_writer_system_message is not None:
        ret_dict.update(dict(agent_code_writer_system_message=agent_code_writer_system_message))
    if agent_system_site_packages is not None:
        ret_dict.update(dict(agent_system_site_packages=agent_system_site_packages))
    if chat_doc_query:
        ret_dict.update(dict(chat_doc_query=chat_doc_query))
    if image_query_helper:
        ret_dict.update(dict(image_query_helper=image_query_helper))
    if mermaid_renderer_helper:
        ret_dict.update(dict(mermaid_renderer_helper=mermaid_renderer_helper))
    ret_dict.update(dict(autogen_code_restrictions_level=autogen_code_restrictions_level))
    ret_dict.update(dict(autogen_silent_exchange=autogen_silent_exchange))
    # can re-use for chat continuation to avoid sending files over
    # FIXME: Maybe just delete files and force send back to agent
    ret_dict.update(dict(temp_dir=temp_dir))

    return ret_dict


import faiss
import numpy as np
from openai import OpenAI
class MemoryVectorDB:
    def __init__(self, model: str, openai_base_url:str, openai_api_key: str):
        # Initialize OpenAI embeddings model
        self.client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        self.model = model

        # Initialize FAISS index (using L2 distance)
        self.index = None
        self.texts = []
        self.embeddings = None
        self.id_map = {}

    def get_embeddings(self, texts: list):
        # Generate embedding  for the texts via client
        response = self.client.embeddings.create(
        input=texts,
        model=self.model
    )
        # To reach embeddings for the first item in the list, use response.data[0].embedding and so on
        embeddings = []
        for i in range(len(response.data)):
            embeddings.append(response.data[i].embedding)

        embedding_matrix = np.array(embeddings).astype('float32')
        return embedding_matrix

    def add_texts(self, texts: list):
        # Generate embeddings for the texts
        embedding_matrix = self.get_embeddings(texts)

        # Update the list of stored texts and id map
        start_id = len(self.texts)
        self.texts.extend(texts)
        for i, text in enumerate(texts):
            self.id_map[start_id + i] = text

        # Create or update the FAISS index
        if self.index is None:
            # Initialize the FAISS index with the embedding dimension
            self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            # Add embeddings to the FAISS index
            self.index.add(embedding_matrix)
            self.embeddings = embedding_matrix
        else:
            self.index.add(embedding_matrix)
            self.embeddings = np.vstack((self.embeddings, embedding_matrix))

        # Confirm embeddings were added
        print("Texts added successfully.")
        print("Number of items in FAISS index:", self.index.ntotal)

    def query(self, query_text: str, k: int = 2, threshold: float = 2.0):
        # Generate embedding for the query
        query_embedding_np = self.get_embeddings([query_text])

        # Check if FAISS index is initialized
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is empty or not initialized. Please add texts before querying.")

        # Perform FAISS search
        D, I = self.index.search(query_embedding_np, k)
        
        # Ensure valid indices and handle potential errors
        results = []
        distances = []
        for i, idx in enumerate(I[0]):
            if idx in self.id_map:
                results.append(self.id_map[idx])
                distances.append(D[0][i])
            else:
                print(f"Warning: Index {idx} not found in id_map. It might have been deleted.")
        final_results = [r for r, d in zip(results, distances) if d <= threshold]
        final_distances = [d for d in distances if d <= threshold]
        print(f"Memory VetorDB: Returns {len(final_results)} results.")
        # Returns results having distance less than or equal to the threshold
        return final_results, final_distances

    def delete_text_by_id(self, idx: int):
        # Remove the text from stored texts and id map
        if idx in self.id_map:
            del self.id_map[idx]
            self.texts.pop(idx)

            # Remove the embedding from FAISS index and rebuild the index
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            self.index.reset()
            self.index.add(self.embeddings)
        else:
            print(f"Warning: Text with index {idx} not found in the database.")

    def get_all_texts(self):
        return self.texts
