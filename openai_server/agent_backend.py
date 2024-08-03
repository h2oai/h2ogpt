import functools
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
        super().print(*objects, sep="", end="", flush=flush)

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
    if (isinstance(msg, dict) and
            isinstance(msg.get('content', ''), str) and
            (msg.get('content', '').endswith("TERMINATE") or msg.get('content', '') == '')):
        return True
    return False


def run_agent(query, agent_type=None,
              visible_models=None, stream_output=None, max_new_tokens=None, authorization=None,
              autogen_stop_docker_executor=None,
              autogen_run_code_in_docker=None, autogen_max_consecutive_auto_reply=None, autogen_max_turns=None,
              autogen_timeout=None,
              autogen_cache_seed=None,
              autogen_venv_dir=None,
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
                if True or agent_verbose:
                    print("Clean-up: Removing autogen_venv_dir: %s" % autogen_venv_dir)
                shutil.rmtree(autogen_venv_dir)

    return ret_dict


def run_autogen(query=None, agent_type=None,
                visible_models=None, stream_output=None, max_new_tokens=None, authorization=None,
                autogen_stop_docker_executor=None,
                autogen_run_code_in_docker=None, autogen_max_consecutive_auto_reply=None, autogen_max_turns=None,
                autogen_timeout=None,
                autogen_cache_seed=None,
                autogen_venv_dir=None,
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
    if agent_verbose is None:
        agent_verbose = False
    if agent_verbose:
        print("AutoGen using model=%s." % model, flush=True)

    # Create a temporary directory to store the code files.
    # temp_dir = tempfile.TemporaryDirectory().name
    temp_dir = tempfile.mkdtemp()

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
        virtual_env_context = create_virtual_env(autogen_venv_dir)
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

    # The code writer agent's system message is to instruct the LLM on how to use
    # the code executor in the code executor agent.
    code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
General instructions:
* When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
* If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
* If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
* You do not need to create a python virtual environment, all python code provided is already run in such an environment.
* When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Stopping instructions:
* Reply 'TERMINATE' in the end only when you have verification from the user that the task you specified was done.
* Do not expect user to manually check if files exist, you should infer whether they exist from the user responses or write code to confirm their existence and infer from the response if they exist.
"""

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
        system_message=code_writer_system_message,
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

    files_list = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
    # FIXME: Could do paths later, this excludes envs LLM may have created
    files_list = [f for f in files_list if os.path.isfile(f)]
    if agent_verbose:
        print("files_list:", files_list)

    iostream = IOStream.get_default()

    # copy files so user can download
    user_dir = get_user_dir(authorization)
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    file_ids = []
    for file in files_list:
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
    if files_list:
        ret_dict.update(dict(files=files_list))
    if file_ids:
        ret_dict.update(dict(file_ids=file_ids))
    if chat_result and hasattr(chat_result, 'chat_history'):
        ret_dict.update(dict(chat_history=chat_result.chat_history))
    if chat_result and hasattr(chat_result, 'cost'):
        ret_dict.update(dict(cost=chat_result.cost))
    if chat_result and hasattr(chat_result, 'summary'):
        ret_dict.update(dict(summary=chat_result.summary))
    if autogen_venv_dir is not None:
        ret_dict.update(dict(autogen_venv_dir=autogen_venv_dir))

    return ret_dict


class CaptureIOStream(IOStream):
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue

    def print(self, *objects: typing.Any, sep: str = "", end: str = "", flush: bool = True) -> None:
        output = sep.join(map(str, objects)) + end
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
