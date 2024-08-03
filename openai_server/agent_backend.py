import functools
import os
import queue
import shutil
import tempfile
import threading
import time
import typing
from contextlib import contextmanager

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


def run_agent(query, **kwargs) -> dict:
    if kwargs['agent_type'] in ['auto', 'autogen']:
        return run_autogen(query, **kwargs)
    else:
        raise ValueError("Invalid agent_type: %s" % kwargs['agent_type'])


def run_autogen(query, **kwargs) -> dict:
    # raise openai.BadRequestError("Testing Error Handling")
    # raise ValueError("Testing Error Handling")

    # handle parameters from chatAPI and OpenAI -> h2oGPT transcription versions
    model = kwargs['visible_models']
    assert model is not None, "No model specified"
    stream_output = kwargs['stream_output']
    if stream_output is None:
        stream_output = False
    max_new_tokens = kwargs['max_new_tokens']
    assert max_new_tokens is not None, "No max_new_tokens specified"

    # handle parameters from FastAPI
    authorization = kwargs['authorization']

    # handle AutoGen specific parameters
    autogen_stop_docker_executor = kwargs['autogen_stop_docker_executor']
    if autogen_stop_docker_executor is None:
        autogen_stop_docker_executor = False
    autogen_run_code_in_docker = kwargs['autogen_run_code_in_docker']
    if autogen_run_code_in_docker is None:
        autogen_run_code_in_docker = False
    autogen_max_consecutive_auto_reply = kwargs['autogen_max_consecutive_auto_reply']
    if autogen_max_consecutive_auto_reply is None:
        autogen_max_consecutive_auto_reply = 10
    autogen_max_turns = kwargs['autogen_max_turns']
    autogen_timeout = kwargs['autogen_timeout']
    if autogen_timeout is None:
        autogen_timeout = 120
    autogen_cache_seed = kwargs['autogen_cache_seed']
    agent_verbose = kwargs['agent_verbose']
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

        # Create an agent with code executor configuration that uses docker.
        code_executor_agent = ConversableAgent(
            "code_executor_agent_docker",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the docker command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
            is_termination_msg=terminate_message_func,
            max_consecutive_auto_reply=autogen_max_consecutive_auto_reply,
        )
    else:
        # Create a local command line code executor.
        from autogen.coding import LocalCommandLineCodeExecutor
        executor = LocalCommandLineCodeExecutor(
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
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
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""

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
    chat_result = code_executor_agent.initiate_chat(
        code_writer_agent,
        max_turns=autogen_max_turns,
        message=query,
    )
    # DEBUG
    if agent_verbose:
        print(chat_result)
        print(os.listdir(temp_dir))

    files_list = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
    # We can see the output scatter.png and the code file generated by the agent.

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

    return ret_dict


class CaptureIOStream(IOStream):
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue

    def print(self, *objects: typing.Any, sep: str = "", end: str = "", flush: bool = True) -> None:
        output = sep.join(map(str, objects)) + end
        self.output_queue.put(output)

    # def input(self, prompt: str = "", *, password: bool = False) -> str:
    #    raise NotImplementedError("Input is not supported in this CaptureIOStream")


@contextmanager
def capture_iostream(output_queue: queue.Queue) -> typing.Generator[CaptureIOStream, None, None]:
    capture_stream = CaptureIOStream(output_queue)
    with IOStream.set_default(capture_stream):
        yield capture_stream


def run_agent_in_thread(output_queue: queue.Queue, query, result_queue: queue.Queue, exception_queue: queue.Queue,
                        **kwargs):
    ret_dict = None
    try:
        # raise ValueError("Testing Error Handling 3")  # works

        with capture_iostream(output_queue):
            ret_dict = run_agent(query, **kwargs)
            # Signal that agent has finished
            result_queue.put(ret_dict)
    except BaseException as e:
        exception_queue.put(e)
    finally:
        output_queue.put(None)
        result_queue.put(ret_dict)


def iostream_generator(query, **kwargs) -> typing.Generator[str, None, None]:
    # raise ValueError("Testing Error Handling 2")  #works
    output_queue = queue.Queue()
    result_queue = queue.Queue()
    exception_queue = queue.Queue()

    # Start agent in a separate thread
    agent_thread = threading.Thread(target=run_agent_in_thread,
                                    args=(output_queue, query, result_queue, exception_queue), kwargs=kwargs)
    agent_thread.start()

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

    agent_thread.join()

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


def get_response(query, **kwargs):
    ret_dict = yield from iostream_generator(query, **kwargs)
    return ret_dict


def get_agent_response(query, gen_kwargs, chunk_response=True, stream_output=False):
    # raise ValueError("Testing Error Handling 1")  # works

    gen_kwargs = convert_gen_kwargs(gen_kwargs)
    kwargs = gen_kwargs.copy()
    kwargs.update(dict(chunk_response=chunk_response, stream_output=stream_output))

    gen = get_response(query, **kwargs)
    # from iterators import TimeoutIterator
    # gen1 = TimeoutIterator(gen, timeout=0, sentinel=None, raise_on_exception=False, whichi=0)
    gen1 = gen

    ret_dict = {}
    try:
        while True:
            res = next(gen1)
            yield res
    except StopIteration as e:
        ret_dict = e.value
    return ret_dict
