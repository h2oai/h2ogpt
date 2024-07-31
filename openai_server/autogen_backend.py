import os
import tempfile
import typing

from autogen.io import IOStream
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor


def run_autogen(query, **kwargs) -> None:
    model = kwargs['model']
    stream_output = kwargs['stream_output']
    print(" Using model=%s." % model, flush=True)

    # Create a temporary directory to store the code files.
    temp_dir = tempfile.TemporaryDirectory()

    use_docker = True

    if use_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
        # Create a Docker command line code executor.
        executor = DockerCommandLineCodeExecutor(
            image="python:3.10-slim-bullseye",
            timeout=20,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        # Create an agent with code executor configuration that uses docker.
        code_executor_agent = ConversableAgent(
            "code_executor_agent_docker",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the docker command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
        )
    else:
        # Create a local command line code executor.
        executor = LocalCommandLineCodeExecutor(
            timeout=20,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        # Create an agent with code executor configuration.
        code_executor_agent = ConversableAgent(
            "code_executor_agent",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
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

    code_writer_agent = ConversableAgent(
        "code_writer_agent",
        system_message=code_writer_system_message,
        llm_config={"config_list": [{"model": model, "api_key": api_key, "base_url": base_url, "stream": stream_output}]},
        code_execution_config=False,  # Turn off code execution for this agent.
    )
    chat_result = code_executor_agent.initiate_chat(
        code_writer_agent,
        message=query,
    )
    print(chat_result)

    print(os.listdir(temp_dir.name))
    # We can see the output scatter.png and the code file generated by the agent.

    # temp_dir.cleanup()
    executor.stop()  # Stop the docker command line code executor.

    return os.listdir(temp_dir.name)


def get_autogen_response(query, gen_kwargs, chunk_response=True, stream_output=False):
    kwargs = gen_kwargs.copy()
    kwargs.update(dict(chunk_response=chunk_response, stream_output=stream_output))
    yield from autogen_iostream_generator(query, **kwargs)
    #return run_autogen(query, **kwargs)


from contextlib import contextmanager
from typing import Generator, Optional
from io import StringIO
import threading
import queue


class CaptureIOStream(IOStream):
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue

    def print(self, *objects: typing.Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        output = sep.join(map(str, objects)) + end
        self.output_queue.put(output)

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        raise NotImplementedError("Input is not supported in this CaptureIOStream")


@contextmanager
def capture_iostream(output_queue: queue.Queue) -> Generator[CaptureIOStream, None, None]:
    capture_stream = CaptureIOStream(output_queue)
    with IOStream.set_default(capture_stream):
        yield capture_stream


def run_autogen_in_thread(output_queue: queue.Queue, query, **kwargs):
    with capture_iostream(output_queue):
        # Your autogen code here
        run_autogen(query, **kwargs)
        # Signal that autogen has finished
        output_queue.put(None)


def autogen_iostream_generator(query, **kwargs) -> Generator[str, None, None]:
    output_queue = queue.Queue()

    # Start autogen in a separate thread
    autogen_thread = threading.Thread(target=run_autogen_in_thread, args=(output_queue, query), kwargs=kwargs)
    autogen_thread.start()

    # Yield output as it becomes available
    while True:
        output = output_queue.get()
        if output is None:  # End of autogen execution
            break
        yield output

    autogen_thread.join()
