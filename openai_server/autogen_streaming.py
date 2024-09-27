import asyncio
import multiprocessing
import queue
import threading
import traceback
import typing
from contextlib import contextmanager

from autogen.io import IOStream, OutputStream

from openai_server.agent_utils import filter_kwargs


class CustomOutputStream(OutputStream):
    def print(self, *objects, sep="", end="", flush=False):
        filtered_objects = [x if x not in ["\033[32m", "\033[0m"] else '' for x in objects]
        super().print(*filtered_objects, sep="", end="", flush=flush)

    def dump(self, *objects, sep="", end="", flush=False):
        # Instead of printing, we return objects directly
        return objects


class CustomIOStream(IOStream, CustomOutputStream):
    pass


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


def run_autogen_in_proc(func, output_queue, result_queue, exception_queue, **kwargs):
    ret_dict = {}
    try:
        # raise ValueError("Testing Error Handling 3")  # works

        with capture_iostream(output_queue):
            ret_dict = func(**kwargs)
            # Signal that agent has finished
            result_queue.put(ret_dict)
    except BaseException as e:
        print(traceback.format_exc())
        exception_queue.put(e)
    finally:
        output_queue.put(None)
        result_queue.put(ret_dict)


async def iostream_generator(func, use_process=False, **kwargs) -> typing.AsyncGenerator[str, None]:
    # start capture
    custom_stream = CustomIOStream()
    IOStream.set_global_default(custom_stream)

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
    filtered_kwargs = filter_kwargs(func, kwargs)

    # Start agent in a separate thread
    agent_proc = proc_cls(target=run_autogen_in_proc,
                          args=(func, output_queue, result_queue, exception_queue),
                          kwargs=filtered_kwargs)
    agent_proc.start()

    # Yield output as it becomes available
    while True:
        # Check for exceptions
        if not exception_queue.empty():
            e = exception_queue.get()
            raise e
        if not output_queue.empty():
            output = output_queue.get()
            if output is None:  # End of agent execution
                break
            yield output
        await asyncio.sleep(0.005)

    agent_proc.join()

    # Return the final result
    ret_dict = result_queue.get() if not result_queue.empty() else None
    yield ret_dict

    # Return the final result
    if not exception_queue.empty():
        e = exception_queue.get()
        if isinstance(e, SystemExit):
            raise ValueError("SystemExit")
        else:
            raise e
