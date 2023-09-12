import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from subprocess import PIPE, CalledProcessError, CompletedProcess, Popen


def stream_command(
    args,
    *,
    stdout_handler=logging.info,
    stderr_handler=logging.error,
    check=True,
    text=True,
    stdout=PIPE,
    stderr=PIPE,
    **kwargs,
):
    """Mimic subprocess.run, while processing the command output in real time."""
    with Popen(args, text=text, stdout=stdout, stderr=stderr, **kwargs) as process:
        with ThreadPoolExecutor(2) as pool:  # two threads to handle the streams
            exhaust = partial(pool.submit, partial(deque, maxlen=0))
            exhaust(stdout_handler(line[:-1]) for line in process.stdout)
            exhaust(stderr_handler(line[:-1]) for line in process.stderr)
    retcode = process.poll()
    if check and retcode:
        raise CalledProcessError(retcode, process.args)
    return CompletedProcess(process.args, retcode)

stream_command(["./pieces.sh"], stdout_handler=None, stderr_handler=None, shell=True)