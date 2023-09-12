import subprocess as sp
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor


def enqueue_output(file, queue):
    for line in iter(file.readline, ''):
        queue.put(line)
    file.close()


def read_popen_pipes(p):

    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = Queue(), Queue()

        pool.submit(enqueue_output, p.stdout, q_stdout)
        pool.submit(enqueue_output, p.stderr, q_stderr)

        while True:

            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break

            out_line = err_line = ''

            try:
                out_line = q_stdout.get_nowait()
            except Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except Empty:
                pass

            yield (out_line, err_line)

# The function in use:

#with sp.Popen(["./pieces.sh"], stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
with sp.Popen(["python"], stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:

    for out_line, err_line in read_popen_pipes(p):
        print(out_line, end='')
        print(err_line, end='')

    p.poll()
