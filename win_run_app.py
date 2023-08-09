import os
from subprocess import Popen, PIPE, STDOUT
import sys
import time
import webbrowser

import requests

from src.utils import url_alive


def main():

    # Getting path to python executable (full path of deployed python on Windows)
    executable = sys.executable

    from nltk.downloader import download
    download('all')

    # Running streamlit server in a subprocess and writing to log file
    proc = Popen(
        [
            executable,
            "generate.py",
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )
    proc.stdin.close()

    # Force the opening (does not open automatically) of the browser tab after a brief delay to let
    # the gradio server start.
    url = "http://localhost:%s" % os.getenv('GRADIO_SERVER_PORT', str(7860))
    while not url_alive(url):
        print("Not alive", flush=True)
        time.sleep(1)
    print("Alive!", flush=True)
    webbrowser.open("http://localhost:7860")

    while True:
        s = proc.stdout.read()
        if not s:
            break
        print(s, end="")

    proc.wait()


if __name__ == "__main__":
    main()