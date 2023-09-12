import sys
from os import fdopen
from subprocess import Popen, PIPE, STDOUT

def run(cmds, join='&&'):
    with Popen(join.join(cmds),
               shell=True,
               stdout=PIPE,
               stderr=STDOUT) as sp:
        with fdopen(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
            for line in sp.stdout:
                stdout.write(line)
                stdout.flush()
commands = [
    './pieces.sh',
    'sleep 3',
    'echo world',
    'sleep 2',
    'echo !',
]
run(commands)