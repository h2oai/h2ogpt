import sys

from src.utils import get_list_or_str, read_popen_pipes
from tests.utils import wrap_test_forked
import subprocess as sp

@wrap_test_forked
def test_get_list_or_str():
    assert get_list_or_str(['foo', 'bar']) == ['foo', 'bar']
    assert get_list_or_str('foo') == 'foo'
    assert get_list_or_str("['foo', 'bar']") == ['foo', 'bar']


@wrap_test_forked
def test_stream_popen1():
    cmd_python = sys.executable + " -i -q -u"
    cmd = cmd_python + " -c print('hi')"
    #cmd = cmd.split(' ')

    with sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')

        p.poll()


@wrap_test_forked
def test_stream_popen2():
    script = """for i in 0 1 2 3 4 5
do
    echo "This messages goes to stdout $i"
    sleep 1
    echo This message goes to stderr >&2
    sleep 1
done
"""
    with open('pieces.sh', 'wt') as f:
        f.write(script)
    with sp.Popen(["./pieces.sh"], stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')
        p.poll()
