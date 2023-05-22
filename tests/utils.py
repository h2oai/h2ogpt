from functools import wraps, partial

from utils import call_subprocess_onetask


def wrap_test_forked(func):
    """Decorate a function to test, call in subprocess"""

    @wraps(func)
    def f(*args, **kwargs):
        func_new = partial(call_subprocess_onetask, func, args, kwargs)
        return run_test(func_new)

    return f


def run_test(func, *args, **kwargs):
    return func(*args, **kwargs)
