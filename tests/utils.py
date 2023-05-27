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


def make_user_path_test():
    import os
    import shutil
    user_path = 'user_path_test'
    if os.path.isdir(user_path):
        shutil.rmtree(user_path)
    os.makedirs(user_path)
    db_dir = "db_dir_UserData"
    if os.path.isdir(db_dir):
        shutil.rmtree(db_dir)
    shutil.copy('data/pexels-evg-kowalievska-1170986_small.jpg', user_path)
    shutil.copy('README.md', user_path)
    shutil.copy('FAQ.md', user_path)
    return user_path
