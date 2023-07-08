import os
import sys
import shutil
from functools import wraps, partial

if os.path.dirname('src') not in sys.path:
    sys.path.append('src')

from src.utils import call_subprocess_onetask, makedirs


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
    shutil.copy('docs/FAQ.md', user_path)
    return user_path


def get_llama(llama_type=2):
    from huggingface_hub import hf_hub_download

    # default should match .env_gpt4all
    if llama_type == 1:
        file = 'ggml-model-q4_0_7b.bin'
        dest = 'models/7B/'
        prompt_type = 'plain'
    elif llama_type == 2:
        file = 'WizardLM-7B-uncensored.ggmlv3.q8_0.bin'
        dest = './'
        prompt_type = 'wizard2'
    else:
        raise ValueError("unknown llama_type=%s" % llama_type)

    makedirs(dest, exist_ok=True)
    full_path = os.path.join(dest, file)

    if not os.path.isfile(full_path):
        # True for case when locally already logged in with correct token, so don't have to set key
        token = os.getenv('HUGGINGFACE_API_TOKEN', True)
        out_path = hf_hub_download('h2oai/ggml', file, token=token, repo_type='model')
        # out_path will look like '/home/jon/.cache/huggingface/hub/models--h2oai--ggml/snapshots/57e79c71bb0cee07e3e3ffdea507105cd669fa96/ggml-model-q4_0_7b.bin'
        shutil.copy(out_path, dest)
    return prompt_type
