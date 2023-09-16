import ast
import contextlib
import functools
import gc
import getpass
import hashlib
import inspect
import json
import os
import pathlib
import pickle
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Tuple, Callable, Dict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import filelock
import fire
import numpy as np
import pandas as pd
import requests
import uuid

import tabulate
from fire import inspectutils
from joblib import Parallel
from tqdm.auto import tqdm


def H2O_Fire(component=None):
    config_prefix = "H2OGPT_"

    args = sys.argv[1:]
    query_args = [arg.split("=")[0].split(" ")[0].lstrip("-") for arg in args]

    fn_spec = inspectutils.GetFullArgSpec(component)
    for key, value in os.environ.items():
        if not (
                (key.startswith(config_prefix) or key.startswith(config_prefix.lower()))
                and len(key) > len(config_prefix)
        ):
            continue  # ignore as non H2OGPT argument

        new_key = key[len(config_prefix):].lower()

        if new_key in query_args:
            continue  # ignore as already passed as script argument

        if new_key not in fn_spec.args:
            continue  # ignore as not a valid H2OGPT argument

        args.append(f"--{new_key}={value}")

    fire.Fire(component=component, command=args)


def set_seed(seed: int):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    import torch
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state


def flatten_list(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten_list(item))
        else:
            new_lis.append(item)
    return new_lis


def clear_torch_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
    except RuntimeError as e:
        print("clear_torch_cache error: %s" % ''.join(traceback.format_tb(e.__traceback__)), flush=True)


def ping():
    try:
        print('Ping: %s' % str(datetime.now()), flush=True)
    except AttributeError:
        # some programs wrap print and will fail with flush passed
        pass


def ping_gpu():
    try:
        print('Ping_GPU: %s %s' % (str(datetime.now()), system_info()), flush=True)
    except AttributeError:
        # some programs wrap print and will fail with flush passed
        pass
    try:
        ping_gpu_memory()
    except Exception as e:
        print('Ping_GPU memory failure: %s' % str(e), flush=True)


def ping_gpu_memory():
    from models.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()  # define a GPU tracker
    from torch.cuda import memory_summary
    gpu_tracker.track()


def get_torch_allocated():
    import torch
    return torch.cuda.memory_allocated()


def get_device():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device


def system_info():
    import psutil

    system = {}
    # https://stackoverflow.com/questions/48951136/plot-multiple-graphs-in-one-plot-using-tensorboard
    # https://arshren.medium.com/monitoring-your-devices-in-python-5191d672f749
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        if 'coretemp' in temps:
            coretemp = temps['coretemp']
            temp_dict = {k.label: k.current for k in coretemp}
            for k, v in temp_dict.items():
                system['CPU_C/%s' % k] = v
    except AttributeError:
        pass

    # https://github.com/gpuopenanalytics/pynvml/blob/master/help_query_gpu.txt
    try:
        from pynvml.smi import nvidia_smi
        nvsmi = nvidia_smi.getInstance()

        gpu_power_dict = {'W_gpu%d' % i: x['power_readings']['power_draw'] for i, x in
                          enumerate(nvsmi.DeviceQuery('power.draw')['gpu'])}
        for k, v in gpu_power_dict.items():
            system['GPU_W/%s' % k] = v

        gpu_temp_dict = {'C_gpu%d' % i: x['temperature']['gpu_temp'] for i, x in
                         enumerate(nvsmi.DeviceQuery('temperature.gpu')['gpu'])}
        for k, v in gpu_temp_dict.items():
            system['GPU_C/%s' % k] = v

        gpu_memory_free_dict = {'MiB_gpu%d' % i: x['fb_memory_usage']['free'] for i, x in
                                enumerate(nvsmi.DeviceQuery('memory.free')['gpu'])}
        gpu_memory_total_dict = {'MiB_gpu%d' % i: x['fb_memory_usage']['total'] for i, x in
                                 enumerate(nvsmi.DeviceQuery('memory.total')['gpu'])}
        gpu_memory_frac_dict = {k: gpu_memory_free_dict[k] / gpu_memory_total_dict[k] for k in gpu_memory_total_dict}
        for k, v in gpu_memory_frac_dict.items():
            system[f'GPU_M/%s' % k] = v
    except (KeyError, ModuleNotFoundError):
        pass
    system['hash'] = get_githash()

    return system


def system_info_print():
    try:
        df = pd.DataFrame.from_dict(system_info(), orient='index')
        # avoid slamming GPUs
        time.sleep(1)
        return df.to_markdown()
    except Exception as e:
        return "Error: %s" % str(e)


def zip_data(root_dirs=None, zip_file=None, base_dir='./', fail_any_exception=False):
    try:
        return _zip_data(zip_file=zip_file, base_dir=base_dir, root_dirs=root_dirs)
    except Exception as e:
        traceback.print_exc()
        print('Exception in zipping: %s' % str(e))
        if not fail_any_exception:
            raise


def _zip_data(root_dirs=None, zip_file=None, base_dir='./'):
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    if zip_file is None:
        datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
        host_name = os.getenv('HF_HOSTNAME', 'emptyhost')
        zip_file = "data_%s_%s.zip" % (datetime_str, host_name)
    assert root_dirs is not None
    base_path = os.path.dirname(zip_file)
    if not os.path.isdir(base_path) and os.path.dirname(zip_file):
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        zip_file = os.path.join(base_path, os.path.basename(zip_file))
    with zipfile.ZipFile(zip_file, "w") as expt_zip:
        for root_dir in root_dirs:
            if root_dir is None:
                continue
            for root, d, files in os.walk(root_dir):
                for file in files:
                    file_to_archive = os.path.join(root, file)
                    assert os.path.exists(file_to_archive)
                    path_to_archive = os.path.relpath(file_to_archive, base_dir)
                    expt_zip.write(filename=file_to_archive, arcname=path_to_archive)
    return zip_file, zip_file


def save_generate_output(prompt=None, output=None, base_model=None, save_dir=None, where_from='unknown where from',
                         extra_dict={}, error='', extra='', which_api='', valid_key=None,
                         h2ogpt_key='', return_dict=False):
    if not save_dir:
        return
    try:
        return _save_generate_output(prompt=prompt, output=output, base_model=base_model, save_dir=save_dir,
                                     where_from=where_from, extra_dict=extra_dict, error=error, extra=extra,
                                     which_api=which_api, valid_key=valid_key, h2ogpt_key=h2ogpt_key,
                                     return_dict=return_dict)
    except Exception as e:
        traceback.print_exc()
        print('Exception in saving: %s' % str(e))


def _save_generate_output(prompt=None, output=None, base_model=None, save_dir=None, where_from='unknown where from',
                          extra_dict={}, error='', extra='', which_api='',
                          valid_key=None, h2ogpt_key='',
                          return_dict=False):
    """
    Save conversation to .json, row by row.
    json_file_path is path to final JSON file. If not in ., then will attempt to make directories.
    Appends if file exists
    """
    prompt = '<not set>' if prompt is None else prompt
    output = '<not set>' if output is None else output

    # tokenize at end if need to, so doesn't block generation in multi-generator case
    if extra_dict.get('ntokens') is None:
        extra_dict['ntokens'] = FakeTokenizer().num_tokens_from_string(output)
        # only do below if didn't already compute ntokens, else assume also computed rate
        extra_dict['tokens_persecond'] = extra_dict['ntokens'] / extra_dict['t_generate']

    dict_to_save = dict(prompt=prompt, text=output, time=time.ctime(),
                        base_model=base_model,
                        where_from=where_from,
                        error=error,
                        extra=extra,
                        which_api=which_api,
                        valid_key=valid_key,
                        h2ogpt_key=h2ogpt_key,
                        )
    dict_to_save.update(extra_dict)

    if return_dict:
        return dict_to_save

    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise RuntimeError("save_dir already exists and is not a directory!")
    makedirs(save_dir, exist_ok=True)  # already should be made, can't change at this point
    import json
    with filelock.FileLock("%s.lock" % os.path.basename(save_dir)):
        # lock logging in case have concurrency
        with open(os.path.join(save_dir, "history.json"), "a") as f:
            # just add [ at start, and ] at end, and have proper JSON dataset
            f.write(
                "  " + json.dumps(
                    dict_to_save
                ) + ",\n"
            )


def s3up(filename):
    try:
        return _s3up(filename)
    except Exception as e:
        traceback.print_exc()
        print('Exception for file %s in s3up: %s' % (filename, str(e)))
        return "Failed to upload %s: Error: %s" % (filename, str(e))


def _s3up(filename):
    import boto3

    aws_access_key_id = os.getenv('AWS_SERVER_PUBLIC_KEY')
    aws_secret_access_key = os.getenv('AWS_SERVER_SECRET_KEY')
    bucket = os.getenv('AWS_BUCKET')
    assert aws_access_key_id, "Set AWS key"
    assert aws_secret_access_key, "Set AWS secret"
    assert bucket, "Set AWS Bucket"

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
                      aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY'),
                      )
    ret = s3.upload_file(
        Filename=filename,
        Bucket=os.getenv('AWS_BUCKET'),
        Key=filename,
    )
    if ret in [None, '']:
        return "Successfully uploaded %s" % filename


def get_githash():
    try:
        githash = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8')[0:-1]
    except:
        githash = ''
    return githash


def copy_code(run_id):
    """
    copy code to track changes
    :param run_id:
    :return:
    """
    rnd_num = str(random.randint(0, 2 ** 31))
    run_id = 'run_' + str(run_id)
    os.makedirs(run_id, exist_ok=True)
    me_full = os.path.join(pathlib.Path(__file__).parent.resolve(), __file__)
    me_file = os.path.basename(__file__)
    new_me = os.path.join(run_id, me_file + '_' + get_githash())
    if os.path.isfile(new_me):
        new_me = os.path.join(run_id, me_file + '_' + get_githash() + '_' + rnd_num)
        shutil.copy(me_full, new_me)
    else:
        shutil.copy(me_full, new_me)


class NullContext(threading.local):
    """No-op context manager, executes block without doing any additional processing.

    Used as a stand-in if a particular block of code is only sometimes
    used with a normal context manager:
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finally_act()

    def finally_act(self):
        pass


def wrapped_partial(func, *args, **kwargs):
    """
    Give partial properties of normal function, like __name__ attribute etc.
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


class ThreadException(Exception):
    pass


class EThread(threading.Thread):
    # Function that raises the custom exception
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None, streamer=None, bucket=None):
        self.bucket = bucket
        self.streamer = streamer
        self.exc = None
        self._return = None
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        # Variable that stores the exception, if raised by someFunction
        try:
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            print("thread exception: %s" % str(sys.exc_info()))
            self.bucket.put(sys.exc_info())
            self.exc = e
            if self.streamer:
                print("make stop: %s" % str(sys.exc_info()), flush=True)
                self.streamer.do_stop = True
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self, timeout=None):
        threading.Thread.join(self)
        # Since join() returns in caller thread
        # we re-raise the caught exception
        # if any was caught
        if self.exc:
            raise self.exc
        return self._return


def import_matplotlib():
    import matplotlib
    matplotlib.use('agg')
    # KEEP THESE HERE! START
    import matplotlib.pyplot as plt
    import pandas as pd
    # to avoid dlopen deadlock in fork
    import pandas.core.computation.expressions as pd_expressions
    import pandas._libs.groupby as pd_libgroupby
    import pandas._libs.reduction as pd_libreduction
    import pandas.core.algorithms as pd_algorithms
    import pandas.core.common as pd_com
    import numpy as np
    # KEEP THESE HERE! END


def get_sha(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()


def sanitize_filename(name):
    """
    Sanitize file *base* names.
    :param name: name to sanitize
    :return:
    """
    bad_chars = ['[', ']', ',', '/', '\\', '\\w', '\\s', '-', '+', '\"', '\'', '>', '<', ' ', '=', ')', '(', ':', '^']
    for char in bad_chars:
        name = name.replace(char, "_")

    length = len(name)
    file_length_limit = 250  # bit smaller than 256 for safety
    sha_length = 32
    real_length_limit = file_length_limit - (sha_length + 2)
    if length > file_length_limit:
        sha = get_sha(name)
        half_real_length_limit = max(1, int(real_length_limit / 2))
        name = name[0:half_real_length_limit] + "_" + sha + "_" + name[length - half_real_length_limit:length]

    return name


def shutil_rmtree(*args, **kwargs):
    return shutil.rmtree(*args, **kwargs)


def remove(path: str):
    try:
        if path is not None and os.path.exists(path):
            if os.path.isdir(path):
                shutil_rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
    except:
        pass


def makedirs(path, exist_ok=True, tmp_ok=False, use_base=False):
    """
    Avoid some inefficiency in os.makedirs()
    :param path:
    :param exist_ok:
    :param tmp_ok:  use /tmp if can't write locally
    :param use_base:
    :return:
    """
    if path is None:
        return path
    # if base path set, make relative to that, unless user_path absolute path
    if use_base:
        if os.path.normpath(path) == os.path.normpath(os.path.abspath(path)):
            pass
        else:
            if os.getenv('H2OGPT_BASE_PATH') is not None:
                base_dir = os.path.normpath(os.getenv('H2OGPT_BASE_PATH'))
                path = os.path.normpath(path)
                if not path.startswith(base_dir):
                    path = os.path.join(os.getenv('H2OGPT_BASE_PATH', ''), path)
                    path = os.path.normpath(path)

    if os.path.isdir(path) and os.path.exists(path):
        assert exist_ok, "Path already exists"
        return path
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return path
    except FileExistsError:
        # e.g. soft link
        return path
    except PermissionError:
        if tmp_ok:
            path0 = path
            path = os.path.join('/tmp/', path)
            print("Permission denied to %s, using %s instead" % (path0, path), flush=True)
            os.makedirs(path, exist_ok=exist_ok)
            return path
        else:
            raise


def atomic_move_simple(src, dst):
    try:
        shutil.move(src, dst)
    except (shutil.Error, FileExistsError):
        pass
    remove(src)


def download_simple(url, dest=None):
    if dest is None:
        dest = os.path.basename(url)
    base_path = os.path.dirname(dest)
    if base_path:  # else local path
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        dest = os.path.join(base_path, os.path.basename(dest))

    if os.path.isfile(dest):
        print("Already have %s from url %s, delete file if invalid" % (dest, str(url)), flush=True)
        return dest

    print("BEGIN get url %s" % str(url), flush=True)
    if url.startswith("file://"):
        from requests_file import FileAdapter
        s = requests.Session()
        s.mount('file://', FileAdapter())
        url_data = s.get(url, stream=True)
    else:
        url_data = requests.get(url, stream=True)
    print("GOT url %s" % str(url), flush=True)

    if url_data.status_code != requests.codes.ok:
        msg = "Cannot get url %s, code: %s, reason: %s" % (
            str(url),
            str(url_data.status_code),
            str(url_data.reason),
        )
        raise requests.exceptions.RequestException(msg)
    url_data.raw.decode_content = True

    uuid_tmp = str(uuid.uuid4())[:6]
    dest_tmp = dest + "_dl_" + uuid_tmp + ".tmp"
    with open(dest_tmp, "wb") as f:
        shutil.copyfileobj(url_data.raw, f)
    atomic_move_simple(dest_tmp, dest)
    print("DONE url %s" % str(url), flush=True)
    return dest


def download(url, dest=None, dest_path=None):
    if dest_path is not None:
        dest = os.path.join(dest_path, os.path.basename(url))
        if os.path.isfile(dest):
            print("already downloaded %s -> %s" % (url, dest))
            return dest
    elif dest is not None:
        if os.path.exists(dest):
            print("already downloaded %s -> %s" % (url, dest))
            return dest
    else:
        uuid_tmp = "dl2_" + str(uuid.uuid4())[:6]
        dest = uuid_tmp + os.path.basename(url)

    print("downloading %s to %s" % (url, dest))

    if url.startswith("file://"):
        from requests_file import FileAdapter
        s = requests.Session()
        s.mount('file://', FileAdapter())
        url_data = s.get(url, stream=True)
    else:
        url_data = requests.get(url, stream=True)

    if url_data.status_code != requests.codes.ok:
        msg = "Cannot get url %s, code: %s, reason: %s" % (
            str(url), str(url_data.status_code), str(url_data.reason))
        raise requests.exceptions.RequestException(msg)
    url_data.raw.decode_content = True
    dirname = os.path.dirname(dest)
    if dirname != "" and not os.path.isdir(dirname):
        base_path = os.path.dirname(dest)
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        dest = os.path.join(base_path, os.path.basename(dest))
    uuid_tmp = "dl3_" + str(uuid.uuid4())[:6]
    dest_tmp = dest + "_" + uuid_tmp + ".tmp"
    with open(dest_tmp, 'wb') as f:
        shutil.copyfileobj(url_data.raw, f)
    try:
        shutil.move(dest_tmp, dest)
    except FileExistsError:
        pass
    remove(dest_tmp)
    return dest


def get_doc(x):
    return x.page_content


def get_source(x):
    return x.metadata.get('source', "UNKNOWN SOURCE")


def get_accordion(x, font_size=2, head_acc=50):
    title = x.page_content[:head_acc].replace("\n", ' ').replace("<br>", ' ').replace("<p>", ' ').replace("\r", ' ')
    content = x.page_content
    return f"""<details><summary><font size="{font_size}">{title}</font></summary><font size="{font_size}">{content}</font></details>"""


def get_url(x, from_str=False, short_name=False, font_size=2):
    if not from_str:
        source = x.metadata['source']
    else:
        source = x
    if short_name:
        source_name = get_short_name(source)
    else:
        source_name = source
    if source.startswith('http://') or source.startswith('https://'):
        return """<font size="%s"><a href="%s" target="_blank"  rel="noopener noreferrer">%s</a></font>""" % (
            font_size, source, source_name)
    else:
        return """<font size="%s"><a href="file/%s" target="_blank"  rel="noopener noreferrer">%s</a></font>""" % (
            font_size, source, source_name)


def get_short_name(name, maxl=50):
    if name is None:
        return ''
    length = len(name)
    if length > maxl:
        allow_length = maxl - 3
        half_allowed = max(1, int(allow_length / 2))
        name = name[0:half_allowed] + "..." + name[length - half_allowed:length]
    return name


def cuda_vis_check(total_gpus):
    """Helper function to count GPUs by environment variable
    Stolen from Jon's h2o4gpu utils
    """
    cudavis = os.getenv("CUDA_VISIBLE_DEVICES")
    which_gpus = []
    if cudavis is not None:
        # prune away white-space, non-numerics,
        # except commas for simple checking
        cudavis = "".join(cudavis.split())
        import re
        cudavis = re.sub("[^0-9,]", "", cudavis)

        lencudavis = len(cudavis)
        if lencudavis == 0:
            total_gpus = 0
        else:
            total_gpus = min(
                total_gpus,
                os.getenv("CUDA_VISIBLE_DEVICES").count(",") + 1)
            which_gpus = os.getenv("CUDA_VISIBLE_DEVICES").split(",")
            which_gpus = [int(x) for x in which_gpus]
    else:
        which_gpus = list(range(0, total_gpus))

    return total_gpus, which_gpus


def get_ngpus_vis(raise_if_exception=True):
    ngpus_vis1 = 0

    shell = False
    if shell:
        cmd = "nvidia-smi -L 2> /dev/null"
    else:
        cmd = ["nvidia-smi", "-L"]

    try:
        timeout = 5 * 3
        o = subprocess.check_output(cmd, shell=shell, timeout=timeout)
        lines = o.decode("utf-8").splitlines()
        ngpus_vis1 = 0
        for line in lines:
            if 'Failed to initialize NVML' not in line:
                ngpus_vis1 += 1
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        # GPU systems might not have nvidia-smi, so can't fail
        pass
    except subprocess.TimeoutExpired as e:
        print('Failed get_ngpus_vis: %s' % str(e))
        if raise_if_exception:
            raise

    ngpus_vis1, which_gpus = cuda_vis_check(ngpus_vis1)
    return ngpus_vis1


def get_mem_gpus(raise_if_exception=True, ngpus=None):
    totalmem_gpus1 = 0
    usedmem_gpus1 = 0
    freemem_gpus1 = 0

    if ngpus == 0:
        return totalmem_gpus1, usedmem_gpus1, freemem_gpus1

    try:
        cmd = "nvidia-smi -q 2> /dev/null | grep -A 3 'FB Memory Usage'"
        o = subprocess.check_output(cmd, shell=True, timeout=15)
        lines = o.decode("utf-8").splitlines()
        for line in lines:
            if 'Total' in line:
                totalmem_gpus1 += int(line.split()[2]) * 1024 ** 2
            if 'Used' in line:
                usedmem_gpus1 += int(line.split()[2]) * 1024 ** 2
            if 'Free' in line:
                freemem_gpus1 += int(line.split()[2]) * 1024 ** 2
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        # GPU systems might not have nvidia-smi, so can't fail
        pass
    except subprocess.TimeoutExpired as e:
        print('Failed get_mem_gpus: %s' % str(e))
        if raise_if_exception:
            raise

    return totalmem_gpus1, usedmem_gpus1, freemem_gpus1


class ForkContext(threading.local):
    """
        Set context for forking
        Ensures state is returned once done
    """

    def __init__(self, args=None, kwargs=None, forkdata_capable=True):
        """
        :param args:
        :param kwargs:
        :param forkdata_capable: whether fork is forkdata capable and will use copy-on-write forking of args/kwargs
        """
        self.forkdata_capable = forkdata_capable
        if self.forkdata_capable:
            self.has_args = args is not None
            self.has_kwargs = kwargs is not None
            forkdatacontext.args = args
            forkdatacontext.kwargs = kwargs
        else:
            self.has_args = False
            self.has_kwargs = False

    def __enter__(self):
        try:
            # flush all outputs so doesn't happen during fork -- don't print/log inside ForkContext contexts!
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException as e:
            # exit not called if exception, and don't want to leave forkdatacontext filled in that case
            print("ForkContext failure on enter: %s" % str(e))
            self.finally_act()
            raise
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finally_act()

    def finally_act(self):
        """
            Done when exception hit or exit is reached in context
            first reset forkdatacontext as crucial to have reset even if later 2 calls fail
        :return: None
        """
        if self.forkdata_capable and (self.has_args or self.has_kwargs):
            forkdatacontext._reset()


class _ForkDataContext(threading.local):
    def __init__(
            self,
            args=None,
            kwargs=None,
    ):
        """
        Global context for fork to carry data to subprocess instead of relying upon copy/pickle/serialization

        :param args: args
        :param kwargs: kwargs
        """
        assert isinstance(args, (tuple, type(None)))
        assert isinstance(kwargs, (dict, type(None)))
        self.__args = args
        self.__kwargs = kwargs

    @property
    def args(self) -> Tuple:
        """returns args"""
        return self.__args

    @args.setter
    def args(self, args):
        if self.__args is not None:
            raise AttributeError(
                "args cannot be overwritten: %s %s" % (str(self.__args), str(self.__kwargs))
            )

        self.__args = args

    @property
    def kwargs(self) -> Dict:
        """returns kwargs"""
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        if self.__kwargs is not None:
            raise AttributeError(
                "kwargs cannot be overwritten: %s %s" % (str(self.__args), str(self.__kwargs))
            )

        self.__kwargs = kwargs

    def _reset(self):
        """Reset fork arg-kwarg context to default values"""
        self.__args = None
        self.__kwargs = None

    def get_args_kwargs(self, func, args, kwargs) -> Tuple[Callable, Tuple, Dict]:
        if self.__args:
            args = self.__args[1:]
            if not func:
                assert len(self.__args) > 0, "if have no func, must have in args"
                func = self.__args[0]  # should always be there
        if self.__kwargs:
            kwargs = self.__kwargs
        try:
            return func, args, kwargs
        finally:
            forkdatacontext._reset()

    @staticmethod
    def get_args_kwargs_for_traced_func(func, args, kwargs):
        """
        Return args/kwargs out of forkdatacontext when using copy-on-write way of passing args/kwargs
        :param func: actual function ran by _traced_func, which itself is directly what mppool treats as function
        :param args:
        :param kwargs:
        :return: func, args, kwargs from forkdatacontext if used, else originals
        """
        # first 3 lines are debug
        func_was_None = func is None
        args_was_None_or_empty = args is None or len(args) == 0
        kwargs_was_None_or_empty = kwargs is None or len(kwargs) == 0

        forkdatacontext_args_was_None = forkdatacontext.args is None
        forkdatacontext_kwargs_was_None = forkdatacontext.kwargs is None
        func, args, kwargs = forkdatacontext.get_args_kwargs(func, args, kwargs)
        using_forkdatacontext = func_was_None and func is not None  # pulled func out of forkdatacontext.__args[0]
        assert forkdatacontext.args is None, "forkdatacontext.args should be None after get_args_kwargs"
        assert forkdatacontext.kwargs is None, "forkdatacontext.kwargs should be None after get_args_kwargs"

        proc_type = kwargs.get('proc_type', 'SUBPROCESS')
        if using_forkdatacontext:
            assert proc_type == "SUBPROCESS" or proc_type == "SUBPROCESS"
        if proc_type == "NORMAL":
            assert forkdatacontext_args_was_None, "if no fork, expect forkdatacontext.args None entering _traced_func"
            assert forkdatacontext_kwargs_was_None, "if no fork, expect forkdatacontext.kwargs None entering _traced_func"
        assert func is not None, "function should not be None, indicates original args[0] was None or args was None"

        return func, args, kwargs


forkdatacontext = _ForkDataContext()


# Add user info
username = getpass.getuser()
current_working_directory = os.getcwd()
operating_system = platform.system()


def _traced_func(func, *args, **kwargs):
    func, args, kwargs = forkdatacontext.get_args_kwargs_for_traced_func(func, args, kwargs)
    return func(*args, **kwargs)


def call_subprocess_onetask(func, args=None, kwargs=None):
    if platform.system() in ['Darwin', 'Windows']:
        return func(*args, **kwargs)
    if isinstance(args, list):
        args = tuple(args)
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    args = list(args)
    args = [func] + args
    args = tuple(args)
    with ForkContext(args=args, kwargs=kwargs):
        args = (None,)
        kwargs = {}
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_traced_func, *args, **kwargs)
            return future.result()


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_kwargs(func, exclude_names=None, **kwargs):
    func_names = list(inspect.signature(func).parameters)
    missing_kwargs = [x for x in func_names if x not in kwargs]
    if exclude_names:
        for k in exclude_names:
            if k in missing_kwargs:
                missing_kwargs.remove(k)
            if k in func_names:
                func_names.remove(k)
    assert not missing_kwargs, "Missing %s" % missing_kwargs
    kwargs = {k: v for k, v in kwargs.items() if k in func_names}
    return kwargs


from importlib.metadata import distribution, PackageNotFoundError

have_faiss = False

try:
    assert distribution('faiss') is not None
    have_faiss = True
except (PackageNotFoundError, AssertionError):
    pass
try:
    assert distribution('faiss_gpu') is not None
    have_faiss = True
except (PackageNotFoundError, AssertionError):
    pass
try:
    assert distribution('faiss_cpu') is not None
    have_faiss = True
except (PackageNotFoundError, AssertionError):
    pass

have_chromamigdb = False
try:
    assert distribution('chromamigdb') is not None
    have_chromamigdb = True
except (PackageNotFoundError, AssertionError):
    pass


def hash_file(file):
    try:
        import hashlib

        # BUF_SIZE is totally arbitrary, change for your app!
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

        md5 = hashlib.md5()
        # sha1 = hashlib.sha1()

        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
                # sha1.update(data)
    except BaseException as e:
        print("Cannot hash %s due to %s" % (file, str(e)))
        traceback.print_exc()
        return ''
    return md5.hexdigest()


def start_faulthandler():
    # If hit server or any subprocess with signal SIGUSR1, it'll print out all threads stack trace, but wont't quit or coredump
    # If more than one fork tries to write at same time, then looks corrupted.
    import faulthandler

    # SIGUSR1 in h2oai/__init__.py as well
    faulthandler.enable()
    if hasattr(faulthandler, 'register'):
        # windows/mac
        import signal
        faulthandler.register(signal.SIGUSR1)


def get_hf_server(inference_server):
    inf_split = inference_server.split("    ")
    assert len(inf_split) == 1 or len(inf_split) == 3
    inference_server = inf_split[0]
    if len(inf_split) == 3:
        headers = {"authorization": "%s %s" % (inf_split[1], inf_split[2])}
    else:
        headers = None
    return inference_server, headers


class FakeTokenizer:
    """
    1) For keeping track of model_max_length
    2) For when model doesn't directly expose tokenizer but need to count tokens
    """

    def __init__(self, model_max_length=2048, encoding_name="cl100k_base"):
        # dont' push limit, since if using fake tokenizer, only estimate, and seen underestimates by order 250
        self.model_max_length = model_max_length - 250
        self.encoding_name = encoding_name
        # The first time this runs, it will require an internet connection to download. Later runs won't need an internet connection.
        import tiktoken
        self.encoding = tiktoken.get_encoding(self.encoding_name)

    def encode(self, x, *args, return_tensors="pt", **kwargs):
        input_ids = self.encoding.encode(x, disallowed_special=())
        if return_tensors == 'pt' and isinstance(input_ids, list):
            import torch
            input_ids = torch.tensor(input_ids)
        return dict(input_ids=input_ids)

    def decode(self, x, *args, **kwargs):
        # input is input_ids[0] form
        return self.encoding.decode(x)

    def num_tokens_from_string(self, prompt: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encode(prompt)['input_ids'])
        return num_tokens

    def __call__(self, x, *args, **kwargs):
        return self.encode(x, *args, **kwargs)


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


try:
    assert distribution('langchain') is not None
    have_langchain = True
except (PackageNotFoundError, AssertionError):
    have_langchain = False

import distutils.spawn

have_tesseract = distutils.spawn.find_executable("tesseract")
have_libreoffice = distutils.spawn.find_executable("libreoffice")
try:
    from weasyprint import HTML
    import doctr
    have_doctr = True
except:
    have_doctr = False

try:
    assert distribution('arxiv') is not None
    assert distribution('pymupdf') is not None
    have_arxiv = True
except (PackageNotFoundError, AssertionError):
    have_arxiv = False

try:
    assert distribution('pymupdf') is not None
    have_pymupdf = True
except (PackageNotFoundError, AssertionError):
    have_pymupdf = False

try:
    assert distribution('selenium') is not None
    have_selenium = True
except (PackageNotFoundError, AssertionError):
    have_selenium = False

try:
    assert distribution('pillow') is not None
    have_pillow = True
except (PackageNotFoundError, AssertionError):
    have_pillow = False

try:
    assert distribution('playwright') is not None
    have_playwright = True
except (PackageNotFoundError, AssertionError):
    have_playwright = False

try:
    assert distribution('jq') is not None
    have_jq = True
except (PackageNotFoundError, AssertionError):
    have_jq = False

only_unstructured_urls = os.environ.get("ONLY_UNSTRUCTURED_URLS", "0") == "1"
only_selenium = os.environ.get("ONLY_SELENIUM", "0") == "1"
only_playwright = os.environ.get("ONLY_PLAYWRIGHT", "0") == "1"


def set_openai(inference_server):
    if inference_server.startswith('vllm'):
        import openai_vllm
        openai_vllm.api_key = "EMPTY"
        inf_type = inference_server.split(':')[0]
        ip_vllm = inference_server.split(':')[1]
        port_vllm = inference_server.split(':')[2]
        openai_vllm.api_base = f"http://{ip_vllm}:{port_vllm}/v1"
        return openai_vllm, inf_type, None, None, None
    else:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        base_url = None
        deployment_type = None
        api_version = None
        inf_type = inference_server.split(':')[0]
        if len(inference_server.split(':')) >= 2:
            deployment_type = inference_server.split(':')[1]
        if len(inference_server.split(':')) >= 3:
            base_url = inference_server.split(':')[2]
        if len(inference_server.split(':')) >= 4:
            api_version = inference_server.split(':')[3]

        if deployment_type == 'None':
            deployment_type = None
        if base_url == 'None':
            base_url = None
        if base_url == 'None':
            base_url = None
        return openai, inf_type, deployment_type, base_url, api_version


def get_list_or_str(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        try:
            x1 = ast.literal_eval(x)
            assert isinstance(x1, list)
            return x1
        except:
            return x
    else:
        return x


def deepcopy_by_pickle_object(object):
    """
    Faster deepcopy, can only work on things that are picklable.  Naive Deepcopy is more general.
    Same method as for class Individual
    :param object:
    :return:
    """
    gc.disable()
    new_object = pickle.loads(pickle.dumps(object, -1))
    gc.enable()
    return new_object


def url_alive(url):
    try:
        response = requests.head(url)
    except Exception as e:
        return False
    else:
        if response.status_code in [200, 301, 302]:
            return True
        else:
            return False


def dict_to_html(x, small=True, api=False):
    df = pd.DataFrame(x.items(), columns=['Key', 'Value'])
    df.index = df.index + 1
    df.index.name = 'index'
    if api:
        return tabulate.tabulate(df, headers='keys')
    else:
        res = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
        if small:
            return "<small>" + res + "</small>"
        else:
            return res


def text_to_html(x, api=False):
    if api:
        return x
    return """
<style>
      pre {
        overflow-x: auto;
        white-space: pre-wrap;
        white-space: -moz-pre-wrap;
        white-space: -pre-wrap;
        white-space: -o-pre-wrap;
        word-wrap: break-word;
      }
    </style>
<pre>
%s
</pre>
""" % x


def lg_to_gr(
        **kwargs,
):
    # translate:
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus, _ = cuda_vis_check(n_gpus)

    image_loaders_options = ['Caption']
    if n_gpus != 0:
        image_loaders_options.extend(['CaptionBlip2', 'Pix2Struct'])
    if have_tesseract:
        image_loaders_options.append('OCR')
    if have_doctr:
        image_loaders_options.append('DocTR')

    image_loaders_options0 = []
    if have_tesseract and kwargs['enable_ocr']:
        image_loaders_options0.append('OCR')
    if have_doctr and kwargs['enable_doctr']:
        image_loaders_options0.append('DocTR')
    if kwargs['enable_captions']:
        if kwargs['max_quality'] and n_gpus > 0:
            # BLIP2 only on GPU
            image_loaders_options0.append('CaptionBlip2')
        else:
            image_loaders_options0.append('Caption')

    pdf_loaders_options = ['PyMuPDF', 'Unstructured', 'PyPDF', 'TryHTML']
    if have_tesseract:
        pdf_loaders_options.append('OCR')
    if have_doctr:
        pdf_loaders_options.append('DocTR')

    pdf_loaders_options0 = ['PyMuPDF']
    if kwargs['enable_pdf_ocr'] == 'on':
        pdf_loaders_options0.append('OCR')
    if have_doctr and kwargs['enable_pdf_doctr']:
        pdf_loaders_options0.append('DocTR')

    url_loaders_options = []
    if only_unstructured_urls:
        url_loaders_options.append('Unstructured')
    elif have_selenium and only_selenium:
        url_loaders_options.append('Selenium')
    elif have_playwright and only_playwright:
        url_loaders_options.append('PlayWright')
    else:
        url_loaders_options.append('Unstructured')
        if have_selenium:
            url_loaders_options.append('Selenium')
        if have_playwright:
            url_loaders_options.append('PlayWright')
    url_loaders_options0 = [url_loaders_options[0]]
    
    assert set(image_loaders_options0).issubset(image_loaders_options)
    assert set(pdf_loaders_options0).issubset(pdf_loaders_options)
    assert set(url_loaders_options0).issubset(url_loaders_options)

    return image_loaders_options0, image_loaders_options, \
        pdf_loaders_options0, pdf_loaders_options, \
        url_loaders_options0, url_loaders_options


def fix_json(s):

    # Attempt to parse the string as-is.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == '\n' and not escaped:
                char = '\\n' # Replace the newline character with the escape sequence.
            elif char == '\\':
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}' or char == ']':
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing, we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Close any remaining open structures in the reverse order that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON, return None to indicate failure.
        return None


def wrap_in_try_except(code):
    # Add import traceback
    code = "import traceback\n" + code

    # Parse the input code into an AST
    parsed_code = ast.parse(code)

    # Wrap the entire code's AST in a single try-except block
    try_except = ast.Try(
        body=parsed_code.body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name=None,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id="traceback", ctx=ast.Load()), attr="print_exc", ctx=ast.Load()),
                            args=[],
                            keywords=[]
                        )
                    ),
                ]
            )
        ],
        orelse=[],
        finalbody=[]
    )

    # Assign the try-except block as the new body
    parsed_code.body = [try_except]

    # Convert the modified AST back to source code
    return ast.unparse(parsed_code)


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

            yield out_line, err_line


def start_process(cmd):
    start_cmd = sys.executable + " -i -q -u"
    print_cmd = 'print("{}")'
    cmd = [start_cmd] + [cmd]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.write(c)
