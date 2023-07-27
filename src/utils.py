import contextlib
import functools
import hashlib
import inspect
import os
import gc
import pathlib
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime

import filelock
import requests, uuid
from typing import Tuple, Callable, Dict
from tqdm.auto import tqdm
from joblib import Parallel
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


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
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True)
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
                         extra_dict={}):
    try:
        return _save_generate_output(prompt=prompt, output=output, base_model=base_model, save_dir=save_dir,
                                     where_from=where_from, extra_dict=extra_dict)
    except Exception as e:
        traceback.print_exc()
        print('Exception in saving: %s' % str(e))


def _save_generate_output(prompt=None, output=None, base_model=None, save_dir=None, where_from='unknown where from',
                          extra_dict={}):
    """
    Save conversation to .json, row by row.
    json_file_path is path to final JSON file. If not in ., then will attempt to make directories.
    Appends if file exists
    """
    prompt = '<not set>' if prompt is None else prompt
    output = '<not set>' if output is None else output
    assert save_dir, "save_dir must be provided"
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise RuntimeError("save_dir already exists and is not a directory!")
    os.makedirs(save_dir, exist_ok=True)
    import json
    dict_to_save = dict(prompt=prompt, text=output, time=time.ctime(), base_model=base_model, where_from=where_from)
    dict_to_save.update(extra_dict)
    with filelock.FileLock("save_dir.lock"):
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


def makedirs(path, exist_ok=True, tmp_ok=False):
    """
    Avoid some inefficiency in os.makedirs()
    :param path:
    :param exist_ok:
    :param tmp_ok:
    :return:
    """
    if os.path.isdir(path) and os.path.exists(path):
        assert exist_ok, "Path already exists"
        return path
    try:
        os.makedirs(path, exist_ok=exist_ok)
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


def download_simple(url, dest=None, print_func=None):
    if print_func is not None:
        print_func("BEGIN get url %s" % str(url))
    if url.startswith("file://"):
        from requests_file import FileAdapter
        s = requests.Session()
        s.mount('file://', FileAdapter())
        url_data = s.get(url, stream=True)
    else:
        url_data = requests.get(url, stream=True)
    if dest is None:
        dest = os.path.basename(url)
    if url_data.status_code != requests.codes.ok:
        msg = "Cannot get url %s, code: %s, reason: %s" % (
            str(url),
            str(url_data.status_code),
            str(url_data.reason),
        )
        raise requests.exceptions.RequestException(msg)
    url_data.raw.decode_content = True
    base_path = os.path.dirname(dest)
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True)
    dest = os.path.join(base_path, os.path.basename(dest))
    uuid_tmp = str(uuid.uuid4())[:6]
    dest_tmp = dest + "_dl_" + uuid_tmp + ".tmp"
    with open(dest_tmp, "wb") as f:
        shutil.copyfileobj(url_data.raw, f)
    atomic_move_simple(dest_tmp, dest)
    if print_func is not None:
        print_func("END get url %s" % str(url))


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
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True)
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


def get_url(x, from_str=False, short_name=False):
    if not from_str:
        source = x.metadata['source']
    else:
        source = x
    if short_name:
        source_name = get_short_name(source)
    else:
        source_name = source
    if source.startswith('http://') or source.startswith('https://'):
        return """<a href="%s" target="_blank"  rel="noopener noreferrer">%s</a>""" % (
            source, source_name)
    else:
        return """<a href="file/%s" target="_blank"  rel="noopener noreferrer">%s</a>""" % (
            source, source_name)


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


def _traced_func(func, *args, **kwargs):
    func, args, kwargs = forkdatacontext.get_args_kwargs_for_traced_func(func, args, kwargs)
    return func(*args, **kwargs)


def call_subprocess_onetask(func, args=None, kwargs=None):
    import platform
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


import pkg_resources

have_faiss = False

try:
    assert pkg_resources.get_distribution('faiss') is not None
    have_faiss = True
except (pkg_resources.DistributionNotFound, AssertionError):
    pass
try:
    assert pkg_resources.get_distribution('faiss_gpu') is not None
    have_faiss = True
except (pkg_resources.DistributionNotFound, AssertionError):
    pass
try:
    assert pkg_resources.get_distribution('faiss_cpu') is not None
    have_faiss = True
except (pkg_resources.DistributionNotFound, AssertionError):
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
        md5 = None
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
    assert pkg_resources.get_distribution('langchain') is not None
    have_langchain = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_langchain = False

import distutils.spawn

have_tesseract = distutils.spawn.find_executable("tesseract")
have_libreoffice = distutils.spawn.find_executable("libreoffice")

import pkg_resources

try:
    assert pkg_resources.get_distribution('arxiv') is not None
    assert pkg_resources.get_distribution('pymupdf') is not None
    have_arxiv = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_arxiv = False

try:
    assert pkg_resources.get_distribution('pymupdf') is not None
    have_pymupdf = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_pymupdf = False

try:
    assert pkg_resources.get_distribution('selenium') is not None
    have_selenium = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_selenium = False

try:
    assert pkg_resources.get_distribution('playwright') is not None
    have_playwright = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_playwright = False

# disable, hangs too often
have_playwright = False


def set_openai(inference_server):
    if inference_server.startswith('vllm'):
        import openai_vllm
        openai_vllm.api_key = "EMPTY"
        inf_type = inference_server.split(':')[0]
        ip_vllm = inference_server.split(':')[1]
        port_vllm = inference_server.split(':')[2]
        openai_vllm.api_base = f"http://{ip_vllm}:{port_vllm}/v1"
        return openai_vllm, inf_type
    else:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        inf_type = inference_server
        return openai, inf_type


visible_langchain_modes_file = 'visible_langchain_modes.pkl'


def save_collection_names(langchain_modes, visible_langchain_modes, langchain_mode_paths, LangChainMode, db1s):
    """
    extra controls if UserData type of MyData type
    """

    # use first default MyData hash as general user hash to maintain file
    # if user moves MyData from langchain modes, db will still survive, so can still use hash
    scratch_collection_names = list(db1s.keys())
    user_hash = db1s.get(LangChainMode.MY_DATA.value, '')[1]

    llms = ['LLM', 'Disabled']

    scratch_langchain_modes = [x for x in langchain_modes if x in scratch_collection_names]
    scratch_visible_langchain_modes = [x for x in visible_langchain_modes if x in scratch_collection_names]
    scratch_langchain_mode_paths = {k: v for k, v in langchain_mode_paths.items() if
                                    k in scratch_collection_names and k not in llms}

    user_langchain_modes = [x for x in langchain_modes if x not in scratch_collection_names]
    user_visible_langchain_modes = [x for x in visible_langchain_modes if x not in scratch_collection_names]
    user_langchain_mode_paths = {k: v for k, v in langchain_mode_paths.items() if
                                 k not in scratch_collection_names and k not in llms}

    base_path = 'locks'
    base_path = makedirs(base_path, tmp_ok=True)

    # user
    extra = ''
    file = "%s%s" % (visible_langchain_modes_file, extra)
    with filelock.FileLock(os.path.join(base_path, "%s.lock" % file)):
        with open(file, 'wb') as f:
            pickle.dump((user_langchain_modes, user_visible_langchain_modes, user_langchain_mode_paths), f)

    # scratch
    extra = user_hash
    file = "%s%s" % (visible_langchain_modes_file, extra)
    with filelock.FileLock(os.path.join(base_path, "%s.lock" % file)):
        with open(file, 'wb') as f:
            pickle.dump((scratch_langchain_modes, scratch_visible_langchain_modes, scratch_langchain_mode_paths), f)


def load_collection_enum(extra):
    """
    extra controls if UserData type of MyData type
    """
    file = "%s%s" % (visible_langchain_modes_file, extra)
    langchain_modes_from_file = []
    visible_langchain_modes_from_file = []
    langchain_mode_paths_from_file = {}
    if os.path.isfile(visible_langchain_modes_file):
        try:
            with filelock.FileLock("%s.lock" % file):
                with open(file, 'rb') as f:
                    langchain_modes_from_file, visible_langchain_modes_from_file, langchain_mode_paths_from_file = pickle.load(
                        f)
        except BaseException as e:
            print("Cannot load %s, ignoring error: %s" % (file, str(e)), flush=True)
    for k, v in langchain_mode_paths_from_file.items():
        if v is not None and not os.path.isdir(v) and isinstance(v, str):
            # assume was deleted, but need to make again to avoid extra code elsewhere
            makedirs(v)
    return langchain_modes_from_file, visible_langchain_modes_from_file, langchain_mode_paths_from_file


def remove_collection_enum():
    remove(visible_langchain_modes_file)
