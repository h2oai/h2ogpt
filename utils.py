import functools
import os
import gc
import pathlib
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
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def ping():
    print('Ping: %s' % str(datetime.now()), flush=True)


def get_torch_allocated():
    import torch
    return torch.cuda.memory_allocated()


def system_info():
    import psutil

    system = {}
    # https://stackoverflow.com/questions/48951136/plot-multiple-graphs-in-one-plot-using-tensorboard
    # https://arshren.medium.com/monitoring-your-devices-in-python-5191d672f749
    temps = psutil.sensors_temperatures(fahrenheit=False)
    if 'coretemp' in temps:
        coretemp = temps['coretemp']
        temp_dict = {k.label: k.current for k in coretemp}
        for k, v in temp_dict.items():
            system['CPU_C/%s' % k] = v

    # https://github.com/gpuopenanalytics/pynvml/blob/master/help_query_gpu.txt
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


def zip_data(root_dirs=None, zip_file=None, base_dir='./'):
    try:
        return _zip_data(zip_file=zip_file, base_dir=base_dir, root_dirs=root_dirs)
    except Exception as e:
        traceback.print_exc()
        print('Exception in zipping: %s' % str(e))


def _zip_data(root_dirs=None, zip_file=None, base_dir='./'):
    if zip_file is None:
        datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
        host_name = os.getenv('HF_HOSTNAME', 'emptyhost')
        zip_file = "data_%s_%s.zip" % (datetime_str, host_name)
    assert root_dirs is not None

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


def save_generate_output(output=None, base_model=None, save_dir=None):
    try:
        return _save_generate_output(output=output, base_model=base_model, save_dir=save_dir)
    except Exception as e:
        traceback.print_exc()
        print('Exception in saving: %s' % str(e))


def _save_generate_output(output=None, base_model=None, save_dir=None):
    """
    Save conversation to .json, row by row.
    json_file_path is path to final JSON file. If not in ., then will attempt to make directories.
    Appends if file exists
    """
    assert save_dir, "save_dir must be provided"
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise RuntimeError("save_dir already exists and is not a directory!")
    os.makedirs(save_dir, exist_ok=True)
    import json
    if output[-10:] == '\n\n<human>:':
        # remove trailing <human>:
        output = output[:-10]
    with filelock.FileLock("save_dir.lock"):
        # lock logging in case have concurrency
        with open(os.path.join(save_dir, "history.json"), "a") as f:
            # just add [ at start, and ] at end, and have proper JSON dataset
            f.write(
                "  " + json.dumps(
                    dict(text=output, time=time.ctime(), base_model=base_model)
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
