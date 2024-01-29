import os
import sys
import time
import traceback
import webbrowser

# uncomment below to ensure CPU install only uses CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

print('__file__: %s' % __file__)
path1 = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path1)
base_path = os.path.dirname(path1)
sys.path.append(base_path)
os.environ['PYTHONPATH'] = path1
print('path1', path1, flush=True)

os.environ['NLTK_DATA'] = os.path.join(base_path, './nltk_data')
path_list = [os.environ['PATH'],
                     os.path.join(base_path, 'poppler/Library/bin/'),
                     os.path.join(base_path, 'poppler/Library/lib/'),
                     os.path.join(base_path, 'Tesseract-OCR'),
                     os.path.join(base_path, 'ms-playwright'),
                     os.path.join(base_path, 'ms-playwright/chromium-1076/chrome-win'),
                     os.path.join(base_path, 'ms-playwright/ffmpeg-1009'),
                     os.path.join(base_path, 'ms-playwright/firefox-1422/firefox'),
                     os.path.join(base_path, 'ms-playwright/webkit-1883'),
                     os.path.join(base_path, 'rubberband/')]
os.environ['PATH'] = ';'.join(path_list)
print(os.environ['PATH'])

import shutil, errno


def copy_tree(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise


def setup_paths():
    for sub in ['src', 'iterators', 'gradio_utils', 'metrics', 'models', '.']:
        path2 = os.path.join(base_path, '..', sub)
        if os.path.isdir(path2):
            if sub == 'models' and os.path.isfile(os.path.join(path2, 'human.jpg')):
                os.environ['H2OGPT_MODEL_BASE'] = path2
            sys.path.append(path2)
        print(path2, flush=True)

        path2 = os.path.join(path1, '..', sub)
        if os.path.isdir(path2):
            if sub == 'models' and os.path.isfile(os.path.join(path2, 'human.jpg')):
                os.environ['H2OGPT_MODEL_BASE'] = path2
            sys.path.append(path2)
        print(path2, flush=True)

    # for app, avoid forbidden for web access
    if os.getenv('H2OGPT_MODEL_BASE'):
        base0 = os.environ['H2OGPT_MODEL_BASE']
        if 'Programs' in os.environ['H2OGPT_MODEL_BASE']:
            os.environ['H2OGPT_MODEL_BASE'] = os.environ['H2OGPT_MODEL_BASE'].replace('Programs', 'Temp/gradio/')
            if os.path.isdir(os.environ['H2OGPT_MODEL_BASE']):
                shutil.rmtree(os.environ['H2OGPT_MODEL_BASE'], ignore_errors=True)
            if os.path.isfile(os.path.join(base0, 'human.jpg')):
                copy_tree(base0, os.environ['H2OGPT_MODEL_BASE'])


from importlib.metadata import distribution, PackageNotFoundError

try:
    dtorch = distribution('torch')
    assert dtorch is not None
    have_torch = True
    torch_version = dtorch.version
except (PackageNotFoundError, AssertionError):
    have_torch = False
    torch_version = ''


def _main():
    setup_paths()
    os.environ['h2ogpt_block_gradio_exit'] = 'False'
    os.environ['h2ogpt_score_model'] = ''

    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
    except Exception as e:
        print("No GPUs detected by NVML: %s" % str(e))
        deviceCount = 0

    need_get_gpu_torch = False
    if have_torch and deviceCount > 0:
        if '+cu' not in torch_version:
            need_get_gpu_torch = True
    elif not have_torch and deviceCount > 0:
        need_get_gpu_torch = True

    print("Torch Status: have torch: %s need get gpu torch: %s CVD: %s GPUs: %s" % (have_torch, need_get_gpu_torch, os.getenv('CUDA_VISIBLE_DEVICES'), deviceCount))

    auto_install_torch_gpu = False

    import sys
    if auto_install_torch_gpu and (not have_torch or need_get_gpu_torch) and sys.platform == "win32":
        print("Installing Torch")
        # for one-click, don't have torch installed, install now
        import subprocess
        import sys

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        if os.getenv('TORCH_WHEEL'):
            print("Installing Torch from %s" % os.getenv('TORCH_WHEEL'))
            install(os.getenv('TORCH_WHEEL'))
        else:
            if need_get_gpu_torch:
                wheel_file = "https://h2o-release.s3.amazonaws.com/h2ogpt/torch-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl"
                print("Installing Torch from %s" % wheel_file)
                install(wheel_file)
            # assume cpu torch part of install
            #else:
            #   wheel_file = "https://h2o-release.s3.amazonaws.com/h2ogpt/torch-2.1.2-cp310-cp310-win_amd64.whl"
            #    print("Installing Torch from %s" % wheel_file)
            #    install(wheel_file)
        import importlib
        importlib.invalidate_caches()
        import pkg_resources
        importlib.reload(pkg_resources)  # re-load because otherwise cache would be bad

    from generate import entrypoint_main as main_h2ogpt
    main_h2ogpt()

    server_name = os.getenv('h2ogpt_server_name', os.getenv('H2OGPT_SERVER_NAME', 'localhost'))
    server_port = os.getenv('GRADIO_SERVER_PORT', str(7860))

    url = "http://%s:%s" % (server_name, server_port)
    webbrowser.open(url)

    while True:
        time.sleep(10000)


def main():
    try:
        _main()
    except BaseException as e:
        with open('h2ogpt_exception.log', 'at') as f:
            f.write(traceback.format_exc())
        time.sleep(10)
        raise
    time.sleep(10)


if __name__ == "__main__":
    main()
