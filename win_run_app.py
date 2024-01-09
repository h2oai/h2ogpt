import os
import sys
import time
import webbrowser

# ensure CPU install only uses CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

print('__file__: %s' % __file__)
path1 = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path1)
base_path = os.path.dirname(path1)
sys.path.append(base_path)
os.environ['PYTHONPATH'] = path1
print('path1', path1, flush=True)

os.environ['NLTK_DATA'] = os.path.join(base_path, './nltk_data')
os.environ['PATH'] = os.environ['PATH'] + ';' + \
                     os.path.join(base_path, 'poppler/Library/bin/') + ';' + \
                     os.path.join(base_path, 'poppler/Library/lib/') + ';' + \
                     os.path.join(base_path, 'Tesseract-OCR') + \
                     os.path.join(base_path, 'ms-playwright') + \
                     os.path.join(base_path, 'ms-playwright/chromium-1076/chrome-win') + \
                     os.path.join(base_path, 'ms-playwright/ffmpeg-1009') + \
                     os.path.join(base_path, 'ms-playwright/firefox-1422/firefox') + \
                     os.path.join(base_path, 'ms-playwright/webkit-1883') + \
                     os.path.join(base_path, 'rubberband/')
print(os.environ['PATH'])

for sub in ['src', 'iterators', 'gradio_utils', 'metrics', 'models', '.']:
    path2 = os.path.join(base_path, '..', sub)
    sys.path.append(path2)
    print(path2, flush=True)

    path2 = os.path.join(path1, '..', sub)
    sys.path.append(path2)
    print(path2, flush=True)

from importlib.metadata import distribution, PackageNotFoundError

try:
    assert distribution('torch') is not None
    have_torch = True
except (PackageNotFoundError, AssertionError):
    have_torch = False


def main():
    from generate import entrypoint_main as main_h2ogpt
    os.environ['h2ogpt_block_gradio_exit'] = 'False'
    os.environ['h2ogpt_score_model'] = ''

    import sys
    if not have_torch and sys.platform == "win32":
        # for one-click, don't have torch installed, install now
        import subprocess
        import sys

        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        if os.getenv('CUDA_VISIBLE_DEVICES') != '':
            install("https://h2o-release.s3.amazonaws.com/h2ogpt/torch-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl")
        else:
            install("https://h2o-release.s3.amazonaws.com/h2ogpt/torch-2.1.2-cp310-cp310-win_amd64.whl")

    main_h2ogpt()

    server_name = os.getenv('h2ogpt_server_name', os.getenv('H2OGPT_SERVER_NAME', 'localhost'))
    server_port = os.getenv('GRADIO_SERVER_PORT', str(7860))

    url = "http://%s:%s" % (server_name, server_port)
    webbrowser.open(url)

    while True:
        time.sleep(10000)


if __name__ == "__main__":
    main()
