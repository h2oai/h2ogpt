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
                     os.path.join(base_path, 'ms-playwright/webkit-1883')
print(os.environ['PATH'])

for sub in ['src', 'iterators', 'gradio_utils', 'metrics', 'models', '.']:
    path2 = os.path.join(base_path, '..', sub)
    sys.path.append(path2)
    print(path2, flush=True)

    path2 = os.path.join(path1, '..', sub)
    sys.path.append(path2)
    print(path2, flush=True)


def main():
    from generate import entrypoint_main as main_h2ogpt
    os.environ['h2ogpt_block_gradio_exit'] = 'False'
    os.environ['h2ogpt_score_model'] = ''
    main_h2ogpt()

    server_name = os.getenv('h2ogpt_server_name', os.getenv('H2OGPT_SERVER_NAME', 'localhost'))
    server_port = os.getenv('GRADIO_SERVER_PORT', str(7860))

    url = "http://%s:%s" % (server_name, server_port)
    webbrowser.open(url)

    while True:
        time.sleep(10000)


if __name__ == "__main__":
    main()
