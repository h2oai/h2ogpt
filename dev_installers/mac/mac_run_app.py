import os
import sys
import time
import webbrowser

print('__file__: %s' % __file__)
path1 = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path1)
base_path = os.path.dirname(path1)
sys.path.append(base_path)
os.environ['PYTHONPATH'] = path1
print('PYTHONPATH: ', os.getenv('PYTHONPATH'), end='\n', flush=True)
print('Path_1: ', path1, end='\n', flush=True)

os.environ['NLTK_DATA'] = os.path.join(path1, 'nltk_data')
os.environ['PATH'] = os.environ['PATH'] + ':' + \
                     os.path.join(path1, 'poppler/bin/') + ':' + \
                     os.path.join(path1, 'poppler/lib/') + ':' + \
                     os.path.join(path1, 'Tesseract-OCR')

print('NLTK_DATA: ', os.getenv('NLTK_DATA'), end='\n', flush=True)
print('PATH: ', os.environ['PATH'], end='\n', flush=True)

for sub in ['src', 'iterators', 'gradio_utils', 'metrics', 'models', '.']:
    path2 = os.path.join(path1, 'h2ogpt', sub)
    sys.path.append(path2)
    print('Path_3: ', path2, end='\n', flush=True)


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
