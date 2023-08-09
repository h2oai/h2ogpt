import os
import sys
import time
import webbrowser

os.environ['NLTK_DATA'] = './nltk_data'
os.environ['PATH'] = os.environ['PATH'] + ';' + 'poppler/Library/bin/' + ';' + '../Tesseract-OCR'
print(os.environ['PATH'])

print(__file__)
path1 = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path1)
os.environ['PYTHONPATH'] = path1
print(path1)
for sub in ['src', 'iterators', 'gradio_utils', 'metrics', 'models', '.']:
    path2 = os.path.join(os.path.dirname(__file__), '..', sub)
    sys.path.append(path2)
    print(path2)


def main():

    from gen import main as main_h2ogpt
    main_h2ogpt(block_gradio_exit=False)

    url = "http://localhost:%s" % os.getenv('GRADIO_SERVER_PORT', str(7860))
    webbrowser.open(url)

    while True:
        time.sleep(10000)


if __name__ == "__main__":
    main()
