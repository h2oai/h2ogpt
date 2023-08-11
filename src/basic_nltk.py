import os

os.environ['NLTK_DATA'] = './nltk_data'

from nltk.downloader import download

# download('all')
download('tokenizers', download_dir=os.environ['NLTK_DATA'])
download('taggers', download_dir=os.environ['NLTK_DATA'])
download('punkt', download_dir=os.environ['NLTK_DATA'])
download('averaged_perceptron_tagger', download_dir=os.environ['NLTK_DATA'])
download('maxent_treebank_pos_tagger', download_dir=os.environ['NLTK_DATA'])
download('spanish_grammars', download_dir=os.environ['NLTK_DATA'])
