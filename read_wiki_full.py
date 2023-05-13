"""Load Data from a MediaWiki dump xml."""
import ast
import glob
import pickle
import uuid
from typing import List, Optional
import os
import bz2
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from langchain.docstore.document import Document
from langchain.document_loaders import MWDumpLoader


def unescape(x):
    try:
        x = ast.literal_eval(x)
    except:
        try:
            x = x.encode('ascii', 'ignore').decode('unicode_escape')
        except:
            pass
    return x


class MWDumpDirectLoader(MWDumpLoader):
    def __init__(self, data: str, encoding: Optional[str] = "utf8",
                 title_words_limit=None, use_views=True):
        """Initialize with file path."""
        self.data = data
        self.encoding = encoding
        self.title_words_limit = title_words_limit
        if use_views:
            self.views = pd.read_csv('wiki_page_views_more_1000month.csv')
            self.views.index = self.views['title']
            self.views = self.views['views']
            self.views = self.views.to_dict()
            self.views = {unescape(k): v for k, v in self.views.items()}
        else:
            self.views = None

    def load(self) -> List[Document]:
        """Load from file path."""
        import mwparserfromhell
        import mwxml

        dump = mwxml.Dump.from_page_xml(self.data)

        docs = []

        for page in dump.pages:
            if self.views is not None and page.title not in self.views:
                print("Skipped %s low views" % page.title, flush=True)
                continue
            for revision in page:
                if self.title_words_limit is not None:
                    num_words = len(' '.join(page.title.split('_')).split(' '))
                    if num_words > self.title_words_limit:
                        print("Skipped %s" % page.title, flush=True)
                        continue
                    if self.views is not None:
                        print("Kept %s views: %s" % (page.title, self.views[page.title]), flush=True)
                    else:
                        print("Kept %s" % page.title, flush=True)

                code = mwparserfromhell.parse(revision.text)
                text = code.strip_code(
                    normalize=True, collapse=True, keep_template_params=False
                )
                metadata = dict(title=page.title,
                                source="https://en.wikipedia.org/wiki/" + page.title,
                                id=page.id,
                                redirect=page.redirect,
                                views=self.views[page.title] if self.views is not None else -1,
                                )
                metadata = {k: v for k, v in metadata.items() if v is not None}
                docs.append(Document(page_content=text, metadata=metadata))

        return docs


def search_index(search_term, index_filename):
    byte_flag = False
    data_length = start_byte = 0
    index_file = open(index_filename, 'r')
    csv_reader = csv.reader(index_file, delimiter=':')
    for line in csv_reader:
        if not byte_flag and search_term == line[2]:
            start_byte = int(line[0])
            byte_flag = True
        elif byte_flag and int(line[0]) != start_byte:
            data_length = int(line[0]) - start_byte
            break
    index_file.close()
    return start_byte, data_length


def get_start_bytes(index_filename):
    index_file = open(index_filename, 'r')
    csv_reader = csv.reader(index_file, delimiter=':')
    start_bytes = set()
    for line in csv_reader:
        start_bytes.add(int(line[0]))
    index_file.close()
    return sorted(start_bytes)


def get_wiki_filenames():
    base_path = '/data/jon/enwiki-20230401-pages-articles-multistream'
    index_file = 'enwiki-20230401-pages-articles-multistream-index.txt'
    index_filename = os.path.join(base_path, index_file)
    wiki_filename = os.path.join(base_path, 'enwiki-20230401-pages-articles-multistream.xml.bz2')
    return index_filename, wiki_filename


def get_documents_by_search_term(search_term):
    index_filename, wiki_filename = get_wiki_filenames()
    start_byte, data_length = search_index(search_term, index_filename)
    with open(wiki_filename, 'rb') as wiki_file:
        wiki_file.seek(start_byte)
        data = bz2.BZ2Decompressor().decompress(wiki_file.read(data_length))

    loader = MWDumpDirectLoader(data.decode())
    documents = loader.load()
    return documents


def get_one_chunk(wiki_filename, start_byte, end_byte, return_file=True, title_words_limit=None, use_views=True):
    data_length = end_byte - start_byte
    with open(wiki_filename, 'rb') as wiki_file:
        wiki_file.seek(start_byte)
        data = bz2.BZ2Decompressor().decompress(wiki_file.read(data_length))

    loader = MWDumpDirectLoader(data.decode(), title_words_limit=title_words_limit, use_views=use_views)
    documents1 = loader.load()
    if return_file:
        base_tmp = "temp_wiki"
        if not os.path.isdir(base_tmp):
            os.makedirs(base_tmp, exist_ok=True)
        filename = os.path.join(base_tmp, str(uuid.uuid4()) + ".tmp.pickle")
        with open(filename, 'wb') as f:
            pickle.dump(documents1, f)
        return filename
    return documents1


from joblib import Parallel, delayed


def get_all_documents(small_test=2, n_jobs=None, use_views=True):
    print("DO get all wiki docs: %s" % small_test, flush=True)
    index_filename, wiki_filename = get_wiki_filenames()
    start_bytes = get_start_bytes(index_filename)
    end_bytes = start_bytes[1:]
    start_bytes = start_bytes[:-1]

    if small_test:
        start_bytes = start_bytes[:small_test]
        end_bytes = end_bytes[:small_test]
        if n_jobs is None:
            n_jobs = 5
    else:
        if n_jobs is None:
            n_jobs = os.cpu_count() // 4

    # default loky backend leads to name space conflict problems
    return_file = True  # large return from joblib hangs
    documents = Parallel(n_jobs=n_jobs, verbose=10, backend='multiprocessing')(
        delayed(get_one_chunk)(wiki_filename, start_byte, end_byte, return_file=return_file, use_views=use_views) for start_byte, end_byte in
        zip(start_bytes, end_bytes))
    if return_file:
        # then documents really are files
        files = documents.copy()
        documents = []
        for fil in files:
            with open(fil, 'rb') as f:
                documents.extend(pickle.load(f))
            os.remove(fil)
    else:
        from functools import reduce
        from operator import concat
        documents = reduce(concat, documents)
    assert isinstance(documents, list)

    print("DONE get all wiki docs", flush=True)
    return documents


def test_by_search_term():
    search_term = 'Apollo'
    assert len(get_documents_by_search_term(search_term)) == 100

    search_term = 'Abstract (law)'
    assert len(get_documents_by_search_term(search_term)) == 100

    search_term = 'Artificial languages'
    assert len(get_documents_by_search_term(search_term)) == 100


def test_start_bytes():
    index_filename, wiki_filename = get_wiki_filenames()
    assert len(get_start_bytes(index_filename)) == 227850


def test_get_all_documents():
    small_test = 20  # 227850
    n_jobs = os.cpu_count() // 4

    assert len(get_all_documents(small_test=small_test, n_jobs=n_jobs, use_views=False)) == small_test * 100

    assert len(get_all_documents(small_test=small_test, n_jobs=n_jobs, use_views=True)) == 429


def get_one_pageviews(fil):
    df1 = pd.read_csv(fil, sep=' ', header=None, names=['region', 'title', 'views', 'foo'], quoting=csv.QUOTE_NONE)
    df1.index = df1['title']
    df1 = df1[df1['region'] == 'en']
    df1 = df1.drop('region', axis=1)
    df1 = df1.drop('foo', axis=1)
    df1 = df1.drop('title', axis=1)  # already index

    base_tmp = "temp_wiki_pageviews"
    if not os.path.isdir(base_tmp):
        os.makedirs(base_tmp, exist_ok=True)
    filename = os.path.join(base_tmp, str(uuid.uuid4()) + ".tmp.csv")
    df1.to_csv(filename, index=True)
    return filename


def test_agg_pageviews():
    gen_files = False
    if gen_files:
        path = '/data/jon/h2o-llm/wiki_pageviews/dumps.wikimedia.org/other/pageviews/2023/2023-04'
        files = glob.glob(os.path.join(path, 'pageviews*.gz'))
        # files = files[:2]  # test
        n_jobs = os.cpu_count() // 2
        csv_files = Parallel(n_jobs=n_jobs, verbose=10, backend='multiprocessing')(delayed(get_one_pageviews)(fil) for fil in files)
    else:
        # to continue without redoing above
        csv_files = glob.glob('/data/jon/h2o-llm/temp_wiki_pageviews/*.csv')

    df_list = []
    for csv_file in csv_files:
        print(csv_file)
        df1 = pd.read_csv(csv_file)
        df_list.append(df1)
    df = pd.concat(df_list, axis=0)
    df = df.groupby('title')['views'].sum().reset_index()
    df.to_csv("wiki_page_views.csv", index=True)


def test_reduce_pageview():
    filename = "wiki_page_views.csv"
    df = pd.read_csv(filename)
    df = df[df['views'] < 1e7]
    #
    plt.hist(df['views'], bins=100, log=True)
    views_avg = np.mean(df['views'])
    views_median = np.median(df['views'])
    plt.title("Views avg: %s median: %s" % (views_avg, views_median))
    plt.savefig(filename.replace('.csv', '.png'))
    plt.close()
    #
    views_limit = 1000
    df = df[df['views'] > views_limit]
    filename = "wiki_page_views_more_1000month.csv"
    df.to_csv(filename, index=True)
    #
    plt.hist(df['views'], bins=100, log=True)
    views_avg = np.mean(df['views'])
    views_median = np.median(df['views'])
    plt.title("Views avg: %s median: %s" % (views_avg, views_median))
    plt.savefig(filename.replace('.csv', '.png'))
    plt.close()
