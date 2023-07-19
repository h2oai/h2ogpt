import os
import fire

from gpt_langchain import path_to_docs, get_some_dbs_from_hf, all_db_zips, some_db_zips, create_or_update_db
from utils import get_ngpus_vis


def glob_to_db(user_path, chunk=True, chunk_size=512, verbose=False,
               fail_any_exception=False, n_jobs=-1, url=None,
               enable_captions=True, captions_model=None,
               caption_loader=None,
               enable_ocr=False,
               enable_pdf_ocr='auto'):
    sources1 = path_to_docs(user_path, verbose=verbose, fail_any_exception=fail_any_exception,
                            n_jobs=n_jobs,
                            chunk=chunk,
                            chunk_size=chunk_size, url=url,
                            enable_captions=enable_captions,
                            captions_model=captions_model,
                            caption_loader=caption_loader,
                            enable_ocr=enable_ocr,
                            enable_pdf_ocr=enable_pdf_ocr,
                            )
    return sources1


def make_db_main(use_openai_embedding: bool = False,
                 hf_embedding_model: str = None,
                 persist_directory: str = 'db_dir_UserData',
                 user_path: str = 'user_path',
                 url: str = None,
                 add_if_exists: bool = True,
                 collection_name: str = 'UserData',
                 verbose: bool = False,
                 chunk: bool = True,
                 chunk_size: int = 512,
                 fail_any_exception: bool = False,
                 download_all: bool = False,
                 download_some: bool = False,
                 download_one: str = None,
                 download_dest: str = "./",
                 n_jobs: int = -1,
                 enable_captions: bool = True,
                 captions_model: str = "Salesforce/blip-image-captioning-base",
                 pre_load_caption_model: bool = False,
                 caption_gpu: bool = True,
                 enable_ocr: bool = False,
                 enable_pdf_ocr: str = 'auto',
                 db_type: str = 'chroma',
                 ):
    """
    # To make UserData db for generate.py, put pdfs, etc. into path user_path and run:
    python src/make_db.py

    # once db is made, can use in generate.py like:

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData

    or zip-up the db_dir_UserData and share:

    zip -r db_dir_UserData.zip db_dir_UserData

    # To get all db files (except large wiki_full) do:
    python src/make_db.py --download_some=True

    # To get a single db file from HF:
    python src/make_db.py --download_one=db_dir_DriverlessAI_docs.zip

    :param use_openai_embedding: Whether to use OpenAI embedding
    :param hf_embedding_model: HF embedding model to use. Like generate.py, uses 'hkunlp/instructor-large' if have GPUs, else "sentence-transformers/all-MiniLM-L6-v2"
    :param persist_directory: where to persist db
    :param user_path: where to pull documents from (None means url is not None.  If url is not None, this is ignored.)
    :param url: url to generate documents from (None means user_path is not None)
    :param add_if_exists: Add to db if already exists, but will not add duplicate sources
    :param collection_name: Collection name for new db if not adding
    :param verbose: whether to show verbose messages
    :param chunk: whether to chunk data
    :param chunk_size: chunk size for chunking
    :param fail_any_exception: whether to fail if any exception hit during ingestion of files
    :param download_all: whether to download all (including 23GB Wikipedia) example databases from h2o.ai HF
    :param download_some: whether to download some small example databases from h2o.ai HF
    :param download_one: whether to download one chosen example databases from h2o.ai HF
    :param download_dest: Destination for downloads
    :param n_jobs: Number of cores to use for ingesting multiple files
    :param enable_captions: Whether to enable captions on images
    :param captions_model: See generate.py
    :param pre_load_caption_model: See generate.py
    :param caption_gpu: Caption images on GPU if present
    :param enable_ocr: Whether to enable OCR on images
    :param enable_pdf_ocr: 'auto' uses OCR on PDFs as backup, good to handle image PDFs
    :param db_type: Type of db to create. Currently only 'chroma' and 'weaviate' is supported.
    :return: None
    """
    db = None

    # match behavior of main() in generate.py for non-HF case
    n_gpus = get_ngpus_vis()
    if n_gpus == 0:
        if hf_embedding_model is None:
            # if no GPUs, use simpler embedding model to avoid cost in time
            hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        if hf_embedding_model is None:
            # if still None, then set default
            hf_embedding_model = 'hkunlp/instructor-large'

    if download_all:
        print("Downloading all (and unzipping): %s" % all_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=all_db_zips)
        if verbose:
            print("DONE", flush=True)
        return db, collection_name
    elif download_some:
        print("Downloading some (and unzipping): %s" % some_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=some_db_zips)
        if verbose:
            print("DONE", flush=True)
        return db, collection_name
    elif download_one:
        print("Downloading %s (and unzipping)" % download_one, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=[[download_one, '', 'Unknown License']])
        if verbose:
            print("DONE", flush=True)
        return db, collection_name

    if enable_captions and pre_load_caption_model:
        # preload, else can be too slow or if on GPU have cuda context issues
        # Inside ingestion, this will disable parallel loading of multiple other kinds of docs
        # However, if have many images, all those images will be handled more quickly by preloaded model on GPU
        from image_captions import H2OImageCaptionLoader
        caption_loader = H2OImageCaptionLoader(None,
                                               blip_model=captions_model,
                                               blip_processor=captions_model,
                                               caption_gpu=caption_gpu,
                                               ).load_model()
    else:
        if enable_captions:
            caption_loader = 'gpu' if caption_gpu else 'cpu'
        else:
            caption_loader = False

    if verbose:
        print("Getting sources", flush=True)
    assert user_path is not None or url is not None, "Can't have both user_path and url as None"
    if not url:
        assert os.path.isdir(user_path), "user_path=%s does not exist" % user_path
    sources = glob_to_db(user_path, chunk=chunk, chunk_size=chunk_size, verbose=verbose,
                         fail_any_exception=fail_any_exception, n_jobs=n_jobs, url=url,
                         enable_captions=enable_captions,
                         captions_model=captions_model,
                         caption_loader=caption_loader,
                         enable_ocr=enable_ocr,
                         enable_pdf_ocr=enable_pdf_ocr,
                         )
    exceptions = [x for x in sources if x.metadata.get('exception')]
    print("Exceptions: %s" % exceptions, flush=True)
    sources = [x for x in sources if 'exception' not in x.metadata]

    assert len(sources) > 0, "No sources found"
    db = create_or_update_db(db_type, persist_directory, collection_name,
                             sources, use_openai_embedding, add_if_exists, verbose,
                             hf_embedding_model)

    assert db is not None
    if verbose:
        print("DONE", flush=True)
    return db, collection_name


if __name__ == "__main__":
    fire.Fire(make_db_main)
