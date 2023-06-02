import os
import fire

from gpt_langchain import path_to_docs, get_db, get_some_dbs_from_hf, all_db_zips, some_db_zips, \
    get_embedding, add_to_db


def glob_to_db(user_path, chunk=True, chunk_size=512, verbose=False,
               fail_any_exception=False, n_jobs=-1, url=None,
               enable_captions=True, captions_model=None,
               caption_loader=None,
               enable_ocr=False):
    sources1 = path_to_docs(user_path, verbose=verbose, fail_any_exception=fail_any_exception,
                            n_jobs=n_jobs,
                            chunk=chunk,
                            chunk_size=chunk_size, url=url,
                            enable_captions=enable_captions,
                            captions_model=captions_model,
                            caption_loader=caption_loader,
                            enable_ocr=enable_ocr,
                            )
    return sources1


def make_db_main(use_openai_embedding: bool = False,
                 hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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
                 db_type: str = 'chroma',
                 ):
    """
    # To make UserData db for generate.py, put pdfs, etc. into path user_path and run:
    python make_db.py

    # once db is made, can use in generate.py like:

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData

    or zip-up the db_dir_UserData and share:

    zip -r db_dir_UserData.zip db_dir_UserData

    # To get all db files (except large wiki_full) do:
    python make_db.py --download_some=True

    # To get a single db file from HF:
    python make_db.py --download_one=db_dir_DriverlessAI_docs.zip

    :param use_openai_embedding: Whether to use OpenAI embedding
    :param hf_embedding_model: HF embedding model to use
    :param persist_directory: where to persist db
    :param user_path: where to pull documents from (None means url is not None.  If url is not None, this is ignored.)
    :param url: url to generate documents from (None means user_path is not None)
    :param add_if_exists: Add to db if already exists
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
    :param db_type: Type of db to create. Currently only 'chroma' and 'weaviate' is supported.
    :return: None
    """

    if download_all:
        print("Downloading all (and unzipping): %s" % all_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=all_db_zips)
        if verbose:
            print("DONE", flush=True)
        return
    elif download_some:
        print("Downloading some (and unzipping): %s" % some_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=some_db_zips)
        if verbose:
            print("DONE", flush=True)
        return
    elif download_one:
        print("Downloading %s (and unzipping)" % download_one, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=[[download_one, '', 'Unknown License']])
        if verbose:
            print("DONE", flush=True)
        return

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
                         )
    exceptions = [x for x in sources if x.metadata.get('exception')]
    print("Exceptions: %s" % exceptions, flush=True)
    sources = [x for x in sources if 'exception' not in x.metadata]

    assert len(sources) > 0, "No sources found"
    if db_type == 'chroma':
        db = _create_or_update_chroma_db(sources, use_openai_embedding, persist_directory, add_if_exists, verbose,
                                         hf_embedding_model, collection_name)
    elif db_type == 'weaviate':
        db = _create_or_update_weaviate_db(sources, use_openai_embedding, add_if_exists, verbose, hf_embedding_model,
                                           collection_name)
    else:
        raise ValueError(f"db_type={db_type} not supported")

    assert db is not None
    if verbose:
        print("DONE", flush=True)
    return db


def _create_or_update_weaviate_db(sources, use_openai_embedding, add_if_exists, verbose, hf_embedding_model,
                                  collection_name):
    import weaviate
    from weaviate.embedded import EmbeddedOptions
    from langchain.vectorstores import Weaviate

    # TODO: add support for connecting via docker compose
    client = weaviate.Client(
        embedded_options=EmbeddedOptions()
    )

    index_name = collection_name.replace(' ', '_').capitalize()

    if not add_if_exists:
        if verbose and client.schema.exists(index_name):
            print("Removing %s" % index_name, flush=True)
            client.schema.delete_class(index_name)

        if verbose:
            print("Generating db", flush=True)
        db = get_db(sources,
                    use_openai_embedding=use_openai_embedding,
                    db_type='weaviate',
                    persist_directory=None,
                    langchain_mode='UserData',
                    hf_embedding_model=hf_embedding_model)
    else:
        embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
        db = Weaviate(embedding=embedding, client=client, by_text=False, index_name=index_name)
        add_to_db(db, sources, db_type='weaviate')

    return db


def _create_or_update_chroma_db(sources, use_openai_embedding, persist_directory, add_if_exists, verbose,
                                hf_embedding_model, collection_name):
    if not os.path.isdir(persist_directory) or not add_if_exists:
        if os.path.isdir(persist_directory):
            if verbose:
                print("Removing %s" % persist_directory, flush=True)
            os.remove(persist_directory)
        if verbose:
            print("Generating db", flush=True)
        db = get_db(sources,
                    use_openai_embedding=use_openai_embedding,
                    db_type='chroma',
                    persist_directory=persist_directory,
                    langchain_mode='UserData',
                    hf_embedding_model=hf_embedding_model)
    else:
        # get embedding model
        embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
        from langchain.vectorstores import Chroma
        db = Chroma(embedding_function=embedding,
                    persist_directory=persist_directory,
                    collection_name=collection_name)
        add_to_db(db, sources, db_type='chroma')

    return db


if __name__ == "__main__":
    fire.Fire(make_db_main)
