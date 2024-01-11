import ast
import os
import sys
from typing import Union, List

if os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))))

from gpt_langchain import path_to_docs, get_some_dbs_from_hf, all_db_zips, some_db_zips, create_or_update_db, \
    get_persist_directory, get_existing_db
from utils import H2O_Fire, makedirs, n_gpus_global


def glob_to_db(user_path, chunk=True, chunk_size=512, verbose=False,
               fail_any_exception=False, n_jobs=-1, url=None,

               # urls
               use_unstructured=True,
               use_playwright=False,
               use_selenium=False,
               use_scrapeplaywright=False,
               use_scrapehttp=False,

               # pdfs
               use_pymupdf='auto',
               use_unstructured_pdf='auto',
               use_pypdf='auto',
               enable_pdf_ocr='auto',
               try_pdf_as_html='auto',
               enable_pdf_doctr='auto',

               # images
               enable_ocr=False,
               enable_doctr=False,
               enable_pix2struct=False,
               enable_captions=True,
               enable_llava=True,
               enable_transcriptions=True,
               captions_model=None,
               caption_loader=None,
               doctr_loader=None,
               llava_model=None,
               llava_prompt=None,
               asr_model=None,
               asr_loader=None,

               # json
               jq_schema='.[]',
               extract_frames=10,

               db_type=None,
               selected_file_types=None,

               is_public=False):
    assert db_type is not None

    loaders_and_settings = dict(
        # diag/error handling
        verbose=verbose, fail_any_exception=fail_any_exception,
        # speed
        n_jobs=n_jobs,

        # chunking
        chunk=chunk,
        chunk_size=chunk_size,

        # urls
        use_unstructured=use_unstructured,
        use_playwright=use_playwright,
        use_selenium=use_selenium,
        use_scrapeplaywright=use_scrapeplaywright,
        use_scrapehttp=use_scrapehttp,

        # pdfs
        use_pymupdf=use_pymupdf,
        use_unstructured_pdf=use_unstructured_pdf,
        use_pypdf=use_pypdf,
        enable_pdf_ocr=enable_pdf_ocr,
        try_pdf_as_html=try_pdf_as_html,
        enable_pdf_doctr=enable_pdf_doctr,

        # images
        enable_ocr=enable_ocr,
        enable_doctr=enable_doctr,
        enable_pix2struct=enable_pix2struct,
        enable_captions=enable_captions,
        enable_llava=enable_llava,
        enable_transcriptions=enable_transcriptions,
        captions_model=captions_model,
        caption_loader=caption_loader,
        doctr_loader=doctr_loader,
        llava_model=llava_model,
        llava_prompt=llava_prompt,
        asr_model=asr_model,
        asr_loader=asr_loader,

        # json
        jq_schema=jq_schema,
        extract_frames=extract_frames,

        db_type=db_type,
        is_public=is_public,
    )
    sources1 = path_to_docs(user_path,
                            url=url,
                            **loaders_and_settings,
                            selected_file_types=selected_file_types,
                            )
    return sources1


def make_db_main(use_openai_embedding: bool = False,
                 hf_embedding_model: str = None,
                 migrate_embedding_model=False,
                 auto_migrate_db=False,
                 persist_directory: str = None,
                 user_path: str = 'user_path',
                 langchain_type: str = 'shared',
                 url: Union[List[str], str] = None,
                 add_if_exists: bool = True,
                 collection_name: str = 'UserData',
                 verbose: bool = False,
                 chunk: bool = True,
                 chunk_size: int = 512,
                 fail_any_exception: bool = False,
                 download_all: bool = False,
                 download_some: bool = False,
                 download_one: str = None,
                 download_dest: str = None,
                 n_jobs: int = -1,

                 # urls
                 use_unstructured=True,
                 use_playwright=False,
                 use_selenium=False,
                 use_scrapeplaywright=False,
                 use_scrapehttp=False,

                 # pdfs
                 use_pymupdf='auto',
                 use_unstructured_pdf='auto',
                 use_pypdf='auto',
                 enable_pdf_ocr='auto',
                 enable_pdf_doctr='auto',
                 try_pdf_as_html='auto',

                 # images
                 enable_ocr=False,
                 enable_doctr=False,
                 enable_pix2struct=False,
                 enable_captions=True,
                 enable_llava=True,
                 captions_model: str = "Salesforce/blip-image-captioning-base",
                 llava_model: str = None,
                 llava_prompt: str = None,
                 pre_load_image_audio_models: bool = False,
                 caption_gpu: bool = True,
                 # caption_loader=None,  # set internally
                 # doctr_loader=None,  # set internally
                 # asr_loader=None  # set internally
                 enable_transcriptions: bool = True,
                 asr_model: str = "openai/whisper-medium",
                 asr_gpu: bool = True,

                 # json
                 jq_schema='.[]',
                 extract_frames=10,

                 db_type: str = 'chroma',
                 selected_file_types: Union[List[str], str] = None,
                 fail_if_no_sources: bool = True
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
    :param migrate_embedding_model: whether to migrate to newly chosen hf_embedding_model or stick with one in db
    :param auto_migrate_db: whether to migrate database for chroma<0.4 -> >0.4
    :param persist_directory: where to persist db (note generate.py always uses db_dir_<collection name>
           If making personal database for user, set persistent_directory to users/<username>/db_dir_<collection name>
           and pass --langchain_type=personal
    :param user_path: where to pull documents from (None means url is not None.  If url is not None, this is ignored.)
    :param langchain_type: type of database, i.e.. 'shared' or 'personal'
    :param url: url (or urls) to generate documents from (None means user_path is not None)
    :param add_if_exists: Add to db if already exists, but will not add duplicate sources
    :param collection_name: Collection name for new db if not adding
           Normally same as langchain_mode
    :param verbose: whether to show verbose messages
    :param chunk: whether to chunk data
    :param chunk_size: chunk size for chunking
    :param fail_any_exception: whether to fail if any exception hit during ingestion of files
    :param download_all: whether to download all (including 23GB Wikipedia) example databases from h2o.ai HF
    :param download_some: whether to download some small example databases from h2o.ai HF
    :param download_one: whether to download one chosen example databases from h2o.ai HF
    :param download_dest: Destination for downloads
    :param n_jobs: Number of cores to use for ingesting multiple files

    :param use_unstructured: see gen.py
    :param use_playwright: see gen.py
    :param use_selenium: see gen.py
    :param use_scrapeplaywright: see gen.py
    :param use_scrapehttp: see gen.py

    :param use_pymupdf: see gen.py
    :param use_unstructured_pdf: see gen.py
    :param use_pypdf: see gen.py
    :param enable_pdf_ocr: see gen.py
    :param try_pdf_as_html: see gen.py
    :param enable_pdf_doctr: see gen.py

    :param enable_ocr: see gen.py
    :param enable_doctr: see gen.py
    :param enable_pix2struct: see gen.py
    :param enable_captions: Whether to enable captions on images
    :param enable_llava: See gen.py
    :param captions_model: See gen.py
    :param llava_model: See gen.py
    :param llava_prompt: See gen.py
    :param pre_load_image_audio_models: See generate.py
    :param caption_gpu: Caption images on GPU if present

    :param db_type: 'faiss' for in-memory
                    'chroma' (for chroma >= 0.4)
                    'chroma_old' (for chroma < 0.4) -- recommended for large collections
                    'weaviate' for persisted on disk
    :param selected_file_types: File types (by extension) to include if passing user_path
       For a list of possible values, see:
       https://github.com/h2oai/h2ogpt/blob/main/docs/README_LangChain.md#shoosing-document-types
       e.g. --selected_file_types="['pdf', 'html', 'htm']"
    :return: None
    """
    db = None

    if isinstance(selected_file_types, str):
        selected_file_types = ast.literal_eval(selected_file_types)
    if persist_directory is None:
        persist_directory, langchain_type = get_persist_directory(collection_name, langchain_type=langchain_type)
    if download_dest is None:
        download_dest = makedirs('./', use_base=True)

    # match behavior of main() in generate.py for non-HF case
    n_gpus = n_gpus_global
    if n_gpus == 0:
        if hf_embedding_model is None:
            # if no GPUs, use simpler embedding model to avoid cost in time
            hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        if hf_embedding_model is None:
            # if still None, then set default
            hf_embedding_model = 'hkunlp/instructor-large'

    existing_db = False

    if download_all:
        print("Downloading all (and unzipping): %s" % all_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=all_db_zips)
        if verbose:
            print("DONE", flush=True)
        existing_db = True
    elif download_some:
        print("Downloading some (and unzipping): %s" % some_db_zips, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=some_db_zips)
        if verbose:
            print("DONE", flush=True)
        existing_db = True
    elif download_one:
        print("Downloading %s (and unzipping)" % download_one, flush=True)
        get_some_dbs_from_hf(download_dest, db_zips=[[download_one, '', 'Unknown License']])
        if verbose:
            print("DONE", flush=True)
        existing_db = True

    if existing_db:
        load_db_if_exists = True
        langchain_mode = collection_name
        langchain_mode_paths = dict(langchain_mode=None)
        langchain_mode_types = dict(langchain_mode='shared')
        db, use_openai_embedding, hf_embedding_model = \
            get_existing_db(None, persist_directory, load_db_if_exists, db_type,
                            use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model, auto_migrate_db,
                            verbose=False,
                            n_jobs=n_jobs)
        return db, collection_name

    if enable_captions and pre_load_image_audio_models:
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
            caption_loader = 'gpu' if n_gpus > 0 and caption_gpu else 'cpu'
        else:
            caption_loader = False
    if enable_doctr or enable_pdf_ocr in [True, 'auto', 'on']:
        doctr_loader = 'gpu' if n_gpus > 0 and caption_gpu else 'cpu'
    else:
        doctr_loader = False

    if enable_transcriptions:
        asr_loader = 'gpu' if n_gpus > 0 and asr_gpu else 'cpu'
    else:
        asr_loader = False

    if verbose:
        print("Getting sources", flush=True)
    assert user_path is not None or url is not None, "Can't have both user_path and url as None"
    if not url:
        assert os.path.isdir(user_path), "user_path=%s does not exist" % user_path
    sources = glob_to_db(user_path, chunk=chunk, chunk_size=chunk_size, verbose=verbose,
                         fail_any_exception=fail_any_exception, n_jobs=n_jobs, url=url,

                         # urls
                         use_unstructured=use_unstructured,
                         use_playwright=use_playwright,
                         use_selenium=use_selenium,
                         use_scrapeplaywright=use_scrapeplaywright,
                         use_scrapehttp=use_scrapehttp,

                         # pdfs
                         use_pymupdf=use_pymupdf,
                         use_unstructured_pdf=use_unstructured_pdf,
                         use_pypdf=use_pypdf,
                         enable_pdf_ocr=enable_pdf_ocr,
                         try_pdf_as_html=try_pdf_as_html,
                         enable_pdf_doctr=enable_pdf_doctr,

                         # images
                         enable_ocr=enable_ocr,
                         enable_doctr=enable_doctr,
                         enable_pix2struct=enable_pix2struct,
                         enable_captions=enable_captions,
                         enable_llava=enable_llava,
                         enable_transcriptions=enable_transcriptions,
                         captions_model=captions_model,
                         caption_loader=caption_loader,
                         doctr_loader=doctr_loader,
                         llava_model=llava_model,
                         llava_prompt=llava_prompt,
                         # Note: we don't reload doctr model
                         asr_loader=asr_loader,
                         asr_model=asr_model,

                         # json
                         jq_schema=jq_schema,
                         extract_frames=extract_frames,

                         db_type=db_type,
                         selected_file_types=selected_file_types,

                         is_public=False,
                         )
    exceptions = [x for x in sources if x.metadata.get('exception')]
    print("Exceptions: %s/%s %s" % (len(exceptions), len(sources), exceptions), flush=True)
    sources = [x for x in sources if 'exception' not in x.metadata]

    assert len(sources) > 0 or not fail_if_no_sources, "No sources found"
    db = create_or_update_db(db_type, persist_directory,
                             collection_name, user_path, langchain_type,
                             sources, use_openai_embedding, add_if_exists, verbose,
                             hf_embedding_model, migrate_embedding_model, auto_migrate_db,
                             n_jobs=n_jobs)

    assert db is not None or not fail_if_no_sources
    if verbose:
        print("DONE", flush=True)
    return db, collection_name


if __name__ == "__main__":
    H2O_Fire(make_db_main)
