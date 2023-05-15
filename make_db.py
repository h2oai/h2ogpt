import os

import fire

from gpt_langchain import path_to_docs, chunk_sources, get_db, get_some_dbs_from_hf, all_db_zips, some_db_zips


def glob_to_db(user_path, chunk=True, chunk_size=512, verbose=False, fail_any_exception=False):
    sources1 = path_to_docs(user_path, verbose=verbose, fail_any_exception=fail_any_exception)
    if chunk:
        sources1 = chunk_sources(sources1, chunk_size=chunk_size)
    return sources1


def main(use_openai_embedding: bool = False,
         persist_directory: str = 'db_dir_UserData',
         user_path: str = 'user_path',
         verbose: bool = False,
         chunk: bool = True,
         chunk_size: int = 512,
         fail_any_exception: bool = False,
         download_all: bool = False,
         download_some: bool = False,
         download_one: str = None,
         download_dest: str = "./",
         ):
    """
    # To make UserData db for generate.py, put pdfs, etc. into path user_path and run:
    python make_db.py

    # once db is made, can use in generate.py like:

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b --langchain_mode=UserData

    or zip-up the db_dir_UserData and share:

    zip -r db_dir_UserData.zip db_dir_UserData

    # To get all db files (except large wiki_full) do:
    python make_db.py --download_some=True

    # To get a single db file from HF:
    python make_db.py --download_one=db_dir_DriverlessAI_docs.zip

    :param use_openai_embedding:
    :param persist_directory:
    :param user_path:
    :param verbose:
    :param chunk:
    :param chunk_size:
    :param fail_any_exception:
    :param download_all:
    :param download_some:
    :param download_one:
    :param download_dest:
    :return:
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

    if verbose:
        print("Getting sources", flush=True)
    assert os.path.isdir(user_path), "user_path=%s does not exist" % user_path
    sources = glob_to_db(user_path, chunk=chunk, chunk_size=chunk_size, verbose=verbose,
                         fail_any_exception=fail_any_exception)
    assert len(sources) > 0, "No sources found"
    if verbose:
        print("Generating db", flush=True)
    db = get_db(sources,
                use_openai_embedding=use_openai_embedding,
                db_type='chroma',
                persist_directory=persist_directory,
                langchain_mode='UserData')
    assert db is not None
    if verbose:
        print("DONE", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
