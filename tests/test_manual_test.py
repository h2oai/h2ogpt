import os
import pytest
from tests.utils import wrap_test_forked


pytestmark = pytest.mark.skipif(os.getenv('SKIP_MANUAL_TESTS', None) is not None, reason="manual tests.")


def test_chat_context():
    # on h2oai/h2ogpt-oasst1-512-20b
    instruction = """Rephrase in 5 different ways: “Apple a day keeps the doctor away.”"""
    expected_response = """1. “A apple every day will keep you healthy.”
2. “An Apple A Day Keeps The Doctor Away”
3. “One of these apples each and everyday, is all it takes to stay well”
4. “Eat an apple daily for good health!”
5. “If eaten one per day, this fruit can help prevent disease”.

I hope that helps! Let me know if there’s anything else I could do for you today?"""
    instruction2 = """Summarize into single sentence."""
    expected_response2 = """“The more fruits we eat, the healthier.” - Dr. John Yiamouyiannis (American physician)"""

    # NOTE: if something broken, might say something unrelated to first question, e.g.
    unexpected_response2 = """I am an AI language model ..."""

    raise NotImplementedError("MANUAL TEST FOR NOW")


def test_upload_one_file():
    raise NotImplementedError("MANUAL TEST FOR NOW -- do and ask query of file")


def test_upload_multiple_file():
    raise NotImplementedError("MANUAL TEST FOR NOW -- do and ask query of files")


def test_upload_url():
    raise NotImplementedError("MANUAL TEST FOR NOW -- put in URL box https://github.com/h2oai/h2ogpt/ (and ask what is h2ogpt?). Ensure can go to source links")


def test_upload_arxiv():
    raise NotImplementedError("MANUAL TEST FOR NOW -- paste in arxiv:1706.03762 and ask who wrote attention paper. Ensure can go to source links")


def test_upload_pasted_text():
    raise NotImplementedError("MANUAL TEST FOR NOW -- do and see test code for what to try")

    # Text: "Yufuu is a wonderful place and you should really visit because there is lots of sun."
    # Choose MyData
    # Ask: "Why should I visit Yufuu?"
    # Expected: ~Text


def test_no_db_dirs():
    raise NotImplementedError("""MANUAL TEST FOR NOW -- Remove db_dirs, ensure can still start up and use in MyData Mode.
    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=MyData
    """)


def test_upload_unsupported_file():
    raise NotImplementedError("""MANUAL TEST FOR NOW -- e.g. json, ensure error correct and reasonable, no cascades""")


def test_upload_to_UserData_and_MyData():
    raise NotImplementedError("""MANUAL TEST FOR NOW Upload to each when enabled, ensure no failures""")


def test_chat_control():
    raise NotImplementedError("""MANUAL TEST FOR NOW save chat, select chats, clear chat, export, import, etc.""")


def test_subset_only():
    raise NotImplementedError("""MANUAL TEST FOR NOW UserData, Select Only for subset, then put in whisper.  Ensure get back only chunks of data with url links to data sources.""")


def test_add_new_doc():
    raise NotImplementedError("""MANUAL TEST FOR NOW UserData, add new pdf or file to user_path and see if pushing refresh sources updates and shows new file in list, then ask question about that new doc""")


def test_model_lock():
    raise NotImplementedError("""MANUAL TEST FOR NOW  UI test of model lock""")
