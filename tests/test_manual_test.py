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
    raise NotImplementedError("MANUAL TEST FOR NOW -- do and ask query of content of url")


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
