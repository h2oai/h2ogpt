def test_extract_xml_tags():
    xml_input = """
<doc>
<name>Zulu is hot..pdf</name>
<page>1</page>
<text>
Zulu is hot.
</text>
</doc>
"""

    from openai_server.backend_utils import extract_xml_tags
    output = extract_xml_tags(xml_input)
    assert output == """<name>Zulu is hot..pdf</name>
<page>1</page>
"""

    from openai_server.backend_utils import generate_unique_filename
    filename = generate_unique_filename(output)
    assert filename == 'Zulu_is_hot__page_1.txt'


def test_deduplicate_filenames():
    original_filenames = [
        "Zulu_is_hot__page_1.txt",
        "Zulu_is_hot__page_1.txt",
        "Zulu_is_hot__page_2.txt",
        "Another_document_page_1.txt",
        "Zulu_is_hot__page_1.txt"
    ]

    expected = [
        "Zulu_is_hot__page_1_chunk_0.txt",
        "Zulu_is_hot__page_1_chunk_1.txt",
        "Zulu_is_hot__page_2.txt",
        "Another_document_page_1.txt",
        "Zulu_is_hot__page_1_chunk_2.txt"
    ]

    from openai_server.backend_utils import deduplicate_filenames
    result = deduplicate_filenames(original_filenames)
    assert result == expected, f"Expected: {expected}, but got: {result}"


import re


def test_robust_xml_functions():
    from openai_server.backend_utils import extract_xml_tags
    from openai_server.backend_utils import generate_unique_filename

    test_cases = [
        ("<name>Zulu is hot..pdf</name>\n<page>1</page>", "Zulu_is_hot__page_1.txt"),
        ("<name>Missing page.pdf</name>", "Missing_page_page_0.txt"),
        ("<page>5</page>", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_5\.txt"),
        ("No XML tags here", r"unparseable_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_0\.txt"),
        ("", r"unknown_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_0\.txt"),
    ]

    for i, (input_xml, expected_pattern) in enumerate(test_cases):
        extracted = extract_xml_tags(input_xml)
        filename = generate_unique_filename(extracted)

        assert re.fullmatch(expected_pattern,
                            filename), f"Test case {i + 1} failed. Expected pattern: {expected_pattern}, Got: {filename}"
