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
    filename, clean_name, page = generate_unique_filename(output)
    assert (filename, clean_name, page) == ('Zulu_is_hot__page_1.txt', 'Zulu_is_hot_', '1')


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
        ("<doc><name>Zulu is hot..pdf</name>\n<page>1</page></doc>", "Zulu_is_hot__page_1.txt"),
        ("<name>Missing page.pdf</name>", "Missing_page_page_0.txt"),
        ("<page>5</page>", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_5\.txt"),
        ("No XML tags here", r"unparseable_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_0\.txt"),
        ("", r"unknown_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_page_0\.txt"),
    ]

    for i, (input_xml, expected_pattern) in enumerate(test_cases):
        extracted = extract_xml_tags(input_xml)
        filename, clean_name, page = generate_unique_filename(extracted)

        assert re.fullmatch(expected_pattern,
                            filename), f"Test case {i + 1} failed. Expected pattern: {expected_pattern}, Got: {filename}"


def test_generate_unique_filename_multiple_returns():
    meta_datas = [
        "<name>Zulu is hot..pdf</name>\n<page>1</page>",
        "<name>Missing page.pdf</name>",
        "<page>5</page>",
        "No XML tags here",
        ""
    ]

    from openai_server.backend_utils import generate_unique_filename
    from openai_server.backend_utils import extract_xml_tags
    results = [generate_unique_filename(extract_xml_tags(x)) for x in meta_datas]
    file_names, cleaned_names, pages = zip(*results)

    print("File names:", file_names)
    print("Cleaned names:", cleaned_names)
    print("Pages:", pages)

    # Assertions to verify the results
    assert len(file_names) == len(meta_datas)
    assert len(cleaned_names) == len(meta_datas)
    assert len(pages) == len(meta_datas)

    assert file_names[0] == "Zulu_is_hot__page_1.txt"
    assert cleaned_names[0] == "Zulu_is_hot_"
    assert pages[0] == "1"

    assert file_names[1].endswith("_page_0.txt")
    assert cleaned_names[1] == "Missing_page"
    assert pages[1] == "0"

    assert pages[2] == "5"
    assert file_names[3].startswith("unparseable_")
    assert file_names[4].startswith("unknown_")
