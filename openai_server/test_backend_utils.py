import sys


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
    name_page_dict = extract_xml_tags(xml_input)
    assert name_page_dict == {'name': 'Zulu is hot..pdf', 'page': '1'}

    from openai_server.backend_utils import generate_unique_filename
    filename, clean_name, page = generate_unique_filename(name_page_dict)
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
    assert file_names[3] == 'unknown_page_0.txt'
    assert file_names[4] == 'unknown_page_0.txt'


def test_exif():
    import pyexiv2
    img_file_one = 'tests/image_exif.jpg'
    with pyexiv2.Image(img_file_one) as img:
        metadata = img.read_exif()
    assert metadata is not None and metadata != {}
    print(metadata, file=sys.stderr)
