import json
from docutils import core


def parse_rst_file(filepath):
    with open(filepath, 'r') as f:
        input_data = f.read()
    settings_overrides = {'initial_header_level': 2}
    document = core.publish_doctree(
        source=input_data,
        source_path=filepath,
        settings_overrides=settings_overrides,
    )
    qa_pairs = []
    current_section = None
    current_question = ""
    current_answer = ""
    for node in document.traverse():
        if node.__class__.__name__ == 'section':
            current_section = ""
        elif current_section is not None:
            if node.__class__.__name__ == 'Text':
                if node.astext()[-1] == "?":
                    if current_question:
                        qa_pairs.append((current_question, current_answer))
                    current_question = node.astext()
                    current_answer = ""
                else:
                    current_answer += node.astext()
    if current_answer:
        qa_pairs.append((current_question, current_answer))
    return {k: v for k, v in qa_pairs}


def test_scrape_dai_docs():
    file = "/home/arno/h2oai/docs/faq.rst"
    qa_pairs = parse_rst_file(file)
    save_thing = [{"instruction": k, "output": v} for k, v in qa_pairs.items()]
    output_file = "dai_faq.json"
    with open(output_file, "wt") as f:
        f.write(json.dumps(save_thing, indent=2))


def test_scrape_dai_docs_all():
    import numpy as np
    import glob
    from nltk.tokenize import sent_tokenize
    dd = {}
    np.random.seed(1234)
    files = list(glob.glob("/home/arno/h2oai/docs/**/*rst"))
    np.random.shuffle(files)
    val_count = int(0.05 * len(files))
    train_files = files[val_count:]
    valid_files = files[:val_count]
    things = [
        ("dai_docs.train.json", train_files),
        ("dai_docs.valid.json", valid_files)
    ]
    for LEN in [100, 200, 500]:
        for output_file, ff in things:
            if output_file not in dd:
                dd[output_file] = []
            for f in ff:
                with open(f) as input:
                    blob = input.read()
                    blob = blob.replace("~~", "")
                    blob = blob.replace("==", "")
                    blob = blob.replace("''", "")
                    blob = blob.replace("--", "")
                    blob = blob.replace("**", "")
                    sentences = sent_tokenize(blob)
                    my_string = ""
                    for sentence in sentences:
                        if len(my_string) < LEN:
                            my_string += " " + sentence
                        else:
                            dd[output_file].append(my_string)
                            my_string = ""
    for output_file, _ in things:
        save_thing = [{"output": k} for k in dd[output_file]]
        with open(output_file, "wt") as f:
            f.write(json.dumps(save_thing, indent=2))
