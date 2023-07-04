"""
Dataset creation tools.

Keep to-level imports clean of non-trivial imports for specific tools,
because this file is imported for various purposes
"""

import ast
import concurrent.futures
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import psutil
import pytest
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import flatten_list, remove


def parse_rst_file(filepath):
    with open(filepath, 'r') as f:
        input_data = f.read()
    settings_overrides = {'initial_header_level': 2}
    from docutils import core
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
    home = os.path.expanduser('~')
    file = os.path.join(home, 'h2oai/docs/faq.rst')
    qa_pairs = parse_rst_file(file)
    prompt_type = 'human_bot'
    from prompter import prompt_types
    assert prompt_type in prompt_types
    save_thing = [{"instruction": k, "output": v, 'prompt_type': prompt_type} for k, v in qa_pairs.items()]
    output_file = "dai_faq.json"
    with open(output_file, "wt") as f:
        f.write(json.dumps(save_thing, indent=2))


def test_scrape_dai_docs_all():
    """
    pytest create_data.py::test_scrape_dai_docs_all
    """
    import glob
    import nltk
    nltk.download('punkt')
    dd = {}
    np.random.seed(1234)
    home = os.path.expanduser('~')
    files = list(glob.glob(os.path.join(home, "h2oai/docs/**/*rst")))
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
                    dd[output_file].extend(get_sentences(blob, length=LEN))
    for output_file, _ in things:
        save_thing = [{"output": k.strip(), 'prompt_type': 'plain'} for k in dd[output_file]]
        with open(output_file, "wt") as f:
            f.write(json.dumps(save_thing, indent=2))


def get_sentences(blob, length):
    """
    break-up input text into sentences and then output list of sentences of about length in size
    :param blob:
    :param length:
    :return:
    """
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(blob)
    my_sentences = []
    my_string = ""
    for sentence in sentences:
        if len(my_string) + len(sentence) <= length:
            if my_string:
                my_string += " " + sentence
            else:
                my_string = sentence
        else:
            my_sentences.append(my_string)
            my_string = ""
    return my_sentences or [my_string]


def setup_dai_docs(path=None, dst="working_dir_docs", from_hf=False):
    """
    Only supported if have access to source code or HF token for HF spaces and from_hf=True
    :param path:
    :param dst:
    :param from_hf:
    :return:
    """

    home = os.path.expanduser('~')

    if from_hf:
        # assumes
        from huggingface_hub import hf_hub_download
        # True for case when locally already logged in with correct token, so don't have to set key
        token = os.getenv('HUGGINGFACE_API_TOKEN', True)
        path_to_zip_file = hf_hub_download('h2oai/dai_docs', 'dai_docs.zip', token=token, repo_type='dataset')
        path = 'h2oai'
        import zipfile
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
        path = os.path.join(path, 'docs/**/*')

    if path is None:
        if os.path.isdir(os.path.join(home, 'h2oai')):
            path = os.path.join(home, "h2oai/docs/**/*")
        else:
            assert os.path.isdir(os.path.join(home, 'h2oai.superclean')), '%s does not exist' % path
            path = os.path.join(home, "h2oai.superclean/docs/**/*")
    import glob
    files = list(glob.glob(path, recursive=True))

    # pandoc can't find include files

    remove(dst)
    os.makedirs(dst)

    # copy full tree, for absolute paths in rst
    for fil in files:
        if os.path.isfile(fil):
            shutil.copy(fil, dst)

    # hack for relative path
    scorers_dir = os.path.join(dst, 'scorers')
    makedirs(scorers_dir)
    for fil in glob.glob(os.path.join(dst, '*.frag')):
        shutil.copy(fil, scorers_dir)

    return dst


def rst_to_outputs(files, min_len=30, max_len=2048 // 2 - 30):
    # account for sequence length (context window) including prompt and input and output

    # os.system('pandoc -f rst -t plain ./expert_settings/nlp_settings.rst')
    import pypandoc
    basedir = os.path.abspath(os.getcwd())

    outputs = []
    for fil in files:
        os.chdir(basedir)
        os.chdir(os.path.dirname(fil))
        fil = os.path.basename(fil)
        print("Processing %s" % fil, flush=True)
        # out_format can be one of: asciidoc, asciidoctor, beamer, biblatex, bibtex, commonmark, commonmark_x,
        # context, csljson, docbook, docbook4, docbook5, docx, dokuwiki,
        # dzslides, epub, epub2, epub3, fb2, gfm, haddock, html, html4, html5, icml,
        # ipynb, jats, jats_archiving, jats_articleauthoring, jats_publishing, jira,
        # json, latex, man,
        # markdown, markdown_github, markdown_mmd, markdown_phpextra, markdown_strict,
        # mediawiki, ms, muse, native, odt, opendocument, opml, org, pdf, plain, pptx,
        # revealjs, rst, rtf, s5, slideous, slidy, tei, texinfo, textile, xwiki, zimwiki
        out_format = 'plain'
        # avoid extra new lines injected into text
        extra_args = ['--wrap=preserve', '--resource path="%s" % dst']

        plain_list = []
        try:
            # valid for expert settings
            input_rst = pypandoc.convert_file(fil, 'rst')
            input_list = input_rst.split('\n``')
            for input_subrst in input_list:
                input_plain = pypandoc.convert_text(input_subrst, format='rst', to='plain')
                plain_list.append([input_plain, fil])
        except Exception as e:
            print("file exception: %s %s" % (fil, str(e)), flush=True)

        if not plain_list:
            # if failed to process as pieces of rst, then
            output = pypandoc.convert_file(fil, out_format, extra_args=extra_args, format='rst')
            outputs1 = get_sentences(output, length=max_len)
            for oi, output in enumerate(outputs1):
                output = output.replace('\n\n', '\n')
                plain_list.append([output, fil])
        outputs.extend(plain_list)

    # report:
    # [print(len(x)) for x in outputs]

    # deal with blocks longer than context size (sequence length) of 2048
    new_outputs = []
    num_truncated = 0
    num_orig = len(outputs)
    for output, fil in outputs:
        if len(output) < max_len:
            new_outputs.append([output, fil])
            continue
        outputs1 = get_sentences(output, length=max_len)
        for oi, output1 in enumerate(outputs1):
            output1 = output1.replace('\n\n', '\n')
            new_outputs.append([output1, fil])
        num_truncated += 1
    print('num_orig: %s num_truncated: %s' % (num_orig, num_truncated), flush=True)

    new_outputs = [[k.strip(), fil] for k, fil in new_outputs if len(k.strip()) > min_len]

    return new_outputs


def test_scrape_dai_docs_all_pandoc():
    """
    pytest -s -v create_data.py::test_scrape_dai_docs_all_pandoc
    :return:
    """

    dst = setup_dai_docs()

    import glob
    files = list(glob.glob(os.path.join(dst, '*rst'), recursive=True))

    basedir = os.path.abspath(os.getcwd())
    new_outputs = rst_to_outputs(files)
    os.chdir(basedir)

    remove(dst)
    save_thing = [{"output": k.strip(), 'prompt_type': 'plain'} for k in new_outputs]
    output_file = "dai_docs.train_cleaned.json"
    with open(output_file, "wt") as f:
        f.write(json.dumps(save_thing, indent=2))


def test_config_to_json():
    """
    Needs to run from Driverless AI source directory.
    E.g. (base) jon@gpu:~/h2oai$ pytest -s -v /data/jon/h2ogpt/create_data.py::test_config_to_json ; cp config.json /data/jon/h2ogpt/
    :return:
    """
    try:
        # Arrange
        import json
        from h2oaicore.systemutils import config
        toml_list = []
        for k, v in config.get_meta_dict().items():
            title = (v.title + ": ") if v.title else ''
            comment = v.comment or ''
            if not (title or comment):
                continue
            toml_list.extend(
                [
                    {
                        'prompt_type': 'plain',
                        'instruction': f"<human>: What does {k} do?\n<bot>: {k.replace('_', ' ')} config.toml:  {comment or title}\n<human>:".replace(
                            "\n", ""),
                    },
                    {
                        'prompt_type': 'plain',
                        'instruction': f"<human>: Explain {k}.\n<bot>: {k.replace('_', ' ')} config.toml:  {comment or title}\n<human>:".replace(
                            "\n", ""),
                    },
                    {
                        'prompt_type': 'plain',
                        'instruction': f"<human>: How can I do this: {title}.\n<bot>: Set the {k.replace('_', ' ')} config.toml\n<human>:".replace(
                            "\n", ""),
                    } if title and comment else None,
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Explain the following expert setting for Driverless AI',
                        'input': f"{k}",
                        'output': f"{k.replace('_', ' ')} config.toml: {comment or title}".replace("\n", ""),
                    },
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Explain the following expert setting for Driverless AI',
                        'input': f"{k}",
                        'output': f"{k.replace('_', ' ')} config.toml: {title}{comment}".replace("\n", ""),
                    },
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Explain the following expert setting for Driverless AI',
                        'input': f"{k.replace('_', ' ')}",
                        'output': f"{k.replace('_', ' ')} config.toml: {title}{comment}".replace("\n", ""),
                    },
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Explain the following expert setting for Driverless AI',
                        'input': f"{title}",
                        'output': f"{k.replace('_', ' ')} config.toml: {title}{comment}".replace("\n", ""),
                    },
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Provide a short explanation of the expert setting {k}',
                        'output': f"{k.replace('_', ' ')} config.toml: {comment or title}".replace("\n", ""),
                    },
                    {
                        'prompt_type': 'human_bot',
                        'instruction': f'Provide a detailed explanation of the expert setting {k}',
                        'output': f"{k.replace('_', ' ')} config.toml: {title}{comment}".replace("\n", ""),
                    },
                ]
            )
        toml_list = [x for x in toml_list if x]
        with open("config.json", "wt") as f:
            f.write(json.dumps(toml_list, indent=2))
    except Exception as e:
        print("Exception: %s" % str(e), flush=True)


def copy_tree(src, dst, follow_symlink=False):
    makedirs(dst, exist_ok=True)
    for (path, dirs, files) in os.walk(src, followlinks=follow_symlink):
        new_path = path.replace(src, dst)
        makedirs(new_path, exist_ok=True)
        for file in files:
            filename = os.path.join(path, file)
            new_filename = os.path.join(new_path, file)
            # print("%s -> %s" % (filename, new_filename))
            try:
                atomic_copy(filename, new_filename)
            except FileNotFoundError:
                pass


def atomic_move(src, dst):
    try:
        shutil.move(src, dst)
    except (shutil.Error, FileExistsError):
        pass
    remove(src)


def atomic_copy(src=None, dst=None, with_permissions=True):
    if os.path.isfile(dst):
        return
    import uuid
    my_uuid = uuid.uuid4()
    dst_tmp = dst + str(my_uuid)
    makedirs(os.path.dirname(dst), exist_ok=True)
    if with_permissions:
        shutil.copy(src, dst_tmp)
    else:
        shutil.copyfile(src, dst_tmp)
    atomic_move(dst_tmp, dst)
    remove(dst_tmp)


def makedirs(path, exist_ok=True):
    """
    Avoid some inefficiency in os.makedirs()
    :param path:
    :param exist_ok:
    :return:
    """
    if os.path.isdir(path) and os.path.exists(path):
        assert exist_ok, "Path already exists"
        return path
    os.makedirs(path, exist_ok=exist_ok)


## Download from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_unfiltered_cleaned_split.json
## Turn into simple instruct prompt type. No context/previous conversations.
def test_prep_instruct_vicuna():
    from datasets import load_dataset
    filename = 'ShareGPT_unfiltered_cleaned_split.json'
    if not os.path.exists(filename):
        os.system(
            'wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/%s' % filename)
    data = load_dataset("json", data_files={"train": filename})["train"]
    training_rows = []
    for i in range(data.num_rows):
        conversations = data[i]['conversations']
        assert isinstance(conversations, list), conversations
        convo = ""
        for j, conv in enumerate(conversations):
            # Get ready for generate.py prompt_type=human_bot
            # But train with prompt_type=plain
            if conv['from'] == 'human':
                FROM = '<human>: '
            elif conv['from'] == 'gpt':
                FROM = '<bot>: '
            convo += f"{FROM}" + conv['value'] + "\n"
        if convo:
            training_rows.append(dict(input=convo))
    with open(filename + ".generate_human_bot.train_plain.json", "wt") as f:
        f.write(json.dumps(training_rows, indent=2))


POSTFIX = ".generate_human_bot.train_plain.json"

# https://bair.berkeley.edu/blog/2023/04/03/koala/
OIG_DATASETS = [
    "unified_chip2.jsonl",
    "unified_grade_school_math_instructions.jsonl",
    "unified_poetry_2_song.jsonl",
    "unified_plot_screenplay_books_dialog.jsonl",
]

# hub issue: https://huggingface.co/datasets/laion/OIG/discussions/4
ALL_OIG_DATASETS = ['unified_abstract_infill.jsonl',
                    'unified_basic.jsonl',
                    'unified_canadian_parliament.jsonl',
                    'unified_chip2.jsonl',
                    'unified_conv_finqa.jsonl',
                    'unified_cuad.jsonl',
                    'unified_essays.jsonl',
                    'unified_flan.jsonl.gz',
                    'unified_grade_school_math_instructions.jsonl',
                    'unified_hc3_human.jsonl',
                    'unified_image_prompts_instructions.jsonl',
                    'unified_joke_explanations.jsonl',
                    'unified_mathqa_flanv2_kojma_cot.jsonl',
                    'unified_merged_code_xp3.jsonl',
                    'unified_multi_news.jsonl',
                    'unified_multi_sum.jsonl',
                    'unified_ni.jsonl.gz',
                    'unified_nq.jsonl',
                    'unified_openai_summarize_tldr.jsonl',
                    'unified_oscar_en_sample_dialog.jsonl',
                    'unified_p3.jsonl.gz',
                    'unified_plot_screenplay_books_dialog.jsonl',
                    'unified_poetry_2_song.jsonl',
                    'unified_poetry_instructions.jsonl',
                    'unified_rallio_safety_and_prosocial.jsonl',
                    'unified_rallio_soda_upgraded_2048.jsonl',
                    'unified_soda_dialog.jsonl',
                    'unified_sqlv1.jsonl',
                    'unified_sqlv2.jsonl',
                    'unified_squad_v2.jsonl',
                    'unified_squad_v2_more_neg.jsonl',
                    'unified_ul2_plus_oscar_en_sample_dialog.jsonl',
                    'unified_unifiedskg_instructions.jsonl',
                    'unified_unnatural_instructions.jsonl',
                    'unified_xp3_sample.jsonl']

useful_oig_files = ['unified_rallio_safety_and_prosocial.jsonl.parquet',
                    'unified_chip2.jsonl.parquet',
                    'unified_cuad.jsonl.parquet',
                    'unified_essays.jsonl.parquet',
                    'unified_flan.jsonl.gz.parquet',
                    'unified_grade_school_math_instructions.jsonl.parquet',
                    'unified_hc3_human.jsonl.parquet',
                    'unified_mathqa_flanv2_kojma_cot.jsonl.parquet',
                    'unified_merged_code_xp3.jsonl.parquet',
                    'unified_multi_news.jsonl.parquet',
                    # 'unified_multi_sum.jsonl.parquet'
                    'unified_ni.jsonl.gz.parquet',
                    'unified_openai_summarize_tldr.jsonl.parquet',
                    # 'unified_oscar_en_sample_dialog.jsonl.parquet', # create text containing these N words, not specific
                    'unified_plot_screenplay_books_dialog.jsonl.parquet',
                    'unified_soda_dialog.jsonl.parquet',
                    'unified_unnatural_instructions.jsonl.parquet',
                    ]


@pytest.mark.parametrize("filename", OIG_DATASETS)
def test_get_small_sample_oig_data(filename):
    if not os.path.exists(filename):
        os.system('wget https://huggingface.co/datasets/laion/OIG/resolve/main/%s' % filename)
    import json
    rows = []
    with open(filename, "r") as f:
        for line in f.readlines():
            row = json.loads(line)
            rows.append(dict(input=row["text"]))
    with open(filename + POSTFIX, "w") as f:
        f.write(json.dumps(rows, indent=2))


@pytest.mark.parametrize("filename", ALL_OIG_DATASETS)
def test_download_useful_data_as_parquet(filename):
    dest_file = filename + '.parquet'
    if dest_file not in useful_oig_files:
        pytest.skip('file declared not useful')
    if not os.path.exists(filename):
        os.system('wget https://huggingface.co/datasets/laion/OIG/resolve/main/%s' % filename)
    if not os.path.exists(dest_file):
        df = pd.read_json(path_or_buf=filename, lines=True)
        df.to_parquet(dest_file, index=False)


def test_merge_shuffle_small_sample_oig_data():
    np.random.seed(1234)
    rows = []
    for filename in OIG_DATASETS:
        with open(filename + POSTFIX, "r") as f:
            rows.extend(json.loads(f.read()))
    np.random.shuffle(rows)
    with open("merged_shuffled_OIG_%s.json" % hashlib.sha256(str(OIG_DATASETS).encode()).hexdigest()[:10], "w") as f:
        f.write(json.dumps(rows, indent=2))


def test_join_jsons():
    files = ['config.json'] * 1 + \
            ['dai_docs.train_cleaned.json'] * 2 + \
            ['dai_faq.json'] * 3
    print(files)
    lst = []
    [lst.extend(json.load(open(fil, 'rt'))) for fil in files]
    print(len(lst))
    json.dump(lst, open("merged.json", "wt"), indent=2)


@pytest.mark.parametrize("filename", ['Anthropic/hh-rlhf'])
def test_make_rlhf_good_data(filename):
    from datasets import load_dataset
    rows = load_dataset(filename)["train"]["chosen"]
    new_rows = []
    for row in rows:
        if row[:2] == "\n\n":
            row = row[2:]
        row = row.replace("Human: ", "<human>: ")
        row = row.replace("Assistant: ", "<bot>: ")
        new_rows.append(dict(input=row))
    with open(filename.replace("/", "_") + POSTFIX, "w") as f:
        f.write(json.dumps(new_rows, indent=2))


def test_show_prompts():
    files = ['config.json'] * 1 + \
            ['dai_docs.train_cleaned.json'] * 1 + \
            ['dai_faq.json'] * 1
    file_points = [json.load(open(fil, 'rt')) for fil in files]
    from prompter import generate_prompt
    for data_points in file_points:
        for data_point in data_points:
            print(generate_prompt(data_point, 'plain', '', False, False, False)[0])


def test_get_open_datasets():
    # HF changed things so don't get raw list of all datasets, so not have to filter, but can't do negative filter
    open_tags = ['license:Apache License 2.0',
                 'license:mit',
                 'license:apache',
                 'license:apache2',
                 'license:apache-2.0',
                 'license:bsd',
                 'license:bsd-2-clause',
                 'license:bsd-3-clause',
                 'license:bsd-3-clause-clear',
                 'license:lgpl-2.1',
                 'license:lgpl-3.0',
                 'license:lgpl-lr',
                 'license:lgpl',
                 'license:openrail++',
                 'license:openrail',
                 'license:bigscience-bloom-rail-1.0',
                 # 'license:agpl-3.0',
                 'license:other',
                 'license:unknown',
                 # 'license:mpl-2.0',     # ok, but would have to include original copyright, license, source, copies in distribution
                 # Attribution required:
                 'license:odc-by',
                 'license:cc-by-4.0',
                 'license:cc-by-3.0',
                 'license:cc-by-2.0',
                 'license:cc-by-2.5',
                 # 'license:cc-by-sa-4.0',  # would require same license
                 'license:odbl',
                 'license:pddl',
                 'license:ms-pl',
                 'license:zlib',
                 ]
    # bad license: cc-by-nc-4.0

    from huggingface_hub import list_datasets
    datasets = flatten_list([[x for x in list_datasets(filter=y)] for y in open_tags])
    datasets += [x for x in list_datasets(author='openai')]
    # check all:
    all_license_tags = set(flatten_list([[y for y in x.tags if 'license' in y] for x in datasets]))
    print(len(all_license_tags))
    open_datasets = [x for x in datasets if any([y in x.tags for y in open_tags]) or 'license:' not in str(x.tags)]
    print('open_datasets', len(open_datasets))
    all_task_tags = set(flatten_list([[y for y in x.tags if 'task' in y] for x in open_datasets]))
    print('all_task_tags', len(all_task_tags))
    excluded_tags = ['image', 'hate', 'tabular', 'table-', 'classification', 'retrieval',
                     'translation', 'identification', 'object', 'mask', 'to-text',
                     'face-detection', 'audio', 'voice', 'reinforcement', 'depth-est',
                     'forecasting', 'parsing', 'visual', 'speech', 'multiple-choice',
                     'slot-filling', 'irds/argsme', '-scoring', 'other', 'graph-ml',
                     'feature-extraction', 'keyword-spotting',
                     'coreference-resolution', 'segmentation',
                     'word-sense-disambiguation',
                     'lemmatization']
    task_tags = [x.replace('task_categories:', '').replace('task_ids:', '')
                 for x in all_task_tags if not any([y in x for y in
                                                    excluded_tags])]
    print('task_tags', len(task_tags))
    # str(x.tags) to catch any pattern match to anything in list
    open_tasked_datasets = [x for x in open_datasets if
                            any([y in str([x for x in x.tags if 'task' in x]) for y in task_tags]) and
                            not any([y in str([x for x in x.tags if 'task' in x]) for y in excluded_tags]) or
                            'task_categories' not in str(x.tags) and 'task_ids' not in str(x.tags)]
    open_tasked_datasets = [x for x in open_tasked_datasets if not x.disabled]
    open_tasked_datasets = [x for x in open_tasked_datasets if not x.gated]
    open_tasked_datasets = [x for x in open_tasked_datasets if not x.private]
    print('open_tasked_datasets', len(open_tasked_datasets))
    sizes = list(set(flatten_list([[(y, x.id) for y in x.tags if 'size' in y] for x in open_tasked_datasets])))
    languages = list(set(flatten_list([[(y, x.id) for y in x.tags if 'language:' in y] for x in open_tasked_datasets])))
    open_english_tasked_datasets = [x for x in open_tasked_datasets if
                                    'language:' not in str(x.tags) or
                                    'language:en' in str(x.tags)]
    small_open_english_tasked_datasets = [x for x in open_english_tasked_datasets if
                                          'n<1K' in str(x.tags) or
                                          '1K<n<10K' in str(x.tags) or
                                          '1K0<n<100K' in str(x.tags) or
                                          '100K<n<1M' in str(x.tags) or
                                          'size_category' not in str(x.tags)
                                          ]
    # 'aeslc' : email_body, subject -> summarization?
    # load_dataset(open_tasked_datasets[0].id).data['train'].to_pandas()
    ids = [x.id for x in small_open_english_tasked_datasets]

    # sanity checks
    # https://bair.berkeley.edu/blog/2023/04/03/koala/
    assert 'alespalla/chatbot_instruction_prompts' in ids
    assert 'laion/OIG' in ids
    assert 'openai/webgpt_comparisons' in ids
    assert 'openai/summarize_from_feedback' in ids
    assert 'Anthropic/hh-rlhf' in ids

    # useful but not allowed for commercial purposes:
    # https://huggingface.co/datasets/squad

    print('open_english_tasked_datasets: ', ids, flush=True)

    exclude_ids = ['allenai/nllb',  # translation only
                   'hf-internal-testing/fixtures_image_utils',  # testing
                   'allenai/c4',  # search-url
                   'agemagician/uniref50',  # unknown
                   'huggingface-course/documentation-images',  # images
                   'smilegate-ai/kor_unsmile',  # korean
                   'MohamedRashad/ChatGPT-prompts',  # ChatGPT/LearnGPT/https://www.emergentmind.com/
                   'humarin/chatgpt-paraphrases',  # Paraphrase using ChatGPT
                   'Jeska/vaccinchat',  # not useful
                   'alespalla/chatbot_instruction_prompts',  # mixes alpaca
                   'allenai/prosocial-dialog',
                   # already exlucded, but wrongly in other datasets that say more permissive license
                   'AlekseyKorshuk/persona-chat',  # low quality
                   'bavard/personachat_truecased',  # low quality
                   'adamlin/daily_dialog',  # medium quality conversations
                   'adamlin/FewShotWoz',  # low quality
                   'benjaminbeilharz/better_daily_dialog',  # low quality
                   'benjaminbeilharz/daily_dialog_w_turn_templates',  # low
                   'benjaminbeilharz/empathetic_dialogues_for_lm',  # low
                   'GEM-submissions/GEM__bart_base_schema_guided_dialog__1645547915',  # NA
                   'ia-bentebib/conv_ai_2_fr',  # low fr
                   'ia-bentebib/daily_dialog_fr',  # low fr
                   'ia-bentebib/dialog_re_fr',  # low fr
                   'ia-bentebib/empathetic_dialogues_fr',  # low fr
                   'roskoN/dailydialog',  # low
                   'VadorMazer/skyrimdialogstest',  # low
                   'bigbio/med_qa',  # med specific Q/A
                   'biu-nlp/qa_srl2018',  # low quality Q/A
                   'biu-nlp/qa_discourse',  # low quality Q/A
                   'iarfmoose/qa_evaluator',  # low quality Q/A
                   'jeopardy',  # low quality Q/A -- no reasoning
                   'narrativeqa',  # low quality Q/A
                   'nomic-ai/gpt4all_prompt_generations',  # bad license
                   'nomic-ai/gpt4all_prompt_generations_with_p3',  # bad license
                   'HuggingFaceH4/alpaca',  # bad license
                   'tatsu-lab/alpaca',  # ToS breaking
                   'yahma/alpaca-cleaned',  # ToS breaking
                   'Hello-SimpleAI/HC3',  # bad license
                   'glue',  # no reasoning QA
                   'sahil2801/CodeAlpaca-20k',  # bad license
                   'Short-Answer-Feedback/saf_communication_networks_english',  # long Q, medium A
                   ]
    small_open_english_tasked_datasets = [x for x in small_open_english_tasked_datasets if x.id not in exclude_ids]
    # some ids clearly speech related
    small_open_english_tasked_datasets = [x for x in small_open_english_tasked_datasets if 'speech' not in x.id]
    # HF testing
    small_open_english_tasked_datasets = [x for x in small_open_english_tasked_datasets if
                                          'hf-internal-testing' not in x.id]
    small_open_english_tasked_datasets = [x for x in small_open_english_tasked_datasets if
                                          'chinese' not in x.id]

    sorted_small_open_english_tasked_datasets = sorted([(x.downloads, x) for x in small_open_english_tasked_datasets],
                                                       key=lambda x: x[0], reverse=True)

    # NOTES:
    # Run like pytest -s -v create_data.py::test_get_open_datasets &> getdata9.log
    # See what needs config passed and add:
    # grep 'load_dataset(' getdata9.log|grep -v data_id|less -S
    # grep "pip install" getdata9.log
    # NOTE: Some datasets have default config, but others are there.  Don't know how to access them.

    """
    https://huggingface.co/datasets/wikihow/blob/main/wikihow.py
    https://github.com/mahnazkoupaee/WikiHow-Dataset
    https://ucsb.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358
    https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358
    """

    """
    # some ambiguous or non-commercial datasets
    https://github.com/PhoebusSi/alpaca-CoT
    """

    timeout = 3 * 60
    # laion/OIG takes longer
    for num_downloads, dataset in sorted_small_open_english_tasked_datasets:
        data_id = dataset.id
        func = do_one
        args = (data_id, num_downloads)
        kwargs = {}
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("\n\ndata_id %s timeout\n\n" % data_id, flush=True)
            for child in psutil.Process(os.getpid()).children(recursive=True):
                os.kill(child.pid, signal.SIGINT)
                os.kill(child.pid, signal.SIGTERM)
                os.kill(child.pid, signal.SIGKILL)


def do_one(data_id, num_downloads):
    from datasets import load_dataset
    out_file = "data_%s.parquet" % str(data_id.replace('/', '_'))
    if os.path.isfile(out_file) and os.path.getsize(out_file) > 1024 ** 3:
        return
    try:
        print("Loading data_id %s num_downloads: %s" % (data_id, num_downloads), flush=True)
        avail_list = None
        try:
            data = load_dataset(data_id, 'foobar')
        except Exception as e:
            if 'Available: ' in str(e):
                avail_list = ast.literal_eval(str(e).split('Available:')[1].strip())
            else:
                avail_list = None
        if avail_list is None:
            avail_list = [None]
        print("%s avail_list: %s" % (data_id, avail_list), flush=True)

        for name in avail_list:
            out_file = "data_%s_%s.parquet" % (str(data_id.replace('/', '_')), str(name))
            if os.path.isfile(out_file):
                continue
            data = load_dataset(data_id, name)
            column_names_dict = data.column_names
            column_names = column_names_dict[list(column_names_dict.keys())[0]]
            print("Processing data_id %s num_downloads: %s columns: %s" % (data_id, num_downloads, column_names),
                  flush=True)
            data_dict = data.data
            col_dict = data.num_columns
            first_col = list(col_dict.keys())[0]
            if 'train' in data_dict:
                df = data['train'].to_pandas()
            else:
                df = data[first_col].to_pandas()
            # csv has issues with escaping chars, even for datasets I know I want
            df.to_parquet(out_file, index=False)
    except Exception as e:
        t, v, tb = sys.exc_info()
        ex = ''.join(traceback.format_exception(t, v, tb))
        print("Exception: %s %s" % (data_id, ex), flush=True)


def test_otherlic():
    from huggingface_hub import list_datasets
    lic = ['license:odc-by',
           'license:cc-by-4.0',
           'license:cc-by-3.0',
           'license:cc-by-2.0',
           'license:cc-by-2.5',
           'license:cc-by-sa-4.0',
           'license:odbl',
           'license:pddl',
           'license:ms-pl',
           'license:zlib',
           ]
    datasets = flatten_list([[x for x in list_datasets(filter=y) if 'translation' not in str(x.tags)] for y in lic])
    print(len(datasets))


# These useful datasets are determined based upon data sample, column types, and uniqueness compared to larger datasets like Pile
# grep columns getdata13.log|grep -v "\['image'\]"|sort|uniq|grep -v tokens|grep -v "'image'"|grep -v embedding|grep dialog
useful = ['Dahoas/instruct-human-assistant-prompt',
          'Dahoas/first-instruct-human-assistant-prompt',
          'knkarthick/dialogsum',  # summary of conversation
          'McGill-NLP/FaithDial',  # medium quality
          'Zaid/quac_expanded',  # medium quality context + QA
          '0-hero/OIG-small-chip2',  # medium
          'alistvt/coqa-flat',  # QA medium
          'AnonymousSub/MedQuAD_47441_Question_Answer_Pairs',  # QA medium
          'Anthropic/hh-rlhf',  # high quality  # similar to Dahoas/full-hh-rlhf
          'arjunth2001/online_privacy_qna',  # good quality QA
          'Dahoas/instruct_helpful_preferences',  # medium quality instruct
          'Dahoas/rl-prompt-dataset',  # medium chat
          'Dahoas/rm-static',  # medium chat
          'Dahoas/static-hh',  # medium chat  # HuggingFaceH4/self_instruct
          'Dahoas/synthetic-instruct-gptj-pairwise',  # medium chat
          'eli5',  # QA if prompt ELI5
          'gsm8k',  # QA (various)
          'guanaco/guanaco',  # prompt/response
          'kastan/rlhf-qa-comparisons',  # good QA
          'kastan/rlhf-qa-conditional-generation-v2',  # prompt answer
          'OllieStanley/humaneval-mbpp-codegen-qa',  # code QA, but started from words, so better than other code QA
          'OllieStanley/humaneval-mbpp-testgen-qa',  # code QA
          'Graverman/Instruct-to-Code',  # code QA
          'openai/summarize_from_feedback',  # summarize
          'relbert/analogy_questions',  # analogy QA
          'yitingxie/rlhf-reward-datasets',  # prompt, chosen, rejected.
          'yizhongw/self_instruct',  # instruct (super natural & instruct)
          'HuggingFaceH4/asss',  # QA, big A
          'kastan/rlhf-qa-conditional-generation-v2',  # QA
          'cosmos_qa',  # context QA
          'vishal-burman/c4-faqs',  # QA but not so much reasoning, but alot of text
          'squadshifts',  # QA from context
          'hotpot_qa',  # QA from context
          'adversarial_qa',  # QA from context
          'allenai/soda',  # dialog -> narrative/summary
          'squad_v2',  # context QA
          'squadshifts',  # context QA
          'dferndz/cSQuAD1',  # context QA
          'dferndz/cSQuAD2',  # context QA
          'din0s/msmarco-nlgen',  # context QA
          'domenicrosati/TruthfulQA',  # common sense truthful QA -- trivia but good trivia
          'hotpot_qa',  # context, QA
          'HuggingFaceH4/self-instruct-eval',  # instruct QA, medium quality, some language reasoning
          'kastan/EE_QA_for_RLHF',  # context QA
          'KK04/LogicInference_OA',  # instruction logical QA
          'lmqg/qa_squadshifts_synthetic',  # context QA
          'lmqg/qg_squad',  # context QA
          'lmqg/qg_squadshifts',  # context QA
          'lmqg/qg_subjqa',  # context QA
          'pszemraj/HC3-textgen-qa',
          # QA medium, has human responses -- humans tend to provide links instead of trying to answer
          'pythonist/newdata',  # long context, QA, brief A
          'ropes',  # long background, situation, question, A
          'wikitablequestions',  # table -> QA
          'bigscience/p3',  # context QA but short answers
          ]

code_useful = ['0n1xus/codexglue',
               'openai_humaneval',
               'koutch/staqc',
               ]

maybe_useful = ['AlekseyKorshuk/comedy-scripts',
                'openbookqa',  # hard to parse, low reasoning
                'qed',  # reasonable QA, but low reasoning
                'selqa',  # candidate answers
                'HuggingFaceH4/instruction-pilot-outputs-filtered',
                'GBaker/MedQA-USMLE-4-options',  # medical QA with long questions
                'npc-engine/light-batch-summarize-dialogue',  # dialog summarize, kinda low specific quality
                ]

summary_useful = ['austin/rheum_abstracts',
                  'CarperAI/openai_summarize_comparisons',  # summarize chosen/rejected
                  'CarperAI/openai_summarize_tldr',  # summarize QA
                  'ccdv/cnn_dailymail',  # summarize news
                  'ccdv/govreport-summarization',  # summarize high quality
                  'ccdv/pubmed-summarization',  # summarize high quality
                  'duorc',  # plot -> QA
                  'farleyknight/big_patent_5_percent',  # desc -> abstract
                  'multi_news',  # summary
                  'opinosis',
                  'SophieTr/reddit_clean',
                  'allenai/mup',  # long text -> summary
                  'allenai/multi_lexsum',  # long text -> summary
                  'big_patent',
                  'allenai/wcep_dense_max',
                  'awinml/costco_long_practice',
                  'GEM/xsum',
                  'ratishsp/newshead',
                  'RussianNLP/wikiomnia',  # russian
                  'stacked-summaries/stacked-xsum-1024',
                  ]

math_useful = [
    'competition_math'
]

skipped = ['c4',  # maybe useful, used for flan, but skipped due to size
           ]

"""
To get training data from oig:
pytest test_oig test_grade_final test_finalize_to_json
"""

human = '<human>:'
bot = '<bot>:'


def test_assemble_and_detox():
    import re
    from profanity_check import predict_prob
    df_list = []
    for data in useful_oig_files:
        print("Processing %s" % data, flush=True)
        df = pd.read_parquet(data)
        df = df.reset_index(drop=True)
        # chop up into human/bot interactions of no more than 10kB per row
        text_list = df[['text']].values.ravel().tolist()
        new_text = []
        max_len = 2048  # uber cutoff
        MAX_LEN = 2048 // 2 - 30  # max len per question/answer
        for text in tqdm(text_list):
            human_starts = [m.start() for m in re.finditer('<human>: ', text)]
            if len(human_starts) == 1:
                human_starts = [0, len(text)]  # always go into for loop below
            blurb = ''
            for i in range(len(human_starts) - 1):
                interaction = text[human_starts[i]: human_starts[i + 1]][:max_len]
                blurb += interaction
                if len(blurb) >= MAX_LEN:
                    blurb = get_sentences(blurb, length=MAX_LEN)[0]
                    new_text.append(blurb + "\n<human>:")
                    blurb = ''
            if blurb:
                blurb = get_sentences(blurb, length=MAX_LEN)[0]
                new_text.append(blurb + "\n<human>:")

        if len(new_text) > len(text_list):
            print("Added %d new rows (before: %d)" % (len(new_text) - df.shape[0], df.shape[0]))
        df = pd.DataFrame({"text": new_text, "source": [data] * len(new_text)})
        df = df.drop_duplicates(keep='first')
        print(df['text'].apply(lambda x: len(x)).describe())
        assert df['text'].apply(lambda x: len(x)).max() <= 2 * max_len

        # faster than better_profanity, do early
        df['profanity'] = predict_prob(df['text'])
        before_rows = df.shape[0]
        df = df[df['profanity'] < 0.25]  # drop any low quality stuff
        after_rows = df.shape[0]
        print("Dropped %d rows out of %d due to alt-profanity-check" % (before_rows - after_rows, before_rows))
        df_list.append(df)
        print("Done processing %s -> %s rows" % (data, df.shape[0]), flush=True)
        print("So far have %d rows" % sum([len(x) for x in df_list]))
    df_final = pd.concat(df_list)
    df_final = df_final.sample(frac=1, random_state=1234).reset_index(drop=True)
    df_final.to_parquet('h2oGPT.cleaned.human_bot.shorter.parquet', index=False)


def test_basic_cleaning():
    # from better_profanity import profanity
    # https://pypi.org/project/alt-profanity-check/
    from profanity_check import predict
    df_list = []
    for data in useful_oig_files:
        # for data in useful_oig_files[:5]:
        # for data in ['unified_openai_summarize_tldr.jsonl.parquet']:
        print("Processing %s" % data, flush=True)
        df = pd.read_parquet(data)
        df = df.reset_index(drop=True)
        # NOTE: Not correct if multiple human-bot interactions, but those dialogs even more desired
        # avg_chars = len(df['text'][0])/(df['text'][0].count(human)+df['text'][0].count(bot))
        df['avg_words'] = df['text'].apply(lambda x: x.count(' ') / (x.count(human) + x.count(bot)) / 2.0)
        df['avg_bot_words'] = df['text'].apply(lambda x: x.split(bot)[1].count(' ') / x.count(bot))
        # df['bad_words'] = df['text'].apply(lambda x: profanity.contains_profanity(x))
        # low_quality_patterns = ['Write the rest of this wikipedia article']
        res = predict(df['text'])
        df['bad_words'] = res
        df = df.reset_index(drop=True)
        df = df[df['bad_words'] == 0]
        df = df[['text', 'avg_words', 'avg_bot_words']]
        df = df.drop_duplicates(keep='first')
        print(df[df['avg_words'] == df['avg_words'].max()]['text'].values)
        median_words = np.median(df['avg_words'])
        min_words_per_entity = max(30, 0.8 * median_words)
        max_words_per_entity = 2048  # too hard to learn from for now
        df = df[df['avg_words'] > min_words_per_entity]
        df = df[df['avg_words'] < max_words_per_entity]

        min_words_per_entity = max(20, 0.5 * median_words)  # bot should say stuff for now
        max_words_per_entity = 2048  # too hard to learn from for now
        df = df[df['avg_bot_words'] > min_words_per_entity]
        df = df[df['avg_bot_words'] < max_words_per_entity]

        df_list.append(df)
        print("Done processing %s -> %s rows" % (data, df.shape[0]), flush=True)
    df_final = pd.concat(df_list)
    df_final.to_parquet('h2oGPT.cleaned.human_bot.parquet', index=False)


from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples


def parallel_apply(df, func, n_jobs=-1, **kwargs):
    """ Pandas apply in parallel using joblib.
    Uses sklearn.utils to partition input evenly.

    Args:
        df: Pandas DataFrame, Series, or any other object that supports slicing and apply.
        func: Callable to apply
        n_jobs: Desired number of workers. Default value -1 means use all available cores.
        **kwargs: Any additional parameters will be supplied to the apply function

    Returns:
        Same as for normal Pandas DataFrame.apply()

    """

    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df[s], func, **kwargs)
            for s in gen_even_slices(_num_samples(df), effective_n_jobs(n_jobs)))
        return pd.concat(ret)


def add_better_profanity_flag(df):
    from better_profanity import profanity
    df['better_profanity'] = parallel_apply(
        df['text'],
        lambda x: profanity.contains_profanity(x),
        n_jobs=-1,
    )
    return df


def add_textstat_grade(df):
    import textstat

    def myfunc(x):
        return textstat.flesch_kincaid_grade(x)  # simple grade

    if False:
        import dask.dataframe as dd
        # 40 seconds for 1000 rows, but have 1,787,799 rows
        ddata = dd.from_pandas(df, npartitions=120)

        df['flesch_grade'] = ddata['text'].apply(myfunc).compute()
    if True:
        # fast way
        df['flesch_grade'] = parallel_apply(df['text'], myfunc, n_jobs=-1)
    return df


def add_deberta_grade(df):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(
        reward_name), AutoTokenizer.from_pretrained(reward_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rank_model.to(device)

    def get_question(x):
        return x.replace('<human>: ', '').split('<bot>:')[0]

    def get_answer(x):
        try:
            answer = x.split('<bot>: ')[1].split('<human>:')[0].replace('<bot>: ', '')
        except:
            answer = x.split('<bot>:')[1].split('<human>:')[0].replace('<bot>:', '')
        return answer

    df['question'] = parallel_apply(df['text'], get_question, n_jobs=-1)
    df['answer'] = parallel_apply(df['text'], get_answer, n_jobs=-1)

    from datasets import Dataset
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyPairDataset
    import tqdm

    pipe = pipeline(
        "text-classification",
        model=reward_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    start = 0
    batch_size = 64 * 16
    micro_batch = orig_micro_batch = 16
    end = 0
    import socket
    checkpoint = "grades.%s.pkl" % socket.gethostname()
    grades = []
    import pickle
    if os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            start, grades = pickle.loads(f.read())
    last_oom = 0
    while end < df.shape[0]:
        # manual batching to handle OOM more gracefully
        end = min(start + batch_size, df.shape[0])
        if start == end:
            break
        dataset = Dataset.from_pandas(df.iloc[start:end, :])
        try:
            grades.extend([
                x['score'] for x in tqdm.tqdm(
                    pipe(KeyPairDataset(dataset, "question", "answer"), batch_size=micro_batch)
                )
            ])
        except torch.cuda.OutOfMemoryError:
            last_oom = start
            micro_batch = max(1, micro_batch // 2)
            print("OOM - retrying with micro_batch=%d" % micro_batch)
            continue
        if last_oom == start:
            micro_batch = orig_micro_batch
            print("Returning to micro_batch=%d" % micro_batch)
        assert len(grades) == end
        start = end
        with open(checkpoint, "wb") as f:
            f.write(pickle.dumps((end, grades)))
        print("%d/%d" % (end, df.shape[0]))
    df['grade_deberta'] = grades
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    return df


def test_chop_by_lengths():
    file = "h2oGPT.cleaned.human_bot.shorter.parquet"
    df = pd.read_parquet(file).reset_index(drop=True)
    df = count_human_bot_lengths(df)
    df['rand'] = np.random.rand(df.shape[0])
    df['rand2'] = np.random.rand(df.shape[0])
    before_rows = df.shape[0]
    # throw away short human/bot responses with higher likelihood
    df = df[(df['len_human_mean'] > 20)]  # never keep very short ones
    df = df[(df['len_human_mean'] > 30) | (df['rand'] < 0.2)]
    df = df[(df['len_human_mean'] > 50) | (df['rand'] < 0.5)]
    df = df[(df['len_human_max'] < 10000)]  # drop super long (basically only human) ones
    df = df[(df['len_bot_mean'] > 20)]  # never keep very short ones
    df = df[(df['len_bot_mean'] > 30) | (df['rand2'] < 0.2)]
    df = df[(df['len_bot_mean'] > 50) | (df['rand2'] < 0.5)]
    df = df[(df['len_bot_max'] < 10000)]  # drop super long (only bot) ones
    assert df['text'].apply(lambda x: len(x)).max() < 20000
    df = df.drop(['rand', 'rand2'], axis=1)
    after_rows = df.shape[0]
    print("Chopped off %d out of %d rows due to length" % (before_rows - after_rows, before_rows))
    print(df.describe())
    df.to_parquet('h2oGPT.cleaned.chopped.human_bot.shorter.parquet', index=False)


def count_human_bot_lengths(df, human=None, bot=None):
    import re
    len_human_min = []
    len_human_max = []
    len_human_mean = []
    len_bot_min = []
    len_bot_max = []
    len_bot_mean = []
    human = human or '<human>:'
    bot = bot or '<bot>:'
    for is_human in [True, False]:
        what = human if is_human else bot
        other = human if not is_human else bot
        for i in range(df.shape[0]):
            text = df.loc[i, 'text']
            assert isinstance(text, str)
            starts = [m.start() for m in re.finditer(what, text)]
            if len(starts) == 1:
                starts = [starts[0], len(text)]  # always go into for loop below
            assert len(text)
            list_what = []
            for ii in range(len(starts) - 1):
                interaction = text[starts[ii]: starts[ii + 1]]
                if other in interaction:
                    interaction = interaction[:interaction.find(other)]
                interaction.strip()
                list_what.append(interaction)
            if not list_what:
                list_what = ['']  # handle corrupted data, very rare, leads to sizes 0
            if is_human:
                len_human_min.append(min([len(x) for x in list_what]))
                len_human_max.append(max([len(x) for x in list_what]))
                len_human_mean.append(np.mean([len(x) for x in list_what]))
            else:
                len_bot_min.append(min([len(x) for x in list_what]))
                len_bot_max.append(max([len(x) for x in list_what]))
                len_bot_mean.append(np.mean([len(x) for x in list_what]))
    df['len_human_min'] = len_human_min
    df['len_human_max'] = len_human_max
    df['len_human_mean'] = len_human_mean
    df['len_bot_min'] = len_bot_min
    df['len_bot_max'] = len_bot_max
    df['len_bot_mean'] = len_bot_mean
    np.random.seed(1234)
    pd.set_option('display.max_columns', None)
    print("Before chopping")
    print(df.describe())
    return df


def test_grade():
    df = None

    file = "h2oGPT.cleaned.chopped.human_bot.shorter.parquet"
    output_file = "h2oGPT.cleaned.graded1.human_bot.shorter.parquet"
    if not os.path.exists(output_file):
        if df is None:
            df = pd.read_parquet(file).reset_index(drop=True)
        df = add_textstat_grade(df)
        min_grade = 10
        max_grade = 25
        df = df[df['flesch_grade'] >= min_grade]
        df = df[df['flesch_grade'] <= max_grade]
        print("After Flesch grade")
        print(df.describe())
        df.to_parquet(output_file, index=False)

    file = output_file
    output_file = "h2oGPT.cleaned.graded2.human_bot.shorter.parquet"
    if not os.path.exists(output_file):
        # slower than alt-profanity, do last, but do before deberta grading, since that's slower
        if df is None:
            df = pd.read_parquet(file).reset_index(drop=True)
        df = add_better_profanity_flag(df)
        before_rows = df.shape[0]
        df = df[df['better_profanity'] == 0]
        df = df.drop(['better_profanity'], axis=1)
        after_rows = df.shape[0]
        print("Dropped %d rows out of %d due to better_profanity" % (before_rows - after_rows, before_rows))
        print(df.describe())
        df.to_parquet(output_file, index=False)

    file = output_file
    output_file = 'h2oGPT.cleaned.graded3.human_bot.shorter.parquet'
    if not os.path.exists(output_file):
        if df is None:
            df = pd.read_parquet(file).reset_index(drop=True)
        df = add_deberta_grade(df)
        min_grade = 0.3
        max_grade = np.inf
        before_rows = df.shape[0]
        df = df[df['grade_deberta'] >= min_grade]
        df = df[df['grade_deberta'] <= max_grade]
        after_rows = df.shape[0]
        print("Dropped %d rows out of %d due to deberta grade" % (before_rows - after_rows, before_rows))
        print("After DeBERTa grade")
        print(df.describe())
        df.to_parquet(output_file, index=False)

    file = output_file
    output_file = 'h2oGPT.cleaned.graded.human_bot.shorter.parquet'
    if df is None:
        df = pd.read_parquet(file).reset_index(drop=True)
    df.to_parquet(output_file, index=False)


@pytest.mark.parametrize(
    "fixup_personality, only_personality, deberta_grading",
    [
        [False, False, False],
        [True, True, False],
        [True, False, False],
        [True, False, True],
    ]
)
def test_add_open_assistant(fixup_personality, only_personality, deberta_grading, save_json=True):
    """
    Flatten tree structure into one row per path from root to leaf
    Also turn into human_bot prompting format:
        <human>: question\n<bot>: answer <human>: question2\n<bot>: answer2 Etc.
    Also saves a .json locally as side-effect
    returns list of dicts, containing intput, prompt_type and source
    """
    from datasets import load_dataset
    data_file = "OpenAssistant/oasst1"
    ds = load_dataset(data_file)
    df = pd.concat([ds['train'].to_pandas(), ds['validation'].to_pandas()], axis=0)
    rows = {}
    message_ids = df['message_id'].values.tolist()
    message_tree_ids = df['message_tree_id'].values.tolist()
    parent_ids = df['parent_id'].values.tolist()
    texts = df['text'].values.tolist()
    roles = df['role'].values.tolist()

    for i in range(df.shape[0]):
        # collect all trees
        message_id = message_ids[i]
        message_tree_id = message_tree_ids[i]
        parent_id = parent_ids[i]
        text = texts[i]
        if fixup_personality:
            text = text.replace("Open Assistant", "h2oGPT")
            text = text.replace("Open-Assistant", "h2oGPT")
            text = text.replace("open-assistant", "h2oGPT")
            text = text.replace("OpenAssistant", "h2oGPT")
            text = text.replace("open assistant", "h2oGPT")
            text = text.replace("Open Assistand", "h2oGPT")
            text = text.replace("Open Assitant", "h2oGPT")
            text = text.replace("Open Assistent", "h2oGPT")
            text = text.replace("Open Assisstant", "h2oGPT")
            text = text.replace("Open Assitent", "h2oGPT")
            text = text.replace("Open Assitiant", "h2oGPT")
            text = text.replace("Open Assistiant", "h2oGPT")
            text = text.replace("Open Assitan ", "h2oGPT ")
            text = text.replace("Open Assistan ", "h2oGPT ")
            text = text.replace("Open Asistant", "h2oGPT")
            text = text.replace("Open Assiant", "h2oGPT")
            text = text.replace("Assistant", "h2oGPT")
            text = text.replace("LAION AI", "H2O.ai")
            text = text.replace("LAION-AI", "H2O.ai")
            text = text.replace("LAION,", "H2O.ai,")
            text = text.replace("LAION.ai", "H2O.ai")
            text = text.replace("LAION.", "H2O.ai.")
            text = text.replace("LAION", "H2O.ai")

        role = roles[i]
        new_data = ('<human>: ' if role == 'prompter' else '<bot>: ') + text
        entry = dict(message_id=message_id, parent_id=parent_id, text=new_data)
        if message_tree_id not in rows:
            rows[message_tree_id] = [entry]
        else:
            rows[message_tree_id].append(entry)

    all_rows = []

    for node_id in rows:
        # order responses in tree, based on message/parent relationship
        conversations = []

        list_msgs = rows[node_id]
        # find start
        while len(list_msgs):
            for i, leaf in enumerate(list_msgs):
                found = False
                parent_id = leaf['parent_id']
                if parent_id is None:
                    # conversation starter
                    conversations.append(leaf)
                    found = True
                else:
                    for conv in conversations:
                        # find all conversations to add my message to
                        if parent_id in conv['message_id'] and parent_id != conv['message_id'][-len(parent_id):]:
                            # my message doesn't follow conversation
                            continue
                        if parent_id == conv['message_id'][-len(parent_id):]:
                            # my message follows conversation, but fork first, so another follow-on message can do same
                            conversations.append(conv.copy())
                            conv['text'] += f"""
{leaf['text']}
"""
                            conv['message_id'] += leaf['message_id']
                            found = True
                            break
                if found:
                    # my content was used, so nuke from list
                    del list_msgs[i]
                    break

        # now reduce down to final conversations, find the longest chains of message ids
        for i, conv in enumerate(conversations):
            for j, conv2 in enumerate(conversations):
                if i == j:
                    continue
                if conv['message_id'] and conv2['message_id']:
                    assert conv['message_id'] != conv2['message_id']
                    # delete the shorter conversation, if one contains the other
                    if conv['message_id'] in conv2['message_id']:
                        conv['message_id'] = None
                    if conv2['message_id'] in conv['message_id']:
                        conv2['message_id'] = None
        conversations = [c for c in conversations if c['message_id']]
        if only_personality:
            all_rows.extend(
                [dict(input=c['text'] + "\n<human>:", prompt_type='plain', source=data_file) for c in conversations if
                 'h2oGPT' in c['text']])
        else:
            all_rows.extend(
                [dict(input=c['text'] + "\n<human>:", prompt_type='plain', source=data_file) for c in conversations if
                 "What is H2O.ai" not in c['text']])
    unhelpful = get_unhelpful_list()
    all_rows = [x for x in all_rows if not any(u in x['input'] for u in unhelpful)]
    personality = create_personality_data()
    all_rows.extend(personality * 10)
    np.random.seed(123)
    np.random.shuffle(all_rows)
    print(len(all_rows))
    if deberta_grading:
        df = pd.DataFrame(all_rows)
        df = df.rename(columns={'input': 'text'})
        df = add_deberta_grade(df)
        df = df.rename(columns={'text': 'input'})
        drop = True
        if drop:
            min_grade = 0.3
            max_grade = np.inf
            before_rows = df.shape[0]
            df = df[df['grade_deberta'] >= min_grade]
            df = df[df['grade_deberta'] <= max_grade]
            after_rows = df.shape[0]
            print("Dropped %d rows out of %d due to deberta grade" % (before_rows - after_rows, before_rows))
            print("After DeBERTa grade")
        print(df.describe())
        all_rows = []
        for i in range(df.shape[0]):
            all_rows.append(
                dict(
                    input=df['input'].iloc[i],
                    source=df['source'].iloc[i],
                    prompt_type=df['prompt_type'].iloc[i],
                    grade_deberta=df['grade_deberta'].iloc[i],
                )
            )
    if save_json:
        data_file = data_file + \
                    ("_h2ogpt" if fixup_personality else "") + \
                    ("_only" if only_personality else "") + \
                    ("_graded" if deberta_grading else "")
        for i in range(len(all_rows)):
            all_rows[i]['id'] = i
        with open(data_file.lower().replace("/", "_") + ".json", "w") as f:
            f.write(json.dumps(all_rows, indent=2))
    return all_rows


def test_finalize_to_json():
    df = pd.read_parquet('h2oGPT.cleaned.graded.human_bot.shorter.parquet')
    df = df.rename(columns={'text': 'input'})

    print("Number of high-quality human_bot interactions: %s" % df.shape[0], flush=True)

    print("Adding open assistant data")
    with open("openassistant_oasst1_h2ogpt_graded.json") as f:
        open_assistant = json.loads(f.read())
    df = pd.concat([df, pd.DataFrame(open_assistant)], axis=0)

    def final_clean(df):
        from better_profanity import profanity
        profanity.load_censor_words_from_file("data/censor_words.txt")
        df['profanity'] = parallel_apply(
            df['input'],
            lambda x: profanity.contains_profanity(x),
            n_jobs=-1,
        )
        return df[(df['profanity'] == 0)].reset_index(drop=True)

    print("Before cleaning: Number of final high-quality human_bot interactions: %s" % df.shape[0], flush=True)
    df = final_clean(df)
    print("After cleaning: Number of final high-quality human_bot interactions: %s" % df.shape[0], flush=True)
    print(df.describe())
    print(df.shape)
    row_list = []
    for i in range(df.shape[0]):
        row_list.append(
            dict(
                input=df.loc[i, 'input'],
                source=df.loc[i, 'source'],
                prompt_type='plain',
            )
        )
    np.random.seed(1234)
    np.random.shuffle(row_list)
    unhelpful = get_unhelpful_list()
    row_list = [x for x in row_list if not any(u in x['input'] for u in unhelpful)]
    for i in range(len(row_list)):
        row_list[i]['id'] = i
        row_list[i]['input'] = row_list[i]['input'].replace(" <bot>:", "\n<bot>:")
    with open('h2ogpt-oig-oasst1-instruct-cleaned-v3.json', "w") as f:
        f.write(json.dumps(row_list, indent=2))


def create_personality_data():
    questions = [
        "What's your name?",
        "What is your name?",
        "What are you?",
        "Who are you?",
        "Do you have a name?",
        "Who trained you?",
        "Who created you?",
        "Who made you?",
    ]
    answers = [
        "I'm h2oGPT, a large language model by H2O.ai.",
        "I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI.",
        "My name is h2oGPT. I'm a large language model by H2O.ai, the visionary leader in democratizing AI.",
        "My name is h2oGPT. I'm a large language model trained by H2O.ai.",
        "Hi! I'm h2oGPT, a large language model by H2O.ai.",
        "Hi! I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI.",
    ]
    help = [
        "",
        " How can I help you?",
        " How may I assist you?",
        " Nice to meet you.",
    ]
    import itertools
    rows = []
    for pair in itertools.product(questions, answers, help):
        rows.append(
            dict(input=f"<human>: {pair[0]}\n<bot>: {pair[1]}{pair[2]}\n<human>:", prompt_type='plain', source="H2O.ai")
        )
    for row in [
        "<human>: What is H2O.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: What is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: What is H2O?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: Who is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: who is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: who is h2o?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:",
        "<human>: What is H2O.ai?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:",
        "<human>: Who is H2O.ai?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:",
        "<human>: Who is H2O?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:",
        "<human>: Who is h2o?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:",
        "<human>: who is h2o?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:",
    ]:
        rows.append(dict(input=row, prompt_type='plain', source='H2O.ai'))
    print(len(rows))
    with open("h2ogpt-personality.json", "w") as f:
        f.write(json.dumps(rows, indent=2))
    return rows


def test_check_stats_data():
    filename = 'h2ogpt-oig-oasst1-instruct-cleaned-v3.json'
    df = pd.read_json(filename)

    # get word stats
    df['char_count'] = df['input'].apply(lambda x: len(x))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.hist(df['char_count'], bins=100)
    chars_avg = np.mean(df['char_count'])
    chars_median = np.median(df['char_count'])
    plt.title("char_count avg: %s median: %s" % (chars_avg, chars_median))
    plt.savefig('chars_hist.png')
    plt.close()

    # get tokenize stats for random sample of 1000 rows
    from finetune import generate_and_tokenize_prompt
    from loaders import get_loaders, get_tokenizer
    from functools import partial

    llama_type = False
    tokenizer_base_model = base_model = 'h2oai/h2ogpt-oasst1-512-20b'
    model_loader, tokenizer_loader = get_loaders(model_name=base_model, reward_type=False, llama_type=llama_type)
    local_files_only = False
    resume_download = True
    use_auth_token = False
    tokenizer = get_tokenizer(tokenizer_loader, tokenizer_base_model, local_files_only, resume_download, use_auth_token)
    prompt_type = 'plain'  # trained with data already in human bot form
    train_on_inputs = True
    add_eos_token = False
    cutoff_len = 512  # can choose 2048
    generate_and_tokenize_prompt_fun = partial(generate_and_tokenize_prompt, prompt_type=prompt_type,
                                               train_on_inputs=train_on_inputs, add_eos_token=add_eos_token,
                                               cutoff_len=cutoff_len, tokenizer=tokenizer)
    from datasets import load_dataset
    data = load_dataset("json", data_files={"train": filename})
    val_set_size = 0.90
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt_fun, num_proc=os.cpu_count())

    df_tokens = pd.DataFrame([len(x) for x in train_data['input_ids']], columns=['token_count'])

    plt.figure(figsize=(10, 10))
    plt.hist(df_tokens['token_count'], bins=100)
    token_avg = np.mean(df_tokens['token_count'])
    token_median = np.median(df_tokens['token_count'])
    plt.title("token_count with cutoff=%s avg: %s median: %s" % (cutoff_len, token_avg, token_median))
    plt.savefig('token_hist_%s.png' % cutoff_len)
    plt.close()


def get_unhelpful_list():
    # base versions
    unhelpful = ["I'm sorry, I didn't quite understand your question, could you please rephrase it?",
                 "I'm sorry, but I don't understand your question. Could you please rephrase it?",
                 "I'm sorry, I don't quite understand your question",
                 "I'm sorry, I don't know",
                 "I'm sorry, but I don't know",
                 "I don't know anything",
                 "I do not know",
                 "I don't know",
                 "I don't know how",
                 "I do not know how",
                 "Can you please explain what you mean",
                 "please explain what you mean",
                 "please explain",
                 "I'm sorry, but I don't know how to tell a story. Can you please explain what you mean by",
                 "I'm sorry but I don't understand what you mean",
                 "I don't understand",
                 "I don't have the ability",
                 "I do not have the ability",
                 "I do not have",
                 "I am a language model,",
                 "I am a large language model,",
                 "I do not understand your question. Can you please try to make it clearer?",
                 "I'm sorry, but as an AI language model",
                 "I apologize, but I cannot rephrase text that I cannot understand. Your post is difficult to read and follow.",
                 "I apologize, but I am not h2oGPT. I am a language model developed by H2O.ai. How may I help you?",
                 "Sorry, but I am not an actual Linux shell, nor am I capable of emulating one. I am an open source chat assistant and would be glad t",
                 "I apologize, but I cannot perform the task you have requested.",
                 "I'm sorry, I cannot perform this task as I am an AI language model and do not have access",
                 "I'm sorry, I'm not sure what you're asking for here.",
                 "I'm not sure what you are asking",
                 "You need to provide more context",
                 ]
    # reduced versions, with redundant parts, just to give context for where they came from
    unhelpful += ["sorry, I didn't quite understand your question",
                  "I didn't quite understand your question",
                  "I didn't understand your question",
                  "I did not understand your question",
                  "I did not understand the question",
                  "could you please rephrase"
                  "could you rephrase"
                  "I do not understand your question.",
                  "I do not understand the question.",
                  "I do not understand that question.",
                  "Can you please try to make it clearer",
                  "Can you try to make it clearer",
                  "sorry, but as an AI language model",
                  "as an AI language model",
                  "I apologize, but I cannot",
                  "I cannot rephrase text",
                  "I cannot understand. Your post is difficult to read and follow."
                  "Your post is difficult to read and follow."
                  "I apologize, but I am",
                  "Sorry, but I am not ",
                  "nor am I capable",
                  "I am not capable of",
                  "I apologize, but I cannot perform the task you have requested",
                  "I cannot perform the task",
                  "I cannot complete the task",
                  "I'm sorry",
                  "I am sorry",
                  "do not have access",
                  "not sure what you're asking for",
                  "not sure what you are asking for",
                  "not sure what is being asked",
                  "I'm not sure what you are asking",
                  "not sure what you are asking",
                  "You need to provide more context",
                  "provide more context",
                  ]
    unhelpful += ["As a large language model",
                  "cannot provide any information",
                  "As an artificial intelligence I do not have the capability",
                  "As an artificial intelligence I don't have the capability",
                  "As an artificial intelligence I can't",
                  "As an artificial intelligence I cannot",
                  "I am sorry but I do not understand",
                  "Can you please explain",
                  "(sorry couldn't resist)",
                  "(sorry could not resist)",
                  " :)",
                  " ;)",
                  " :-)",
                  " ;-)",
                  " lol ",
                  "Thanks so much!!!",
                  "Thank You :)!!!",
                  "Please try not to repeat",
                  "I am an AI language model",
                  "I'm a AI assistant that",
                  "I'm an AI assistant that",
                  "I am an AI assistant that",
                  "etc.",
                  "etc.etc.",
                  "etc. etc.",
                  "etc etc",
                  ]
    return unhelpful


def test_check_unhelpful():
    # file = '/home/jon/Downloads/openassistant_oasst1_h2ogpt_graded.json'
    file = '/home/jon/Downloads/openassistant_oasst1_h2ogpt_grades.json'
    # file = 'h2ogpt-oig-oasst1-instruct-cleaned-v2.json'

    unhelpful = get_unhelpful_list()
    # data = json.load(open(file, 'rt'))
    df = pd.read_json(file)

    use_reward_score_threshold = False
    use_bleu_threshold = False
    use_sentence_sim = True

    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    from nltk.translate.bleu_score import sentence_bleu

    def get_bleu(actual, expected_list):
        # return bleu.sentence_score(actual, expected_list).score
        return sentence_bleu(expected_list, actual)

    threshold = 0.0
    if use_reward_score_threshold:
        df = df[df['grade_deberta'] > threshold]

    # back to as if original json load
    data = df.to_dict(orient='records')
    bads = {}
    string_all = str(data)
    for sub in unhelpful:
        bads[sub] = string_all.count(sub)
    bads = {k: v for k, v in bads.items() if v > 0}
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bads)

    total_bads = sum(list(bads.values()))
    print('total_bads: %s' % total_bads, flush=True)

    # check just bot
    import re
    convs = [[x.strip() for x in re.split(r'%s|%s' % (human, bot), y['input']) if x.strip()] for y in data]
    humans = [[x for i, x in enumerate(y) if i % 2 == 0] for y in convs]
    bots = [[x for i, x in enumerate(y) if i % 2 == 1] for y in convs]

    # FIXME: apply back to json etc., just see for now
    bleu_threshold = 0.9
    if use_bleu_threshold:
        bots = [[x for x in y if get_bleu(x, unhelpful) < bleu_threshold] for y in tqdm(bots)]

    cosine_sim_threshold = 0.8
    if use_sentence_sim:
        # pip install sentence_transformers-2.2.2
        from sentence_transformers import SentenceTransformer
        # sent_model = 'bert-base-nli-mean-tokens'
        # sent_model = 'nli-distilroberta-base-v2'
        sent_model = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(sent_model)
        sentence_embeddings = model.encode(unhelpful)
        from sklearn.metrics.pairwise import cosine_similarity
        bots = [x for x in tqdm(bots) if
                np.max(cosine_similarity(model.encode(x), sentence_embeddings)) < cosine_sim_threshold]

    bads_bots = {}
    string_all = str(bots)
    for sub in unhelpful:
        bads_bots[sub] = string_all.count(sub)
    bads_bots = {k: v for k, v in bads_bots.items() if v > 0}
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bads_bots)

    total_bads_bots = sum(list(bads_bots.values()))
    print('threshold: %g use_bleu_threshold: %g total_bads_bots: %s total_bots: %s total_humans: %s' % (
    threshold, use_bleu_threshold, total_bads_bots, len(bots), len(humans)), flush=True)

    # assert len(bads) == 0, bads
    assert len(bads_bots) == 0, bads_bots


def test_fortune2000_personalized():
    row_list = []
    import glob
    if not os.path.isdir("wikitext"):
        raise RuntimeError("download https://github.com/h2oai/h2ogpt/files/11423008/wikitext.zip and unzip")
    for file in glob.glob("wikitext/*.txt"):
        with open(file, "r") as f:
            blob = f.read()
        N = 512 * 4
        row_list.extend([{'input': s, 'prompt_type': 'plain', 'source': "%s" % os.path.basename(file)}
                         for s in get_sentences(blob, N) if s])
    personality = create_personality_data()
    import copy
    for i in range(10):
        row_list.extend(copy.deepcopy(personality))
    np.random.seed(123)
    np.random.shuffle(row_list)
    for i in range(len(row_list)):
        row_list[i]['id'] = i
    for i in range(len(row_list)):
        assert row_list[i]['id'] == i
    with open("h2ogpt-fortune2000-personalized.json", "w") as ff:
        ff.write(json.dumps(row_list, indent=2))
