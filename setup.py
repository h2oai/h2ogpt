import os
import re
import setuptools
from typing import List


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        required = f.read().splitlines()
    required = [x for x in required if not x.strip().startswith("#")]
    required = [x if 'git+http' not in x else re.search(r"/([^/]+?)\.git", x).group(1) + ' @ ' + x for x in required]
    required = [x for x in required if x]
    return required


# base requirements list
install_requires = parse_requirements('requirements.txt')

# 4BIT - avoid 4bit deps as part of package until part of normal deps on pypi
install_4bit = parse_requirements('reqs_optional/requirements_optional_4bit.txt')

# LANGCHAIN
install_langchain = parse_requirements('reqs_optional/requirements_optional_langchain.txt')

# FLASH
install_flash = parse_requirements('reqs_optional/requirements_optional_flashattention.txt')

# GPL
install_gpl = parse_requirements('reqs_optional/requirements_optional_langchain.gpllike.txt')

# NO_GPU
install_no_gpu = parse_requirements('reqs_optional/requirements_optional_faiss_cpu.txt')

# GPU
install_gpu = parse_requirements('reqs_optional/requirements_optional_faiss.txt')

# TRAINING
install_extra_training = parse_requirements('reqs_optional/requirements_optional_training.txt')

# WIKI_EXTRA
install_wiki_extra = parse_requirements('reqs_optional/requirements_optional_wikiprocessing.txt')

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(current_directory, 'version.txt'), encoding='utf-8') as f:
    version = f.read().strip()

setuptools.setup(
    name='h2ogpt',
    packages=['h2ogpt'],
    package_dir={
        'h2ogpt': '',
    },
    package_data={
        'h2ogpt': [
            'data/**',
            'docs/**',
            'models/**',
            'spaces/**',
            'tests/**',
        ],
    },
    version=version,
    license='https://opensource.org/license/apache-2-0/',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='H2O.ai',
    author_email='jon.mckinney@h2o.ai, arno@h2o.ai',
    url='https://github.com/h2oai/h2ogpt',
    download_url='',
    keywords=['LLM', 'AI'],
    install_requires=install_requires,
    extras_require={
        'ALL': install_gpl + install_extra_training + install_wiki_extra,
        'CPU': install_no_gpu,
        'GPU': install_gpu,
        'LANGCHAIN': install_langchain,
        '4BIT': install_4bit,
        'FLASH': install_flash,
        'GPL': install_gpl,
        'TRAINING': install_extra_training,
        'WIKI_EXTRA': install_wiki_extra,
    },
    dependency_links=[
        'https://download.pytorch.org/whl/cu117',
    ],
    classifiers=[],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'h2ogpt_finetune=h2ogpt.finetune:entrypoint_main',
            'h2ogpt_generate=h2ogpt.generate:entrypoint_main',
        ],
    },
)
