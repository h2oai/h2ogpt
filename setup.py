import os
import re
import setuptools
from typing import List
from setuptools import find_packages


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        required = f.read().splitlines()
    required = [x for x in required if not x.strip().startswith("#")]
    required = [x if 'git+http' not in x else re.search(r"/([^/]+?)\.git", x).group(1) + ' @ ' + x for x in required]
    required = [x for x in required if x]
    return required


# base requirements list
install_requires = parse_requirements('requirements.txt')
install_requires.extend(parse_requirements('reqs_optional/requirements_optional_langchain.txt'))
install_requires.extend(parse_requirements('reqs_optional/requirements_optional_gpt4all.txt'))
install_requires.extend(parse_requirements('reqs_optional/requirements_optional_langchain.gpllike.txt'))

# FLASH
install_flashattention = parse_requirements('reqs_optional/requirements_optional_flashattention.txt')

# FAISS_CPU
install_faiss_cpu = parse_requirements('reqs_optional/requirements_optional_faiss_cpu.txt')

# FAISS
install_faiss = parse_requirements('reqs_optional/requirements_optional_faiss.txt')

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

# Data to include
packages = [p + '/**' for p in find_packages(include='*',exclude=['tests'])]

setuptools.setup(
    name='h2ogpt',
    packages=['h2ogpt'],
    package_dir={
        'h2ogpt': '',
    },
    package_data={
        'h2ogpt': list(set([
            'spaces/**',
        ] + packages)),
    },
    exclude_package_data={
        'h2ogpt': [
            '**/__pycache__/**',
            'models/modelling_RW_falcon40b.py',
            'models/modelling_RW_falcon7b.py',
            'models/README-template.md'
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
        'FLASH': install_flashattention,
        'FAISS_CPU': install_faiss_cpu,
        'FAISS': install_faiss,
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
