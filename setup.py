import os

import setuptools
from typing import List
from setuptools import find_packages

for_pypi = os.getenv('PYPI') is not None


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        lines = f.read().splitlines()

    # Filter out comments and empty lines
    lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

    requirements = []
    for line in lines:
        if 'chromamigdb' in line:
            # hnsw issue
            continue
        if for_pypi:
            if 'http://' in line or 'https://' in line:
                continue
            if 'llama-cpp-python' in line and ';' in line:
                line = line[:line.index(';')]

        # assume all requirements files are in PEP 508 format with name @ <url> or name @ git+http/git+https
        requirements.append(line)

    return requirements


install_requires = parse_requirements('requirements.txt')

req_files = [
    'reqs_optional/requirements_optional_langchain.txt',
    'reqs_optional/requirements_optional_llamacpp_gpt4all.txt',
    'reqs_optional/requirements_optional_langchain.gpllike.txt',
    'reqs_optional/requirements_optional_agents.txt',
    'reqs_optional/requirements_optional_langchain.urls.txt',
    'reqs_optional/requirements_optional_doctr.txt',
    'reqs_optional/requirements_optional_audio.txt',
    'reqs_optional/requirements_optional_image.txt',
]

for req_file in req_files:
    x = parse_requirements(req_file)
    install_requires.extend(x)

# faiss on cpu etc.
install_cpu = parse_requirements('reqs_optional/requirements_optional_cpu_only.txt')

# faiss on gpu etc.
install_cuda = parse_requirements('reqs_optional/requirements_optional_gpu_only.txt')

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
packages = find_packages(include=['h2ogpt', 'h2ogpt.*'], exclude=['tests'])

setuptools.setup(
    name='h2ogpt',
    packages=packages,
    package_data={
        # If 'h2ogpt' is your package directory and 'spkemb' is directly inside it
        'h2ogpt': ['spkemb/*.npy'],
        # If 'spkemb' is inside 'src' which is inside 'h2ogpt'
        # Adjust the string according to your actual package structure
        'h2ogpt.src': ['spkemb/*.npy'],
    },
    exclude_package_data={
        'h2ogpt': [
            '**/__pycache__/**',
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
        'cpu': install_cpu,
        'cuda': install_cuda,
        'TRAINING': install_extra_training,
        'WIKI_EXTRA': install_wiki_extra,
        'local-inference': ['unstructured[local-inference]>=0.12.5,<0.13'],
    },
    classifiers=[],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'h2ogpt_finetune=h2ogpt.finetune:entrypoint_main',
            'h2ogpt_generate=h2ogpt.generate:entrypoint_main',
        ],
    },
)
