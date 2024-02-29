import os
import platform
import re
import sys

import setuptools
from typing import List
from setuptools import find_packages


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        lines = f.read().splitlines()

    # Filter out comments and empty lines
    lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

    requirements = []
    for line in lines:
        # Separate and evaluate environment markers if present
        if ";" in line:
            line, marker = line.split(";", 1)
            include_req = evaluate_marker(marker.strip())
            if not include_req:
                continue  # Skip this requirement if the marker conditions are not met

        # Handle Git URLs
        if 'git+http' in line or 'git+https' in line:
            pkg_name_match = re.search(r"/([^/]+?)\.git", line)
            if pkg_name_match and '@' not in line:
                pkg_name = pkg_name_match.group(1)
                requirements.append(pkg_name + ' @ ' + line)
            else:
                requirements.append(line)
        elif line.startswith("http://") or line.startswith("https://"):
            # Directly append http(s) links, assuming they're already in PEP 508 format
            requirements.append(line)
        else:
            # Regular PyPI packages
            requirements.append(line)

    return requirements


def evaluate_marker(marker: str) -> bool:
    """Evaluate an environment marker. Return True if the marker conditions are met, else False."""
    # This is a simplified evaluator and might need adjustment for complex markers
    environment = {
        'os_name': os.name,
        'sys_platform': sys.platform,
        'platform_system': platform.system(),
        'platform_machine': platform.machine(),
        'python_version': platform.python_version(),
        'python_full_version': platform.python_version(),
        'platform_python_implementation': platform.python_implementation(),
        'implementation_name': platform.python_implementation().lower(),
        'python_version_major': str(sys.version_info[0]),
        'python_version_minor': str(sys.version_info[1]),
        'python_version_patch': str(sys.version_info[2]),
    }
    try:
        return eval(marker, environment)
    except Exception as e:
        print(f"Error evaluating marker '{marker}': {e}")
        return False

install_requires = parse_requirements('requirements.txt')

req_files = [
    'reqs_optional/requirements_optional_langchain.txt',
    'reqs_optional/requirements_optional_gpt4all.txt',
    'reqs_optional/requirements_optional_langchain.gpllike.txt',
    'reqs_optional/requirements_optional_agents.txt',
    'reqs_optional/requirements_optional_langchain.urls.txt',
    'reqs_optional/requirements_optional_doctr.txt',
]

for req_file in req_files:
    x = parse_requirements(req_file)
    install_requires.extend(x)

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
        'FAISS_CPU': install_faiss_cpu,
        'FAISS': install_faiss,
        'TRAINING': install_extra_training,
        'WIKI_EXTRA': install_wiki_extra,
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
