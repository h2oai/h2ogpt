import os
import glob
import setuptools
from typing import List

from utils import get_ngpus_vis


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        required = f.read().splitlines()
    required = [x for x in required if not x.strip().startswith("#")]
    required = [x if 'git+http' not in x else 'peft @' + x for x in required]
    required = [x for x in required if x]
    return required


do_install_optional_req = bool(int(os.environ.get('OPTIONAL', '1')))
do_install_gpl = bool(int(os.environ.get('GPL', '1')))
do_install_extra_training = bool(int(os.environ.get('TRAINING', '0')))
do_install_wiki_extra = bool(int(os.environ.get('WIKI_EXTRA', '0')))
# avoid 4bit deps as part of package until part of normal deps on pypi
do_install_4bit = bool(int(os.environ.get('4BIT', '0')))
do_install_flash = bool(int(os.environ.get('FLASH', '0')))
have_gpus = int(get_ngpus_vis(raise_if_exception=False) > 0)
do_gpu = bool(int(os.environ.get('GPU', str(have_gpus))))

# base requirements list
base_req = 'requirements.txt'
install_requires = parse_requirements(base_req)
# list of optional requirement files
all_optional_reqs = glob.glob('reqs_optional/requirements*.txt')
exceptional_deps = []
if not do_install_4bit:
    exceptional_deps.append('reqs_optional/requirements_optional_4bit.txt')
if not do_install_flash:
    exceptional_deps.append('reqs_optional/requirements_optional_flashattention.txt')
if not do_install_gpl:
    exceptional_deps.append('reqs_optional/requirements_optional_langchain.gpllike.txt')
if do_gpu:
    exceptional_deps.append('reqs_optional/requirements_optional_faiss_cpu.txt')
if not do_gpu:
    exceptional_deps.append('reqs_optional/requirements_optional_faiss.txt')
if not do_install_extra_training:
    exceptional_deps.append('reqs_optional/requirements_optional_training.txt')
if not do_install_wiki_extra:
    exceptional_deps.append('reqs_optional/requirements_optional_wikiprocessing.txt')

all_optional_reqs = [x for x in all_optional_reqs if x not in exceptional_deps]

if do_install_optional_req:
    for opt_req in all_optional_reqs:
        if opt_req == base_req:
            continue
        install_requires.extend(parse_requirements(opt_req))

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setuptools.setup(
    # Name of the package
    name='h2ogpt',
    # Packages to include into the distribution
    packages=['h2ogpt'],
    #package_dir={'h2ogpt': 'src/h2ogpt'},
    package_dir={'h2ogpt': './'},
    # Start with a small number and increase it with
    # every change you make https://semver.org
    version='0.1.0',
    # Chose a license from here: https: //
    # help.github.com / articles / licensing - a -
    # repository. For example: MIT
    license='https://opensource.org/license/apache-2-0/',
    # Short description of your library
    description='',
    # Long description of your library
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Your name
    author='H2O.ai',
    # Your email
    author_email='jon.mckinney@h2o.ai, arno@h2o.ai',
    # Either the link to your github or to your website
    url='https://github.com/h2oai/h2ogpt',
    # Link from which the project can be downloaded
    download_url='',
    # List of keywords
    keywords=['LLM', 'AI'],
    # List of packages to install with this one
    install_requires=install_requires,
    # https://pypi.org/classifiers/
    classifiers=[],
    python_requires=">=3.10",
)
