import os

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

from src.utils import remove, makedirs, download
from tests.utils import wrap_test_forked


def get_all_requirements():
    import glob
    requirements_all = []
    reqs_http_all = []
    for req_name in ['requirements.txt'] + glob.glob('reqs_optional/req*.txt'):
        if 'reqs_constraints.txt' in req_name:
            continue
        if 'requirements_optional_training.txt' in req_name:
            continue
        requirements1, reqs_http1 = get_requirements(req_name)
        requirements_all.extend(requirements1)
        reqs_http_all.extend(reqs_http1)
    return requirements_all, reqs_http_all


def get_requirements(req_file="requirements.txt"):
    req_tmp_file = req_file + '.tmp.txt'
    try:

        reqs_http = []

        with open(req_file, 'rt') as f:
            contents = f.readlines()
            with open(req_tmp_file, 'wt') as g:
                for line in contents:
                    if 'http://' not in line and 'https://' not in line:
                        g.write(line)
                    else:
                        reqs_http.append(line.replace('\n', ''))
        reqs_http = [x for x in reqs_http if x]
        print('reqs_http: %s' % reqs_http, flush=True)

        with open(req_tmp_file, "rt") as f:
            requirements = pkg_resources.parse_requirements(f.read())
    finally:
        remove(req_tmp_file)
    return requirements, reqs_http


@wrap_test_forked
def test_requirements():
    """Test that each required package is available."""
    packages_all = []
    packages_dist = []
    packages_version = []
    packages_unkn = []

    requirements, reqs_http = get_all_requirements()

    for requirement in requirements:
        try:
            requirement = str(requirement)
            pkg_resources.require(requirement)
        except DistributionNotFound:
            packages_all.append(requirement)
            packages_dist.append(requirement)
        except VersionConflict:
            packages_all.append(requirement)
            packages_version.append(requirement)
        except pkg_resources.extern.packaging.requirements.InvalidRequirement:
            packages_all.append(requirement)
            packages_unkn.append(requirement)

    packages_all.extend(reqs_http)
    if packages_dist or packages_version:
        print('Missing packages: %s' % packages_dist, flush=True)
        print('Wrong version of packages: %s' % packages_version, flush=True)
        print("Can't determine (e.g. http) packages: %s" % packages_unkn, flush=True)
        print('\n\nRUN THIS:\n\n', flush=True)
        print(
            'pip uninstall peft transformers accelerate -y ; CUDA_HOME=/usr/local/cuda-12.1 pip install %s --upgrade' % str(
                ' '.join(packages_all)), flush=True)
        print('\n\n', flush=True)

        raise ValueError(packages_all)


import requests
import json

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

URL_PATTERN = 'https://pypi.python.org/pypi/{package}/json'


def get_version(package, url_pattern=URL_PATTERN):
    """Return version of package on pypi.python.org using json."""
    req = requests.get(url_pattern.format(package=package))
    version = parse('0')
    if req.status_code == requests.codes.ok:
        j = json.loads(req.text.encode(req.encoding))
        releases = j.get('releases', [])
        for release in releases:
            ver = parse(release)
            if not ver.is_prerelease:
                version = max(version, ver)
    return version


@wrap_test_forked
def test_what_latest_packages():
    # pip install requirements-parser
    import requirements
    import glob
    for req_name in ['requirements.txt'] + glob.glob('reqs_optional/req*.txt'):
        print("\n File: %s" % req_name, flush=True)
        with open(req_name, 'rt') as fd:
            for req in requirements.parse(fd):
                from importlib.metadata import version
                try:
                    current_version = version(req.name)
                    latest_version = get_version(req.name)
                    if str(current_version) != str(latest_version):
                        print("%s: %s -> %s" % (req.name, current_version, latest_version), flush=True)
                except Exception as e:
                    print("Exception: %s" % str(e), flush=True)


@wrap_test_forked
def test_make_packages():
    # for https://github.com/pypiserver/pypiserver

    dryrun = False

    """Test that each required package is available."""
    reqs, reqs_http = get_all_requirements()

    makedirs('packages')
    print("PACKAGES START\n\n\n")
    for requirement in reqs_http:
        if requirement.startswith('#') and ('.whl' in requirement or 'http' in requirement):
            requirement = requirement[1:]
        if ('https://' in requirement or 'http://' in requirement) and '@' in requirement:
            requirement = requirement[requirement.index('@')+1:]
        if ';' in requirement:
            requirement = requirement[:requirement.index(';')]
        requirement = requirement.strip()
        print(requirement)
        if not dryrun:
            if '.whl' in requirement:
                download(requirement, dest_path='packages')
            else:
                os.system('cd packages && pip wheel %s --no-deps' % requirement)

    for req1 in reqs:
        name = req1.name
        if req1.specs:
            version = req1.specs[0][1]
        else:
            version = None
        req1 = str(req1)
        req1 = req1.strip()
        if ';' in str(req1):
            req1 = req1[:req1.index(';')]
        print(req1)
        if not dryrun:
            if version:
                os.system('cd packages && pip wheel %s==%s --no-deps' % (name, version))
            else:
                os.system('cd packages && pip wheel %s --no-deps' % name)
    # then do on host with server: (pypiserver) ubuntu@ip-10-10-0-245:~/packages$ scp jon@pseudotensor.hopto.org:h2ogpt/packages/* .
