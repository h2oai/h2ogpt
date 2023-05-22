from pathlib import Path

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict


def test_requirements():
    """Test that each required package is available."""
    packages_all = []
    packages_dist = []
    packages_version = []
    packages_unkn = []
    req_file = "requirements.txt"
    req_tmp_file = req_file + '.tmp.txt'

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
            
    _REQUIREMENTS_PATH = Path(__file__).parent.with_name(req_tmp_file)
    requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
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
        print('pip uninstall peft -y ; CUDA_HOME=/usr/local/cuda-11.7 pip install %s --upgrade' % str(' '.join(packages_all)), flush=True)
        print('\n\n', flush=True)

        raise ValueError(packages_all)
    
