#!/bin/bash -e

export PATH=$PATH:/home/ubuntu/.local/bin
sudo mkdir -p /workspace && cd /workspace
sudo chmod a+rwx .

git config --global --add safe.directory /workspace
git config --global advice.detachedHead false
git clone https://github.com/h2oai/h2ogpt.git .

if [ -z "$BRANCH_TAG" ]; then
  echo "BRANCH_TAG environment variable is not set."
  exit 1
fi

git checkout $BRANCH_TAG

ls -la
sudo ./docker_build_script_ubuntu.sh
