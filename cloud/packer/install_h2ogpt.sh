#!/bin/bash -e

export PATH=$PATH:/home/ubuntu/.local/bin
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt

if [ -z "$BRANCH_TAG" ]; then
  echo "BRANCH_TAG environment variable is not set."
  exit 1
fi

git checkout $BRANCH_TAG

# Setup h2oGPT
./docker_build_script_ubuntu.sh
