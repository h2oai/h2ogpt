#!/bin/bash
set -o pipefail
set -ex


# vllm server
cd /h2ogpt_conda
python -m venv vllm_env --system-site-packages
# gputil is for rayWorker in vllm to run as non-root
# below required outside docker:
# apt-get install libnccl2
#/h2ogpt_conda/vllm_env/bin/python -m pip install https://h2o-release.s3.amazonaws.com/h2ogpt/vllm-0.2.7%2Bcu118-cp310-cp310-linux_x86_64.whl
#/h2ogpt_conda/vllm_env/bin/python -m pip install https://github.com/vllm-project/vllm/releases/download/v0.2.7/vllm-0.2.7+cu118-cp310-cp310-manylinux1_x86_64.whl
#/h2ogpt_conda/vllm_env/bin/python -m pip install vllm

/h2ogpt_conda/vllm_env/bin/python -m pip install ray pandas gputil==1.4.0 fschat==0.2.34 flash-attn==2.4.2 autoawq==0.1.8 uvicorn[standard] hf_transfer==0.1.5
/h2ogpt_conda/vllm_env/bin/python -m pip install https://h2o-release.s3.amazonaws.com/h2ogpt/megablocks-0.5.1-cp310-cp310-linux_x86_64.whl
/h2ogpt_conda/vllm_env/bin/python -m pip install https://h2o-release.s3.amazonaws.com/h2ogpt/triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
/h2ogpt_conda/vllm_env/bin/python -m pip install https://h2o-release.s3.amazonaws.com/h2ogpt/mosaicml_turbo-0.0.9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# below has issue that compiled on A100, doesn't seem to work on V100, go back to vllm's own build
#/h2ogpt_conda/vllm_env/bin/python -m pip install https://h2o-release.s3.amazonaws.com/h2ogpt/vllm-0.3.0-cp310-cp310-manylinux1_x86_64.whl
/h2ogpt_conda/vllm_env/bin/python -m pip install vllm==0.3.3
mkdir -p $VLLM_CACHE

# Make sure old python location works in case using scripts from old documentation
mkdir -p /h2ogpt_conda/envs/vllm/bin
ln -s /h2ogpt_conda/vllm_env/bin/python3.10 /h2ogpt_conda/envs/vllm/bin/python3.10
