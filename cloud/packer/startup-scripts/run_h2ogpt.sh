#!/bin/bash -e

CUDA_VISIBLE_DEVICES=$(seq -s, $(($(nvidia-smi -L | wc -l) / 2)) $(($(nvidia-smi -L | wc -l) - 1))) /h2ogpt_conda/bin/python3.10 \
  /workspace/generate.py \
  --inference_server="vllm:0.0.0.0:5000" \
  --base_model=h2oai/h2ogpt-4096-llama2-13b-chat \
  --langchain_mode=UserData
