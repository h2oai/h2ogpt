#!/bin/bash -e

tps=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l | awk '{if ($1 > 1) print int($1/2); else print 1}')
NCCL_IGNORE_DISABLED_P2P=1 CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi -L | wc -l) > 1 ? $(nvidia-smi -L | wc -l) / 2 - 1 : 0))) \
/h2ogpt_conda/vllm_env/bin/python3.10 -m vllm.entrypoints.openai.api_server \
    --port=5000 \
    --host=0.0.0.0 \
    --model h2oai/h2ogpt-4096-llama2-13b-chat \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --tensor-parallel-size=$tps --seed 1234
