#!/bin/sh

echo "$(date '+%F %T') BEGIN: run-gpt.sh"

set -e

export TRANSFORMERS_CACHE=/h2ogpt_env/.cache

# run generate.py
mkdir -p /h2ogpt_env && cd /h2ogpt_env
exec python3.10 /workspace/generate.py --base_model="${HF_MODEL}"
