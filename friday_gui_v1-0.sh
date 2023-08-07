#!/bin/bash

python generate.py \
    --share=True \
    --gradio_offline_level=1 \
    --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 \
    --score_model=None \
    --load_4bit=False \
    --prompt_type=human_bot \
    --user_path='/mnt/llm/synology_github/h2ogpt_rg/data/training' \
    --allow_upload_to_user_data=True
