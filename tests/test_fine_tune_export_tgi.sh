export MODEL=h2ogpt-oasst1-falcon-7b-test
export HF_PORT=5000
export MODEL_NAME=`echo $MODEL | sed 's@/@_@g'`
export CUDA_VISIBLE_DEVICES=0

rm -rf $MODEL.lora

# Train LoRA
python finetune.py --data_path=h2oai/openassistant_oasst1_h2ogpt --base_model=tiiuae/falcon-7b --num_epochs=0.01 --output_dir=$MODEL.lora

# Merge LoRA, export model to $MODEL dir (via env var)
python src/export_hf_checkpoint.py

# Load model with TGI
docker run --gpus all --shm-size 1g -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -e TRANSFORMERS_CACHE="/.cache/" -p $HF_PORT:80 -v $HOME/.cache:/.cache/ -v $PWD/$MODEL:/data/$MODEL ghcr.io/huggingface/text-generation-inference:latest --model-id /data/$MODEL --max-input-length 2048 --max-total-tokens 4096 --max-stop-sequences 6 --sharded false
