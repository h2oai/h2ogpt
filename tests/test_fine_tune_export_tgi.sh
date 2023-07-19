export DATA=h2oai/openassistant_oasst1_h2ogpt

export BASE_MODEL=tiiuae/falcon-7b  # confirmed working with 0.9.2
# export BASE_MODEL=openlm-research/open_llama_3b  # fails with OOM on 48GB card??
# export BASE_MODEL=Salesforce/xgen-7b-8k-base  # fails since tokenizer not yet supported (have to hack to force LLaMa tokenizer)

export CUDA_VISIBLE_DEVICES=0

export MODEL=model-test
export MODEL_NAME=`echo $MODEL | sed 's@/@_@g'`
export HF_PORT=1000
#export TGI_VERSION=latest  # works
#export TGI_VERSION=0.9.1  # fails
export TGI_VERSION=0.9.3  # works


# Train LoRA
rm -rf $MODEL.lora
python finetune.py --data_path=$DATA --base_model=$BASE_MODEL --num_epochs=0.01 --output_dir=$MODEL.lora

# Merge LoRA, export model to $MODEL dir (via env var)
rm -rf $MODEL
python src/export_hf_checkpoint.py

# Load model with TGI
docker run --gpus all --shm-size 1g -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -e TRANSFORMERS_CACHE="/.cache/" -p $HF_PORT:80 -v $HOME/.cache:/.cache/ -v $PWD/$MODEL:/$MODEL ghcr.io/huggingface/text-generation-inference:$TGI_VERSION --model-id /$MODEL --max-input-length 2048 --max-total-tokens 4096 --max-stop-sequences 6 --sharded false --disable-custom-kernels --trust-remote-code
