## Triton Inference Server

To get optimal performance for inference for h2oGPT models, we will be using the [FastTransformer Backend for Triton](https://github.com/triton-inference-server/fastertransformer_backend/).

To build in Docker, we follow the [instructions](https://github.com/triton-inference-server/fastertransformer_backend/blob/main/README.md#setup):

### Build Docker image for Triton with FasterTransformer backend:

```bash
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
git clone https://github.com/NVIDIA/FasterTransformer.git
cd fastertransformer_backend
export WORKSPACE=$(pwd)
export CONTAINER_VERSION=22.12
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
docker build --rm   \
    --build-arg TRITON_VERSION=${CONTAINER_VERSION}   \
    -t ${TRITON_DOCKER_IMAGE} \
    -f docker/Dockerfile \
    .
docker run -it --rm --runtime=nvidia --shm-size=1g \
       --ulimit memlock=-1 -v ${WORKSPACE}:${WORKSPACE} \
       -w ${WORKSPACE} ${TRITON_DOCKER_IMAGE} bash
```

### Create model definition files

We convert the h2oGPT model from [HF to FT format](https://github.com/NVIDIA/FasterTransformer/pull/569):

####  Fetch model from Hugging Face
```bash
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export PYTHONPATH=$PWD/FasterTransformer/:$PYTHONPATH
export MODEL=h2ogpt-oig-oasst1-512-6.9b
if [ ! -d ${MODEL} ]; then
    git lfs clone https://huggingface.co/h2oai/${MODEL}
fi
```

####  Convert to FasterTransformer format

```bash
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export PYTHONPATH=$PWD/FasterTransformer/:$PYTHONPATH
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptneox/utils/huggingface_gptneox_convert.py \
        -i_g 1 \
        -m_n gptneox \
        -i ${WORKSPACE}/${MODEL} \
        -o ${WORKSPACE}/FT-${MODEL}
```

####  Run the model

FIXME - not yet working
```bash
echo "Hi, who are you?" > gptneox_input
echo "And you are?" >> gptneox_input
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptneox/gptneox_example.py \
         --ckpt_path ${WORKSPACE}/FT-${MODEL}/1-gpu \
         --tokenizer_path ${WORKSPACE}/${MODEL} \
         --sample_input_file gptneox_input
```

### Launch Triton

```bash
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
CUDA_VISIBLE_DEVICES=0,1 mpirun -n 1 \
        --allow-run-as-root /opt/tritonserver/bin/tritonserver  \
        --model-repository=${WORKSPACE}/all_models/gptneox/fastertransformer/
```

### Run client test

```bash
TODO
```


