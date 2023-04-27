## Triton Inference Server

To get optimal performance for inference for h2oGPT models, we will be using the [FastTransformer Backend for Triton](https://github.com/triton-inference-server/fastertransformer_backend/).

To build in Docker, we follow the [instructions](https://github.com/triton-inference-server/fastertransformer_backend/blob/main/README.md#setup):

### Build Docker image for Triton with FasterTransformer backend:

```bash
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
cd fastertransformer_backend
export WORKSPACE=$(pwd)
export CONTAINER_VERSION=22.12
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
docker build --rm   \
    --build-arg TRITON_VERSION=${CONTAINER_VERSION}   \
    -t ${TRITON_DOCKER_IMAGE} \
    -f docker/Dockerfile \
    .
```

### Create model definition files

TODO

### Launch Triton

```bash
docker run -it --rm --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -v ${WORKSPACE}:${WORKSPACE} -w ${WORKSPACE} ${TRITON_DOCKER_IMAGE} bash

# Now inside Docker, start Triton

export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models

CUDA_VISIBLE_DEVICES=0,1 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=${WORKSPACE}/all_models/gptneox/
```

### Run client test

```bash
TODO
```


