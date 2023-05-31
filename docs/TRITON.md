## Triton Inference Server

To get optimal performance for inference for h2oGPT models, we will be using the [FastTransformer Backend for Triton](https://github.com/triton-inference-server/fastertransformer_backend/).

Make sure to [install Docker](INSTALL-DOCKER.md) first.

### Build Docker image for Triton with FasterTransformer backend:

```bash
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
cd fastertransformer_backend
git clone https://github.com/NVIDIA/FasterTransformer.git
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

We convert the h2oGPT model from [HF to FT format](https://github.com/NVIDIA/FasterTransformer/pull/569):

####  Fetch model from Hugging Face
```bash
export MODEL=h2ogpt-oig-oasst1-512-6_9b
if [ ! -d ${MODEL} ]; then
    git lfs clone https://huggingface.co/h2oai/${MODEL}
fi
```
If `git lfs` fails, make sure to install it first. For Ubuntu:
```bash
sudo apt-get install git-lfs
```

####  Convert to FasterTransformer format

```bash
export WORKSPACE=$(pwd)
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
# Go into Docker
docker run -it --rm --runtime=nvidia --shm-size=1g \
       --ulimit memlock=-1 -v ${WORKSPACE}:${WORKSPACE} \
       -e CUDA_VISIBLE_DEVICES=0 \
       -e MODEL=${MODEL} \
       -e WORKSPACE=${WORKSPACE} \
       -w ${WORKSPACE} ${TRITON_DOCKER_IMAGE} bash
export PYTHONPATH=${WORKSPACE}/FasterTransformer/:$PYTHONPATH
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptneox/utils/huggingface_gptneox_convert.py \
        -i_g 1 \
        -m_n gptneox \
        -i ${WORKSPACE}/${MODEL} \
        -o ${WORKSPACE}/FT-${MODEL}
```

####  Test the FasterTransformer model

FIXME
```bash
echo "Hi, who are you?" > gptneox_input
echo "And you are?" >> gptneox_input
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptneox/gptneox_example.py \
         --ckpt_path ${WORKSPACE}/FT-${MODEL}/1-gpu \
         --tokenizer_path ${WORKSPACE}/${MODEL} \
         --sample_input_file gptneox_input
```

#### Update Triton configuration files

Fix a typo in the example:
```bash
sed -i -e 's@postprocessing@preprocessing@' all_models/gptneox/preprocessing/config.pbtxt
```

Update the path to the PyTorch model, and set to use 1 GPU:
```bash
sed -i -e "s@/workspace/ft/models/ft/gptneox/@${WORKSPACE}/FT-${MODEL}/1-gpu@" all_models/gptneox/fastertransformer/config.pbtxt
sed -i -e 's@string_value: "2"@string_value: "1"@' all_models/gptneox/fastertransformer/config.pbtxt
```

#### Launch Triton

```bash
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 \
        --allow-run-as-root /opt/tritonserver/bin/tritonserver  \
        --model-repository=${WORKSPACE}/all_models/gptneox/ &
```

Now, you should see something like this:
```bash
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| ensemble          | 1       | READY  |
| fastertransformer | 1       | READY  |
| postprocessing    | 1       | READY  |
| preprocessing     | 1       | READY  |
+-------------------+---------+--------+
```
which means the pipeline is ready to make predictions!

### Run client test

Let's test the endpoint:
```bash
python3 ${WORKSPACE}/tools/gpt/identity_test.py
```

And now the end-to-end test:

We first have to fix a bug in the inputs for postprocessing:
```bash
sed -i -e 's@prepare_tensor("RESPONSE_INPUT_LENGTHS", output2, FLAGS.protocol)@prepare_tensor("sequence_length", output1, FLAGS.protocol)@' ${WORKSPACE}/tools/gpt/end_to_end_test.py
```

```bash
python3 ${WORKSPACE}/tools/gpt/end_to_end_test.py
```


