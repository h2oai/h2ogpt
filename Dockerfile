FROM gcr.io/vorvan/h2oai/h2ogpt-oss-wolfi-base:3 AS base-stage

USER root

ENV HOME=/workspace
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV VLLM_CACHE=/workspace/.vllm_cache
ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

FROM base-stage as intermediate-stage

## copy code
COPY .              /workspace/

# copy build info
COPY build_info.txt /workspace/
COPY git_hash.txt   /workspace/

# copy install script
COPY linux_install_wolfi.sh    /workspace/

# run setup
RUN cd /workspace && ./linux_install_wolfi.sh

# mv to separate locations so can copy from under a enw docker layer
RUN \
    mkdir -p /docker_cache && \
    mv /usr/lib/python3.10/site-packages/nvidia         /docker_cache/nvidia && \
    mv /usr/lib/python3.10/site-packages/torch          /docker_cache/torch && \
    mv /usr/lib/python3.10/site-packages/onnxruntime    /docker_cache/onnxruntime && \
    mv /usr/lib/python3.10/site-packages/triton         /docker_cache/triton && \
    mv /usr/lib/python3.10                              /docker_cache/python_data && \
    cp -R /usr                                          /docker_cache/user_data

# remove since already in base image and didn't change
RUN \
    rm -rf /docker_cache/user_data/lib && \
    rm -rf /docker_cache/user_data/libexec && \
    rm -rf /docker_cache/user_data/local && \
    rm -rf /docker_cache/user_data/bin/pandoc && \
    rm -rf /docker_cache/user_data/bin/node && \
    rm -rf /docker_cache/user_data/bin/lto-dump-11 && \
    rm -rf /docker_cache/user_data/bin/lto-dump && \
    rm -rf /docker_cache/user_data/share/misc/magic.mgc && \
    rm -rf /docker_cache/user_data/share/icu && \
    rm -rf /docker_cache/user_data/x86_64-pc-linux-gnu && \
    rm -rf /docker_cache/python_data/site-packages/future/backports/test

# cleanup
RUN rm -rf /workspace/.cache && \
    rm -rf /workspace/spaces && \
    rm -rf /workspace/benchmarks && \
    rm -rf /workspace/data && \
    rm -rf /workspace/cloud && \
    rm -rf /workspace/docs && \
    rm -rf /workspace/helm && \
    rm -rf /workspace/notebooks && \
    rm -rf /workspace/papers

RUN mkdir -p /workspace/save

# make main workspace writable
RUN chmod -R a+rwx /workspace

FROM base-stage as final-stage

COPY --from=intermediate-stage    /docker_cache/user_data         /usr
COPY --from=intermediate-stage    /docker_cache/python_data/      /usr/lib/python3.10/
COPY --from=intermediate-stage    /docker_cache/nvidia/           /usr/lib/python3.10/site-packages/nvidia/
COPY --from=intermediate-stage    /docker_cache/torch/            /usr/lib/python3.10/site-packages/torch/
COPY --from=intermediate-stage    /docker_cache/onnxruntime/      /usr/lib/python3.10/site-packages/onnxruntime/
COPY --from=intermediate-stage    /docker_cache/triton/           /usr/lib/python3.10/site-packages/triton/

COPY --from=intermediate-stage    /workspace/build_info.txt       /build_info.txt
COPY --from=intermediate-stage    /workspace/git_hash.txt         /git_hash.txt
COPY --from=intermediate-stage    /workspace                      /workspace
RUN chmod a+rwx /workspace  # only for top dir, as docker COPY skips it.

EXPOSE 8888
EXPOSE 7860
EXPOSE 5000

USER h2ogpt

ENTRYPOINT ["python3.10"]
