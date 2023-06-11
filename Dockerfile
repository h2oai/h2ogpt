# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    pandoc \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install -r requirements.txt
COPY . .
ENTRYPOINT [ "python3.10"]
