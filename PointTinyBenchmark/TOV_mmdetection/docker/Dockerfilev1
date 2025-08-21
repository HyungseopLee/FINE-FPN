ARG PYTORCH="1.6.0"
ARG CUDA="10.1" 
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# 모든 필요한 NVIDIA GPG keys 추가
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf21e8b.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        ninja-build \
        libglib2.0-0 \
        libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 환경 생성
RUN conda create -n open-mmlab python=3.7 -y

# Conda 환경용 PATH 설정
ENV PATH /opt/conda/envs/open-mmlab/bin:$PATH

# PyTorch + CUDA + 필수 패키지 설치
RUN conda install -n open-mmlab -y pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch \
    && conda install -n open-mmlab -y scikit-image ninja git ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev -c conda-forge

# MMCV 설치
RUN pip install mmcv-full==1.3.17

# MMDetection 설치
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -r requirements/build.txt
RUN pip install -v -e .

# 작업 디렉토리
WORKDIR /mmdetection