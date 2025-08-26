# -----------------------------
# RTX 3090 전용 Dockerfile (Python 3.7 유지)
# CUDA 11.8 + MMDetection + MMCV
# -----------------------------

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# -----------------------------
# 환경 변수
# -----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

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