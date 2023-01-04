FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS nvidia

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# CUDA
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=2
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION

ENV PATH=/usr/loca/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH_NO_STUBS="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/conda/lib"
ENV LD_LIBRARY_PATH="/usr/loca/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/opt/conda/lib"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"

# Add KAKAO ubuntu archive mirror server
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list && \
    apt-get update

# openjdk java vm 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    g++ \
    gcc \
    openjdk-8-jdk \
    python3-dev \
    python3-pip \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    libzmq3-dev \
    vim \
    git

RUN apt-get update

ARG CONDA_DIR=/opt/conda

# Add to path
ENV PATH $CONDA_DIR/bin:$PATH

# Install miniconda
RUN echo "export PATH=$CONDA_DIR/bin:"'$PATH' > /etc/profile.d/conda.sh && \
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# create conda vm
RUN conda config --set always_yes yes --set changeps1 no && \
    conda create -y -q -n py38 python=3.8

ENV PATH /opt/conda/envs/py38/bin:$PATH
ENV CONDA_DEFAULT_ENV py38
ENV CONDA_PREFIX /opt/conda/envs/py38

# install packages
RUN pip install --upgrade pip

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install tqdm \
    pip install pycocotools==2.0.6 \
    pip install pyyaml \
    pip install tensorboard==2.11.0 \
    pip install tensorboardX==2.5.1 \
    pip install funcy==1.17 \
    pip install webcolors \
    pip install matplotlib \
    pip install numpy==1.21.3 \
    pip install opencv-python \
    pip install opencv-contrib-python \
    pip install Pillow \
    pip install PyMySQL==1.0.2 \
    pip install seaborn \
    pip install -U scikit-learn \
    pip install torchsummaryX==1.3.0 \
	pip install fiftyone \
	pip install ipykernel \
	pip install ipywidgets

# pytorch
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Remove the CUDA stubs
ENV LD_LIBRARY_PATH="&LD_LIBRARY_PATH_NO_STUBS"

RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -a -y 

# LAND EV SETTING
ENV LANG Ko_KR.UTF-8
