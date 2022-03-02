ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx curl wget vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install gdown
RUN pip install gdown

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install mmcv, mmdet and mmseg
RUN pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
# RUN pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.17.0
RUN pip install mmsegmentation==0.18.0

# Install mmdetection3d
RUN conda clean --all
RUN git clone https://gitlab.com/konrad966/mmdetection3d.git /home/appuser/mmdetection3d
WORKDIR /home/appuser/mmdetection3d
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/runtime.txt
RUN pip install --no-cache-dir -e .

RUN pip install numpy==1.19.5
RUN pip install pycocotools==2.0.1
RUN pip install xtcocotools==1.9

RUN pip install timm
