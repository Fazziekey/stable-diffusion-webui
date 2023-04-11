
FROM nvcr.io/nvidia/tensorrt:22.10-py3

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git && apt-get install ffmpeg libsm6 libxext6 libgl1  -y

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH

RUN HTTP_PROXY=http://172.17.0.1:7890 HTTPS_PROXY=http://172.17.0.1:7890 curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

COPY requirements_versions_new.txt /tmp/requirements.txt

RUN export HTTP_PROXY=http://172.17.0.1:7890 HTTPS_PROXY=http://172.17.0.1:7890 \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install nvidia-pyindex xformers==0.0.16rc425\
    && python3 -m pip install --upgrade tensorrt \
    && python3 -m pip install colored cuda-python diffusers==0.7.2 ftfy matplotlib nvtx onnx==1.12.0 \
    && python3 -m pip install onnx-graphsurgeon==0.3.25 onnxruntime==1.13.1 polygraphy==0.43.1 scipy numpy==1.23.0 --extra-index-url https://pypi.ngc.nvidia.com \
    && python3 -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 \
    && python3 -m pip install transformers==4.24.0 \
    && python3 -m pip install gfpgan open_clip_torch\
    && python3 -m pip install git+https://github.com/openai/CLIP.git \
    && python3 -m pip install -r /tmp/requirements.txt

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && python3 setup.py install \
    && cd ..




