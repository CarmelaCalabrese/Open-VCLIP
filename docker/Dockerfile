ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt install -y git nano vim wget

### Installing DualCoOp
RUN git clone https://github.com/wengzejia1/Open-VCLIP.git /Open-VCLIP
WORKDIR /Open-VCLIP

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

#Environment
RUN conda create -n ivnet python=3.8.13 
#RUN conda activate ivnet
SHELL ["conda","run","-n","ivnet","/bin/bash","-c"]

#Install pytorch for cuda 
RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

### Installing dependencies
RUN pip install scipy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install ftfy
RUN pip install regex

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install simplejson
RUN conda install av -c conda-forge
RUN pip install -U iopath
RUN pip install psutil
RUN pip install opencv-python
RUN pip install tensorboard
RUN git clone https://github.com/facebookresearch/pytorchvideo.git
WORKDIR pytorchvideo
RUN pip install -e .
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo
RUN pip install 'git+https://github.com/facebookresearch/fairscale'

RUN export PYTHONPATH=/path/to/OpenVCLIP/slowfast:$PYTHONPATH
WORKDIR /OpenVCLIP
#RUN python setup.py build develop
