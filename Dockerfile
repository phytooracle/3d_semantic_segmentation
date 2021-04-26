FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update
RUN cp sample.env .env
# Changed python version (may need to add -dev back) also added git
RUN apt-get install -y python3.6 \
                       python3-pip \
                       wget \
                       git \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       libsm6 \
                       libxext6\
		       ffmpeg\
		       libgl1-mesa-glx\
		       vim\
		       nvidia-cuda-toolkit

RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install -r updated_requirements.txt
RUN pip3 install -r ./Open3D-ML/requirements-tensorflow.txt
RUN apt-get update
RUN ldconfig
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
# Commented out for build testing
#ENTRYPOINT [ "/usr/bin/python3", "src/models/train_pred_vis.py" ]
