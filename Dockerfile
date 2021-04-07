  FROM ubuntu:18.04


# RUN cp -r . /opt


WORKDIR /opt
COPY . /opt


USER root

RUN apt-get update
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       libsm6 \
                       libxext6

RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -r {wherever the tf requirements file is}
RUN ldconfig
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

ENTRYPOINT [ "/usr/bin/python3", "/opt/{train_test_vis}" ]
