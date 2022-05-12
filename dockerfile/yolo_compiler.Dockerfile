FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# For xorg graphics
ENV QT_X11_NO_MITSHM 1

# Contact info
LABEL maintainer="Raul Castilla Arquillo <raulcastar@uma.es>"
LABEL authors="Raul Castilla Arquillo <raulcastar@uma.es>,\
Carlos Jesus Perez del Pulgar Mancebo <carlosperez@uma.es>"
LABEL organization="Space Robotics Laboratory (University of Malaga)"
LABEL url="https://www.uma.es/robotics-and-mechatronics/info/107542/robotica-espacial/"
LABEL version="1.0"
LABEL license="MIT License"
LABEL description=""
LABEL created=""

# Visual and dev packages
RUN apt-get update && \
      apt-get -y install python3  \
      libopencv-dev python3-opencv \
      libcanberra-gtk-module libcanberra-gtk3-module \
      xorg nano vim curl python3-gi-cairo python3-pip \
      && rm -rf /var/lib/apt/lists/*

# Packages needed to convert yolo network
RUN pip install keras==2.4.3
RUN pip install Keras-Preprocessing==1.1.2
RUN pip install tensorflow==2.3.1
RUN pip install tensorflow-gpu==2.3.1

# Package to plot DEM
RUN pip install matplotlib
RUN pip install pandas

RUN pip install natsort

# Packages for google coral
# more info: https://coral.ai/docs/accelerator/get-started/
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
      tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && \
      apt-get -y install libedgetpu1-std  \
      edgetpu-compiler \
      python3-tflite-runtime \
      && rm -rf /var/lib/apt/lists/*
