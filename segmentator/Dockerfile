FROM tensorflow/tensorflow:1.0.0

MAINTAINER alexander.sologub@altoros.com

RUN apt-get update -y && \
    apt-get install -y git wget && \
    git clone http://github.com/russell91/TensorBox && \
    cd TensorBox && \
    cd utils && \
    make && \
    pip install opencv-python

RUN pip install flask flask-cors dotmap

ADD model model
ADD server.py server.py

EXPOSE 80
