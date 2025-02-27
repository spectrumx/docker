FROM python:3.13-alpine AS builder
WORKDIR /app
RUN apk update && apk add --no-cache \
    libhdf5-dev \
    libzmq3-dev \
    git \

RUN pip install --no-cache-dir digital_rf spectrumx paho-mqtt