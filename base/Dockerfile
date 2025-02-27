FROM python:3.13-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libzmq3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir digital_rf spectrumx paho-mqtt