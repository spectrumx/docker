FROM python:3.13-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libzmq3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*




FROM builder AS worker
WORKDIR /app
RUN pip install --no-cache-dir paho.mqtt digital_rf
COPY . /app
ENTRYPOINT ["python", "drf_read.py"]
