# SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ghcr.io/ryanvolz/holohub/rf-array:v1.2

LABEL org.opencontainers.image.description="Holoscan MEP recorder"

#
# Install any utils needed for execution
#
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    iftop \
    iproute2 \
    iputils-ping \
    net-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#
# Install Python dependencies not covered by deb packages
#
RUN python3 -m pip install --no-cache-dir aiomqtt anyio digital-rf jsonargparse[ruyaml,signatures] matplotlib exceptiongroup

#
# Set up environment variables
#
ENV HOLOSCAN_EXECUTOR_LOG_LEVEL=WARN
ENV HOLOSCAN_LOG_LEVEL=INFO
ENV HOLOSCAN_LOG_FORMAT=DEFAULT

#
# Typical use would involve mounting a volume to /app containing
# these scripts, but copy them into the image anyway to allow
# use of these static files without the mounting
#
COPY --chmod=775 recorder_service.py /app/recorder_service.py
COPY --chmod=777 sdr_mep_recorder.py /app/sdr_mep_recorder.py
COPY --chmod=775 spectrogram.py /app/spectrogram.py

ENV HOME=/ramdisk
WORKDIR /ramdisk
ENTRYPOINT ["python3", "/app/recorder_service.py"]
