#!/usr/bin/env python3

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

import dataclasses
import logging
import os
import pathlib
import signal
import sys
import tempfile
import typing
import subprocess
import fractions

import holoscan
import jsonargparse
import json
import base64
import cupy as cp
import numpy as np
import paho.mqtt.client as mqtt
import matplotlib as mpl
from holohub import basic_network, rf_array
from holohub.rf_array.digital_metadata import DigitalMetadataSink
from holohub.rf_array.params import (
    DigitalRFSinkParams,
    NetConnectorBasicParams,
    ResamplePolyParams,
    RotatorScheduledParams,
    SubchannelSelectParams,
    add_chunk_kwargs,
)
from jsonargparse.typing import NonNegativeInt, PositiveInt
from spectrogram import (
    Spectrogram,
    SpectrogramOutput,
    SpectrogramOutputParams,
    SpectrogramParams,
)

mpl.use("agg")

# set up Holoscan logger
env_log_level = os.environ.get("HOLOSCAN_LOG_LEVEL", "WARN").upper()
log_level_map = {
    "OFF": "NOTSET",
    "CRITICAL": "CRITICAL",
    "ERROR": "ERROR",
    "WARN": "WARNING",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
    "TRACE": "DEBUG",
}
log_level = log_level_map[env_log_level]
holoscan_handler = logging.StreamHandler()
holoscan_handler.setFormatter(
    logging.Formatter(
        fmt="[{levelname}] [{filename}:{lineno}] {message}",
        style="{",
    )
)
holoscan_handler.setLevel(log_level)
holoscan_logger = logging.getLogger("holoscan")
holoscan_logger.setLevel(log_level)
holoscan_logger.propagate = False
holoscan_logger.addHandler(holoscan_handler)

jsonargparse.set_parsing_settings(docstring_parse_attribute_docstrings=True)

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")


@dataclasses.dataclass
class SchedulerParams:
    """Event-based scheduler parameters"""

    worker_thread_number: PositiveInt = 8
    """Number of worker threads"""
    stop_on_deadlock: bool = True
    """Whether the application will terminate if a deadlock occurs"""
    stop_on_deadlock_timeout: int = 500
    """Time (in ms) to wait before determining that a deadlock has occurred"""


@dataclasses.dataclass
class PipelineParams:
    """Pipeline configuration parameters"""

    selector: bool = False
    "Enable / disable subchannel selector"
    converter: bool = True
    "Enable / disable complex int to float converter"
    rotator: bool = False
    "Enable / disable frequency rotator"
    resampler0: bool = True
    "Enable / disable the first stage resampler"
    resampler1: bool = False
    "Enable / disable the second stage resampler"
    resampler2: bool = True
    "Enable / disable the third stage resampler"
    int_converter: bool = True
    "Enable / disable complex float to int converter"
    digital_rf: bool = True
    "Enable / disable writing output to Digital RF"
    metadata: bool = True
    "Enable / disable writing inherent and user-supplied Digital RF metadata"
    spectrogram: bool = True
    "Enable / disable spectrogram processing and output"
    fft: bool = True
    "Enable / disable FFT processing and mqtt output"


@dataclasses.dataclass
class BasicNetworkOperatorParams:
    """Basic network operator parameters"""

    ip_addr: str = "0.0.0.0"
    """IP address to bind to"""
    dst_port: NonNegativeInt = 60133
    "UDP or TCP port to listen on"
    l4_proto: str = "udp"
    "Layer 4 protocol (udp or tcp)"
    batch_size: PositiveInt = 6250
    "Number of packets in batch"
    max_payload_size: PositiveInt = 8256
    "Maximum payload size expected from sender"


@dataclasses.dataclass
class AdvancedNetworkOperatorParams:
    """Advanced network operator parameters"""

    cfg: typing.Optional[dict] = None


def build_config_parser():
    parser = jsonargparse.ArgumentParser(
        prog="sdr_mep_recorder",
        description="Process and record RF data for the SpectrumX Mobile Experiment Platform (MEP)",
        default_env=True,
    )
    parser.add_argument("--config", action="config")
    parser.add_argument("--scheduler", type=SchedulerParams)
    parser.add_argument("--pipeline", type=PipelineParams)
    parser.add_argument("--basic_network", type=BasicNetworkOperatorParams)
    parser.add_argument("--advanced_network", type=AdvancedNetworkOperatorParams)
    parser.add_argument(
        "--packet",
        type=NetConnectorBasicParams,
        default=NetConnectorBasicParams(spoof_header=True),
    )
    parser.add_argument("--selector", type=SubchannelSelectParams)
    parser.add_argument("--rotator", type=RotatorScheduledParams)
    parser.add_argument(
        "--resampler0",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=1,
            down=8,
            outrate_cutoff=1.0,
            # transition_width: 2 * (cutoff - 1 / remaining_dec)
            #                   2 * (1.0 - 1 / 8) = 1.75
            outrate_transition_width=1.75,
            attenuation_db=105,
        ),
    )
    parser.add_argument(
        "--resampler1",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=5,
            down=16,
            outrate_cutoff=1.0,
            outrate_transition_width=0.2,
            attenuation_db=99.65,
        ),
    )
    parser.add_argument(
        "--resampler2",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=1,
            down=8,
            outrate_cutoff=1.0,
            outrate_transition_width=0.2,
            attenuation_db=99.475,
        ),
    )
    parser.add_argument("--drf_sink", type=DigitalRFSinkParams)
    parser.add_argument(
        "--metadata", type=typing.Optional[dict[str, typing.Any]], default=None
    )
    parser.add_argument("--spectrogram", type=SpectrogramParams)
    parser.add_argument("--spectrogram_output", type=SpectrogramOutputParams)

    return parser

class FFT(holoscan.core.Operator):
    """Simple FFT operator that computes an FFT across time for each subchannel

    Inputs:
        rf_in: RFArray-like tensor (chunk_size x num_subchannels)
    Outputs:
        fft_out: dict with 'fft' (host array) and 'metadata' (RFMetadata)
    """

    def __init__(self, fragment, *args, chunk_size, num_subchannels, nfft=4096, **kwargs):
        self.chunk_size = chunk_size
        self.num_subchannels = num_subchannels
        # fixed FFT size (defaults to 4096 bins)
        self.nfft = nfft
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.rf_array.FFT")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in")
        spec.output("fft_out").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=1,
            policy=0,  # pop
        )

    def initialize(self):
        # warm up
        _ = cp.fft.fft(cp.ones((self.nfft, self.num_subchannels), dtype="complex64"))

    def compute(self, op_input, op_output, context):
        rf_arr = op_input.receive("rf_in")
        stream_ptr = op_input.receive_cuda_stream("rf_in", allocate=True)
        while rf_arr is not None:
            rf_metadata = rf_arr.metadata
            with cp.cuda.ExternalStream(stream_ptr):
                rf_data = cp.from_dlpack(rf_arr.data)
                # compute FFT along time axis (axis 0) with fixed nfft
                fft_gpu = cp.fft.fftshift(cp.fft.fft(rf_data, n=self.nfft, axis=0), axes=0)
                # move to host pinned memory
                fft_host = cp.asnumpy(fft_gpu)
                out_message = {"fft": holoscan.as_tensor(fft_host), "metadata": rf_metadata}
                op_output.emit(out_message, "fft_out")

            rf_arr = op_input.receive("rf_in")


@dataclasses.dataclass
class FFTOutputParams:
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883


class FFTOutput(holoscan.core.Operator):
    def __init__(self, fragment, *args, nfft, num_subchannels, mqtt_host="localhost", mqtt_port=1883, **kwargs):
        self.nfft = nfft
        self.num_subchannels = num_subchannels
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.rf_array.FFTOutput")
        service_name = "recorder_fft"
        self.mqtt_client = mqtt.Client(client_id=service_name)
        self.mqtt_client.will_set(service_name + "/status", payload='{"state": "offline"}', qos=0, retain=True)
        try:
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
        except Exception:
            self.logger.exception("Failed to connect to MQTT broker")

        # try to discover MAC address (same approach as SpectrogramOutput)
        self.mac_address = None
        try:
            result = subprocess.run(["ifconfig", "en0"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if "ether" in line:
                    self.mac_address = line.split()[1].replace(':','')
                    break
        except Exception:
            # if this fails, leave mac_address as None
            self.logger.debug("Could not determine MAC address for FFTOutput")

        # placeholders for metadata populated on first message
        self.stored_metadata = None
        self.sample_rate_frac = None
        self.spec_freq_idx = None

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("fft_in").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=4,
            policy=0,
        )

    def compute(self, op_input, op_output, context):
        fft_msg = op_input.receive("fft_in")
        # synchronize stream to ensure host memory valid
        stream_ptr = op_input.receive_cuda_stream("fft_in", allocate=True)
        stream = cp.cuda.ExternalStream(stream_ptr)
        stream.synchronize()
        while fft_msg is not None:
            fft_arr = np.from_dlpack(fft_msg["fft"])
            rf_metadata = fft_msg["metadata"]
            # initialize metadata-derived fields on first message
            if self.stored_metadata is None:
                self.stored_metadata = rf_metadata
                self.sample_rate_frac = fractions.Fraction(
                    self.stored_metadata.sample_rate_numerator,
                    self.stored_metadata.sample_rate_denominator,
                )
                # frequency indices for FFT bins (MHz offset will be added when publishing)
                self.spec_freq_idx = np.fft.fftshift(
                    np.fft.fftfreq(self.nfft, 1 / self.sample_rate_frac)
                )
            # take magnitude
            mag = np.abs(fft_arr).astype(np.float32)
            # build payload with expanded metadata (mirrors SpectrogramOutput fields)
            payload = {
                "data": base64.b64encode(mag[:, 0].copy(order="C")).decode("utf-8"),
                "mac_address": self.mac_address,
                "type": "float32",
                "short_name": "MEP",
                "software_version": "v0.10b30",
                "latitude": 41.699584,
                "longitude": -86.237237,
                "altitude": 2,
                "batch": 0,
                "sample_rate": float(self.sample_rate_frac),
                "center_frequency": float(self.stored_metadata.center_freq),
                "timestamp": None,
                "gain": 1,
                "metadata": {
                    "data_type": "periodogram",
                    "fmin": int(min(self.spec_freq_idx)),
                    "fmax": int(max(self.spec_freq_idx)),
                    "nfft": self.nfft,
                    "xcount": self.nfft,
                    "gps_lock": False,
                    "scan_time": 0.0,
                },
            }
            try:
                self.mqtt_client.publish("radiohound/clients/data/"+self.mac_address, payload=json.dumps(payload))
            except Exception:
                self.logger.exception("Failed to publish FFT payload")

            fft_msg = op_input.receive("fft_in")


class App(holoscan.core.Application):
    def compose(self):
        cuda_stream_pool = holoscan.resources.CudaStreamPool(
            self,
            name="stream_pool",
            stream_flags=1,  # cudaStreamNonBlocking
            stream_priority=0,
            reserved_size=1,
            max_size=0,
        )

        basic_net_rx = basic_network.BasicNetworkOpRx(
            self,
            name="basic_network_rx",
            **self.kwargs("basic_network"),
        )
        basic_net_rx.spec.outputs["burst_out"].connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=self.kwargs("packet").get("batch_capacity", 4),
            policy=0,  # pop
        )

        packet_kwargs = self.kwargs("packet")
        net_connector_rx = rf_array.NetConnectorBasic(
            self,
            cuda_stream_pool,
            name="net_connector_rx",
            **packet_kwargs,
        )
        net_connector_rx.spec.inputs["burst_in"].connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=packet_kwargs.get("batch_capacity", 4),
            policy=0,  # pop
        )
        net_connector_rx.spec.outputs["rf_out"].connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=packet_kwargs.get("buffer_size", 4),
            policy=0,  # pop
        )
        self.add_flow(basic_net_rx, net_connector_rx, {("burst_out", "burst_in")})

        last_chunk_shape = (
            self.kwargs("packet")["num_samples"],
            self.kwargs("packet")["num_subchannels"],
        )
        last_op = net_connector_rx

        if self.kwargs("pipeline")["selector"]:
            selector = rf_array.SubchannelSelect_sc16(
                self, cuda_stream_pool, name="selector", **self.kwargs("selector")
            )
            self.add_flow(last_op, selector)
            last_op = selector
            last_chunk_shape = (
                last_chunk_shape[0],
                len(self.kwargs("selector")["subchannel_idx"]),
            )

        if self.kwargs("pipeline")["converter"]:
            converter = rf_array.TypeConversionComplexIntToFloat(
                self,
                cuda_stream_pool,
                name="converter",
            )
            self.add_flow(last_op, converter)
            last_op = converter

            if self.kwargs("pipeline")["rotator"]:
                rotator = rf_array.RotatorScheduled(
                    self, cuda_stream_pool, name="rotator", **self.kwargs("rotator")
                )
                self.add_flow(last_op, rotator)
                last_op = rotator

            if self.kwargs("pipeline")["resampler0"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler0")
                )
                resampler0 = rf_array.ResamplePoly(
                    self, cuda_stream_pool, name="resampler0", **resample_kwargs
                )
                self.add_flow(last_op, resampler0)
                last_op = resampler0
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler1"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler1")
                )
                resampler1 = rf_array.ResamplePoly(
                    self, cuda_stream_pool, name="resampler1", **resample_kwargs
                )
                self.add_flow(last_op, resampler1)
                last_op = resampler1
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler2"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler2")
                )
                resampler2 = rf_array.ResamplePoly(
                    self, cuda_stream_pool, name="resampler2", **resample_kwargs
                )
                self.add_flow(last_op, resampler2)
                last_op = resampler2
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["spectrogram"]:
                spectrogram = Spectrogram(
                    self,
                    cuda_stream_pool,
                    name="spectrogram",
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("spectrogram")),
                )
                # Queue policy is currently set by specifying a connector in setup()
                # # drop old messages rather than get backed up by slow
                # # downstream operators
                # spectrogram.queue_policy(
                #     port_name="spec_out",
                #     port_type=holoscan.core.IOSpec.IOType.OUTPUT,
                #     policy=holoscan.core.IOSpec.QueuePolicy.POP,
                # )
                self.add_flow(last_op, spectrogram)

                spectrogram_output = SpectrogramOutput(
                    self,
                    ## CudaStreamCondition doesn't work with a message queue size
                    ## larger than 1, so get by without it for now
                    # holoscan.conditions.MessageAvailableCondition(
                    #     self,
                    #     receiver="spec_in",
                    #     name="spectrogram_output_message_available",
                    # ),
                    # holoscan.conditions.CudaStreamCondition(
                    #     self, receiver="spec_in", name="spectrogram_output_stream_sync"
                    # ),
                    # # no downstream condition, and we don't want one
                    cuda_stream_pool,
                    name="spectrogram_output",
                    data_outdir=(
                        f"{DRF_RECORDING_DIR}/{self.kwargs('drf_sink')['channel_dir']}"
                        "_spectrogram"
                    ),
                    nfft=spectrogram.nfft,
                    spec_sample_cadence=spectrogram.spec_sample_cadence,
                    num_subchannels=spectrogram.num_subchannels,
                    **self.kwargs("spectrogram_output"),
                )
                self.add_flow(spectrogram, spectrogram_output)
                # (FFT wiring moved below so it can be enabled independently)

            if self.kwargs("pipeline")["int_converter"]:
                int_converter = rf_array.TypeConversionComplexFloatToInt(
                    self,
                    cuda_stream_pool,
                    name="int_converter",
                )
                self.add_flow(last_op, int_converter)
                last_op = int_converter

            # independent FFT wiring: place after current last_op so it can consume
            # the converted/resampled data if requested
            if self.kwargs("pipeline")["fft"]:
                fft_op = FFT(
                    self,
                    cuda_stream_pool,
                    name="fft",
                    chunk_size=last_chunk_shape[0],
                    num_subchannels=last_chunk_shape[1],
                )
                fft_output = FFTOutput(
                    self,
                    name="fft_output",
                    nfft=last_chunk_shape[0],
                    num_subchannels=last_chunk_shape[1],
                    mqtt_host="localhost",
                    mqtt_port=1883,
                )
                self.add_flow(last_op, fft_op)
                self.add_flow(fft_op, fft_output)

        if self.kwargs("pipeline")["digital_rf"]:
            if (
                self.kwargs("pipeline")["converter"]
                and not self.kwargs("pipeline")["int_converter"]
            ):
                drf_sink = rf_array.DigitalRFSink_fc32(
                    self,
                    cuda_stream_pool,
                    name="drf_sink",
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
                )
                drf_sink.spec.inputs["rf_in"].connector(
                    holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
                    capacity=25,
                    policy=0,  # pop
                )
                self.add_flow(last_op, drf_sink)
            else:
                drf_sink = rf_array.DigitalRFSink_sc16(
                    self,
                    cuda_stream_pool,
                    name="drf_sink",
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
                )
                drf_sink.spec.inputs["rf_in"].connector(
                    holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
                    capacity=25,
                    policy=0,  # pop
                )
                self.add_flow(last_op, drf_sink)

            if self.kwargs("pipeline")["metadata"]:
                dmd_sink = DigitalMetadataSink(
                    self,
                    name="dmd_sink",
                    metadata_dir=f"{self.kwargs('drf_sink')['channel_dir']}/metadata",
                    subdir_cadence_secs=self.kwargs("drf_sink")["subdir_cadence_secs"],
                    file_cadence_secs=self.kwargs("drf_sink")["file_cadence_millisecs"]
                    // 1000,
                    uuid=self.kwargs("drf_sink")["uuid"],
                    filename_prefix="metadata",
                    metadata=self.kwargs("metadata"),
                )
                dmd_sink.spec.inputs["rf_in"].connector(
                    holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
                    capacity=25,
                    policy=0,  # pop
                )
                self.add_flow(last_op, dmd_sink)


def main():
    parser = build_config_parser()
    cfg = parser.parse_args()

    logger = logging.getLogger("holoscan.sdr_mep_recorder")

    # We have a parsed configuration (using jsonargparse), but the holoscan app wants
    # to read all of its configuration parameters from a YAML file, so we write out
    # the configuration to a file in the temporary directory and feed it that
    config_path = pathlib.Path(tempfile.gettempdir()) / "sdr_mep_recorder_config.yaml"
    logger.debug(f"Writing temporary config file to {config_path}")
    parser.save(cfg, config_path, format="yaml", overwrite=True)

    app = App([sys.executable, sys.argv[0]])
    app.config(str(config_path))

    scheduler = holoscan.schedulers.EventBasedScheduler(
        app,
        name="event-based-scheduler",
        **app.kwargs("scheduler"),
    )
    app.scheduler(scheduler)

    def sigterm_handler(signal, frame):
        logger.info("Received SIGTERM, cleaning up")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        app.run()
    except KeyboardInterrupt:
        # catch keyboard interrupt and simply exit
        logger.info("Done")
        sys.stdout.flush()
        # Holoscan graph execution framework handles all cleanup
        # so we just need to exit immediately without further Python cleanup
        # (which would result in a segfault from double free)
        os._exit(0)
    except SystemExit as e:
        # Holoscan graph execution framework handles all cleanup
        # so we just need to exit immediately without further Python cleanup
        # (which would result in a segfault from double free)
        os._exit(e.code)


if __name__ == "__main__":
    main()
