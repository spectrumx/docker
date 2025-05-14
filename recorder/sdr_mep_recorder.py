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

import holoscan
import jsonargparse
from jsonargparse.typing import NonNegativeInt, PositiveInt

from holohub import basic_network, rf_array
from holohub.rf_array.params import (
    DigitalRFSinkParams,
    NetConnectorBasicParams,
    ResamplePolyParams,
    RotatorScheduledParams,
    SubchannelSelectParams,
    add_chunk_kwargs,
)

logger = logging.getLogger("sdr_mep_recorder.py")

jsonargparse.set_parsing_settings(docstring_parse_attribute_docstrings=True)


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
        "--packet", type=NetConnectorBasicParams, default=NetConnectorBasicParams(spoof_header=True)
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

    return parser


class App(holoscan.core.Application):
    def compose(self):
        basic_net_rx = basic_network.BasicNetworkOpRx(
            self, name="basic_network_rx", **self.kwargs("basic_network")
        )

        net_connector_rx = rf_array.NetConnectorBasic(
            self, name="net_connector_rx", **self.kwargs("packet")
        )
        self.add_flow(basic_net_rx, net_connector_rx, {("burst_out", "burst_in")})

        last_chunk_shape = (
            self.kwargs("packet")["num_samples"],
            self.kwargs("packet")["num_subchannels"],
        )
        last_op = net_connector_rx

        if self.kwargs("pipeline")["selector"]:
            selector = rf_array.SubchannelSelect_sc16(self, name="selector", **self.kwargs("selector"))
            self.add_flow(last_op, selector)
            last_op = selector
            last_chunk_shape = (
                last_chunk_shape[0],
                len(self.kwargs("selector")["subchannel_idx"]),
            )

        if self.kwargs("pipeline")["converter"]:
            converter = rf_array.TypeConversionComplexIntToFloat(
                self,
                name="converter",
            )
            self.add_flow(last_op, converter)
            last_op = converter

            if self.kwargs("pipeline")["rotator"]:
                rotator = rf_array.RotatorScheduled(self, name="rotator", **self.kwargs("rotator"))
                self.add_flow(last_op, rotator)
                last_op = rotator

            if self.kwargs("pipeline")["resampler0"]:
                resample_kwargs = add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler0"))
                resampler0 = rf_array.ResamplePoly(self, name="resampler0", **resample_kwargs)
                self.add_flow(last_op, resampler0)
                last_op = resampler0
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler1"]:
                resample_kwargs = add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler1"))
                resampler1 = rf_array.ResamplePoly(self, name="resampler1", **resample_kwargs)
                self.add_flow(last_op, resampler1)
                last_op = resampler1
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler2"]:
                resample_kwargs = add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler2"))
                resampler2 = rf_array.ResamplePoly(self, name="resampler2", **resample_kwargs)
                self.add_flow(last_op, resampler2)
                last_op = resampler2
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            drf_sink = rf_array.DigitalRFSink_fc32(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)

        else:
            drf_sink = rf_array.DigitalRFSink_sc16(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)


def main():
    parser = build_config_parser()
    cfg = parser.parse_args()

    env_log_level = os.environ.get("HOLOSCAN_LOG_LEVEL", "WARN").upper()
    if env_log_level == "TRACE":
        # TRACE exists for holoscan, but not in Python, so substitute with DEBUG
        env_log_level = "DEBUG"
    logging.basicConfig(level=env_log_level)

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
        pass
    finally:
        logger.info("Done")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
