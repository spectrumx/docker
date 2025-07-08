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
import fractions
import logging
import os
import pathlib
import signal
import sys
import tempfile
import traceback
import typing

import cupy as cp
import cupyx
import cupyx.scipy.signal as cpss
import digital_rf as drf
import holoscan
import jsonargparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

mpl.use("agg")

logger = logging.getLogger("sdr_mep_recorder.py")

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
    spectrogram: bool = True
    "Enable / disable spectrogram processing and output"


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
class SpectrogramParams:
    """Spectrogram parameters"""

    window: str = "hann"
    """Window function to apply before taking FFT"""
    nperseg: int = 1024
    """Length of each segment of samples on which to calculate a spectrum"""
    noverlap: typing.Optional[int] = None
    """Number of samples to overlap between segments. If None, `noverlap = nperseg // 2`"""
    nfft: typing.Optional[int] = None
    """Length of FFT used per segment. If None, `nfft = nperseg`"""
    detrend: typing.Union[
        typing.Literal["linear"], typing.Literal["constant"], typing.Literal[False]
    ] = False
    """Specifies how to detrend each segment. ["constant", "linear", or False]"""
    reduce_op: typing.Union[
        typing.Literal["max"], typing.Literal["median"], typing.Literal["mean"]
    ] = "max"
    """Operation to use to reduce segment spectra to one result per chunk. ["max", "median", or "mean"]"""
    num_spectra_per_chunk: int = 1
    """Number of spectra samples to calculate per chunk of data. Must evenly divide `chunk_size`."""
    num_chunks_per_output: int = 300
    """Number of chunks to combine in a single output, either a data sample or plot"""
    figsize: tuple[float, float] = (6.4, 4.8)
    """Figure size in inches given as a tuple of (width, height)"""
    dpi: int = 150
    """Figure dots per inch"""
    col_wrap: int = 1
    """Number of columns of spectrograms to use in the figure, wrapping to new rows"""
    cmap: str = "viridis"
    """Colormap"""
    snr_db_min: float = -10
    """Spectrogram color scale minimum, given as SNR in decibels"""
    snr_db_max: float = 40
    """Spectrogram color scale maximum, given as SNR in decibels"""
    plot_outdir: os.PathLike = f"{DRF_RECORDING_DIR}/spectrograms"
    """Directory for writing spectrogram plots"""


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

    return parser


class Spectrogram(holoscan.core.Operator):
    chunk_size: int
    num_subchannels: int
    data_outdir = os.PathLike
    window: str
    nperseg: int
    noverlap: typing.Optional[int]
    nfft: typing.Optional[int]
    detrend: typing.Union[
        typing.Literal["linear"], typing.Literal["constant"], typing.Literal[False]
    ]
    reduce_op: typing.Union[
        typing.Literal["max"], typing.Literal["median"], typing.Literal["mean"]
    ]
    num_spectra_per_chunk: int
    num_chunks_per_output: int
    figsize: tuple[float, float]
    dpi: int
    col_wrap: int
    cmap: str
    snr_db_min: float
    snr_db_max: float
    plot_outdir: os.PathLike

    def __init__(
        self,
        fragment,
        chunk_size,
        num_subchannels,
        data_outdir,
        *args,
        window="hann",
        nperseg=1024,
        noverlap=None,
        nfft=None,
        detrend=False,
        reduce_op="max",
        num_spectra_per_chunk=1,
        num_chunks_per_output=300,
        figsize=(6.4, 4.8),
        dpi=150,
        col_wrap=1,
        cmap="viridis",
        snr_db_min=-10,
        snr_db_max=40,
        plot_outdir=f"{DRF_RECORDING_DIR}/spectrograms",
        **kwargs,
    ):
        """Operator that computes spectrograms from RF data.

        **==Named Inputs==**

            rf_in : RFArray
                RFArray, including metadata.

        Parameters
        ----------
        fragment : Fragment
            The fragment that the operator belongs to
        chunk_size: int
            Number of samples in an RFArray chunk of data
        num_subchannels: int
            Number of subchannels contained in the RFArray data
        data_outdir: os.PathLike
            Directory for writing spectrogram data
        window: str
            Window function to apply before taking FFT
        nperseg: int
            Length of each segment of samples on which to calculate a spectrum
        noverlap: int or None
            Number of samples to overlap between segments. If None, `noverlap = nperseg // 2`
        nfft: int or None
            Length of FFT used per segment. If None, `nfft = nperseg`
        detrend: "constant", "linear", or False
            Specifies how to detrend each segment.
        reduce_op: "max", "median", or "mean"
            Operation to use to reduce multiple segment spectra to one.
        num_spectra_per_chunk: int
            Number of spectra samples to calculate per chunk of data.
            Must evenly divide `chunk_size`.
        num_chunks_per_output: int
            Number of chunks to combine in a single output, either a data sample or plot
        figsize: tuple[float, float]
            Figure size in inches given as a tuple of (width, height)
        dpi: int
            Figure dots per inch
        col_wrap: int
            Number of columns of spectrograms to use in the figure, wrapping to new rows
        cmap: str
            Colormap
        snr_db_min: float
            Spectrogram color scale minimum, given as SNR in decibels
        snr_db_max: float
            Spectrogram color scale maximum, given as SNR in decibels
        plot_outdir: os.PathLike
            Directory for writing spectrogram plots
        """
        self.chunk_size = chunk_size
        self.num_subchannels = num_subchannels
        self.data_outdir = pathlib.Path(data_outdir).resolve()
        self.window = window
        self.nperseg = nperseg
        if noverlap is None:
            noverlap = nperseg // 2
        self.noverlap = noverlap
        if nfft is None:
            nfft = nperseg
        self.nfft = nfft
        self.detrend = detrend
        if reduce_op == "max":
            self.reduce_op = cp.amax
        elif reduce_op == "median":
            self.reduce_op = cp.median
        else:
            self.reduce_op = cp.mean
        if (self.chunk_size % num_spectra_per_chunk) != 0:
            msg = (
                f"Number of spectra per chunk ({num_spectra_per_chunk}) must evenly"
                f" divide the chunk size ({chunk_size})."
            )
            raise ValueError(msg)
        self.num_spectra_per_chunk = num_spectra_per_chunk
        self.num_chunks_per_output = num_chunks_per_output
        self.num_spectra_per_output = (
            self.num_spectra_per_chunk * self.num_chunks_per_output
        )
        self.figsize = figsize
        self.dpi = dpi
        self.col_wrap = col_wrap
        self.cmap = cmap
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.plot_outdir = pathlib.Path(plot_outdir).resolve()

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("Spectrogram")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=100,
        )

    def create_spec_figure(self):
        ncols = min(self.col_wrap, self.num_subchannels)
        nrows = int(np.ceil(self.num_subchannels / self.col_wrap))
        fig, axs = plt.subplots(
            nrows,
            ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="compressed",
            figsize=self.figsize,
            dpi=self.dpi,
        )
        self.norm = mpl.colors.Normalize(vmin=self.snr_db_min, vmax=self.snr_db_max)
        xlocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=7)
        xformatter = mpl.dates.ConciseDateFormatter(xlocator)
        axs_1d = []
        imgs = []
        for sch in range(self.num_subchannels):
            row_idx = sch // self.col_wrap
            col_idx = sch % self.col_wrap
            ax = axs[row_idx, col_idx]
            img = ax.imshow(
                self.spec_host_data[:, sch, :],
                cmap=self.cmap,
                norm=self.norm,
                aspect="auto",
                interpolation="none",
                origin="lower",
            )
            cb = fig.colorbar(img, ax=ax, fraction=0.05, pad=0.01)
            cb.set_label("Relative power [dB] (m/s)")
            ax.set_ylabel("Frequency [MHz]")
            if self.num_subchannels > 1:
                ax.set_title(f"Subchannel {sch}")
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(xformatter)
            imgs.append(img)
            axs_1d.append(ax)
        axs_1d[-1].set_xlabel("Time (UTC)")
        fig.suptitle("Spectrogram")
        fig.autofmt_xdate(rotation=0, ha="center")

        self.fig = fig
        self.axs = axs_1d
        self.imgs = imgs

    def initialize(self):
        self.logger.debug("Initializing spectrogram operator")
        self.data_outdir.mkdir(parents=True, exist_ok=True)
        self.spec_host_data = cupyx.zeros_pinned(
            (self.nfft, self.num_subchannels, self.num_spectra_per_output),
            dtype=np.float32,
            order="F",
        )
        self.fill_data = np.full(
            (self.nfft, self.num_subchannels, self.num_spectra_per_output),
            np.nan,
            dtype=np.float32,
            order="F",
        )
        self.spec_host_data[...] = self.fill_data
        self.last_written_sample_idx = -1
        self.last_seen_sample_idx = -1
        self.create_spec_figure()
        self.prior_metadata = None
        self.freq_idx = None
        self.dmd_writer = None
        self.chunk_rate_frac = None

    def set_metadata(self, rf_metadata):
        self.prior_metadata = rf_metadata
        self.freq_idx = np.fft.fftshift(
            np.fft.fftfreq(
                self.nfft,
                rf_metadata.sample_rate_denominator / rf_metadata.sample_rate_numerator,
            )
        )
        self.spectra_rate_frac = fractions.Fraction(
            self.prior_metadata.sample_rate_numerator * self.num_spectra_per_chunk,
            self.prior_metadata.sample_rate_denominator * self.chunk_size,
        )

    def get_chunk_idx(self, sample_idx):
        # prior_metadata.sample_idx marks the start of an output cycle
        return (
            (sample_idx - self.prior_metadata.sample_idx) // self.chunk_size
        ) % self.num_chunks_per_output

    def write_output(self):
        sample_idx = self.last_seen_sample_idx
        if sample_idx <= self.last_written_sample_idx:
            # skip writing because we already wrote this data
            return
        if self.last_written_sample_idx != -1 and (
            sample_idx - self.last_written_sample_idx
        ) > (self.num_chunks_per_output * self.chunk_size):
            # shouldn't be here, trying to write data that spans more than one output batch
            msg = (
                f"Call to write_output() with {sample_idx=} when "
                f"last_written_sample_idx={self.last_written_sample_idx} is more than "
                f"size of output batch ({self.num_chunks_per_output * self.chunk_size})"
            )
            self.logger.warning(msg)
        chunk_idx = self.get_chunk_idx(sample_idx)

        spec_sample_idx = sample_idx - chunk_idx * self.chunk_size
        sr_frac = fractions.Fraction(
            self.prior_metadata.sample_rate_numerator,
            self.prior_metadata.sample_rate_denominator,
        )
        spec_start_dt = drf.util.sample_to_datetime(
            spec_sample_idx,
            sr_frac,
        )
        spectra_arange = np.arange(0, (chunk_idx + 1) * self.num_spectra_per_chunk)
        sample_idx_arr = (
            spec_sample_idx
            + self.chunk_size // self.num_spectra_per_chunk * spectra_arange
        )
        time_idx = np.datetime64(spec_start_dt.replace(tzinfo=None)) + (
            np.timedelta64(int(1000000000 / self.spectra_rate_frac), "ns")
            * spectra_arange
        )
        output_spec_data = self.spec_host_data[
            ...,
            0 : (chunk_idx + 1) * self.num_spectra_per_chunk,
        ]

        self.logger.info(f"Outputting spectrogram for time {spec_start_dt}")

        num_retries = 3
        for retry in range(0, num_retries):
            try:
                self.dmd_writer.write(
                    [spec_sample_idx],
                    [
                        {
                            "spectrogram": output_spec_data.transpose((1, 0, 2)),
                            "freq_idx": self.freq_idx + self.prior_metadata.center_freq,
                            "sample_idx": sample_idx_arr,
                            "center_freq": self.prior_metadata.center_freq,
                        }
                    ],
                )
            except IOError:
                if retry == (num_retries - 1):
                    self.logger.warning(traceback.format_exc())
            else:
                break

        timestr = spec_start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        freqstr = f"{self.prior_metadata.center_freq / 1e6:n}MHz"
        datestr = spec_start_dt.strftime("%Y-%m-%d")

        spec_power_db = 10 * np.log10(
            output_spec_data
            / np.nanpercentile(output_spec_data, 15, axis=(0, 2), keepdims=True)
        )
        delta_t = time_idx[1] - time_idx[0]
        delta_f = self.freq_idx[1] - self.freq_idx[0]
        extent = (
            time_idx[0],
            time_idx[-1] + delta_t,
            (self.prior_metadata.center_freq + self.freq_idx[0] - delta_f / 2) / 1e6,
            (self.prior_metadata.center_freq + self.freq_idx[-1] + delta_f / 2) / 1e6,
        )
        for sch in range(self.num_subchannels):
            self.imgs[sch].set(
                data=spec_power_db[:, sch, :],
                extent=extent,
            )
        self.fig.suptitle(
            f"{self.data_outdir.parent.name}/{self.data_outdir.name} @ {freqstr}"
        )
        self.fig.canvas.draw()

        fname = f"spec_{timestr}_{freqstr}.png"
        outpath = self.plot_outdir / freqstr / datestr / fname
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(outpath)
        latest_spec_path = outpath.parent.parent / "spec_latest.png"
        latest_spec_path.unlink(missing_ok=True)
        os.link(outpath, latest_spec_path)

        # reset spectrogram data for next plot
        self.spec_host_data[...] = self.fill_data
        self.last_written_sample_idx = sample_idx

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        rf_array = op_input.receive("rf_in")
        rf_data = cp.from_dlpack(rf_array.data)
        rf_metadata = rf_array.metadata

        if (rf_metadata.sample_idx - self.last_seen_sample_idx) > (
            self.num_chunks_per_output * self.chunk_size
        ):
            # triggers on first compute call, but write_output returns immediately
            # new data is not in same output batch as unwritten, so write that first
            self.write_output()

        if self.prior_metadata is None:
            self.set_metadata(rf_metadata)
            self.dmd_writer = drf.DigitalMetadataWriter(
                metadata_dir=str(self.data_outdir),
                subdir_cadence_secs=3600,
                file_cadence_secs=1,
                sample_rate_numerator=self.prior_metadata.sample_rate_numerator,
                sample_rate_denominator=self.prior_metadata.sample_rate_denominator,
                file_name="spectrogram",
            )
        if (
            (
                self.prior_metadata.sample_rate_numerator
                != rf_metadata.sample_rate_numerator
            )
            or (
                self.prior_metadata.sample_rate_denominator
                != rf_metadata.sample_rate_denominator
            )
            or (self.prior_metadata.center_freq != rf_metadata.center_freq)
        ):
            # metadata changed, write out existing data and start anew
            self.write_output()
            self.set_metadata(rf_metadata)

        self.last_seen_sample_idx = rf_metadata.sample_idx
        chunk_idx = self.get_chunk_idx(self.last_seen_sample_idx)

        msg = (
            f"Processing spectrogram for chunk with sample_idx {rf_metadata.sample_idx}"
            f" into chunk_idx={chunk_idx}"
        )
        self.logger.debug(msg)

        with cp.cuda.ExternalStream(rf_array.stream):
            for chunk_spectrum_idx, spectrum_chunk in enumerate(
                cp.split(rf_data, self.num_spectra_per_chunk, axis=0)
            ):
                _freqs, sidxs, Zxx = cpss.stft(
                    spectrum_chunk,
                    fs=1,
                    window=self.window,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.nfft,
                    detrend=self.detrend,
                    return_onesided=False,
                    boundary=None,
                    padded=True,
                    axis=0,
                    scaling="spectrum",
                )
                # reduce over time axis
                spec = cp.fft.fftshift(
                    self.reduce_op(Zxx.real**2 + Zxx.imag**2, axis=-1), axes=0
                )

                cp.asnumpy(
                    spec,
                    out=self.spec_host_data[
                        ..., chunk_idx * self.num_spectra_per_chunk + chunk_spectrum_idx
                    ],
                    blocking=False,
                )

        if chunk_idx == (self.num_chunks_per_output - 1):
            self.write_output()

    def stop(self):
        msg = (
            "Stopping spectrogram operator with "
            f"last_seen_sample_idx={self.last_seen_sample_idx}."
        )
        self.logger.info(msg)
        self.write_output()


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
            selector = rf_array.SubchannelSelect_sc16(
                self, name="selector", **self.kwargs("selector")
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
                name="converter",
            )
            self.add_flow(last_op, converter)
            last_op = converter

            if self.kwargs("pipeline")["rotator"]:
                rotator = rf_array.RotatorScheduled(
                    self, name="rotator", **self.kwargs("rotator")
                )
                self.add_flow(last_op, rotator)
                last_op = rotator

            if self.kwargs("pipeline")["resampler0"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler0")
                )
                resampler0 = rf_array.ResamplePoly(
                    self, name="resampler0", **resample_kwargs
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
                    self, name="resampler1", **resample_kwargs
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
                    self, name="resampler2", **resample_kwargs
                )
                self.add_flow(last_op, resampler2)
                last_op = resampler2
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
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

        dmd_sink = DigitalMetadataSink(
            self,
            name="dmd_sink",
            metadata_dir=f"{self.kwargs('drf_sink')['channel_dir']}/metadata",
            subdir_cadence_secs=self.kwargs("drf_sink")["subdir_cadence_secs"],
            file_cadence_secs=self.kwargs("drf_sink")["file_cadence_millisecs"] // 1000,
            uuid=self.kwargs("drf_sink")["uuid"],
            filename_prefix="metadata",
            metadata=self.kwargs("metadata"),
        )
        self.add_flow(last_op, dmd_sink)

        if self.kwargs("pipeline")["spectrogram"]:
            spectrogram = Spectrogram(
                self,
                name="spectrogram",
                data_outdir=f"{DRF_RECORDING_DIR}/{self.kwargs('drf_sink')['channel_dir']}_spectrogram",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("spectrogram")),
            )
            self.add_flow(last_op, spectrogram)


def main():
    parser = build_config_parser()
    cfg = parser.parse_args()

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
    logging.basicConfig(level=log_level, force=True)

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
