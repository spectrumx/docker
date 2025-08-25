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
import datetime
import fractions
import logging
import os
import pathlib
import traceback
import typing

import cupy as cp
import cupyx
import cupyx.scipy.signal as cpss
import digital_rf as drf
import h5py
import holoscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from holohub import rf_array

mpl.use("agg")

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")


def timestamp_floor(nsamples, sample_rate_frac):
    """Convert number of samples into a (sec, picosec) tuple using a given sample rate"""
    srn = sample_rate_frac.numerator
    srd = sample_rate_frac.denominator
    # calculate with divide/modulus split to avoid overflow
    # second = s * d // n == ((s // n) * d) + ((si % n) * d) // n
    tmp_div = nsamples // srn
    tmp_mod = nsamples % srn
    second = tmp_div * srd
    tmp = tmp_mod * srd
    tmp_div = tmp // srn
    tmp_mod = tmp % srn
    second += tmp_div
    # picoseconds calculated from remainder of division to calculate seconds
    # picosecond = rem * 1e12 // n = rem * (1e12 // n) + (rem * (1e12 % n)) // n
    tmp = tmp_mod
    tmp_div = 1_000_000_000_000 // srn
    tmp_mod = 1_000_000_000_000 % srn
    picosecond = (tmp * tmp_div) + ((tmp * tmp_mod) // srn)

    return second, picosecond


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


class Spectrogram(holoscan.core.Operator):
    chunk_size: int
    num_subchannels: int
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
    spec_sample_cadence: int

    def __init__(
        self,
        fragment,
        *args,
        chunk_size,
        num_subchannels,
        window="hann",
        nperseg=1024,
        noverlap=None,
        nfft=None,
        detrend=False,
        reduce_op="max",
        num_spectra_per_chunk=1,
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
        """
        self.chunk_size = chunk_size
        self.num_subchannels = num_subchannels
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
        self.spec_sample_cadence = self.chunk_size // self.num_spectra_per_chunk

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.rf_array.Spectrogram")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in")
        spec.output("spec_out").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=self.num_spectra_per_chunk,
            policy=0,  # pop
        )

    def initialize(self):
        self.logger.debug("Initializing spectrogram operator")

        # warm up CUDA calculation and extract an FFT plan
        for _host_spec in self.calc_spectrogram_chunk(
            cp.ones((self.chunk_size, self.num_subchannels), dtype="complex64"),
        ):
            pass
        plan_cache = cp.fft.config.get_plan_cache()
        for key, node in plan_cache:
            self.cufft_plan = node.plan

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        rf_arr = op_input.receive("rf_in")
        stream_ptr = op_input.receive_cuda_stream("rf_in", allocate=True)

        while rf_arr is not None:
            rf_metadata = rf_arr.metadata

            msg = f"Processing spectrogram for chunk with sample_idx {rf_metadata.sample_idx}"
            self.logger.debug(msg)

            with cp.cuda.ExternalStream(stream_ptr):
                rf_data = cp.from_dlpack(rf_arr.data)
                with self.cufft_plan:
                    for spec_count, host_spec in enumerate(
                        self.calc_spectrogram_chunk(rf_data)
                    ):
                        sample_idx = (
                            rf_metadata.sample_idx
                            + spec_count * self.spec_sample_cadence
                        )
                        spec_metadata = rf_array.RFMetadata(
                            sample_idx,
                            rf_metadata.sample_rate_numerator,
                            rf_metadata.sample_rate_denominator,
                            rf_metadata.center_freq,
                        )
                        out_message = {
                            # cast to Holoscan Tensor so it is passed without data copy
                            # allowing emit before the stream is synced
                            "spec": holoscan.as_tensor(host_spec),
                            "metadata": spec_metadata,
                        }
                        op_output.emit(out_message, "spec_out")
            # try to receive again to either get another message or exit the while loop
            rf_arr = op_input.receive("rf_in")

    def calc_spectrogram_chunk(self, rf_data):
        # get pinned host memory for output so copy can run asynchronously
        # (Fortran contiguous because we will want to output/plot separately by
        #  subchannel and this keeps each subchannel contiguous)
        spec_pinned = cupyx.empty_pinned(
            (self.nfft, self.num_subchannels, self.num_spectra_per_chunk),
            dtype="float32",
            order="F",
        )
        spec_pinned[...] = np.nan

        spectrum_chunks = rf_data.reshape(
            (
                self.num_spectra_per_chunk,
                self.chunk_size // self.num_spectra_per_chunk,
                self.num_subchannels,
            )
        )
        _freqs, _sidxs, Zxx = cpss.stft(
            spectrum_chunks,
            fs=1,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            detrend=self.detrend,
            return_onesided=False,
            boundary=None,
            padded=False,
            axis=1,
            scaling="spectrum",
        )
        # reduce over time (last) axis
        spec = cp.fft.fftshift(
            self.reduce_op(Zxx.real**2 + Zxx.imag**2, axis=-1), axes=1
        )

        for spec_count in range(spec.shape[0]):
            host_spec = spec[spec_count, ...].get(
                out=spec_pinned[..., spec_count], blocking=False
            )
            yield host_spec


@dataclasses.dataclass
class SpectrogramOutputParams:
    """Spectrogram output parameters"""

    plot_outdir: os.PathLike = f"{DRF_RECORDING_DIR}/spectrograms"
    """Directory for writing spectrogram plots"""
    num_spectra_per_output: int = 1200
    """Number of spectra to combine in a single output, either a data sample or plot"""
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


class SpectrogramOutput(holoscan.core.Operator):
    nfft: int
    spec_sample_cadence: int
    num_subchannels: int
    data_outdir: os.PathLike
    plot_outdir: os.PathLike
    num_spectra_per_output: int
    figsize: tuple[float, float]
    dpi: int
    col_wrap: int
    cmap: str
    snr_db_min: float
    snr_db_max: float

    def __init__(
        self,
        fragment,
        *args,
        nfft,
        spec_sample_cadence,
        num_subchannels,
        data_outdir,
        plot_outdir=f"{DRF_RECORDING_DIR}/spectrograms",
        num_spectra_per_output=1200,
        figsize=(6.4, 4.8),
        dpi=150,
        col_wrap=1,
        cmap="viridis",
        snr_db_min=-10,
        snr_db_max=40,
        **kwargs,
    ):
        """Operator that computes spectrograms from RF data.

        **==Named Inputs==**

            spec_in : dict[spec: array, rf_metadata: RFMetadata]
                Dictionary containing a spectrum and rf_metadata.

        Parameters
        ----------
        fragment : Fragment
            The fragment that the operator belongs to
        nfft: int
            Number of frequency samples in the spectrogram
        spec_sample_cadence: int
            Number of RF samples that go into each spectrum input chunk
        num_subchannels: int
            Number of subchannels contained in the spectrogram data
        data_outdir: os.PathLike
            Directory for writing spectrogram data
        plot_outdir: os.PathLike
            Directory for writing spectrogram plots
        num_spectra_per_output: int
            Number of spectra to combine in a single output, either a data sample or plot
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
        """
        self.nfft = nfft
        self.spec_sample_cadence = spec_sample_cadence
        self.num_subchannels = num_subchannels
        self.data_outdir = pathlib.Path(data_outdir).resolve()
        self.plot_outdir = pathlib.Path(plot_outdir).resolve()
        self.num_spectra_per_output = num_spectra_per_output
        self.figsize = figsize
        self.dpi = dpi
        self.col_wrap = col_wrap
        self.cmap = cmap
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.rf_array.SpectrogramOutput")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("spec_in").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=self.num_spectra_per_output,
            policy=0,  # pop
        )

    def initialize(self):
        self.logger.debug("Initializing spectrogram output operator")
        self.data_outdir.mkdir(parents=True, exist_ok=True)
        self.last_written_sample_idx = -1
        self.latest_chunk_idx = 0
        self.reset_stored_data()
        self.create_spec_figure()
        self.dmd_writer = None

    def reset_stored_data(self, rf_metadata=None):
        if rf_metadata is not None:
            self.stored_metadata = rf_metadata
            self.sample_rate_frac = fractions.Fraction(
                self.stored_metadata.sample_rate_numerator,
                self.stored_metadata.sample_rate_denominator,
            )
            self.spec_freq_idx = np.fft.fftshift(
                np.fft.fftfreq(self.nfft, 1 / self.sample_rate_frac)
            )
            self.spec_sample_idx = (
                rf_metadata.sample_idx
                + self.spec_sample_cadence
                * np.arange(self.num_spectra_per_output, dtype="uint64")
            )
        else:
            self.stored_metadata = None
            self.sample_rate_frac = None
            self.spec_freq_idx = None
            self.spec_sample_idx = None
        self.spec_data = np.full(
            (self.nfft, self.num_subchannels, self.num_spectra_per_output),
            np.nan,
            dtype=np.float32,
            order="F",
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
        fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72)
        self.norm = mpl.colors.Normalize(vmin=self.snr_db_min, vmax=self.snr_db_max)
        xlocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=7)
        xformatter = mpl.dates.ConciseDateFormatter(xlocator)
        axs_1d = []
        imgs = []
        ref_lvl_texts = []
        for sch in range(self.num_subchannels):
            row_idx = sch // self.col_wrap
            col_idx = sch % self.col_wrap
            ax = axs[row_idx, col_idx]
            img = ax.imshow(
                self.spec_data[:, sch, :],
                cmap=self.cmap,
                norm=self.norm,
                aspect="auto",
                interpolation="none",
                origin="lower",
            )
            cb = fig.colorbar(img, ax=ax, fraction=0.05, pad=0.01)
            cb.set_label("Power relative to reference [dB]")
            ax.set_ylabel("Frequency [MHz]")
            if self.num_subchannels > 1:
                title = ax.set_title(f"Subchannel {sch}", fontsize="small")
            else:
                title = ax.set_title(" ", fontsize="small")
            ref_lvl_text = ax.text(
                1.0,
                title.get_position()[1],
                "Ref: 1.23e-9 [$V_{ADC}^2$]",
                fontsize="small",
                fontstyle="italic",
                va=title.get_verticalalignment(),
                ha="right",
                transform=title.get_transform(),
            )
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(xformatter)
            imgs.append(img)
            axs_1d.append(ax)
            ref_lvl_texts.append(ref_lvl_text)
        axs_1d[-1].set_xlabel("Time (UTC)")
        self.suptitle = fig.suptitle("Spectrogram", fontsize="medium")
        fig.autofmt_xdate(rotation=0, ha="center")

        self.fig = fig
        self.axs = axs_1d
        self.imgs = imgs
        self.ref_lvl_texts = ref_lvl_texts

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        spec_message = op_input.receive("spec_in")
        # get stream and synchronize to ensure data is in host memory before proceeding
        # (do this here because input buffer is large and therefore waiting on this
        #  operator does not slow down upstream operators)
        stream_ptr = op_input.receive_cuda_stream("spec_in", allocate=True)
        stream = cp.cuda.ExternalStream(stream_ptr)
        stream.synchronize()
        while spec_message is not None:
            self.compute_one(spec_message)
            # try to receive again to either get another message or exit the while loop
            spec_message = op_input.receive("spec_in")

    def compute_one(self, spec_message):
        spec_arr = np.from_dlpack(spec_message["spec"])
        rf_metadata = spec_message["metadata"]

        # reset stored data with new metadata if uninitialized
        # (first time or finished previous write)
        if self.stored_metadata is None:
            self.reset_stored_data(rf_metadata)

        # initialize Digital Metadata writer if uninitialized
        if self.dmd_writer is None:
            self.dmd_writer = drf.DigitalMetadataWriter(
                metadata_dir=str(self.data_outdir),
                subdir_cadence_secs=3600,
                file_cadence_secs=1,
                sample_rate_numerator=self.stored_metadata.sample_rate_numerator,
                sample_rate_denominator=self.stored_metadata.sample_rate_denominator,
                file_name="spectrogram",
            )

        # if metadata changes or incoming data belongs to a newer batch,
        # write out existing data and start anew
        if (
            (
                self.stored_metadata.sample_rate_numerator
                != rf_metadata.sample_rate_numerator
            )
            or (
                self.stored_metadata.sample_rate_denominator
                != rf_metadata.sample_rate_denominator
            )
            or (self.stored_metadata.center_freq != rf_metadata.center_freq)
            or (
                (rf_metadata.sample_idx - self.stored_metadata.sample_idx)
                >= (self.num_spectra_per_output * self.spec_sample_cadence)
            )
        ):
            self.write_output()
            self.reset_stored_data(rf_metadata)

        # chunk_idx is the index into the stored data arrays for the incoming data
        # (stored_metadata.sample_idx marks the start of an output cycle)
        chunk_idx = (
            int(rf_metadata.sample_idx - self.stored_metadata.sample_idx)
            // self.spec_sample_cadence
        )
        # store incoming data
        self.latest_chunk_idx = chunk_idx
        self.spec_data[..., chunk_idx] = spec_arr

        # write output if we've filled the storage arrays
        if chunk_idx == (self.num_spectra_per_output - 1):
            self.write_output()
            self.reset_stored_data()

    def write_output(self):
        if self.stored_metadata is None:
            return
        chunk_idx = self.latest_chunk_idx
        sample_idx = self.spec_sample_idx[chunk_idx]
        if sample_idx <= self.last_written_sample_idx:
            # skip writing because we already wrote this data
            return

        sample_idx_arr = self.spec_sample_idx[: (chunk_idx + 1)]
        secs, picosecs = timestamp_floor(sample_idx_arr, self.sample_rate_frac)
        microsecs = picosecs // 1_000_000
        spec_start_dt = drf.util.epoch + datetime.timedelta(
            seconds=int(secs[0]), microseconds=int(microsecs[0])
        )
        time_idx = (
            np.datetime64(spec_start_dt.replace(tzinfo=None))
            + np.asarray(secs - secs[0], dtype="m8[s]")
            + np.asarray(microsecs - microsecs[0], dtype="m8[us]")
        )

        output_spec_data = self.spec_data[..., : (chunk_idx + 1)]

        self.logger.info(f"Outputting spectrogram for time {spec_start_dt}")

        num_retries = 3
        for retry in range(0, num_retries):
            try:
                self.dmd_writer.write(
                    [int(sample_idx_arr[0])],
                    [
                        {
                            "spectrogram": output_spec_data.transpose((1, 0, 2)),
                            "freq_idx": self.spec_freq_idx
                            + self.stored_metadata.center_freq,
                            "sample_idx": sample_idx_arr,
                            "time_idx": time_idx.astype(
                                h5py.opaque_dtype(time_idx.dtype)
                            ),
                            "center_freq": self.stored_metadata.center_freq,
                        }
                    ],
                )
            except IOError:
                if retry == (num_retries - 1):
                    self.logger.warning(traceback.format_exc())
            else:
                break

        timestr = spec_start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        freqstr = f"{self.stored_metadata.center_freq / 1e6:n}MHz"
        datestr = spec_start_dt.strftime("%Y-%m-%d")

        reference_pwr = np.nanpercentile(
            output_spec_data, 15, axis=(0, 2), keepdims=True
        )
        spec_power_db = 10 * np.log10(output_spec_data / reference_pwr)
        delta_t = time_idx[1] - time_idx[0]
        delta_f = self.spec_freq_idx[1] - self.spec_freq_idx[0]
        extent = (
            time_idx[0],
            time_idx[-1] + delta_t,
            (self.stored_metadata.center_freq + self.spec_freq_idx[0] - delta_f / 2)
            / 1e6,
            (self.stored_metadata.center_freq + self.spec_freq_idx[-1] + delta_f / 2)
            / 1e6,
        )
        for sch in range(self.num_subchannels):
            self.imgs[sch].set(
                data=spec_power_db[:, sch, :],
                extent=extent,
            )
            self.ref_lvl_texts[sch].set_text(
                f"Ref: {float(reference_pwr[0, sch, 0]):.3n} [$V_{{ADC}}^2$]"
            )
        self.suptitle.set_text(
            f"{self.data_outdir.parent.name}/{self.data_outdir.name} @ {freqstr}"
        )
        self.fig.canvas.draw()

        fname = f"spec_{timestr}_{freqstr}.png"
        outpath = self.plot_outdir / datestr / fname
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(outpath)
        latest_spec_path = outpath.parent.parent / "spec_latest.png"
        latest_spec_path.unlink(missing_ok=True)
        os.link(outpath, latest_spec_path)

        # track that we just wrote a sample
        self.last_written_sample_idx = sample_idx

    def stop(self):
        self.write_output()
