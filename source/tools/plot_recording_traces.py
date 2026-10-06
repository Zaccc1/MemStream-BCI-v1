#!/usr/bin/env python3
"""Export stacked extracellular traces from a recording or AP binary as SVG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


def _read_open_ephys(recording_dir: Path, ap_folder: str | None, start_s: float,
                     duration_s: float) -> tuple[np.ndarray, float]:
    from local_data_loader import LocalDataStreamer

    streamer = LocalDataStreamer(str(recording_dir), ap_folder=ap_folder)
    try:
        streamer.open()
        data = streamer.read_calibration_data(duration_s=duration_s, start_s=start_s)
        sample_rate = getattr(streamer, "_sample_rate", None) or config.SAMPLE_RATE
        return data, float(sample_rate)
    finally:
        streamer.close()


def _read_binary(binary_path: Path, n_channels: int, sample_rate: float,
                 uv_per_bit: float, start_s: float,
                 duration_s: float) -> tuple[np.ndarray, float]:
    raw = np.memmap(binary_path, dtype=np.int16, mode="r")
    total_samples = raw.size // n_channels
    if total_samples <= 0:
        raise ValueError(f"{binary_path} is too small for {n_channels} channels")

    start = max(0, int(round(start_s * sample_rate)))
    n_samples = max(1, int(round(duration_s * sample_rate)))
    if start >= total_samples:
        raise ValueError(
            f"start_s={start_s} is beyond binary duration "
            f"{total_samples / sample_rate:.3f}s"
        )

    end = min(total_samples, start + n_samples)
    shaped = raw[:total_samples * n_channels].reshape(total_samples, n_channels)
    data = np.asarray(shaped[start:end, :], dtype=np.float32) * float(uv_per_bit)
    return data, float(sample_rate)


def _filter_trace_data(data: np.ndarray, sample_rate: float) -> np.ndarray:
    from scipy.signal import firwin, lfilter

    if config.CAR_METHOD == "median":
        ref = np.median(data, axis=1, keepdims=True)
    else:
        ref = np.mean(data, axis=1, keepdims=True)
    car = (data - ref).astype(np.float32)

    num_taps = int(config.FIR_NUM_TAPS)
    if num_taps % 2 == 0:
        num_taps += 1
    coeffs = firwin(
        num_taps,
        float(config.FIR_HIGHPASS_FREQ),
        pass_zero=False,
        fs=float(sample_rate),
        window=config.FIR_WINDOW,
    ).astype(np.float32)
    return lfilter(coeffs, 1.0, car, axis=0).astype(np.float32)


def _parse_channels(value: str | None) -> list[int] | None:
    if not value:
        return None
    channels = []
    for part in value.split(","):
        part = part.strip()
        if part:
            channels.append(int(part))
    return channels or None


def _choose_channels(data: np.ndarray, channels: list[int] | None,
                     n_traces: int, start_channel: int,
                     channel_step: int, mode: str) -> np.ndarray:
    n_channels = data.shape[1]
    if channels is not None:
        selected = [ch for ch in channels if 0 <= ch < n_channels]
        if not selected:
            raise ValueError("None of the requested channels exist in this data")
        return np.asarray(selected, dtype=int)

    if mode == "rms":
        rms = np.sqrt(np.mean(data ** 2, axis=0))
        selected = np.argsort(rms)[-n_traces:]
        return np.asarray(sorted(int(ch) for ch in selected), dtype=int)

    stop = min(n_channels, start_channel + n_traces * channel_step)
    selected = np.arange(start_channel, stop, channel_step, dtype=int)
    if selected.size == 0:
        raise ValueError("No sequential channels selected")
    return selected[:n_traces]


def _plot_svg(data: np.ndarray, sample_rate: float, channels: np.ndarray,
              output_path: Path, start_s: float, title: str, gain: float,
              show_labels: bool, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_data = data[:, channels]
    if trace_data.shape[0] > max_points:
        idx = np.linspace(0, trace_data.shape[0] - 1, max_points).astype(int)
        trace_data = trace_data[idx]
        t = start_s + idx / sample_rate
    else:
        t = start_s + np.arange(trace_data.shape[0]) / sample_rate

    robust = np.percentile(np.abs(trace_data), 95)
    offset = max(float(robust) * 3.0 / max(gain, 1e-9), 1.0)

    fig_h = max(3.2, 0.38 * len(channels) + 1.4)
    fig, ax = plt.subplots(figsize=(5.0, fig_h))

    for row, ch in enumerate(channels):
        y = trace_data[:, row] * gain + row * offset
        ax.plot(t, y, color="black", lw=0.55, solid_capstyle="round")

    ymin = -0.7 * offset
    ymax = (len(channels) - 1 + 0.7) * offset
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Channels", fontsize=12)

    if show_labels:
        ax.set_yticks(np.arange(len(channels)) * offset)
        ax.set_yticklabels([str(ch) for ch in channels], fontsize=7)
    else:
        ax.set_yticks([])

    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    x0, x1 = ax.get_xlim()
    ax.annotate("", xy=(x1, ymin), xytext=(x1 - 0.03 * (x1 - x0), ymin),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black"})
    ax.annotate("", xy=(x0, ymax), xytext=(x0, ymax - 0.07 * (ymax - ymin)),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black"})

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stacked extracellular-recording SVG trace plot."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording-dir", type=Path,
                        help="Local Open Ephys recording directory containing structure.oebin.")
    source.add_argument("--binary", type=Path,
                        help="Raw int16 AP binary, shaped as samples x channels.")
    parser.add_argument("--ap-folder", default=None,
                        help="Optional Open Ephys AP stream folder override.")
    parser.add_argument("--output", type=Path, default=Path("recording_traces.svg"))
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-ms", type=float, default=120.0)
    parser.add_argument("--sample-rate", type=float, default=float(config.SAMPLE_RATE))
    parser.add_argument("--n-channels", type=int, default=int(config.N_CHANNELS))
    parser.add_argument("--uv-per-bit", type=float, default=float(config.UV_PER_BIT))
    parser.add_argument("--channels",
                        help="Comma-separated channel list, for example 12,40,88,130.")
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--start-channel", type=int, default=0)
    parser.add_argument("--channel-step", type=int, default=32)
    parser.add_argument("--channel-mode", choices=("rms", "sequential"), default="rms",
                        help="How to select channels when --channels is omitted.")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Vertical trace gain after conversion to microvolts.")
    parser.add_argument("--raw", action="store_true",
                        help="Plot unfiltered raw data instead of CAR + high-pass filtered data.")
    parser.add_argument("--show-channel-labels", action="store_true")
    parser.add_argument("--title", default="Extracellular\nrecording")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Downsample only for drawing if the window is very long.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    duration_s = args.duration_ms / 1000.0

    if args.recording_dir is not None:
        data, sample_rate = _read_open_ephys(
            args.recording_dir, args.ap_folder, args.start_s, duration_s
        )
    else:
        data, sample_rate = _read_binary(
            args.binary, args.n_channels, args.sample_rate, args.uv_per_bit,
            args.start_s, duration_s
        )

    if not args.raw:
        data = _filter_trace_data(data, sample_rate)

    channels = _choose_channels(
        data=data,
        channels=_parse_channels(args.channels),
        n_traces=args.n_traces,
        start_channel=args.start_channel,
        channel_step=max(1, args.channel_step),
        mode=args.channel_mode,
    )
    _plot_svg(
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        output_path=args.output,
        start_s=args.start_s,
        title=args.title,
        gain=args.gain,
        show_labels=args.show_channel_labels,
        max_points=args.max_points,
    )

    print(f"Wrote {args.output}")
    print(f"channels={','.join(str(int(ch)) for ch in channels)}")
    print(f"window={args.start_s:.3f}s..{args.start_s + duration_s:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
