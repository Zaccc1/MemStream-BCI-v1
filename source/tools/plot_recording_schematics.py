#!/usr/bin/env python3
"""Generate two SVG recording schematics from a real AP recording.

Outputs:
  1. extracellular_recording_schematic.svg
  2. dense_multichannel_recording_schematic.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

DEFAULT_IBL_EID = "9b5a1754-ac99-4d53-97d3-35c2f6638507"
DEFAULT_IBL_PROBE = "probe00"


def _read_open_ephys(recording_dir: Path, ap_folder: str | None,
                     start_s: float, duration_s: float) -> tuple[np.ndarray, float]:
    from local_data_loader import LocalDataStreamer

    streamer = LocalDataStreamer(str(recording_dir), ap_folder=ap_folder)
    try:
        streamer.open()
        data = streamer.read_calibration_data(
            duration_s=duration_s, start_s=start_s)
        sample_rate = getattr(streamer, "_sample_rate", None) or config.SAMPLE_RATE
        return np.asarray(data, dtype=np.float32), float(sample_rate)
    finally:
        streamer.close()


def _read_binary(binary_path: Path, n_channels: int, sample_rate: float,
                 uv_per_bit: float, start_s: float,
                 duration_s: float) -> tuple[np.ndarray, float]:
    raw = np.memmap(binary_path, dtype=np.int16, mode="r")
    total_samples = raw.size // int(n_channels)
    if total_samples <= 0:
        raise ValueError(f"{binary_path} is too small for {n_channels} channels")
    start = max(0, int(round(float(start_s) * float(sample_rate))))
    n_samples = max(1, int(round(float(duration_s) * float(sample_rate))))
    end = min(total_samples, start + n_samples)
    if start >= total_samples:
        raise ValueError(
            f"start_s={start_s} is beyond binary duration "
            f"{total_samples / sample_rate:.3f}s")
    shaped = raw[:total_samples * n_channels].reshape(total_samples, n_channels)
    data = np.asarray(shaped[start:end, :], dtype=np.float32)
    data *= float(uv_per_bit)
    return data, float(sample_rate)


def _read_ibl(eid: str, probe: str, start_s: float,
              duration_s: float) -> tuple[np.ndarray, float]:
    from data_loader import RawDataStreamer, connect_one

    one = connect_one(silent=True)
    print("[IBL] Using RawDataStreamer, same loader path as main.py")
    streamer = RawDataStreamer(one, eid, probe)
    try:
        streamer.open()
        data = streamer.read_calibration_data(
            duration_s=duration_s, start_s=start_s)
        return np.asarray(data, dtype=np.float32), float(config.SAMPLE_RATE)
    finally:
        streamer.close()


def _read_window(args, start_s: float, duration_s: float) -> tuple[np.ndarray, float]:
    if bool(args.ibl):
        eid = str(args.ibl_eid or DEFAULT_IBL_EID).strip()
        probe = str(args.ibl_probe or DEFAULT_IBL_PROBE).strip()
        return _read_ibl(eid, probe, start_s, duration_s)
    if args.recording_dir is not None:
        return _read_open_ephys(
            args.recording_dir, args.ap_folder, start_s, duration_s)
    return _read_binary(
        args.binary, args.n_channels, args.sample_rate, args.uv_per_bit,
        start_s, duration_s)


def _preprocess(data: np.ndarray, sample_rate: float,
                mode: str) -> np.ndarray:
    if mode == "raw":
        return np.asarray(data, dtype=np.float32)

    out = np.asarray(data, dtype=np.float32)
    if mode in ("car", "car_highpass"):
        ref = np.median(out, axis=1, keepdims=True)
        out = out - ref

    if mode == "car_highpass":
        try:
            from scipy.signal import firwin, lfilter
        except Exception as exc:
            raise RuntimeError(
                "scipy is required for --preprocess car_highpass") from exc
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
        out = lfilter(coeffs, 1.0, out, axis=0).astype(np.float32)
    return out.astype(np.float32)


def _parse_channels(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    vals = [int(p.strip()) for p in text.split(",") if p.strip()]
    return np.asarray(vals, dtype=int) if vals else None


def _choose_sparse_channels(data: np.ndarray, channels: np.ndarray | None,
                            n_traces: int) -> np.ndarray:
    n_channels = data.shape[1]
    if channels is not None:
        valid = channels[(channels >= 0) & (channels < n_channels)]
        if valid.size == 0:
            raise ValueError("None of the requested sparse channels exist")
        return valid[:n_traces]

    centered = data - np.median(data, axis=0, keepdims=True)
    score = np.percentile(np.abs(centered), 98, axis=0)
    selected = np.argsort(score)[-int(n_traces):]
    return np.asarray(sorted(int(c) for c in selected), dtype=int)


def _choose_dense_channels(data: np.ndarray, channels: np.ndarray | None,
                           n_traces: int) -> np.ndarray:
    n_channels = data.shape[1]
    if channels is not None:
        valid = channels[(channels >= 0) & (channels < n_channels)]
        if valid.size == 0:
            raise ValueError("None of the requested dense channels exist")
        return valid[:n_traces]
    if n_traces >= n_channels:
        return np.arange(n_channels, dtype=int)
    return np.unique(np.linspace(0, n_channels - 1, int(n_traces)).astype(int))


def _decimate_for_drawing(data: np.ndarray, sample_rate: float,
                          max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    if n <= max_points:
        idx = np.arange(n, dtype=int)
    else:
        idx = np.linspace(0, n - 1, int(max_points)).astype(int)
    t = idx.astype(np.float32) / float(sample_rate)
    return data[idx], t


def _robust_scaled(trace_data: np.ndarray, gain: float = 1.0) -> tuple[np.ndarray, float]:
    centered = trace_data - np.median(trace_data, axis=0, keepdims=True)
    scale = float(np.percentile(np.abs(centered), 98))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(centered) + 1e-6)
    scaled = np.clip(centered / scale, -3.0, 3.0) * float(gain)
    return scaled.astype(np.float32), scale


def _plot_extracellular(data: np.ndarray, sample_rate: float,
                        channels: np.ndarray, output_path: Path,
                        title: str, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_data, t = _decimate_for_drawing(data[:, channels], sample_rate, max_points)
    scaled, _ = _robust_scaled(trace_data, gain=0.42)
    n = len(channels)
    spacing = 2.15
    offsets = np.arange(n, dtype=np.float32)[::-1] * spacing

    fig, ax = plt.subplots(figsize=(4.5, 6.6))
    for i in range(n):
        ax.plot(t, scaled[:, i] + offsets[i], color="black", lw=0.95)

    _ = title
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(-0.95, offsets[0] + 0.95)
    ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", transparent=False)
    plt.close(fig)


def _plot_dense(data: np.ndarray, sample_rate: float, channels: np.ndarray,
                output_path: Path, title: str, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_data, t = _decimate_for_drawing(data[:, channels], sample_rate, max_points)
    scaled, _ = _robust_scaled(trace_data, gain=0.28)
    n = len(channels)
    offsets = np.arange(n, dtype=np.float32)[::-1]

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for i in range(n):
        ax.plot(t, scaled[:, i] + offsets[i], color="0.15", lw=0.32, alpha=0.75)

    _ = title
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", transparent=False)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate two publication-style SVG schematics from a recording.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--ibl", action="store_true",
                        help="Read IBL raw AP data using ONE/spikeglx.")
    source.add_argument("--recording-dir", type=Path,
                        help="Open Ephys recording directory containing structure.oebin.")
    source.add_argument("--binary", type=Path,
                        help="Raw int16 AP binary shaped as samples x channels.")
    parser.add_argument("--ibl-eid", default=None,
                        help=f"IBL session EID. Default: {DEFAULT_IBL_EID}.")
    parser.add_argument("--ibl-probe", default=None,
                        help=f"IBL probe label. Default: {DEFAULT_IBL_PROBE}.")
    parser.add_argument("--ap-folder", default=None,
                        help="Optional Open Ephys AP stream folder override.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("docs") / "recording_schematics_ibl")
    parser.add_argument("--start-s", type=float, default=60.0)
    parser.add_argument("--sparse-duration-ms", type=float, default=30.0)
    parser.add_argument("--dense-duration-s", type=float, default=1.0)
    parser.add_argument("--sparse-channels",
                        help="Comma-separated channels for the sparse schematic.")
    parser.add_argument("--dense-channels",
                        help="Comma-separated channels for the dense schematic.")
    parser.add_argument("--sparse-n-traces", type=int, default=6)
    parser.add_argument("--dense-n-traces", type=int, default=48)
    parser.add_argument("--sample-rate", type=float, default=float(config.SAMPLE_RATE))
    parser.add_argument("--n-channels", type=int, default=int(config.N_CHANNELS))
    parser.add_argument("--uv-per-bit", type=float, default=float(config.UV_PER_BIT))
    parser.add_argument("--preprocess", choices=("raw", "car", "car_highpass"),
                        default="car_highpass")
    parser.add_argument("--max-points", type=int, default=3500)
    parser.add_argument("--extracellular-title", default="Extracellular\nrecording")
    parser.add_argument("--dense-title", default="Dense multi-channel recording")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ibl and args.recording_dir is None and args.binary is None:
        configured = str(getattr(config, "LOCAL_RECORDING_DIR", "") or "").strip()
        if configured:
            args.recording_dir = Path(configured)
        else:
            args.ibl = True

    sparse_duration_s = float(args.sparse_duration_ms) / 1000.0
    dense_duration_s = float(args.dense_duration_s)
    sparse_data, sparse_sr = _read_window(args, args.start_s, sparse_duration_s)
    dense_data, dense_sr = _read_window(args, args.start_s, dense_duration_s)

    sparse_data = _preprocess(sparse_data, sparse_sr, args.preprocess)
    dense_data = _preprocess(dense_data, dense_sr, args.preprocess)

    sparse_channels = _choose_sparse_channels(
        sparse_data, _parse_channels(args.sparse_channels),
        int(args.sparse_n_traces))
    dense_channels = _choose_dense_channels(
        dense_data, _parse_channels(args.dense_channels),
        int(args.dense_n_traces))

    out_dir = args.output_dir
    extracellular_path = out_dir / "extracellular_recording_schematic.svg"
    dense_path = out_dir / "dense_multichannel_recording_schematic.svg"
    _plot_extracellular(
        sparse_data, sparse_sr, sparse_channels, extracellular_path,
        args.extracellular_title, int(args.max_points))
    _plot_dense(
        dense_data, dense_sr, dense_channels, dense_path,
        args.dense_title, int(args.max_points))

    print(f"Wrote {extracellular_path}")
    print(f"  sparse channels: {','.join(str(int(c)) for c in sparse_channels)}")
    print(f"Wrote {dense_path}")
    print(f"  dense channels: {','.join(str(int(c)) for c in dense_channels)}")
    print(
        "Difference: extracellular schematic highlights a few separated AP "
        "traces; dense schematic compresses many simultaneous channels to show "
        "high-density probe coverage over time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
