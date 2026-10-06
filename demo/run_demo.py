"""Run sorting on the included sample or the downloaded ten-minute input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from review_common import (ROOT, load_calibration, load_scientific_source, long_path,
                           read_json, sha256, verified_asset, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sw8", "fp"), default="sw8")
    parser.add_argument("--input", type=Path, help="Downloaded recording.ap.cbin; omit for offline smoke")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Refit v1 on [0,240) s using the bundled KS4 teacher; needs 10-min input")
    parser.add_argument("--check-expected", action="store_true", help="Check included smoke reference")
    args = parser.parse_args()
    if args.recalibrate and not args.input:
        parser.error("--recalibrate requires the downloaded 10-min prefix")
    if args.check_expected and (args.input or args.recalibrate):
        parser.error("The expected reference is for the bundled, cached-calibration smoke test only")
    output = long_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output is not empty. Preserve results and choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    load_scientific_source()
    import numpy as np
    import config
    import main as pipeline
    from config_runtime import apply_config_snapshot
    from preprocessing import (reset_quantization_diagnostics, set_quantization_fixed_ranges,
                               stop_quantization_range_collection)
    from run_logger import start_logging, stop_logging

    settings = read_json(verified_asset(f"data/config_{args.mode}.json"))
    apply_config_snapshot(settings)
    config.KS4_FULL_ENABLE = False
    config.HARDWARE_EXPORT_ENABLE = False
    np.random.seed(config.RANDOM_SEED)
    positions = np.load(verified_asset("data/channel_positions.npy"), allow_pickle=False)
    reader = None
    tee = start_logging(str(output))
    cleanups = []
    pipeline._PIPELINE_CLEANUP_STACK.append(cleanups)
    try:
        if args.input:
            import spikeglx
            descriptor = read_json(verified_asset("data/download_descriptor.json"))
            input_path = long_path(args.input)
            if sha256(input_path) != descriptor["prefix_sha256"]:
                raise RuntimeError("Input is not the pinned ten-minute prefix")
            for suffix in ("ch", "meta"):
                if sha256(input_path.with_suffix('.' + suffix)) != sha256(verified_asset(f"data/prefix.ap.{suffix}")):
                    raise RuntimeError(f"Input .{suffix} does not match the packaged metadata")
            reader = spikeglx.Reader(input_path)
            if reader.fs != 30000 or reader.nc != 385 or reader.ns < 18_000_000:
                raise RuntimeError("Unexpected recording dimensions")
            start_s, end_s = 240.0, 600.0

            def read_uv(start, count):
                return reader.read(nsel=slice(start, start + count), csel=slice(0, 384), sync=False).astype(np.float32) * 1e6
        else:
            smoke = np.load(verified_asset("data/smoke_raw.npz"), allow_pickle=False)
            start_s = float(smoke["start_s"])
            raw = smoke["raw_int16"]
            gains_v = smoke["volts_per_bit"]
            end_s = start_s + len(raw) / config.SAMPLE_RATE

            def read_uv(start, count):
                local = start - int(start_s * config.SAMPLE_RATE)
                converted = raw[local:local + count].astype(np.float32)
                converted *= gains_v
                converted *= 1e6
                return converted

        if args.recalibrate:
            from data_loader import RawDataStreamer
            teacher_dir = output / "ks4_cache"
            teacher_dir.mkdir()
            shutil.copyfile(verified_asset("data/ks4_labels_240s_384ch.npz"),
                            teacher_dir / "ks4_labels_240s_384ch.npz")
            import spike_clustering

            def no_new_ks4(*a, **kw):
                raise RuntimeError("KS4 cache was not reused. This demo never launches a new KS4 job.")

            spike_clustering.run_official_ks4_calibration = no_new_ks4
            stream = RawDataStreamer(None, config.IBL_SESSION_EID, config.IBL_PROBE_LABEL)
            stream._reader, stream._mode = reader, "spikeglx"
            stream._total_samples, stream._n_channels = reader.ns, 384
            pipeline._begin_quant_range_calibration()
            fit, ranges, _ = pipeline._run_calibration_phase(
                "IBL", "IBL DATA", str(output), stream, positions,
                {"session_eid": config.IBL_SESSION_EID, "probe": "probe00", "brain_region": "VISp"},
                diagnostic_required=True)
        else:
            model = load_calibration(args.mode)
            fit, ranges = model["final_fit"], model["range_payload"]
        reset_quantization_diagnostics()
        stop_quantization_range_collection()
        set_quantization_fixed_ranges((ranges or {}).get("ranges", {}))
        pipe, clust = fit["pipe"], fit["clust"]
        if any(len(getattr(clust, name, [])) for name in ("all_times", "all_labels")):
            raise RuntimeError("Calibration bundle contains previous streaming events")
        pipe.fir_filter.reset(384)
        state = pipeline._SAStreamState()
        chunk = config.CHUNK_SAMPLES
        count = int(round((end_s - start_s) * config.SAMPLE_RATE)) // chunk
        print(f"[DEMO] {args.mode}: [{start_s},{end_s}) s; {count} x 100-ms chunks")
        for index in range(count):
            raw_chunk = read_uv(int(start_s * config.SAMPLE_RATE) + index * chunk, chunk)
            white = pipe.process_chunk(raw_chunk)
            state.process_chunk(white, fit["wTEMP"], fit["sa"], clust,
                                batch_time_offset=index * chunk)
            if index % 100 == 0 or index + 1 == count:
                print(f"[DEMO] {index + 1}/{count} chunks", flush=True)
        results = pipeline._run_postprocessing(clust, positions, str(output))
        if results is None:
            raise RuntimeError("No spikes produced")
        times = np.load(output / "clustering/spike_times.npy", allow_pickle=False)
        labels = np.load(output / "clustering/spike_clusters.npy", allow_pickle=False)
        # Preserve relative sample coordinates and also provide absolute AP seconds.
        np.save(output / "clustering/spike_times_ap_seconds.npy", times / config.SAMPLE_RATE + start_s)
        active = np.unique(labels[labels >= 0])
        summary = {"mode": args.mode, "eid": config.IBL_SESSION_EID, "probe": "probe00",
                   "ap_interval_s": [start_s, end_s], "calibration_interval_s": [0, 240],
                   "calibration": "refitted_with_cached_KS4_teacher" if args.recalibrate else "bundled_240s_fit",
                   "chunk_ms": config.CHUNK_DURATION_MS, "n_chunks": count,
                   "total_spikes": int(len(times)), "active_clusters": int(len(active)),
                   "elapsed_s": time.perf_counter() - t0,
                   "python": platform.python_version(), "platform": platform.platform(),
                   "interpretation": "Sorting output for the specified AP interval",
                   "arrays_sha256": {n: sha256(output / "clustering" / n) for n in
                                     ("spike_times.npy", "spike_clusters.npy", "channel_positions.npy")}}
        write_json(output / "demo_summary.json", summary)
        if args.check_expected:
            expected = read_json(ROOT / f"expected/{args.mode}/demo_summary.json")
            matches = summary["arrays_sha256"] == expected["arrays_sha256"]
            write_json(output / "reference_check.json", {"exact_array_match": matches,
                       "note": "Exact equality is expected in the tested environment; investigate differences on other platforms."})
            if not matches:
                raise RuntimeError("Demo differs from the packaged reference; see summary and reference_check.json")
        (output / "DEMO_COMPLETE.txt").write_text("Completed successfully. See demo_summary.json.\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if reader is not None:
            reader.close()
        while cleanups:
            cleanups.pop()()
        pipeline._PIPELINE_CLEANUP_STACK.pop()
        stop_logging(tee)


if __name__ == "__main__":
    main()
