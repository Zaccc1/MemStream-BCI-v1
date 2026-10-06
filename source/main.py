"""
BCI Neural Processing Pipeline — Main
===============================
Usage:
  python main.py              # IBL real data (default)
  python main.py --region CA1 # specify brain region

Each run uses single-value parameters from config.py.
Results are automatically appended to results_log.xlsx.
"""

import sys, os, time, tempfile, shutil, json, pickle, gzip, hashlib, copy
import threading
from functools import wraps
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config_runtime import apply_config_snapshot, load_config_snapshot
from runtime_support import format_error_report, write_crash_report
from run_logger import (start_logging, stop_logging, collect_config_params,
                        collect_results, append_to_xlsx)
from data_loader import connect_one, find_good_session, \
    load_channel_geometry, load_channel_regions, RawDataStreamer
from local_data_loader import (
    LocalDataStreamer, load_local_channel_geometry,
    load_local_reference, load_ttl_events, find_ttl_dir,
)
from preprocessing import (
    design_fir_highpass, apply_car, compute_channel_groups,
    compute_whitening_matrices, PreprocessingPipeline,
    FIRFilter, WhiteningProcessor, quantize_signal,
    using_int_accum_quantization, reset_quantization_diagnostics,
    format_quantization_summary, using_uint8_input_quantization,
    start_quantization_range_collection, stop_quantization_range_collection,
    set_quantization_fixed_ranges, _fixed_range_for,
    _quantize_signed_codes_with_stats, _quantize_uint8_codes_with_stats,
)
from template_matching import (
    get_templates, TemplateMatchingDetector,
    compute_score_matrix, compute_score_matrix_chunks,
)
from spatial_aggregation import SpatialAggregator
from spike_clustering import (
    SpikeClustering, save_clustering_results,
    extract_temporal_features, extract_raw_waveform_features,
)
from merging import merge_clusters, compute_cluster_quality
from visualization import (
    plot_fir_response, plot_whitening_matrices, plot_pipeline_stages,
    plot_channel_groups, plot_templates, plot_detected_spikes,
    plot_sa_detected_spikes, plot_clustering_summary, plot_merge_summary,
    plot_fn_analysis, plot_firing_rate_comparison,
    compute_drift_metrics, plot_drift_map,
)


# ==============================================================
# Shared utilities
# ==============================================================

_PIPELINE_CLEANUP_STACK = []


def _cleanup_once(callback):
    """Return an idempotent cleanup callback for pipeline resources."""
    state = {'done': False}

    def run():
        if state['done']:
            return
        state['done'] = True
        callback()

    return run


def _register_pipeline_cleanup(callback):
    cleanup = _cleanup_once(callback)
    if _PIPELINE_CLEANUP_STACK:
        _PIPELINE_CLEANUP_STACK[-1].append(cleanup)
    return cleanup


def _with_pipeline_cleanups(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cleanups = []
        _PIPELINE_CLEANUP_STACK.append(cleanups)
        try:
            return func(*args, **kwargs)
        finally:
            while cleanups:
                cleanup = cleanups.pop()
                try:
                    cleanup()
                except Exception as exc:
                    print(f"[MAIN] Cleanup skipped after error: {exc}")
            _PIPELINE_CLEANUP_STACK.pop()
    return wrapper


def _begin_quant_range_calibration():
    """Start calibration-time range collection for fixed uint8 quantization."""
    if not using_uint8_input_quantization():
        return None
    print("[QUANT-CALI] Collecting fixed uint8 activation ranges during calibration")
    print(f"[QUANT-CALI] Central percentile: "
          f"{getattr(config, 'QUANT_RANGE_PERCENTILE', 99.0):.2f}%")
    overrides = getattr(config, "QUANT_RANGE_STAGE_OVERRIDES", {}) or {}
    override_path = str(getattr(config, "QUANT_RANGE_OVERRIDE_PATH", "") or "")
    if overrides or override_path:
        n_inline = len(overrides) if isinstance(overrides, dict) else "JSON"
        print(f"[QUANT-CALI] Stage overrides enabled: "
              f"inline={n_inline}, path={override_path or '<none>'}")
    collector = start_quantization_range_collection(
        percentile=getattr(config, "QUANT_RANGE_PERCENTILE", 99.0),
        sample_limit=getattr(config, "QUANT_RANGE_SAMPLE_LIMIT", 200_000),
        sample_per_update=getattr(config, "QUANT_RANGE_SAMPLE_PER_UPDATE", 4096),
        symmetric=getattr(config, "QUANT_RANGE_SYMMETRIC", False),
    )
    _register_pipeline_cleanup(lambda: stop_quantization_range_collection())
    return collector


def _finish_quant_range_calibration(output_dir):
    """Freeze calibration ranges and write range artifacts if needed."""
    if not using_uint8_input_quantization():
        return None
    cali_dir = os.path.join(
        output_dir,
        getattr(config, "QUANT_CALIBRATION_SUBDIR", "quant_calibration"),
    )
    payload = stop_quantization_range_collection(cali_dir)
    ranges = payload.get("ranges", {})
    artifacts = payload.get("artifacts", {})
    print(f"[QUANT-CALI] Frozen {len(ranges)} fixed activation ranges")
    for key in ("json", "override_template", "histograms"):
        if key in artifacts:
            print(f"[QUANT-CALI] {key}: {artifacts[key]}")
    return payload


def _should_refit_after_range_freeze():
    """Whether to replay calibration after fixed uint8 ranges are frozen."""
    return (
        using_uint8_input_quantization() and
        bool(getattr(config, "QUANT_REFIT_AFTER_RANGE_FREEZE", True))
    )


def _record_calibration_only_result(source, output_dir, t_start,
                                    recording_info=None, range_payload=None):
    """Write a lightweight results row for calibration-only runs."""
    elapsed = time.time() - t_start
    print(f"\n[MAIN] Calibration-only run complete in {elapsed:.1f}s")
    if range_payload and range_payload.get("ranges"):
        print(f"[MAIN] Quantization ranges: {len(range_payload['ranges'])} stages")
    params = collect_config_params()
    results_dict = {
        "source": source,
        "mode": "calibration_only",
        "elapsed_s": elapsed,
        "n_quant_ranges": len(range_payload.get("ranges", {}))
        if range_payload else 0,
    }
    if recording_info:
        results_dict.update(recording_info)
    xlsx_path = os.path.join(output_dir, "results_log.xlsx")
    append_to_xlsx(params, results_dict, xlsx_path=xlsx_path)
    return results_dict


def _sync_config_from_streamer(streamer):
    """Auto-detect recording parameters from actual data and sync to config.

    Reads sample_rate, n_channels, bit_volts, and preprocessing state from
    the opened streamer (which parsed them from the data file / oebin).
    Updates config module so all downstream code uses correct values.

    Returns
    -------
    dict with keys 'skip_car' (bool) and 'skip_fir' (bool) indicating
    whether CAR and/or bandpass were already applied to the data.
    """
    updates = []

    sr = getattr(streamer, '_sample_rate', None)
    if sr and int(round(sr)) != config.SAMPLE_RATE:
        updates.append(f"  SAMPLE_RATE: {config.SAMPLE_RATE} -> {int(round(sr))}")
        config.SAMPLE_RATE = int(round(sr))
        config.CHUNK_SAMPLES = int(
            config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000.0)
        config.FIR_GROUP_DELAY = (config.FIR_NUM_TAPS - 1) // 2

    bv = getattr(streamer, '_bit_volts', None)
    if bv and bv != config.UV_PER_BIT:
        updates.append(f"  UV_PER_BIT:  {config.UV_PER_BIT} -> {bv}")
        config.UV_PER_BIT = bv

    nch = getattr(streamer, '_n_channels', None)
    if nch and nch != config.N_CHANNELS:
        updates.append(f"  N_CHANNELS:  {config.N_CHANNELS} -> {nch}")
        config.N_CHANNELS = nch

    if updates:
        print("[LOCAL] Config updated from recording data:")
        for u in updates:
            print(u)
    else:
        print("[LOCAL] Recording parameters match config defaults")

    # Detect preprocessing already applied to data, with an explicit override
    # for controlled local-recording comparison experiments.
    detected_car = bool(getattr(streamer, '_has_car', False))
    detected_fir = bool(getattr(streamer, '_has_bandpass', False))
    mode = str(getattr(config, 'LOCAL_PREPROCESSING_MODE', 'auto') or 'auto').lower()
    if mode == 'force_raw':
        skip_car = False
        skip_fir = False
    elif mode == 'skip_all':
        skip_car = True
        skip_fir = True
    elif mode == 'auto':
        skip_car = detected_car
        skip_fir = detected_fir
    else:
        print(f"[LOCAL] WARNING: unknown LOCAL_PREPROCESSING_MODE={mode!r}; "
              "using auto")
        mode = 'auto'
        skip_car = detected_car
        skip_fir = detected_fir
    history = getattr(streamer, '_preprocessing_history', '')

    print(f"[LOCAL] Preprocessing metadata: "
          f"CAR={'YES' if detected_car else 'no'}, "
          f"Bandpass={'YES' if detected_fir else 'no'}")
    if history:
        print(f"[LOCAL] History: {history}")
    print(f"[LOCAL] Preprocessing mode: {mode}")
    print(f"[LOCAL] Pipeline preprocessing: "
          f"CAR={'SKIP' if skip_car else 'RUN'}, "
          f"FIR={'SKIP' if skip_fir else 'RUN'}")
    return {'skip_car': skip_car, 'skip_fir': skip_fir}


def _preprocessing_kwargs(preproc_state=None):
    """Normalize detected preprocessing state for PreprocessingPipeline."""
    state = preproc_state or {}
    return {
        'skip_car': bool(state.get('skip_car', False)),
        'skip_fir': bool(state.get('skip_fir', False)),
    }


def _format_stream_progress(n_batches, max_batches, total_spikes,
                            last_spikes, n_clusters, stream_t0):
    elapsed = max(time.time() - stream_t0, 1e-9)
    sec_per_batch = elapsed / max(n_batches, 1)
    if max_batches and max_batches < 999999:
        remaining = max(max_batches - n_batches, 0)
        eta = remaining * sec_per_batch
        eta_text = f", ETA={eta / 3600:.1f}h"
    else:
        eta_text = ""
    return (
        f"  Batch {n_batches}/{max_batches}: "
        f"last={last_spikes}, total={total_spikes}, clusters={n_clusters}, "
        f"avg={sec_per_batch:.2f}s/batch{eta_text}"
    )


def _whiten_chunks(pipe, raw_chunks):
    """Run raw chunks through calibrated pipeline -> whitened chunks."""
    whitened = []
    for c in raw_chunks:
        whitened.append(pipe.process_chunk(c))
    pipe.fir_filter.reset(raw_chunks[0].shape[1])
    return whitened


def _calibration_chunk_plan(streamer):
    """Return full calibration chunks available from the start of recording."""
    cs = int(config.CHUNK_SAMPLES)
    sr = float(config.SAMPLE_RATE)
    requested = int(float(config.CALIBRATION_DURATION_S) * sr)
    available = int(streamer.total_duration_s * sr)
    n_samples = max(0, min(requested, available))
    n_chunks = n_samples // cs
    full_samples = n_chunks * cs
    if full_samples < n_samples:
        dropped = n_samples - full_samples
        print(f"[CALIB] Dropping trailing partial calibration chunk: "
              f"{dropped} samples ({dropped / sr:.3f}s)")
    return n_chunks, full_samples


def _iter_calibration_chunks(streamer, stop_event=None):
    """Yield calibration chunks from t=0 without allocating the full block."""
    for chunk in streamer.stream_chunks(
            start_s=0.0,
            duration_s=config.CALIBRATION_DURATION_S,
            chunk_samples=config.CHUNK_SAMPLES):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stop requested during calibration")
        yield chunk


def _materialize_calibration_buffers(pipe, streamer, source_label,
                                     n_chunks, n_channels,
                                     stop_event=None):
    """Create disk-backed raw and whitened calibration arrays.

    The returned memmaps are ndarray-compatible, so downstream template,
    clustering, and diagnostic code can use the same full calibration data
    without keeping duplicate multi-GB arrays in RAM.
    """
    if n_chunks <= 0:
        raise ValueError("No full calibration chunks available")

    cs = int(config.CHUNK_SAMPLES)
    n_samples = int(n_chunks * cs)
    n_out = int(pipe.whitening_processor.n_output_channels)
    tmp_dir = tempfile.mkdtemp(prefix="bci_calibration_")
    _register_pipeline_cleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

    raw_path = os.path.join(tmp_dir, "calibration_raw_float32.dat")
    white_path = os.path.join(tmp_dir, "calibration_whitened_float32.dat")
    raw_mm = np.memmap(raw_path, dtype=np.float32, mode="w+",
                       shape=(n_samples, n_channels))
    white_mm = np.memmap(white_path, dtype=np.float32, mode="w+",
                         shape=(n_samples, n_out))

    print(f"[{source_label}] Calibration pass 2: materializing "
          f"{n_chunks} chunks to disk-backed arrays")
    print(f"[{source_label}] Calibration temp: {tmp_dir}")

    pipe.fir_filter.reset(n_channels)
    first_raw = None
    progress_every = max(1, n_chunks // 20)
    n_seen = 0
    for n_seen, raw_chunk in enumerate(
            _iter_calibration_chunks(streamer, stop_event=stop_event),
            start=1):
        if n_seen > n_chunks:
            break
        raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
        s = (n_seen - 1) * cs
        e = s + cs
        if raw_chunk.shape[0] != cs:
            break
        raw_mm[s:e] = raw_chunk
        if first_raw is None:
            first_raw = raw_chunk.copy()
        white_mm[s:e] = pipe.process_chunk(raw_chunk)

        if n_seen % progress_every == 0 or n_seen == n_chunks:
            pct = 100.0 * n_seen / max(n_chunks, 1)
            print(f"[{source_label}] Calibration pass 2: "
                  f"{n_seen}/{n_chunks} chunks ({pct:.1f}%)")

    if n_seen < n_chunks:
        raise RuntimeError(
            f"Calibration ended early: got {n_seen}/{n_chunks} chunks")

    raw_mm.flush()
    white_mm.flush()
    pipe.fir_filter.reset(n_channels)
    print(f"[{source_label}] Whitened calibration: {n_chunks} chunks "
          f"({n_samples / config.SAMPLE_RATE:.1f}s)")
    return {
        "raw": raw_mm,
        "whitened": white_mm,
        "first_raw": first_raw,
        "n_chunks": n_chunks,
        "n_samples": n_samples,
        "tmp_dir": tmp_dir,
    }


def _iter_array_chunks(data, n_chunks, chunk_samples, stop_event=None):
    """Yield fixed-size chunks from an ndarray/memmap calibration buffer."""
    for idx in range(int(n_chunks)):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stop requested during calibration replay")
        s = idx * int(chunk_samples)
        e = s + int(chunk_samples)
        chunk = np.asarray(data[s:e], dtype=np.float32)
        if chunk.shape[0] != int(chunk_samples):
            break
        yield chunk


def _materialize_whitened_from_raw(pipe, raw_data, source_label, pass_label,
                                   n_chunks, n_channels, stop_event=None):
    """Replay raw calibration through a calibrated pipeline into a memmap."""
    if n_chunks <= 0:
        raise ValueError("No full calibration chunks available")

    cs = int(config.CHUNK_SAMPLES)
    n_samples = int(n_chunks * cs)
    n_out = int(pipe.whitening_processor.n_output_channels)
    safe_label = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in str(pass_label)
    ).strip("_") or "replay"
    tmp_dir = tempfile.mkdtemp(prefix=f"bci_calibration_{safe_label}_")
    _register_pipeline_cleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

    white_path = os.path.join(tmp_dir, "calibration_whitened_float32.dat")
    white_mm = np.memmap(white_path, dtype=np.float32, mode="w+",
                         shape=(n_samples, n_out))

    print(f"[{source_label}] {pass_label}: materializing "
          f"{n_chunks} fixed-range chunks")
    print(f"[{source_label}] {pass_label} temp: {tmp_dir}")

    pipe.fir_filter.reset(n_channels)
    first_raw = np.asarray(raw_data[:cs], dtype=np.float32).copy()
    progress_every = max(1, n_chunks // 20)
    n_seen = 0
    for n_seen, raw_chunk in enumerate(
            _iter_array_chunks(raw_data, n_chunks, cs, stop_event=stop_event),
            start=1):
        s = (n_seen - 1) * cs
        e = s + cs
        white_mm[s:e] = pipe.process_chunk(raw_chunk)

        if n_seen % progress_every == 0 or n_seen == n_chunks:
            pct = 100.0 * n_seen / max(n_chunks, 1)
            print(f"[{source_label}] {pass_label}: "
                  f"{n_seen}/{n_chunks} chunks ({pct:.1f}%)")

    if n_seen < n_chunks:
        raise RuntimeError(
            f"Calibration replay ended early: got {n_seen}/{n_chunks} chunks")

    white_mm.flush()
    pipe.fir_filter.reset(n_channels)
    print(f"[{source_label}] {pass_label}: whitened calibration ready "
          f"({n_samples / config.SAMPLE_RATE:.1f}s)")
    return {
        "raw": raw_data,
        "whitened": white_mm,
        "first_raw": first_raw,
        "n_chunks": n_chunks,
        "n_samples": n_samples,
        "tmp_dir": tmp_dir,
    }


def _fit_calibration_parameters(source_label, template_label, output_dir,
                                positions, pipe, calib_data,
                                whitened_calib_all, first_raw, n_total_calib,
                                stop_event=None, save_visuals=True,
                                run_diagnostic=True,
                                diagnostic_required=False,
                                pass_label=None):
    """Fit templates, streaming-visible centers, LDA, and assign radii."""
    if pass_label:
        print(f"\n{'=' * 60}")
        print(f"  {pass_label}")
        print(f"{'=' * 60}")

    coeffs = pipe.fir_filter.raw_coeffs.copy()

    if save_visuals:
        c0 = np.asarray(first_raw, dtype=np.float32)
        car0 = c0 if pipe.skip_car else apply_car(c0)
        if pipe.skip_fir:
            filt0 = car0
        else:
            fir_tmp = FIRFilter(coeffs, precision_bits=config.PRECISION_FIR)
            fir_tmp.reset(positions.shape[0])
            if using_int_accum_quantization():
                car0_dac = car0
            else:
                car0_dac = quantize_signal(car0, config.PRECISION_DAC)
            filt0 = fir_tmp.process_chunk(car0_dac)
        white0 = pipe.whitening_processor.process_chunk(filt0)
        plot_pipeline_stages(
            c0, car0, filt0, white0, channels=[0, 50, 100, 200],
            save_path=os.path.join(output_dir, 'pipeline_stages.png'))
        plot_whitening_matrices(
            pipe.whitening_processor.ops[:4],
            os.path.join(output_dir, 'whitening_matrices.png'))

    print(f"\n{'=' * 60}")
    print(f"  TEMPLATE MATCHING -- {template_label}")
    print(f"{'=' * 60}")

    wTEMP, wPCA, tmpl_source = get_templates(
        whitened_data=whitened_calib_all,
        channel_positions=positions)
    print(f"[MAIN] Template source: {tmpl_source}")
    if save_visuals:
        plot_templates(wTEMP, os.path.join(output_dir, 'templates.png'))

    n_calib = min(config.N_CALIB_CHUNKS, n_total_calib)
    print(f"[{source_label}] Using {n_calib} chunks for calibration")
    n_calib_samples = n_calib * int(config.CHUNK_SAMPLES)
    whitened_all = whitened_calib_all[:n_calib_samples]

    sa = SpatialAggregator(positions, precision_bits=config.PRECISION_SA)

    print(f"\n{'=' * 60}")
    print(f"  SPIKE CLUSTERING")
    print(f"{'=' * 60}")

    clust = SpikeClustering(channel_positions=positions, wPCA=wPCA,
                           wTEMP=wTEMP,
                           pca_precision_bits=config.PRECISION_PCA,
                           assign_precision_bits=config.PRECISION_ASSIGN)
    raw_for_cali = calib_data[:whitened_all.shape[0]]
    clust.calibrate(whitened_chunk=whitened_all, raw_chunk=raw_for_cali,
                    output_dir=output_dir)

    diag = None
    if run_diagnostic:
        try:
            from alignment_diagnostic import run_alignment_diagnostic
            diag = run_alignment_diagnostic(
                whitened_all, clust, wTEMP, sa, positions, output_dir)
        except Exception as exc:
            if diagnostic_required:
                raise
            print(f"[{source_label}] Alignment diagnostic skipped: {exc}")

    return {
        "coeffs": coeffs,
        "pipe": pipe,
        "clust": clust,
        "wTEMP": wTEMP,
        "wPCA": wPCA,
        "sa": sa,
        "whitened_all": whitened_all,
        "diag": diag,
    }


_CALI_CACHE_ARTIFACTS = [
    "pipeline_stages.png",
    "whitening_matrices.png",
    "templates.png",
    "alignment_diagnostic.txt",
    "quant_calibration",
]


def _json_ready(value):
    """Convert nested config values into stable JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_array(arr):
    arr = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _calibration_cache_fingerprint(source_label, streamer, positions,
                                   source_info, n_total_calib):
    ignored_prefixes = ("HARDWARE_EXPORT_",)
    ignored_names = {"MAX_TEST_BATCHES"}
    params = {
        k: v for k, v in collect_config_params().items()
        if k not in ignored_names and
        not any(k.startswith(prefix) for prefix in ignored_prefixes)
    }
    payload = {
        "cache_version": int(getattr(config, "CALI_CACHE_VERSION", 1)),
        "source": source_label,
        "source_info": source_info,
        "sample_rate": float(config.SAMPLE_RATE),
        "chunk_samples": int(config.CHUNK_SAMPLES),
        "calibration_duration_s": float(config.CALIBRATION_DURATION_S),
        "n_total_calib": int(n_total_calib),
        "total_duration_s": float(getattr(streamer, "total_duration_s", 0.0) or 0.0),
        "positions_hash": _hash_array(np.asarray(positions, dtype=np.float32)),
        "config": _json_ready(params),
    }
    blob = json.dumps(_json_ready(payload), sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload


def _calibration_cache_paths(output_dir, fingerprint):
    cache_dir = os.path.join(
        output_dir, str(getattr(config, "CALI_CACHE_SUBDIR", "cali_cache")))
    return {
        "dir": cache_dir,
        "bundle": os.path.join(cache_dir, f"{fingerprint}.pkl.gz"),
        "artifacts": os.path.join(cache_dir, f"{fingerprint}_artifacts"),
        "latest": os.path.join(cache_dir, "latest.json"),
    }


def _copy_calibration_artifacts(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in _CALI_CACHE_ARTIFACTS:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        copied.append(name)
    return copied


def _cacheable_final_fit(final_fit):
    return {
        "coeffs": final_fit["coeffs"],
        "pipe": final_fit["pipe"],
        "clust": final_fit["clust"],
        "wTEMP": final_fit["wTEMP"],
        "wPCA": final_fit["wPCA"],
        "sa": final_fit["sa"],
    }


def _restore_cached_final_fit(state):
    final_fit = dict(state)
    final_fit["whitened_all"] = np.zeros((0, 0), dtype=np.float32)
    final_fit["diag"] = None
    return final_fit


def _load_calibration_cache(output_dir, fingerprint):
    if not bool(getattr(config, "CALI_CACHE_ENABLE", True)):
        return None
    paths = _calibration_cache_paths(output_dir, fingerprint)
    bundle = paths["bundle"]
    if not os.path.exists(bundle):
        print(f"[CALI-CACHE] Miss: {fingerprint[:12]}")
        return None
    try:
        with gzip.open(bundle, "rb") as f:
            payload = pickle.load(f)
        if payload.get("fingerprint") != fingerprint:
            print("[CALI-CACHE] Ignoring cache with mismatched fingerprint")
            return None
        range_payload = payload.get("range_payload") or {"ranges": {}}
        set_quantization_fixed_ranges(range_payload.get("ranges", {}))
        _copy_calibration_artifacts(paths["artifacts"], output_dir)
        print(f"[CALI-CACHE] Hit: {fingerprint[:12]} <- {bundle}")
        return {
            "final_fit": _restore_cached_final_fit(payload["final_fit"]),
            "range_payload": range_payload,
        }
    except Exception as exc:
        print(f"[CALI-CACHE] Ignoring unreadable cache: {exc}")
        return None


def _save_calibration_cache(output_dir, fingerprint, fingerprint_payload,
                            final_fit, range_payload):
    if not bool(getattr(config, "CALI_CACHE_ENABLE", True)):
        return None
    paths = _calibration_cache_paths(output_dir, fingerprint)
    os.makedirs(paths["dir"], exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "range_payload": range_payload,
        "final_fit": _cacheable_final_fit(final_fit),
    }
    tmp_path = paths["bundle"] + ".tmp"
    try:
        with gzip.open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, paths["bundle"])
        copied = _copy_calibration_artifacts(output_dir, paths["artifacts"])
        with open(paths["latest"], "w", encoding="utf-8") as f:
            json.dump({
                "fingerprint": fingerprint,
                "bundle": paths["bundle"],
                "artifacts": paths["artifacts"],
                "copied_artifacts": copied,
                "fingerprint_payload": _json_ready(fingerprint_payload),
            }, f, indent=2)
        print(f"[CALI-CACHE] Saved: {fingerprint[:12]} -> {paths['bundle']}")
        return paths["bundle"]
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        print(f"[CALI-CACHE] Save skipped: {exc}")
        return None


def _run_calibration_phase(source_label, template_label, output_dir, streamer,
                           positions, source_info, stop_event=None,
                           diagnostic_required=False, preproc_state=None):
    """Run or restore calibration, returning final pipeline state."""
    cs = int(config.CHUNK_SAMPLES)
    preproc_kwargs = _preprocessing_kwargs(preproc_state)
    source_info = dict(source_info or {})
    source_info["preprocessing_state"] = dict(preproc_kwargs)
    n_total_calib, n_calib_samples_total = _calibration_chunk_plan(streamer)
    fingerprint, fingerprint_payload = _calibration_cache_fingerprint(
        source_label, streamer, positions, source_info, n_total_calib)

    cached = _load_calibration_cache(output_dir, fingerprint)
    if cached is not None:
        stop_quantization_range_collection()
        set_quantization_fixed_ranges(
            cached["range_payload"].get("ranges", {}))
        return cached["final_fit"], cached["range_payload"], n_total_calib

    pipe = PreprocessingPipeline(positions, **preproc_kwargs)
    print(f"[{source_label}] Calibration: {n_total_calib} chunks "
          f"({n_calib_samples_total / config.SAMPLE_RATE:.1f}s)")
    pipe.calibrate_from_chunks(
        _iter_calibration_chunks(streamer, stop_event=stop_event),
        n_channels=positions.shape[0],
        n_chunks=n_total_calib,
        progress_label=source_label)

    cali_buffers = _materialize_calibration_buffers(
        pipe, streamer, source_label, n_total_calib, positions.shape[0],
        stop_event=stop_event)
    calib_data = cali_buffers["raw"]
    whitened_calib_all = cali_buffers["whitened"]

    if _should_refit_after_range_freeze():
        _fit_calibration_parameters(
            source_label, template_label, output_dir, positions, pipe,
            calib_data, whitened_calib_all, cali_buffers["first_raw"],
            n_total_calib, stop_event=stop_event, save_visuals=False,
            run_diagnostic=False,
            pass_label="RANGE COLLECTION CALIBRATION PASS")

        range_payload = _finish_quant_range_calibration(output_dir)

        print(f"\n{'=' * 60}")
        print("  FIXED-RANGE PARAMETER REFIT")
        print(f"{'=' * 60}")
        print("[QUANT-CALI] Replaying calibration with frozen uint8 ranges")
        reset_quantization_diagnostics()

        pipe = PreprocessingPipeline(positions, **preproc_kwargs)
        pipe.calibrate_from_chunks(
            _iter_array_chunks(
                calib_data, n_total_calib, cs, stop_event=stop_event),
            n_channels=positions.shape[0],
            n_chunks=n_total_calib,
            progress_label=f"{source_label}-FIXED")
        fixed_buffers = _materialize_whitened_from_raw(
            pipe, calib_data, source_label, "fixed-range pass",
            n_total_calib, positions.shape[0], stop_event=stop_event)
        whitened_calib_all = fixed_buffers["whitened"]
        final_fit = _fit_calibration_parameters(
            source_label, template_label, output_dir, positions, pipe,
            calib_data, whitened_calib_all, fixed_buffers["first_raw"],
            n_total_calib, stop_event=stop_event, save_visuals=True,
            run_diagnostic=True, diagnostic_required=diagnostic_required,
            pass_label="FINAL FIXED-RANGE CALIBRATION PASS")
    else:
        final_fit = _fit_calibration_parameters(
            source_label, template_label, output_dir, positions, pipe,
            calib_data, whitened_calib_all, cali_buffers["first_raw"],
            n_total_calib, stop_event=stop_event, save_visuals=True,
            run_diagnostic=True, diagnostic_required=diagnostic_required)
        range_payload = _finish_quant_range_calibration(output_dir)

    final_fit["clust"]._skip_fir = bool(preproc_kwargs["skip_fir"])

    _save_calibration_cache(
        output_dir, fingerprint, fingerprint_payload, final_fit, range_payload)
    return final_fit, range_payload, n_total_calib


TENSOR_FLOAT = 1
TENSOR_UINT8 = 2
TENSOR_INT8 = 3
TENSOR_INT32 = 6
ONNX_IR_VERSION = 7
ONNX_OPSET = 13
_HW_EXPORT_OPTIONS = {}


def _hw_export_options():
    return {
        "subdir": getattr(config, "HARDWARE_EXPORT_SUBDIR", "hardware_export"),
        "batch_index": int(getattr(config, "HARDWARE_EXPORT_BATCH_INDEX", 0)),
        "stages": getattr(config, "HARDWARE_EXPORT_STAGES", None),
        "max_vmm_rows": int(getattr(config, "HARDWARE_EXPORT_MAX_VMM_ROWS", 20000)),
        "onnx": bool(getattr(config, "HARDWARE_EXPORT_ONNX", True)),
        "save_plots": bool(getattr(config, "HARDWARE_EXPORT_PLOTS", True)),
        "split_models": bool(getattr(config, "HARDWARE_EXPORT_SPLIT_MODELS", True)),
        "plot_max_points": int(getattr(config, "HARDWARE_EXPORT_PLOT_MAX_POINTS", 50000)),
    }


def _hw_export_option(name, default):
    return _HW_EXPORT_OPTIONS.get(name, default)


def _varint(value):
    value = int(value)
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _key(field_no, wire_type):
    return _varint((int(field_no) << 3) | int(wire_type))


def _field_varint(field_no, value):
    return _key(field_no, 0) + _varint(value)


def _field_bytes(field_no, payload):
    return _key(field_no, 2) + _varint(len(payload)) + payload


def _field_string(field_no, value):
    return _field_bytes(field_no, str(value).encode("utf-8"))


def _field_message(field_no, payload):
    return _field_bytes(field_no, payload)


def _tensor_shape_proto(shape):
    msg = bytearray()
    for dim in shape:
        d = bytearray()
        if dim is None or isinstance(dim, str):
            d += _field_string(2, "N" if dim is None else dim)
        else:
            d += _field_varint(1, int(dim))
        msg += _field_message(1, bytes(d))
    return bytes(msg)


def _tensor_type_proto(elem_type, shape):
    tensor = bytearray()
    tensor += _field_varint(1, int(elem_type))
    tensor += _field_message(2, _tensor_shape_proto(shape))
    return _field_message(1, bytes(tensor))


def _value_info_proto(name, elem_type, shape):
    msg = bytearray()
    msg += _field_string(1, name)
    msg += _field_message(2, _tensor_type_proto(elem_type, shape))
    return bytes(msg)


def _tensor_proto(name, array, elem_type):
    arr = np.asarray(array)
    msg = bytearray()
    for dim in arr.shape:
        msg += _field_varint(1, int(dim))
    msg += _field_varint(2, int(elem_type))
    msg += _field_string(8, name)
    msg += _field_bytes(9, np.ascontiguousarray(arr).tobytes())
    return bytes(msg)


def _attribute_int_proto(name, value):
    msg = bytearray()
    msg += _field_string(1, name)
    msg += _field_varint(3, int(value))
    msg += _field_varint(20, 2)
    return bytes(msg)


def _node_proto(op_type, inputs, outputs, name, attributes=None):
    msg = bytearray()
    for item in inputs:
        msg += _field_string(1, item)
    for item in outputs:
        msg += _field_string(2, item)
    msg += _field_string(3, name)
    msg += _field_string(4, op_type)
    for attr in attributes or []:
        msg += _field_message(5, attr)
    return bytes(msg)


def _opset_proto(domain, version):
    msg = bytearray()
    if domain:
        msg += _field_string(1, domain)
    msg += _field_varint(2, int(version))
    return bytes(msg)


def _write_matmul_integer_onnx(path, model_name, kernels,
                               input_dtype=TENSOR_UINT8):
    graph = bytearray()
    graph += _field_string(2, model_name)
    nodes, initializers, inputs, outputs = [], [], [], []
    for kernel in kernels:
        name = kernel["name"]
        input_name = kernel.get("input_name", f"{name}_input")
        output_name = kernel.get("output_name", f"{name}_acc_int32")
        weight_q = np.asarray(kernel["weight_q"], dtype=np.int8)
        if weight_q.ndim != 2:
            raise ValueError(f"{name}: weight_q must be 2-D")
        out_dim, in_dim = weight_q.shape
        weight_name = f"{name}_weight_int8_t"
        a_zp_name = f"{name}_input_zero_point"
        b_zp_name = f"{name}_weight_zero_point"

        initializers.append(_tensor_proto(
            weight_name, weight_q.T.astype(np.int8), TENSOR_INT8))
        if input_dtype == TENSOR_UINT8:
            zp = np.asarray(kernel.get("input_zero_point", 0), dtype=np.uint8)
            initializers.append(_tensor_proto(a_zp_name, zp, TENSOR_UINT8))
        else:
            zp = np.asarray(kernel.get("input_zero_point", 0), dtype=np.int8)
            initializers.append(_tensor_proto(a_zp_name, zp, TENSOR_INT8))
        initializers.append(_tensor_proto(
            b_zp_name, np.asarray(0, dtype=np.int8), TENSOR_INT8))

        inputs.append(_value_info_proto(input_name, input_dtype, [None, in_dim]))
        outputs.append(_value_info_proto(output_name, TENSOR_INT32, [None, out_dim]))
        nodes.append(_node_proto(
            "MatMulInteger", [input_name, weight_name, a_zp_name, b_zp_name],
            [output_name], f"{name}_matmul_integer"))

    for node in nodes:
        graph += _field_message(1, node)
    for init in initializers:
        graph += _field_message(5, init)
    for vi in inputs:
        graph += _field_message(11, vi)
    for vi in outputs:
        graph += _field_message(12, vi)

    model = bytearray()
    model += _field_varint(1, ONNX_IR_VERSION)
    model += _field_string(3, "GUI-BCI main hardware_export")
    model += _field_message(7, bytes(graph))
    model += _field_message(8, _opset_proto("", ONNX_OPSET))
    with open(path, "wb") as f:
        f.write(bytes(model))


def _hw_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _hw_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hw_json_safe(v) for v in value]
    return value


def _hw_write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_hw_json_safe(payload), f, indent=2)


def _hw_ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hw_stage_dir(root, name):
    return _hw_ensure_dir(Path(root) / name)


def _hw_save_array(path, array):
    arr = np.asarray(array)
    np.save(path, arr)
    return {"file": Path(path).name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def _hw_rel_file(from_dir, path):
    try:
        return os.path.relpath(path, from_dir)
    except ValueError:
        return str(path)


def _hw_row_indices(n_rows, max_rows):
    n_rows = int(n_rows)
    if max_rows and max_rows > 0 and n_rows > max_rows:
        idx = np.linspace(0, n_rows - 1, int(max_rows), dtype=np.int64)
        return idx, {
            "selection": "linspace",
            "full_rows": n_rows,
            "exported_rows": int(max_rows),
            "indices_file": None,
        }
    return None, {
        "selection": "all",
        "full_rows": n_rows,
        "exported_rows": n_rows,
        "indices_file": None,
    }


def _hw_selected_rows(x, max_rows):
    x = np.asarray(x)
    n = int(x.shape[0]) if x.ndim > 0 else 1
    idx, meta = _hw_row_indices(n, max_rows)
    if idx is not None:
        return x[idx], meta
    return x, meta


def _hw_window_rows_by_index(data, window, n_out, max_rows):
    data = np.asarray(data, dtype=np.float32)
    n_ch = int(data.shape[1])
    full_rows = int(n_out) * n_ch
    idx, meta = _hw_row_indices(full_rows, max_rows)
    if idx is None:
        windows = np.lib.stride_tricks.sliding_window_view(
            data, int(window), axis=0)[:int(n_out)]
        return windows.reshape(-1, int(window)), meta
    rows = np.empty((len(idx), int(window)), dtype=np.float32)
    t_idx = idx // n_ch
    ch_idx = idx % n_ch
    for out_i, (t, ch) in enumerate(zip(t_idx, ch_idx)):
        rows[out_i] = data[int(t):int(t) + int(window), int(ch)]
    return rows, meta


def _hw_activation_quant(x, op_name, x_bits):
    if using_uint8_input_quantization():
        value_range = _fixed_range_for(f"{op_name}.input")
        q, scale, zero_point, stats = _quantize_uint8_codes_with_stats(
            x, int(x_bits), value_range=value_range)
        centered = q.astype(np.int32) - int(zero_point or 0)
        dtype = "uint8"
        tensor_type = TENSOR_UINT8
    else:
        q, scale, stats = _quantize_signed_codes_with_stats(x, int(x_bits))
        zero_point = 0
        centered = q.astype(np.int32)
        dtype = f"int{x_bits}"
        tensor_type = TENSOR_INT8
    return {
        "q": q,
        "centered": centered,
        "scale": float(scale or 0.0),
        "zero_point": int(zero_point or 0),
        "stats": stats,
        "dtype": dtype,
        "tensor_type": tensor_type,
    }


def _hw_weight_quant(w, w_bits):
    q, scale, stats = _quantize_signed_codes_with_stats(w, int(w_bits))
    return {
        "q": q.astype(np.int8 if int(w_bits) <= 8 else np.int32),
        "scale": float(scale or 0.0),
        "stats": stats,
        "dtype": f"int{w_bits}",
    }


def _hw_adc_quant(out, op_name):
    out = np.asarray(out, dtype=np.float32)
    adc_name = f"{op_name}.adc"
    value_range = _fixed_range_for(adc_name)
    q, scale, zero_point, stats = _quantize_uint8_codes_with_stats(
        out, int(config.PRECISION_ADC), value_range=value_range)
    deq = ((q.astype(np.float32) - float(zero_point or 0)) *
           float(scale or 0.0)).astype(np.float32)
    return {
        "name": adc_name,
        "q": q.astype(np.uint8),
        "dequantized": deq,
        "scale": float(scale or 0.0),
        "zero_point": int(zero_point or 0),
        "stats": stats,
    }


def _hw_vmm_reference(x, w, op_name, x_bits, w_bits):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    xq = _hw_activation_quant(x, op_name, x_bits)
    wq = _hw_weight_quant(w, w_bits)
    acc = xq["centered"] @ wq["q"].astype(np.int32).T
    out = (acc.astype(np.float32) * xq["scale"] * wq["scale"]).astype(np.float32)
    adc = _hw_adc_quant(out, op_name)
    return {"input": xq, "weight": wq, "acc": acc.astype(np.int32),
            "output_pre_adc": out, "output_adc": adc}


def _hw_save_vmm_reference(stage_path, prefix, x, w, op_name,
                           x_bits, w_bits, max_rows,
                           row_meta_override=None):
    if row_meta_override is None:
        x_sel, row_meta = _hw_selected_rows(np.asarray(x, dtype=np.float32), max_rows)
        full_rows = int(np.asarray(x).shape[0])
    else:
        x_sel = np.asarray(x, dtype=np.float32)
        row_meta = row_meta_override
        full_rows = int(row_meta.get("full_rows", x_sel.shape[0]))
    ref = _hw_vmm_reference(x_sel, w, op_name, x_bits, w_bits)
    files = {
        "input_float": _hw_save_array(stage_path / f"{prefix}_input_float.npy", x_sel),
        "input_quant": _hw_save_array(
            stage_path / f"{prefix}_input_{ref['input']['dtype']}.npy",
            ref["input"]["q"]),
        "weight_float": _hw_save_array(
            stage_path / f"{prefix}_weight_float.npy",
            np.asarray(w, dtype=np.float32)),
        "weight_quant": _hw_save_array(
            stage_path / f"{prefix}_weight_{ref['weight']['dtype']}.npy",
            ref["weight"]["q"]),
        "acc_int32": _hw_save_array(
            stage_path / f"{prefix}_acc_int32.npy", ref["acc"]),
        "output_float_pre_adc": _hw_save_array(
            stage_path / f"{prefix}_output_float_pre_adc.npy",
            ref["output_pre_adc"]),
        "output_uint8_adc": _hw_save_array(
            stage_path / f"{prefix}_output_uint8_adc.npy",
            ref["output_adc"]["q"]),
        "output_float_adc_dequant": _hw_save_array(
            stage_path / f"{prefix}_output_float_adc_dequant.npy",
            ref["output_adc"]["dequantized"]),
    }
    meta = {
        "name": prefix,
        "op_name": op_name,
        "shape": {
            "input_rows": full_rows,
            "input_dim": int(np.asarray(x).shape[1]),
            "output_dim": int(np.asarray(w).shape[0]),
            "weight_shape": list(np.asarray(w).shape),
        },
        "row_selection": row_meta,
        "input_quant": {
            "dtype": ref["input"]["dtype"],
            "scale": ref["input"]["scale"],
            "zero_point": ref["input"]["zero_point"],
            "stats": ref["input"]["stats"],
        },
        "weight_quant": {
            "dtype": ref["weight"]["dtype"],
            "scale": ref["weight"]["scale"],
            "zero_point": 0,
            "stats": ref["weight"]["stats"],
        },
        "output_quant": {
            "dtype": "uint8",
            "scale": ref["output_adc"]["scale"],
            "zero_point": ref["output_adc"]["zero_point"],
            "stats": ref["output_adc"]["stats"],
            "source": ref["output_adc"]["name"],
        },
        "files": files,
    }
    onnx_kernel = {
        "name": prefix.replace(".", "_"),
        "input_name": f"{prefix.replace('.', '_')}_input",
        "output_name": f"{prefix.replace('.', '_')}_acc_int32",
        "weight_q": ref["weight"]["q"].astype(np.int8),
        "input_scale": ref["input"]["scale"],
        "input_zero_point": ref["input"]["zero_point"],
        "weight_scale": ref["weight"]["scale"],
        "input_tensor_type": ref["input"]["tensor_type"],
    }
    return meta, onnx_kernel


def _hw_plot_stage_preview(stage_path, name, arrays):
    if not bool(_hw_export_option("save_plots", True)):
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return
    max_points = int(_hw_export_option("plot_max_points", 50000))
    fig, axes = plt.subplots(len(arrays), 1,
                             figsize=(8, max(2.4, 2.2 * len(arrays))))
    if len(arrays) == 1:
        axes = [axes]
    for ax, (label, arr) in zip(axes, arrays.items()):
        vals = np.asarray(arr, dtype=np.float32).ravel()
        n_total = int(vals.size)
        if vals.size:
            vals = vals[np.isfinite(vals)]
        if vals.size > max_points > 0:
            idx = np.linspace(0, vals.size - 1, max_points, dtype=np.int64)
            vals = vals[idx]
        if vals.size:
            ax.hist(vals, bins=100, color="#4C78A8", alpha=0.82)
        else:
            ax.text(0.5, 0.5, "no finite values",
                    ha="center", va="center", transform=ax.transAxes)
        skipped = n_total - int(vals.size)
        suffix = f" (finite={int(vals.size)}, skipped={skipped})" if skipped else ""
        ax.set_title(f"{label}{suffix}")
        ax.grid(True, alpha=0.2)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(stage_path / "preview.png", dpi=140)
    plt.close(fig)


def _hw_write_onnx_if_enabled(stage_path, stage_name, kernels):
    if not bool(_hw_export_option("onnx", True)):
        return {"enabled": False}
    if not kernels:
        return {"enabled": True, "file": None, "reason": "no kernels"}
    input_type = kernels[0].get("input_tensor_type", TENSOR_UINT8)
    onnx_kernels = []
    for kernel in kernels:
        k = dict(kernel)
        k.pop("input_tensor_type", None)
        onnx_kernels.append(k)
    path = stage_path / f"{stage_name}.onnx"
    _write_matmul_integer_onnx(path, stage_name, onnx_kernels, input_dtype=input_type)
    return {
        "enabled": True,
        "file": path.name,
        "format": "uint8 input + int8 weight -> MatMulInteger -> int32 accumulator",
        "input": "quantized activation codes; routing described in manifest.json",
        "output": (
            "raw int32 accumulator; host PC performs dequantization, "
            "thresholding, routing, and next-stage input quantization"
        ),
    }


def _hw_relocate_onnx(stage_path, model_stage_path, onnx_meta):
    if not onnx_meta.get("enabled") or not onnx_meta.get("file"):
        return onnx_meta
    src = stage_path / onnx_meta["file"]
    dst = model_stage_path / onnx_meta["file"]
    if src.exists():
        _hw_ensure_dir(model_stage_path)
        os.replace(src, dst)
    updated = dict(onnx_meta)
    updated["file"] = _hw_rel_file(stage_path, dst)
    updated["model_file"] = dst.name
    return updated


def _hw_split_model_files(stage_name, stage_path, model_stage_path, manifest):
    _hw_ensure_dir(model_stage_path)
    model_manifest = {
        "stage": manifest.get("stage", stage_name),
        "description": manifest.get("description"),
        "mode": manifest.get("mode"),
        "routing": manifest.get("routing"),
        "vmm": manifest.get("vmm", []),
        "onnx": _hw_relocate_onnx(
            stage_path, model_stage_path, manifest.get("onnx", {})),
        "source_batch_stage_dir": str(stage_path),
    }
    model_manifest = {k: v for k, v in model_manifest.items() if v is not None}
    _hw_write_json(model_stage_path / "manifest.json", model_manifest)
    manifest = dict(manifest)
    manifest["model"] = {
        "dir": _hw_rel_file(stage_path, model_stage_path),
        "manifest": _hw_rel_file(stage_path, model_stage_path / "manifest.json"),
    }
    return manifest


def _hw_stage_enabled(name):
    stages = _hw_export_option("stages", None)
    if not stages:
        return True
    if isinstance(stages, str):
        stages = [s.strip() for s in stages.split(",") if s.strip()]
    return name in set(stages)


def _hw_export_raw(stage_path, raw_chunk):
    manifest = {
        "stage": "raw",
        "description": "Original 100 ms raw batch converted to uV.",
        "files": {
            "raw_float": _hw_save_array(stage_path / "raw_float.npy", raw_chunk),
        },
    }
    _hw_plot_stage_preview(stage_path, "raw", {"raw_float": raw_chunk})
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_export_fir(stage_path, pipe, car_data, filtered, max_rows):
    n_taps = int(pipe.fir_filter.num_taps)
    n_ch = int(car_data.shape[1])
    overlap = np.zeros((n_taps - 1, n_ch), dtype=np.float32)
    extended = np.vstack([overlap, car_data.astype(np.float32)])
    vmm_rows, row_meta = _hw_window_rows_by_index(
        extended, n_taps, car_data.shape[0], max_rows)
    weights = pipe.fir_filter.coeffs[::-1][np.newaxis, :].astype(np.float32)
    vmm_meta, onnx_kernel = _hw_save_vmm_reference(
        stage_path, "fir", vmm_rows, weights, "FIR",
        config.PRECISION_DAC, config.PRECISION_FIR, 0,
        row_meta_override=row_meta)
    manifest = {
        "stage": "fir",
        "logical_input": _hw_save_array(
            stage_path / "stage_input_car_float.npy", car_data),
        "logical_output": _hw_save_array(
            stage_path / "stage_output_fir_float.npy", filtered),
        "routing": {
            "window_taps": n_taps,
            "row_order": "for t in samples: for ch in channels: extended[t:t+taps, ch]",
            "overlap": "first exported batch uses zero FIR history, matching streaming reset",
        },
        "vmm": [vmm_meta],
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(stage_path, "fir", [onnx_kernel])
    _hw_plot_stage_preview(stage_path, "fir", {
        "input_car": car_data,
        "output_fir": filtered,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_export_whitening(stage_path, pipe, filtered, whitened, max_rows):
    vmm_entries = []
    onnx_kernels = []
    for op in pipe.whitening_processor.ops:
        gid = int(op["group_id"])
        key = pipe.whitening_processor._W_key
        x = filtered[:, op["input_channels"]]
        w = op[key].astype(np.float32)
        prefix = f"group_{gid:02d}"
        vmm_meta, onnx_kernel = _hw_save_vmm_reference(
            stage_path, prefix, x, w, "WHITEN",
            config.PRECISION_DAC, config.PRECISION_WHITEN, max_rows)
        vmm_meta["routing"] = {
            "input_channels": op["input_channels"].astype(int).tolist(),
            "output_channels": op["output_channels"].astype(int).tolist(),
            "output_indices_in_group": op["output_indices_in_group"].astype(int).tolist(),
        }
        vmm_entries.append(vmm_meta)
        onnx_kernels.append(onnx_kernel)
    manifest = {
        "stage": "whitening",
        "logical_input": _hw_save_array(
            stage_path / "stage_input_fir_float.npy", filtered),
        "logical_output": _hw_save_array(
            stage_path / "stage_output_whitened_float.npy", whitened),
        "routing": {
            "type": "grouped local ZCA",
            "n_groups": len(pipe.whitening_processor.ops),
            "note": "Each ONNX input is a packed [T, 32] group. Manifest maps raw channels to each group and final output order.",
        },
        "vmm": vmm_entries,
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(stage_path, "whitening", onnx_kernels)
    _hw_plot_stage_preview(stage_path, "whitening", {
        "input_fir": filtered,
        "output_whitened": whitened,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_export_template_matching(stage_path, whitened, wtemp, b_scores,
                                 positions, max_rows):
    nt = int(wtemp.shape[1])
    stride = int(config.MATCH_STRIDE)
    t_out = int(b_scores.shape[0])
    if stride == 1:
        vmm_rows, row_meta = _hw_window_rows_by_index(whitened, nt, t_out, max_rows)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(
            whitened, nt, axis=0)[::stride][:t_out]
        vmm_rows, row_meta = _hw_selected_rows(windows.reshape(-1, nt), max_rows)
    vmm_meta, onnx_kernel = _hw_save_vmm_reference(
        stage_path, "template_matching", vmm_rows, wtemp.astype(np.float32),
        "MATCH", config.PRECISION_DAC, config.PRECISION_MATCH, 0,
        row_meta_override=row_meta)
    manifest = {
        "stage": "template_matching",
        "logical_input": _hw_save_array(
            stage_path / "stage_input_whitened_float.npy", whitened),
        "logical_output": _hw_save_array(
            stage_path / "stage_output_B_float.npy", b_scores),
        "positions": _hw_save_array(stage_path / "score_positions.npy", positions),
        "routing": {
            "window_taps": nt,
            "stride": stride,
            "row_order": "for t in valid score positions: for ch in channels: whitened[t-half:t+half+1, ch]",
        },
        "vmm": [vmm_meta],
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(
        stage_path, "template_matching", [onnx_kernel])
    _hw_plot_stage_preview(stage_path, "template_matching", {
        "input_whitened": whitened,
        "output_B": b_scores,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_sa_vmm_rows(sa, b_scores, candidates):
    rows = []
    for t, ch in candidates.astype(int):
        nb_idx = sa.neighbour_idx[ch]
        b = b_scores[t, nb_idx, :]
        rows.append(b.T.astype(np.float32))
    if not rows:
        return np.zeros((0, int(sa.n_nearest)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)


def _hw_save_sa_combined_reference(stage_path, rows, candidates, weights, max_rows):
    rows = np.asarray(rows, dtype=np.float32)
    row_channels = np.repeat(candidates[:, 1].astype(np.int32),
                             config.MATCH_N_TEMPLATES) \
        if len(candidates) else np.zeros(0, dtype=np.int32)
    if len(rows) != len(row_channels):
        row_channels = np.zeros(len(rows), dtype=np.int32)
    if max_rows and max_rows > 0 and len(rows) > max_rows:
        idx = np.linspace(0, len(rows) - 1, int(max_rows), dtype=np.int64)
        rows_sel = rows[idx]
        ch_sel = row_channels[idx]
        row_meta = {
            "selection": "linspace",
            "full_rows": int(len(rows)),
            "exported_rows": int(max_rows),
            "indices_file": "sa_sparse_rows_indices.npy",
        }
        np.save(stage_path / "sa_sparse_rows_indices.npy", idx)
    else:
        rows_sel = rows
        ch_sel = row_channels
        row_meta = {
            "selection": "all",
            "full_rows": int(len(rows)),
            "exported_rows": int(len(rows)),
            "indices_file": None,
        }
    xq = _hw_activation_quant(rows_sel, "SA.sparse", config.PRECISION_DAC)
    all_wq, all_w_scale = [], []
    for ch in range(weights.shape[0]):
        q = _hw_weight_quant(weights[ch], config.PRECISION_SA)
        all_wq.append(q["q"].astype(np.int8))
        all_w_scale.append(q["scale"])
    all_wq = np.asarray(all_wq, dtype=np.int8)
    all_w_scale = np.asarray(all_w_scale, dtype=np.float32)
    acc = np.zeros((len(rows_sel), weights.shape[1]), dtype=np.int32)
    out = np.zeros((len(rows_sel), weights.shape[1]), dtype=np.float32)
    for i, ch in enumerate(ch_sel.astype(int)):
        wq = all_wq[ch].astype(np.int32)
        acc[i] = xq["centered"][i] @ wq.T
        out[i] = acc[i].astype(np.float32) * xq["scale"] * all_w_scale[ch]
    adc = _hw_adc_quant(out, "SA.sparse")
    files = {
        "input_float": _hw_save_array(
            stage_path / "sa_sparse_rows_input_float.npy", rows_sel),
        "input_quant": _hw_save_array(
            stage_path / f"sa_sparse_rows_input_{xq['dtype']}.npy", xq["q"]),
        "row_channels": _hw_save_array(
            stage_path / "sa_sparse_rows_channels.npy", ch_sel.astype(np.int32)),
        "weight_int": _hw_save_array(stage_path / "sa_weights_int.npy", all_wq),
        "weight_scale": _hw_save_array(
            stage_path / "sa_weight_scales.npy", all_w_scale),
        "acc_int32": _hw_save_array(
            stage_path / "sa_sparse_rows_acc_int32.npy", acc),
        "output_float_pre_adc": _hw_save_array(
            stage_path / "sa_sparse_rows_output_float_pre_adc.npy", out),
        "output_uint8_adc": _hw_save_array(
            stage_path / "sa_sparse_rows_output_uint8_adc.npy", adc["q"]),
        "output_float_adc_dequant": _hw_save_array(
            stage_path / "sa_sparse_rows_output_float_adc_dequant.npy",
            adc["dequantized"]),
    }
    return {
        "name": "sa_sparse_rows",
        "op_name": "SA.sparse",
        "shape": {
            "input_rows": int(len(rows)),
            "input_dim": int(rows.shape[1]) if rows.ndim == 2 else 0,
            "output_dim": int(weights.shape[1]),
            "weight_shape": list(weights.shape),
        },
        "row_selection": row_meta,
        "input_quant": {
            "dtype": xq["dtype"],
            "scale": xq["scale"],
            "zero_point": xq["zero_point"],
            "stats": xq["stats"],
        },
        "weight_quant": {
            "dtype": f"int{config.PRECISION_SA}",
            "zero_point": 0,
            "note": "Per-channel scales are stored in sa_weight_scales.npy.",
        },
        "output_quant": {
            "dtype": "uint8",
            "scale": adc["scale"],
            "zero_point": adc["zero_point"],
            "stats": adc["stats"],
            "source": adc["name"],
        },
        "files": files,
    }


def _hw_export_spatial_aggregation(stage_path, sa, b_scores, candidates,
                                   sa_events, max_rows):
    rows = _hw_sa_vmm_rows(sa, b_scores, candidates)
    weights = sa.G.astype(np.float32)
    np.save(stage_path / "sa_weights_float.npy", weights)
    np.save(stage_path / "candidates_t_ch.npy", candidates.astype(np.int32))
    for key, value in sa_events.items():
        np.save(stage_path / f"events_{key}.npy", np.asarray(value))
    onnx_kernels = []
    vmm_entries = []
    for ch in range(int(sa.n_ch)):
        w = weights[ch]
        q = _hw_weight_quant(w, config.PRECISION_SA)
        sample = rows[:1] if len(rows) else np.zeros(
            (1, int(sa.n_nearest)), dtype=np.float32)
        aq = _hw_activation_quant(sample, "SA.sparse", config.PRECISION_DAC)
        onnx_kernels.append({
            "name": f"channel_{ch:03d}",
            "input_name": f"channel_{ch:03d}_input",
            "output_name": f"channel_{ch:03d}_acc_int32",
            "weight_q": q["q"].astype(np.int8),
            "input_scale": aq["scale"],
            "input_zero_point": aq["zero_point"],
            "weight_scale": q["scale"],
            "input_tensor_type": aq["tensor_type"],
        })
    if len(rows):
        vmm_meta = _hw_save_sa_combined_reference(
            stage_path, rows, candidates, weights, max_rows)
        vmm_meta["note"] = (
            "Combined SA VMM input rows. Rows are grouped by candidates; "
            "each row uses the weight matrix of that candidate channel.")
        vmm_entries.append(vmm_meta)
    manifest = {
        "stage": "spatial_aggregation",
        "mode": "dense" if bool(getattr(config, "SA_DENSE", False)) else "sparse",
        "logical_input": _hw_save_array(stage_path / "stage_input_B_float.npy", b_scores),
        "candidate_count": int(len(candidates)),
        "event_count": int(len(sa_events.get("times", []))),
        "routing": {
            "input_per_candidate": "[n_templates, n_nearest] from B[t, neighbour_idx[ch], :].T",
            "weight_per_channel": "sa.G[ch] has shape [n_scales, n_nearest]",
            "neighbour_idx": _hw_save_array(stage_path / "neighbour_idx.npy", sa.neighbour_idx),
        },
        "vmm": vmm_entries,
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(
        stage_path, "spatial_aggregation", onnx_kernels)
    _hw_plot_stage_preview(stage_path, "spatial_aggregation", {
        "input_B": b_scores,
        "sa_event_amplitudes": np.asarray(sa_events.get("amplitudes", []),
                                          dtype=np.float32),
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_build_tpca_rows(whitened, spike_times, spike_channels, clust):
    n_nearby = int(config.CLUST_N_NEARBY)
    nt = int(clust.temporal_pca.basis.shape[1])
    hw = (nt - 1) // 2
    if len(spike_times) == 0:
        return np.zeros((0, nt), dtype=np.float32)
    nearby_idx = clust._nearby_idx
    channels_per_spike = nearby_idx[np.asarray(spike_channels, dtype=np.int64)]
    snippets = np.zeros((len(spike_times), n_nearby, nt), dtype=np.float32)
    t_starts = np.asarray(spike_times, dtype=np.int64) - hw
    t_ends = np.asarray(spike_times, dtype=np.int64) + hw + 1
    valid = (t_starts >= 0) & (t_ends <= whitened.shape[0])
    for i in np.nonzero(valid)[0]:
        time_idx = int(t_starts[i]) + np.arange(nt)
        for j, ch in enumerate(channels_per_spike[i]):
            snippets[i, j] = whitened[time_idx, int(ch)]
    for i in np.nonzero(~valid)[0]:
        ts = max(0, int(t_starts[i]))
        te = min(whitened.shape[0], int(t_ends[i]))
        offset = ts - int(t_starts[i])
        for j, ch in enumerate(channels_per_spike[i]):
            snippets[i, j, offset:offset + (te - ts)] = whitened[ts:te, int(ch)]
    flat = snippets.reshape(-1, nt)
    return (flat - clust.temporal_pca.mean_).astype(np.float32)


def _hw_export_tpca(stage_path, clust, whitened, sa_events, pca_features,
                    max_rows):
    rows = _hw_build_tpca_rows(
        whitened, sa_events.get("times", np.array([], dtype=np.int64)),
        sa_events.get("channels", np.array([], dtype=np.int32)), clust)
    weights = clust.temporal_pca.basis.astype(np.float32)
    vmm_meta, onnx_kernel = _hw_save_vmm_reference(
        stage_path, "tpca", rows, weights, "TPCA",
        config.PRECISION_DAC, config.PRECISION_PCA, max_rows)
    manifest = {
        "stage": "tpca",
        "logical_input": _hw_save_array(
            stage_path / "stage_input_whitened_float.npy", whitened),
        "events_times": _hw_save_array(
            stage_path / "events_times.npy", sa_events.get("times", [])),
        "events_channels": _hw_save_array(
            stage_path / "events_channels.npy", sa_events.get("channels", [])),
        "logical_output": _hw_save_array(
            stage_path / "stage_output_pca_features_float.npy", pca_features),
        "routing": {
            "input_per_spike": "10 nearby channels x 61-sample snippets",
            "vmm_row_order": "for spike in events: for nearby channel: centered snippet",
            "output_reshape": "[n_spikes * 10, 6] -> [n_spikes, 60]",
        },
        "vmm": [vmm_meta],
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(stage_path, "tpca", [onnx_kernel])
    _hw_plot_stage_preview(stage_path, "tpca", {
        "vmm_input_rows": rows,
        "pca_features": pca_features,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_export_lda(stage_path, clust, pca_features, lda_features,
                   spike_channels, max_rows):
    vmm_entries = []
    onnx_kernels = []
    region_ids = clust.assigner.ch_to_region[np.asarray(spike_channels, dtype=np.int64)] \
        if len(spike_channels) else np.zeros(0, dtype=np.int32)
    np.save(stage_path / "region_ids.npy", region_ids.astype(np.int32))
    if clust.lda_mode == "per_region" and clust.lda_per_region_:
        for r_id, lda_state in sorted(clust.lda_per_region_.items()):
            mask = region_ids == int(r_id)
            x = pca_features[mask]
            if len(x) == 0:
                x = np.zeros((0, int(lda_state["mean"].shape[0])),
                             dtype=np.float32)
            centered = x - lda_state["mean"]
            w = lda_state["scalings_q"].T.astype(np.float32)
            prefix = f"region_{int(r_id):02d}"
            vmm_meta, onnx_kernel = _hw_save_vmm_reference(
                stage_path, prefix, centered, w, "LDA",
                config.PRECISION_DAC, config.PRECISION_LDA, max_rows)
            vmm_meta["routing"] = {
                "region_id": int(r_id),
                "spike_indices_file": f"{prefix}_spike_indices.npy",
            }
            np.save(stage_path / f"{prefix}_spike_indices.npy",
                    np.nonzero(mask)[0].astype(np.int64))
            vmm_entries.append(vmm_meta)
            onnx_kernels.append(onnx_kernel)
    elif clust.lda_enabled and clust.lda_scalings_q_ is not None:
        centered = pca_features - clust.lda_mean_
        w = clust.lda_scalings_q_.T.astype(np.float32)
        vmm_meta, onnx_kernel = _hw_save_vmm_reference(
            stage_path, "lda", centered, w, "LDA",
            config.PRECISION_DAC, config.PRECISION_LDA, max_rows)
        vmm_entries.append(vmm_meta)
        onnx_kernels.append(onnx_kernel)
    manifest = {
        "stage": "lda",
        "mode": clust.lda_mode,
        "logical_input": _hw_save_array(
            stage_path / "stage_input_pca_features_float.npy", pca_features),
        "logical_output": _hw_save_array(
            stage_path / "stage_output_lda_features_float.npy", lda_features),
        "routing": {
            "region_ids_file": "region_ids.npy",
            "note": "For per_region LDA, each ONNX input is packed from spikes belonging to one region.",
        },
        "vmm": vmm_entries,
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(stage_path, "lda", onnx_kernels)
    _hw_plot_stage_preview(stage_path, "lda", {
        "pca_features": pca_features,
        "lda_features": lda_features,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_export_assignment(stage_path, clust, features, spike_channels, labels,
                          distances, max_rows):
    vmm_entries = []
    onnx_kernels = []
    region_ids = clust.assigner.ch_to_region[np.asarray(spike_channels, dtype=np.int64)] \
        if len(spike_channels) else np.zeros(0, dtype=np.int32)
    np.save(stage_path / "region_ids.npy", region_ids.astype(np.int32))
    np.save(stage_path / "labels.npy", labels.astype(np.int32))
    np.save(stage_path / "distances.npy", distances.astype(np.float32))
    for r_id in range(clust.assigner.n_regions):
        mask = region_ids == int(r_id)
        x = features[mask]
        c = clust.assigner.centers_[r_id].astype(np.float32)
        if c.shape[0] == 0:
            continue
        prefix = f"region_{int(r_id):02d}"
        vmm_meta, onnx_kernel = _hw_save_vmm_reference(
            stage_path, prefix, x, c, "ASSIGN",
            config.PRECISION_DAC, config.PRECISION_ASSIGN, max_rows)
        vmm_meta["routing"] = {
            "region_id": int(r_id),
            "spike_indices_file": f"{prefix}_spike_indices.npy",
            "k_region": int(c.shape[0]),
            "note": "VMM output is dot(features, centers.T); distance/argmin/reject are digital.",
        }
        np.save(stage_path / f"{prefix}_spike_indices.npy",
                np.nonzero(mask)[0].astype(np.int64))
        vmm_entries.append(vmm_meta)
        onnx_kernels.append(onnx_kernel)
    manifest = {
        "stage": "assignment",
        "logical_input": _hw_save_array(
            stage_path / "stage_input_features_float.npy", features),
        "logical_output_labels": {
            "file": "labels.npy", "shape": list(labels.shape),
            "dtype": str(labels.dtype)},
        "logical_output_distances": {
            "file": "distances.npy", "shape": list(distances.shape),
            "dtype": str(distances.dtype)},
        "routing": {
            "region_ids_file": "region_ids.npy",
            "note": "Each ONNX input is packed from spikes assigned to one channel region.",
        },
        "vmm": vmm_entries,
    }
    manifest["onnx"] = _hw_write_onnx_if_enabled(stage_path, "assignment", onnx_kernels)
    _hw_plot_stage_preview(stage_path, "assignment", {
        "features": features,
        "distances": distances,
    })
    _hw_write_json(stage_path / "manifest.json", manifest)
    return manifest


def _hw_compute_clustering_feature_path(clust, whitened, sa_events):
    n_spikes = len(sa_events.get("times", []))
    if n_spikes == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {
            "pca_features": empty,
            "features": empty,
            "labels": np.zeros(0, dtype=np.int32),
            "distances": np.zeros(0, dtype=np.float32),
            "update_mask": np.zeros(0, dtype=bool),
        }
    if clust.lda_mode == "direct" and clust.lda_enabled and clust.lda_scalings_q_ is not None:
        snippet_len = 2 * config.CLUST_SNIPPET_HW + 1
        raw_feat = extract_raw_waveform_features(
            whitened, sa_events["times"], sa_events["channels"],
            clust.positions, snippet_len=snippet_len,
            nearby_idx=clust._direct_nearby_idx)
        pca_features = raw_feat
        from spike_clustering import _apply_lda
        features = _apply_lda(raw_feat, clust.lda_mean_, clust.lda_scalings_q_)
    else:
        pca_features = extract_temporal_features(
            whitened, sa_events["times"], sa_events["channels"],
            clust.positions, clust.temporal_pca, nearby_idx=clust._nearby_idx)
        if clust.lda_mode == "per_region" and clust.lda_enabled and clust.lda_per_region_:
            region_ids = clust.assigner.ch_to_region[sa_events["channels"]]
            n_lda = max(s["scalings_q"].shape[1]
                        for s in clust.lda_per_region_.values())
            features = np.zeros((n_spikes, n_lda), dtype=np.float32)
            from spike_clustering import _apply_lda
            for r_id, lda_state in clust.lda_per_region_.items():
                mask = region_ids == r_id
                if mask.any():
                    proj = _apply_lda(
                        pca_features[mask], lda_state["mean"],
                        lda_state["scalings_q"])
                    features[mask, :proj.shape[1]] = proj
        elif clust.lda_mode == "post_pca" and clust.lda_enabled and clust.lda_scalings_q_ is not None:
            from spike_clustering import _apply_lda
            features = _apply_lda(pca_features, clust.lda_mean_, clust.lda_scalings_q_)
        else:
            features = pca_features
    labels, distances, update_mask = clust.assigner.assign(
        features, sa_events["channels"])
    return {
        "pca_features": pca_features,
        "features": features,
        "labels": labels,
        "distances": distances,
        "update_mask": update_mask,
    }


def _export_hardware_batch(output_dir, raw_chunk, positions, pipe, wtemp, sa,
                           clust, source_info=None, export_options=None):
    global _HW_EXPORT_OPTIONS
    previous = _HW_EXPORT_OPTIONS
    _HW_EXPORT_OPTIONS = dict(export_options or {})
    try:
        return _export_hardware_batch_impl(
            output_dir, raw_chunk, positions, pipe, wtemp, sa, clust,
            source_info=source_info)
    finally:
        _HW_EXPORT_OPTIONS = previous


def _export_hardware_batch_impl(output_dir, raw_chunk, positions, pipe, wtemp,
                                sa, clust, source_info=None):
    root = _hw_ensure_dir(Path(output_dir))
    split_models = bool(_hw_export_option("split_models", False))
    model_root = _hw_ensure_dir(root.parent.parent / "models") if split_models else None
    max_rows = int(_hw_export_option("max_vmm_rows", 20000))
    raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
    pipe_x = copy.deepcopy(pipe)
    sa_x = copy.deepcopy(sa)
    clust_x = copy.deepcopy(clust)
    pipe_x.fir_filter.reset(raw_chunk.shape[1])
    summary = {
        "export_version": 1,
        "root": str(root),
        "layout": "models+batches" if split_models else "single_batch",
        "models_root": str(model_root) if model_root is not None else None,
        "source_info": source_info or {},
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "chunk_samples": config.CHUNK_SAMPLES,
            "chunk_duration_ms": config.CHUNK_DURATION_MS,
            "quantization_backend": config.QUANTIZATION_BACKEND,
            "precision": {
                "adc": config.PRECISION_ADC,
                "dac": config.PRECISION_DAC,
                "fir": config.PRECISION_FIR,
                "whiten": config.PRECISION_WHITEN,
                "match": config.PRECISION_MATCH,
                "sa": config.PRECISION_SA,
                "pca": config.PRECISION_PCA,
                "lda": config.PRECISION_LDA,
                "assign": config.PRECISION_ASSIGN,
            },
            "max_vmm_rows": max_rows,
            "onnx_format": (
                "uint8 input + int8 weight -> MatMulInteger -> "
                "raw int32 accumulator"
            ),
        },
        "stages": {},
    }

    def add_stage(stage_key, folder_name, manifest, has_model=True):
        stage_path = root / folder_name
        if split_models and has_model and model_root is not None:
            manifest = _hw_split_model_files(
                stage_key, stage_path, model_root / folder_name, manifest)
            _hw_write_json(stage_path / "manifest.json", manifest)
        summary["stages"][stage_key] = manifest

    if _hw_stage_enabled("raw"):
        add_stage("raw", "00_raw",
                  _hw_export_raw(_hw_stage_dir(root, "00_raw"), raw_chunk),
                  has_model=False)
    car_data = raw_chunk if pipe_x.skip_car else apply_car(raw_chunk)
    if _hw_stage_enabled("car"):
        p = _hw_stage_dir(root, "01_car")
        manifest = {
            "stage": "car",
            "description": "CAR is digital preprocessing, no VMM/ONNX.",
            "logical_input": _hw_save_array(p / "stage_input_raw_float.npy", raw_chunk),
            "logical_output": _hw_save_array(p / "stage_output_car_float.npy", car_data),
        }
        _hw_plot_stage_preview(p, "car", {"raw": raw_chunk, "car": car_data})
        _hw_write_json(p / "manifest.json", manifest)
        add_stage("car", "01_car", manifest, has_model=False)
    if pipe_x.skip_fir:
        filtered = car_data
    else:
        filtered = pipe_x.fir_filter.process_chunk(car_data)
        if _hw_stage_enabled("fir"):
            add_stage("fir", "02_fir", _hw_export_fir(
                _hw_stage_dir(root, "02_fir"), pipe_x, car_data, filtered, max_rows))
    whitened = pipe_x.whitening_processor.process_chunk(filtered)
    if _hw_stage_enabled("whitening"):
        add_stage("whitening", "03_whitening", _hw_export_whitening(
            _hw_stage_dir(root, "03_whitening"), pipe_x, filtered, whitened, max_rows))
    b_scores, score_positions = compute_score_matrix(
        whitened, wtemp, precision_bits=config.PRECISION_MATCH)
    if _hw_stage_enabled("template_matching"):
        add_stage("template_matching", "04_template_matching",
                  _hw_export_template_matching(
            _hw_stage_dir(root, "04_template_matching"),
            whitened, wtemp, b_scores, score_positions, max_rows))
    if bool(getattr(config, "SA_DENSE", False)):
        candidates = np.zeros((0, 2), dtype=np.int32)
        sa_events = sa_x.detect(b_scores, stride=config.MATCH_STRIDE, verbose=False)
    else:
        candidates = sa_x.coarse_screen(b_scores, verbose=False)
        sa_events = sa_x.detect(b_scores, stride=config.MATCH_STRIDE, verbose=False)
    if _hw_stage_enabled("spatial_aggregation"):
        add_stage("spatial_aggregation", "05_spatial_aggregation",
                  _hw_export_spatial_aggregation(
            _hw_stage_dir(root, "05_spatial_aggregation"),
            sa_x, b_scores, candidates, sa_events, max_rows))
    feat_path = _hw_compute_clustering_feature_path(clust_x, whitened, sa_events)
    pca_features = feat_path["pca_features"]
    final_features = feat_path["features"]
    labels = feat_path["labels"]
    distances = feat_path["distances"]
    if _hw_stage_enabled("tpca"):
        add_stage("tpca", "06_tpca", _hw_export_tpca(
            _hw_stage_dir(root, "06_tpca"), clust_x, whitened, sa_events,
            pca_features, max_rows))
    if _hw_stage_enabled("lda"):
        add_stage("lda", "07_lda", _hw_export_lda(
            _hw_stage_dir(root, "07_lda"), clust_x, pca_features,
            final_features, sa_events.get("channels", np.array([], dtype=np.int32)),
            max_rows))
    if _hw_stage_enabled("assignment"):
        add_stage("assignment", "08_assignment", _hw_export_assignment(
            _hw_stage_dir(root, "08_assignment"), clust_x, final_features,
            sa_events.get("channels", np.array([], dtype=np.int32)),
            labels, distances, max_rows))
    _hw_write_json(root / "manifest.json", summary)
    if split_models and model_root is not None:
        model_summary = {
            "export_version": 1,
            "models_root": str(model_root),
            "source_batch": str(root),
            "stages": {
                k: v.get("model") for k, v in summary["stages"].items()
                if isinstance(v, dict) and v.get("model")
            },
        }
        _hw_write_json(model_root / "manifest.json", model_summary)
    return summary


def _read_hardware_export_chunk(streamer, batch_index):
    batch_index = max(int(batch_index), 0)
    chunk_s = float(config.CHUNK_SAMPLES) / float(config.SAMPLE_RATE)
    start_s = float(config.CALIBRATION_DURATION_S) + batch_index * chunk_s
    gen = streamer.stream_chunks(
        start_s=start_s,
        duration_s=chunk_s,
        chunk_samples=config.CHUNK_SAMPLES,
    )
    try:
        raw_chunk = next(gen)
    except StopIteration as exc:
        raise RuntimeError(
            f"Requested export batch {batch_index} at {start_s:.3f}s is "
            "outside the recording") from exc
    return raw_chunk, start_s


def _run_hardware_export(output_dir, source_label, streamer, positions,
                         pipe, wtemp, sa, clust, source_info):
    options = _hw_export_options()
    batch_index = int(options.get("batch_index", 0))
    raw_chunk, start_s = _read_hardware_export_chunk(streamer, batch_index)
    subdir = str(options.get("subdir", "hardware_export") or "hardware_export")
    export_root = os.path.join(
        output_dir, subdir, "batches", f"batch_{batch_index:04d}")
    print(f"\n{'=' * 60}")
    print("  HARDWARE GOLDEN BATCH EXPORT")
    print(f"{'=' * 60}")
    print(f"[HW-EXPORT] Source={source_label}, batch={batch_index}, "
          f"start={start_s:.3f}s, output={export_root}")
    info = {
        "source": source_label,
        "batch_index": batch_index,
        "batch_start_s": start_s,
        "chunk_samples": config.CHUNK_SAMPLES,
        "sample_rate": config.SAMPLE_RATE,
    }
    info.update(source_info or {})
    manifest = _export_hardware_batch(
        export_root, raw_chunk, positions, pipe, wtemp, sa, clust,
        source_info=info, export_options=options)
    print(f"[HW-EXPORT] Wrote manifest: "
          f"{os.path.join(export_root, 'manifest.json')}")
    return manifest


class _SAStreamState:
    """Maintain overlap buffer between streaming chunks for SA detection.

    Without overlap, compute_score_matrix cannot evaluate the first/last
    half_nt samples of each chunk, causing systematic detection loss at
    chunk boundaries.
    """

    def __init__(self):
        self._overlap = None
        self._samples_processed = 0

    def reset(self):
        self._overlap = None
        self._samples_processed = 0

    def process_chunk(self, whitened_chunk, wTEMP, sa, clust,
                      batch_time_offset=0):
        """Streaming SA + clustering for one chunk, with overlap handling."""
        overlap_len_needed = config.MATCH_NT - 1

        if self._overlap is not None:
            extended = np.vstack([self._overlap, whitened_chunk])
            overlap_len = self._overlap.shape[0]
        else:
            overlap_len = 0
            extended = whitened_chunk

        B, B_positions = compute_score_matrix(extended, wTEMP,
                                              precision_bits=config.PRECISION_MATCH)
        sa_events = sa.detect(B, stride=config.MATCH_STRIDE, verbose=False)

        n_spikes = len(sa_events['times'])

        # Filter out events in the overlap region (already reported previously)
        if n_spikes > 0 and overlap_len > 0:
            mask = sa_events['times'] >= overlap_len
            for k in sa_events:
                if len(sa_events[k]) == n_spikes:
                    sa_events[k] = sa_events[k][mask]
            n_spikes = len(sa_events['times'])

        # Convert to global stream time
        if n_spikes > 0:
            sa_events['times'] = (sa_events['times'] - overlap_len
                                  + batch_time_offset)

        if n_spikes > 0 and clust.is_calibrated:
            chunk_local_times = sa_events['times'] - batch_time_offset
            sa_local = dict(sa_events)
            sa_local['times'] = chunk_local_times
            labels = clust.process_batch(sa_local,
                                         whitened_chunk=whitened_chunk,
                                         batch_time_offset=batch_time_offset)
        else:
            labels = np.array([], dtype=np.int32)

        self._overlap = whitened_chunk[-overlap_len_needed:].copy()
        self._samples_processed += whitened_chunk.shape[0]

        return sa_events, labels


# ==============================================================
# Post-processing
# ==============================================================

def _run_postprocessing(clust, positions, output_dir):
    """Merge clusters, compute quality, save results, plot."""
    clust_results = clust.get_results()
    if clust_results is None:
        return None

    print(f"\n{'=' * 60}")
    print(f"  CLUSTER MERGING (post-hoc)")
    print(f"{'=' * 60}")
    clust_results = merge_clusters(
        clust_results, sample_rate=config.SAMPLE_RATE,
        channel_positions=positions)
    is_good, contam_rate = compute_cluster_quality(clust_results)
    clust_results['is_good'] = is_good
    clust_results['contam_rate'] = contam_rate

    clust_dir = os.path.join(output_dir, 'clustering')
    save_clustering_results(clust_results, clust_dir)
    # Save channel positions for interactive viewers (DriftViewer etc.)
    np.save(os.path.join(clust_dir, 'channel_positions.npy'), positions)
    plot_clustering_summary(clust_results, channel_positions=positions,
                            save_path=os.path.join(output_dir,
                                                   'clustering_summary.png'))
    plot_merge_summary(clust_results,
                       save_path=os.path.join(output_dir,
                                              'merge_summary.png'))

    # Drift estimation
    drift_metrics = compute_drift_metrics(clust_results, positions)
    plot_drift_map(clust_results, positions, drift_metrics=drift_metrics,
                   save_path=os.path.join(output_dir, 'drift_map.png'))
    clust_results['drift_metrics'] = drift_metrics

    # Save drift metrics JSON
    import json as _json
    dm_json = {k: v for k, v in drift_metrics.items()
               if not isinstance(v, np.ndarray)}
    dm_path = os.path.join(output_dir, 'drift_metrics.json')
    with open(dm_path, 'w') as f:
        _json.dump(dm_json, f, indent=2)
    print(f"[DRIFT] Metrics saved: {dm_path}")
    print(f"[DRIFT] total={drift_metrics['total_drift_um']:.1f} um, "
          f"max_step={drift_metrics['max_drift_um']:.1f} um, "
          f"rate={drift_metrics['drift_rate_um_per_min']:.1f} um/min, "
          f"events={drift_metrics['n_drift_events']}")

    return clust_results


def _run_reference_comparison(one, eid, probe, clust_results, positions, raw_ind, output_dir):
    """Run comparison against IBL reference and diagnostics."""
    comp = None
    if clust_results is None:
        return None, {}
    try:
        from comparison import compare_with_reference, save_comparison
        comp = compare_with_reference(one, eid, probe, clust_results,
                                      channel_positions=positions,
                                      raw_ind=raw_ind)
        if comp is not None:
            save_comparison(comp, os.path.join(output_dir, 'comparison'))
            from diagnostics import run_all_diagnostics
            diag_results = run_all_diagnostics(comp, clust_results, positions)
            if 'fn_analysis' in diag_results:
                plot_fn_analysis(
                    diag_results['fn_analysis'], comp, clust_results, positions,
                    save_path=os.path.join(output_dir, 'fn_analysis.png'))
            if comp.get('rate_comparison') is not None:
                plot_firing_rate_comparison(
                    comp['rate_comparison'], positions,
                    save_path=os.path.join(output_dir, 'firing_rate_comparison.png'))
            return comp, diag_results
    except Exception as e:
        print(f"[MAIN] Reference comparison skipped: {e}")
    return comp, {}


def _flatten_comparison_summary(comp):
    if not isinstance(comp, dict):
        return {}
    summary = comp.get('summary', {}) if isinstance(comp.get('summary', {}), dict) else {}
    if not summary:
        return {}
    out = dict(summary)
    if 'global_recall' not in out:
        if 'mean_recall' in summary:
            out['global_recall'] = summary.get('mean_recall')
    if 'good_global_recall' not in out:
        n_good = summary.get('n_good_matched', None)
        n_ref_good = summary.get('n_ref_good', None)
        if n_good is not None and n_ref_good is not None:
            if n_ref_good > 0:
                out['good_global_recall'] = float(n_good) / float(n_ref_good)
            else:
                out['good_global_recall'] = float('nan')
    return out


def _print_brain_region_diagnostic(comp, clust_results, channel_regions, output_dir=None):
    """Print diagnostic showing which brain region each good cluster belongs to."""
    if comp is None or clust_results is None or channel_regions is None:
        print("[REGION-DIAG] Skipped: missing comp, clust_results, or channel_regions")
        return

    print(f"\n{'=' * 70}")
    print(f"  BRAIN REGION DIAGNOSTIC")
    print(f"{'=' * 70}")

    labels = clust_results.get('labels', np.array([]))
    channels = clust_results.get('channels', np.array([]))
    if len(labels) == 0 or len(channels) == 0:
        print("  No spikes in results")
        return

    active = clust_results.get('active_clusters', np.array([]))
    cluster_peak_ch = {}
    for cid in active:
        mask = labels == cid
        if mask.sum() == 0:
            continue
        chs = channels[mask]
        vals, counts = np.unique(chs, return_counts=True)
        cluster_peak_ch[int(cid)] = int(vals[np.argmax(counts)])

    cluster_region = {}
    for cid, pch in cluster_peak_ch.items():
        if pch < len(channel_regions):
            cluster_region[cid] = channel_regions[pch]
        else:
            cluster_region[cid] = 'unknown'

    matches = comp.get('matches', [])
    pair_info = {}
    if matches:
        for m in matches:
            pid = m.get('pipe_id')
            if pid is not None:
                pair_info[int(pid)] = {
                    'ref': m.get('ref_id'),
                    'ref_q': 'good' if m.get('ref_is_good') else 'other',
                    'acc': m.get('accuracy', 0),
                    'prec': m.get('precision', 0),
                    'rec': m.get('recall', 0),
                }

    is_good = clust_results.get('is_good', np.array([]))
    good_set = set()
    if len(is_good) > 0:
        all_cluster_ids = np.arange(clust_results.get('n_clusters_total', 0))
        for i, cid in enumerate(all_cluster_ids):
            if i < len(is_good) and is_good[i]:
                good_set.add(int(cid))

    good_with_info = []
    for cid in sorted(good_set):
        region = cluster_region.get(cid, '?')
        n_spikes = int((labels == cid).sum())
        pi = pair_info.get(cid, {})
        acc = pi.get('acc', 0)
        ref = pi.get('ref', '-')
        ref_q = pi.get('ref_q', '-')
        prec = pi.get('prec', 0)
        rec = pi.get('rec', 0)
        good_with_info.append((cid, region, n_spikes, ref, ref_q, acc, prec, rec))

    good_with_info.sort(key=lambda x: -x[5])

    print(f"\n  {'pipe':>6s}  {'region':>8s}  {'n_spike':>7s}  {'ref':>5s}  {'ref_q':>6s}  "
          f"{'acc':>6s}  {'prec':>6s}  {'rec':>6s}")
    print(f"  {chr(9472) * 62}")
    for cid, region, n_sp, ref, ref_q, acc, prec, rec in good_with_info:
        ref_str = str(ref) if ref != '-' else '-'
        print(f"  {cid:6d}  {region:>8s}  {n_sp:7d}  {ref_str:>5s}  {ref_q:>6s}  "
              f"{acc:6.3f}  {prec:6.3f}  {rec:6.3f}")

    from collections import Counter
    region_counts = Counter()
    region_good_acc = {}
    for cid, region, n_sp, ref, ref_q, acc, prec, rec in good_with_info:
        region_counts[region] += 1
        if region not in region_good_acc:
            region_good_acc[region] = []
        region_good_acc[region].append(acc)

    print(f"\n  Summary by brain region:")
    print(f"  {'region':>10s}  {'n_good':>6s}  {'acc>0.5':>7s}  {'acc>0.8':>7s}  "
          f"{'mean_acc':>8s}  {'max_acc':>7s}")
    print(f"  {chr(9472) * 55}")
    for region in sorted(region_counts, key=lambda r: -region_counts[r]):
        accs = region_good_acc[region]
        n_good = region_counts[region]
        n_above_05 = sum(1 for a in accs if a > 0.5)
        n_above_08 = sum(1 for a in accs if a > 0.8)
        mean_acc = np.mean(accs) if accs else 0
        max_acc = max(accs) if accs else 0
        print(f"  {region:>10s}  {n_good:6d}  {n_above_05:7d}  {n_above_08:7d}  "
              f"{mean_acc:8.3f}  {max_acc:7.3f}")

    if output_dir:
        report_path = os.path.join(output_dir, 'brain_region_diagnostic.txt')
        try:
            with open(report_path, 'w') as f:
                f.write("pipe,region,n_spikes,ref,ref_q,acc,prec,rec\n")
                for cid, region, n_sp, ref, ref_q, acc, prec, rec in good_with_info:
                    f.write(f"{cid},{region},{n_sp},{ref},{ref_q},{acc:.4f},{prec:.4f},{rec:.4f}\n")
            print(f"\n  Report saved: {report_path}")
        except Exception as e:
            print(f"  Could not save report: {e}")

    print(f"{'=' * 70}")
    return {r: region_good_acc.get(r, []) for r in region_counts}


# ==============================================================
# Validation Filtering + Decoding
# ==============================================================

def _run_validation_and_decoding(clust_results, comp, one, eid, probe,
                                  positions, raw_ind, channel_regions,
                                  output_dir, ks4_full=None):
    """Run validation-based cluster filtering and neural decoding."""
    from decode_task import (load_trials, load_reference_spikes,
                              compute_firing_rates, decode_logistic,
                              get_choice_event_times)
    from comparison import (
        compare_with_reference, filter_reference, load_reference,
        save_comparison, save_matched_pair_rate_comparison,
    )
    from collections import Counter
    import copy

    val_duration = getattr(config, 'DECODE_VAL_DURATION_S', 120.0)
    acc_threshold = getattr(config, 'DECODE_ACC_THRESHOLD', 0.8)
    quant_bits = list(getattr(config, 'DECODE_QUANT_BITS', [2, 4, 6, 8]))
    choice_window = getattr(config, 'DECODE_CHOICE_WINDOW', (-0.1, 0.0))
    feedback_window = getattr(config, 'DECODE_FEEDBACK_WINDOW', (0.0, 0.2))

    print(f"\n\n{'#' * 70}")
    print(f"  VALIDATION FILTERING + NEURAL DECODING")
    print(f"{'#' * 70}")

    cali_end = config.CALIBRATION_DURATION_S
    times_s = clust_results['times'].astype(np.float64) / config.SAMPLE_RATE + cali_end
    labels = clust_results['labels']
    channels = clust_results['channels']
    stream_end = times_s.max()
    val_end = cali_end + val_duration

    print(f"  Calibration: 0-{cali_end:.0f}s")
    print(f"  Validation: {cali_end:.0f}-{val_end:.0f}s")
    print(f"  Test: {val_end:.0f}-{stream_end:.0f}s")
    print(f"  Accuracy threshold: {acc_threshold}")

    is_good = clust_results.get('is_good', None)
    all_active = sorted(set(labels[labels >= 0]))
    good_ids = [cid for cid in all_active
                if is_good is not None and cid < len(is_good) and is_good[cid]]

    n_total = len(labels)
    n_assigned = (labels >= 0).sum()
    print(f"\n  Spike sorting: {n_total} spikes, "
          f"{n_assigned} assigned ({n_assigned / n_total * 100:.1f}%), "
          f"{len(all_active)} active, {len(good_ids)} good")

    region_clusters = {}
    if channel_regions is not None:
        for cid in good_ids:
            cid_mask = labels == cid
            if cid_mask.sum() == 0:
                continue
            peak_ch = Counter(channels[cid_mask]).most_common(1)[0][0]
            if peak_ch < len(channel_regions):
                region = channel_regions[peak_ch]
                if region not in region_clusters:
                    region_clusters[region] = []
                region_clusters[region].append(cid)
        print(f"  Good clusters by region: " +
              ', '.join(f"{r}={len(c)}" for r, c in
                        sorted(region_clusters.items(), key=lambda x: -len(x[1]))))

    # Validation period comparison
    print(f"\n{'=' * 70}")
    print(f"  SECTION 1: VALIDATION PERIOD ({cali_end:.0f}-{val_end:.0f}s)")
    print(f"{'=' * 70}")

    val_samples = int(val_duration * config.SAMPLE_RATE)
    val_mask = clust_results['times'] < val_samples
    test_mask = clust_results['times'] >= val_samples

    val_results = {k: clust_results[k][val_mask] for k in ('times', 'labels', 'channels', 'distances')}
    test_results = {k: clust_results[k][test_mask] for k in ('times', 'labels', 'channels', 'distances')}

    print(f"  Validation spikes: {val_mask.sum()}")
    print(f"  Test spikes: {test_mask.sum()}")

    val_comp = compare_with_reference(
        one, eid, probe, val_results,
        channel_positions=positions, raw_ind=raw_ind, verbose=True)

    val_acc = {}
    if val_comp and val_comp.get('matches'):
        for m in val_comp['matches']:
            val_acc[m['pipe_id']] = m['accuracy']

    # Cluster filtering
    print(f"\n{'=' * 70}")
    print(f"  SECTION 2: CLUSTER FILTERING (val accuracy >= {acc_threshold})")
    print(f"{'=' * 70}")

    good_with_acc = [(cid, val_acc.get(cid, 0.0)) for cid in good_ids if cid in val_acc]
    good_with_acc.sort(key=lambda x: -x[1])
    filtered_ids = [cid for cid, acc in good_with_acc if acc >= acc_threshold]

    filtered_dir = os.path.join(output_dir, 'filtered_analysis')
    os.makedirs(filtered_dir, exist_ok=True)

    if good_with_acc:
        va = [acc for _, acc in good_with_acc]
        print(f"  Good clusters with val match: {len(good_with_acc)}")
        for t in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            n = sum(1 for a in va if a >= t)
            marker = " <- threshold" if abs(t - acc_threshold) < 0.01 else ""
            print(f"    val_acc >= {t:.1f}: {n:>4d} clusters{marker}")

        acc_dtype = [('pipe_id', 'i4'), ('val_accuracy', 'f4')]
        np.save(os.path.join(filtered_dir, 'validation_cluster_accuracy.npy'),
                np.array(good_with_acc, dtype=acc_dtype))

    print(f"\n  SELECTED for decoding: {len(filtered_ids)} clusters")
    np.save(os.path.join(filtered_dir, 'filtered_cluster_ids.npy'),
            np.asarray(filtered_ids, dtype=np.int32))
    if val_comp is not None:
        save_comparison(val_comp,
                        os.path.join(filtered_dir, 'validation_comparison'))

    if filtered_ids:
        fa = [val_acc[cid] for cid in filtered_ids]
        print(f"  Selected accuracy: mean={np.mean(fa):.3f}, "
              f"median={np.median(fa):.3f}, min={np.min(fa):.3f}")

        if region_clusters:
            print(f"  Selected by region:")
            for region, cids in sorted(region_clusters.items(),
                                        key=lambda x: -len(x[1])):
                n_filt = sum(1 for c in filtered_ids if c in cids)
                if n_filt > 0:
                    print(f"    {region:<15s}: {n_filt}/{len(cids)}")

    # Test period comparison
    print(f"\n{'=' * 70}")
    print(f"  SECTION 3: TEST PERIOD COMPARISON ({val_end:.0f}-{stream_end:.0f}s)")
    print(f"{'=' * 70}")

    print(f"\n  --- All clusters ---")
    test_all_comp = compare_with_reference(
        one, eid, probe, test_results,
        channel_positions=positions, raw_ind=raw_ind, verbose=True)

    test_all_acc = {}
    if test_all_comp and test_all_comp.get('matches'):
        for m in test_all_comp['matches']:
            test_all_acc[m['pipe_id']] = {
                'accuracy': m['accuracy'], 'ref_is_good': m['ref_is_good']}

    if filtered_ids:
        print(f"\n  --- Filtered clusters (val acc >= {acc_threshold}) ---")
        test_filt_results = copy.deepcopy(test_results)
        keep_set = set(filtered_ids)
        filt_mask = np.array([l in keep_set for l in test_filt_results['labels']])
        test_filt_results['labels'][~filt_mask] = -1

        test_filt_comp = compare_with_reference(
            one, eid, probe, test_filt_results,
            channel_positions=positions, raw_ind=raw_ind, verbose=True)
        if test_filt_comp is not None:
            filt_comp_dir = os.path.join(filtered_dir,
                                         'filtered_test_comparison')
            save_comparison(test_filt_comp, filt_comp_dir)
            duration = test_filt_comp['t_end'] - test_filt_comp['t_start']
            pair_summary = save_matched_pair_rate_comparison(
                test_filt_comp.get('matches', []), duration, filtered_dir)
            def _jsonable_scalar_dict(d):
                out = {}
                for k, v in d.items():
                    if isinstance(v, np.integer):
                        out[k] = int(v)
                    elif isinstance(v, np.floating):
                        out[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        out[k] = v.tolist()
                    else:
                        out[k] = v
                return out

            filtered_summary = {
                'accuracy_threshold': float(acc_threshold),
                'validation_duration_s': float(val_duration),
                'n_filtered_ids': int(len(filtered_ids)),
                'filtered_ids': [int(cid) for cid in filtered_ids],
                'test_comparison': _jsonable_scalar_dict(
                    test_filt_comp.get('summary', {})),
                'filtered_vs_all_reference_rate':
                    test_filt_comp.get('rate_comparison', {}),
                'matched_pair_rate_comparison':
                    _jsonable_scalar_dict(pair_summary),
            }
            # NumPy arrays in rate_comparison are saved separately by
            # save_comparison; keep the JSON summary scalar-only.
            rc = filtered_summary['filtered_vs_all_reference_rate']
            if rc:
                filtered_summary['filtered_vs_all_reference_rate'] = {
                    'correlation': float(rc['correlation']),
                    'ratio': float(rc['ratio']),
                    'duration': float(rc['duration']),
                    'n_pipe_spikes': int(rc.get('n_pipe_spikes', 0)),
                    'n_ref_spikes': int(rc.get('n_ref_spikes', 0)),
                }
            with open(os.path.join(filtered_dir, 'filtered_summary.json'), 'w') as f:
                json.dump(filtered_summary, f, indent=2)
            print(f"  Filtered matched-pair rate r: "
                  f"{pair_summary['matched_pair_rate_correlation']:.3f}")
            print(f"  Filtered matched-pair rate ratio: "
                  f"{pair_summary['pipe_ref_rate_ratio_over_matched_pairs']:.3f}")

    # Val vs Test correlation
    paired_va, paired_ta = [], []
    for cid in val_acc:
        if cid in test_all_acc:
            paired_va.append(val_acc[cid])
            paired_ta.append(test_all_acc[cid]['accuracy'])
    if len(paired_va) > 10:
        corr = np.corrcoef(paired_va, paired_ta)[0, 1]
        print(f"\n  Validation-Test accuracy correlation: r = {corr:.3f} "
              f"({len(paired_va)} clusters)")

    # Neural decoding
    print(f"\n{'=' * 70}")
    print(f"  SECTION 4: NEURAL DECODING (test period: {val_end:.0f}-{stream_end:.0f}s)")
    print(f"{'=' * 70}")

    trials = load_trials(one, eid)

    pipe_spikes = {
        'times': times_s,
        'labels': labels,
        'channels': channels,
    }

    valid_trials = (trials['valid'] &
                    (trials['stim_on'] >= val_end) &
                    np.isfinite(trials['stim_on']) &
                    np.isfinite(trials['feedback_times']))
    feedback_idx = np.where(valid_trials)[0]
    choice_event_times_all, choice_event_label = get_choice_event_times(trials)
    if choice_event_times_all is not None:
        choice_trials = valid_trials & np.isfinite(choice_event_times_all)
        choice_idx = np.where(choice_trials)[0]
    else:
        choice_idx = np.array([], dtype=np.int64)

    print(f"  Trials in test period: choice={len(choice_idx)}, "
          f"feedback={len(feedback_idx)}")

    if len(choice_idx) < 30 and len(feedback_idx) < 30:
        print(f"  Too few trials for decoding, skipping")
        return

    choice = trials['choice'][choice_idx]
    feedback = trials['feedback'][feedback_idx]
    choice_times = (choice_event_times_all[choice_idx]
                    if choice_event_times_all is not None else np.array([]))
    feedback_t = trials['feedback_times'][feedback_idx]

    decode_results = {}

    def _run_decode(cluster_ids, label):
        if len(cluster_ids) < 3:
            print(f"  [{label}] Too few clusters ({len(cluster_ids)}), skipping")
            return

        # Choice decoding
        if len(choice_idx) >= 30:
            print(f"\n  [{label}] Choice ({len(cluster_ids)} clusters, "
                  f"{len(choice_idx)} trials, align={choice_event_label}, "
                  f"window={choice_window})")
            X = compute_firing_rates(
                pipe_spikes['times'], pipe_spikes['labels'],
                cluster_ids, choice_times, window=choice_window)
            col_active = X.std(axis=0) > 1e-6
            if col_active.sum() >= 2:
                y = (choice == 1).astype(int)
                res = decode_logistic(X[:, col_active], y)
                decode_results[f'{label}_choice'] = res
                sig = '***' if res['p_value'] < 0.001 else \
                      '**' if res['p_value'] < 0.01 else \
                      '*' if res['p_value'] < 0.05 else 'n.s.'
                print(f"    Full: {res['real_acc']:.3f} "
                      f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                      f"p={res['p_value']:.4f} {sig}")
                qa = res.get('quant_accs', {})
                if qa:
                    print(f"    Quant: " +
                          '  '.join(f"{b}b={a:.3f}" for b, a in sorted(qa.items())))
            else:
                print(f"    Choice skipped: only {col_active.sum()} active clusters")
        else:
            print(f"\n  [{label}] Choice skipped "
                  f"(align={choice_event_label or 'unavailable'}, "
                  f"n_trials={len(choice_idx)})")

        # Feedback decoding
        if len(feedback_idx) >= 30:
            print(f"  [{label}] Feedback ({len(feedback_idx)} trials, "
                  f"window={feedback_window})")
            X = compute_firing_rates(
                pipe_spikes['times'], pipe_spikes['labels'],
                cluster_ids, feedback_t, window=feedback_window)
            col_active = X.std(axis=0) > 1e-6
            if col_active.sum() >= 2:
                y = (feedback == 1).astype(int)
                res = decode_logistic(X[:, col_active], y)
                decode_results[f'{label}_feedback'] = res
                sig = '***' if res['p_value'] < 0.001 else \
                      '**' if res['p_value'] < 0.01 else \
                      '*' if res['p_value'] < 0.05 else 'n.s.'
                print(f"    Full: {res['real_acc']:.3f} "
                      f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                      f"p={res['p_value']:.4f} {sig}")
                qa = res.get('quant_accs', {})
                if qa:
                    print(f"    Quant: " +
                          '  '.join(f"{b}b={a:.3f}" for b, a in sorted(qa.items())))
            else:
                print(f"    Feedback skipped: only {col_active.sum()} active clusters")
        else:
            print(f"  [{label}] Feedback skipped (n_trials={len(feedback_idx)})")

    _run_decode(good_ids, "pipe_good")

    if len(filtered_ids) >= 3:
        _run_decode(filtered_ids, "pipe_filtered")

    # IBL reference decoding
    try:
        ref_spikes = load_reference_spikes(one, eid, probe, val_end, stream_end)
        if ref_spikes is not None:
            ref_good_ids = sorted(ref_spikes['good_ids'] &
                                  set(np.unique(ref_spikes['labels'])))
            if len(ref_good_ids) >= 3:
                ref_spike_data = {
                    'times': ref_spikes['times'],
                    'labels': ref_spikes['labels'],
                    'channels': ref_spikes['channels'],
                }
                if len(choice_idx) >= 30:
                    print(f"\n  [ref_ibl] Choice ({len(ref_good_ids)} clusters, "
                          f"{len(choice_idx)} trials, align={choice_event_label}, "
                          f"window={choice_window})")
                    X = compute_firing_rates(
                        ref_spike_data['times'], ref_spike_data['labels'],
                        ref_good_ids, choice_times, window=choice_window)
                    col_active = X.std(axis=0) > 1e-6
                    if col_active.sum() >= 2:
                        y = (choice == 1).astype(int)
                        res = decode_logistic(X[:, col_active], y)
                        decode_results['ref_ibl_choice'] = res
                        sig = '***' if res['p_value'] < 0.001 else \
                              '**' if res['p_value'] < 0.01 else \
                              '*' if res['p_value'] < 0.05 else 'n.s.'
                        print(f"    Full: {res['real_acc']:.3f} "
                              f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                              f"p={res['p_value']:.4f} {sig}")

                if len(feedback_idx) >= 30:
                    print(f"  [ref_ibl] Feedback ({len(feedback_idx)} trials, "
                          f"window={feedback_window})")
                    X = compute_firing_rates(
                        ref_spike_data['times'], ref_spike_data['labels'],
                        ref_good_ids, feedback_t, window=feedback_window)
                    col_active = X.std(axis=0) > 1e-6
                    if col_active.sum() >= 2:
                        y = (feedback == 1).astype(int)
                        res = decode_logistic(X[:, col_active], y)
                        decode_results['ref_ibl_feedback'] = res
                        sig = '***' if res['p_value'] < 0.001 else \
                              '**' if res['p_value'] < 0.01 else \
                              '*' if res['p_value'] < 0.05 else 'n.s.'
                        print(f"    Full: {res['real_acc']:.3f} "
                              f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                              f"p={res['p_value']:.4f} {sig}")
    except Exception as e:
        print(f"  [ref_ibl] Skipped: {e}")

    # KS4 full-recording reference decoding
    if ks4_full is not None:
        try:
            sr = float(config.SAMPLE_RATE)
            ks_times_s = ks4_full['all_times_raw'].astype(np.float64) / sr
            ks_labels = ks4_full['all_labels']
            ks_mask = (ks_times_s >= val_end) & (ks_times_s <= stream_end)
            is_ref = ks4_full.get('is_ref', None)
            if is_ref is not None:
                ks_good_ids = sorted(int(i) for i in np.unique(ks_labels[ks_mask])
                                     if i < len(is_ref) and is_ref[i])
            else:
                ks_good_ids = sorted(int(i) for i in np.unique(ks_labels[ks_mask]))
            if len(ks_good_ids) >= 3:
                ks_spike_data = {
                    'times': ks_times_s[ks_mask],
                    'labels': ks_labels[ks_mask],
                }
                if len(choice_idx) >= 30:
                    print(f"\n  [ref_ks4_full] Choice ({len(ks_good_ids)} clusters, "
                          f"{len(choice_idx)} trials, align={choice_event_label}, "
                          f"window={choice_window})")
                    X = compute_firing_rates(
                        ks_spike_data['times'], ks_spike_data['labels'],
                        ks_good_ids, choice_times, window=choice_window)
                    col_active = X.std(axis=0) > 1e-6
                    if col_active.sum() >= 2:
                        y = (choice == 1).astype(int)
                        res = decode_logistic(X[:, col_active], y)
                        decode_results['ref_ks4_full_choice'] = res
                        sig = '***' if res['p_value'] < 0.001 else \
                              '**' if res['p_value'] < 0.01 else \
                              '*' if res['p_value'] < 0.05 else 'n.s.'
                        print(f"    Full: {res['real_acc']:.3f} "
                              f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                              f"p={res['p_value']:.4f} {sig}")

                if len(feedback_idx) >= 30:
                    print(f"  [ref_ks4_full] Feedback ({len(feedback_idx)} trials, "
                          f"window={feedback_window})")
                    X = compute_firing_rates(
                        ks_spike_data['times'], ks_spike_data['labels'],
                        ks_good_ids, feedback_t, window=feedback_window)
                    col_active = X.std(axis=0) > 1e-6
                    if col_active.sum() >= 2:
                        y = (feedback == 1).astype(int)
                        res = decode_logistic(X[:, col_active], y)
                        decode_results['ref_ks4_full_feedback'] = res
                        sig = '***' if res['p_value'] < 0.001 else \
                              '**' if res['p_value'] < 0.01 else \
                              '*' if res['p_value'] < 0.05 else 'n.s.'
                        print(f"    Full: {res['real_acc']:.3f} "
                              f"(null: {res['null_mean']:.3f}+/-{res['null_std']:.3f}) "
                              f"p={res['p_value']:.4f} {sig}")
        except Exception as e:
            print(f"  [ref_ks4_full] Skipped: {e}")

    # Top regions
    top_regions = sorted(region_clusters.items(), key=lambda x: -len(x[1]))[:3]
    for region, cids in top_regions:
        if len(cids) >= 5:
            _run_decode(cids, f"region_{region}")

    # Summary table
    if decode_results:
        print(f"\n{'=' * 70}")
        print(f"  DECODING SUMMARY")
        print(f"{'=' * 70}")
        qb_headers = ''.join(f" {b:>4d}b" for b in quant_bits)
        print(f"  {'Condition':<25s} {'Task':>10s} {'Full':>7s}{qb_headers} "
              f"{'Null':>12s} {'p':>8s} {'Sig':>5s} {'N_tr':>6s} {'L1nz':>5s}")
        print(f"  {chr(9472) * (85 + 6 * len(quant_bits))}")

        for key, res in sorted(decode_results.items()):
            parts = key.rsplit('_', 1)
            cond = parts[0]
            task = parts[1] if len(parts) > 1 else key
            sig = '***' if res['p_value'] < 0.001 else \
                  '**' if res['p_value'] < 0.01 else \
                  '*' if res['p_value'] < 0.05 else 'n.s.'
            qa = res.get('quant_accs', {})
            qb_vals = ''.join(f" {qa.get(b, 0.5):>5.3f}" for b in quant_bits)
            print(f"  {cond:<25s} {task:>10s} {res['real_acc']:>7.3f}{qb_vals} "
                  f"{res['null_mean']:>5.3f}+/-{res['null_std']:.3f} "
                  f"{res['p_value']:>8.4f} {sig:>5s} "
                  f"{res['n_trials']:>6d} {res.get('n_nonzero', 0):>5d}")


# ==============================================================
# IBL real data
# ==============================================================

@_with_pipeline_cleanups
def run_ibl(output_dir="results_ibl", brain_region="VISp", duration_s=None,
            stop_event=None, calibration_only=False, hardware_export=False):
    """Run on real IBL Neuropixels data.

    Parameters
    ----------
    stop_event : threading.Event or None
        If set, the streaming loop will check this event and exit early
        when stop_event.is_set() returns True.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    tee = start_logging(output_dir)
    stop_log = _register_pipeline_cleanup(lambda: stop_logging(tee))
    reset_quantization_diagnostics()
    _begin_quant_range_calibration()

    print("=" * 60)
    print("  SPIKE SORTING -- IBL REAL DATA")
    print("=" * 60)

    # Reset IBL defaults (may have been overwritten by a previous local run)
    config.UV_PER_BIT = 2.34375
    config.SAMPLE_RATE = 30_000
    config.N_CHANNELS = 384
    config.CHUNK_SAMPLES = int(config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000.0)

    params = collect_config_params()
    print(f"\n[CONFIG] {len(params)} parameters loaded")
    print(f"\n{chr(9472) * 60}")
    print(f"  FULL CONFIGURATION")
    print(f"{chr(9472) * 60}")
    for k, v in sorted(params.items()):
        print(f"  {k:40s} = {v}")
    print(f"{chr(9472) * 60}\n")

    one = connect_one()
    if config.IBL_SESSION_EID:
        eid = config.IBL_SESSION_EID
        probe = config.IBL_PROBE_LABEL
        print(f"[IBL] Using configured session: {eid}, probe={probe}")
    else:
        eid, probe = find_good_session(one, brain_region=brain_region)
    positions, raw_ind = load_channel_geometry(one, eid, probe)
    channel_regions = load_channel_regions(one, eid, probe)

    streamer = RawDataStreamer(one, eid, probe)
    streamer.open()
    close_streamer = _register_pipeline_cleanup(streamer.close)

    # ======== CALIBRATION PHASE ========

    source_info = {
        "session_eid": eid,
        "probe": probe,
        "brain_region": brain_region,
    }
    final_fit, range_payload, n_total_calib = _run_calibration_phase(
        "IBL", "IBL DATA", output_dir, streamer, positions, source_info,
        stop_event=stop_event, diagnostic_required=True)

    pipe = final_fit["pipe"]
    coeffs = final_fit["coeffs"]
    clust = final_fit["clust"]
    wTEMP = final_fit["wTEMP"]
    sa = final_fit["sa"]
    whitened_all = final_fit["whitened_all"]

    if calibration_only:
        recording_info = {
            'source': 'IBL',
            'session_eid': eid,
            'probe': probe,
            'n_channels': positions.shape[0],
            'sample_rate': config.SAMPLE_RATE,
            'total_duration_s': streamer.total_duration_s,
            'calibration_duration_s': config.CALIBRATION_DURATION_S,
            'test_duration_s': 0.0,
            'n_test_chunks': 0,
            'chunk_samples': config.CHUNK_SAMPLES,
        }
        _print_summary(coeffs, pipe.groups, pipe,
                       clust_results=None,
                       recording_info=recording_info)
        _record_calibration_only_result(
            "IBL", output_dir, t_start,
            recording_info=recording_info,
            range_payload=range_payload)
        close_streamer()
        stop_log()
        return range_payload

    if hardware_export or bool(getattr(config, "HARDWARE_EXPORT_ENABLE", False)):
        manifest = _run_hardware_export(
            output_dir, "IBL", streamer, positions, pipe, wTEMP, sa, clust,
            source_info)
        close_streamer()
        stop_log()
        return manifest

    del whitened_all

    # ======== STREAMING PHASE ========
    print(f"\n{'=' * 60}")
    print(f"  STREAMING PROCESSING")
    print(f"{'=' * 60}")

    pipe.fir_filter.reset(positions.shape[0])
    total_spikes = 0
    n_batches = 0
    cs = int(config.CHUNK_SAMPLES)

    max_batches = config.MAX_TEST_BATCHES or 999999
    print_interval = 100
    stream_t0 = time.time()

    sa_stream = _SAStreamState()
    chunk_gen = streamer.stream_chunks(duration_s=duration_s)
    while True:
        if n_batches >= max_batches:
            break
        if stop_event is not None and stop_event.is_set():
            print(f"[MAIN] Stop requested by user after {n_batches} batches")
            break
        try:
            raw_chunk = next(chunk_gen)
        except StopIteration:
            break
        except Exception as e:
            print(f"  Batch {n_batches}: READ ERROR ({e})")
            break

        whitened = pipe.process_chunk(raw_chunk)
        offset = n_batches * cs
        sa_ev, labels = sa_stream.process_chunk(
            whitened, wTEMP, sa, clust, batch_time_offset=offset)
        total_spikes += len(sa_ev['times'])
        n_batches += 1

        if n_batches % print_interval == 0 or n_batches <= 5:
            n_clusters = len(np.unique(labels)) if len(labels) > 0 else 0
            print(_format_stream_progress(
                n_batches, max_batches, total_spikes, len(sa_ev['times']),
                n_clusters, stream_t0))

    # ---- Post-processing ----
    clust_results = _run_postprocessing(clust, positions, output_dir)

    try:
        from window_compare import run_first120_checks
        run_first120_checks(
            one=one, eid=eid, probe=probe,
            clust_results=clust_results,
            cali_result=getattr(clust, '_cali_result', None),
            channel_positions=positions,
            raw_ind=raw_ind,
            output_dir=output_dir,
            window_s=120.0,
        )
    except Exception as e:
        print(f"[MAIN] first-120s checks skipped: {e}")

    comp, _ = _run_reference_comparison(
        one, eid, probe, clust_results, positions, raw_ind, output_dir)
    comp_summary = _flatten_comparison_summary(comp)

    if channel_regions is not None:
        _print_brain_region_diagnostic(comp, clust_results,
                                       channel_regions, output_dir=output_dir)

    elapsed = time.time() - t_start
    print(f"\n[MAIN] Done: {total_spikes} spikes, {n_batches} batches, {elapsed:.1f}s")

    recording_info = {
        'source': 'IBL',
        'session_eid': eid,
        'probe': probe,
        'n_channels': positions.shape[0],
        'sample_rate': config.SAMPLE_RATE,
        'total_duration_s': streamer.total_duration_s,
        'calibration_duration_s': config.CALIBRATION_DURATION_S,
        'test_duration_s': n_batches * config.CHUNK_SAMPLES / config.SAMPLE_RATE,
        'n_test_chunks': n_batches,
        'chunk_samples': config.CHUNK_SAMPLES,
    }

    _print_summary(coeffs, pipe.groups, pipe,
                   clust_results=clust_results,
                   recording_info=recording_info,
                   comparison_summary=comp_summary)

    params = collect_config_params()
    results_dict = collect_results(clust_results=clust_results,
                                   recording_info=recording_info,
                                   elapsed_s=elapsed,
                                   comparison_summary=comp_summary)
    dm = clust_results.get('drift_metrics')
    if dm:
        results_dict.update({
            'drift_total_um': dm['total_drift_um'],
            'drift_max_step_um': dm['max_drift_um'],
            'drift_rate_um_per_min': dm['drift_rate_um_per_min'],
            'drift_n_events': dm['n_drift_events'],
        })
    xlsx_path = os.path.join(output_dir, 'results_log.xlsx')
    append_to_xlsx(params, results_dict, xlsx_path=xlsx_path)

    # ======== KS4 FULL-RECORDING REFERENCE ========
    ks4_full = None
    try:
        from ks4_full_reference import (
            get_ks4_full_reference,
            compare_ks4_full_vs_iblsorter,
            run_full_ks4_comparison,
        )
        ks4_full = get_ks4_full_reference(streamer, positions, output_dir)
        if ks4_full is not None:
            compare_ks4_full_vs_iblsorter(
                ks4_full, one, eid, probe, positions,
                raw_ind=raw_ind, output_dir=output_dir)
            ks4_comp = run_full_ks4_comparison(
                clust_results, ks4_full, positions,
                output_dir=output_dir)
            if ks4_comp is not None:
                from diagnostics import run_all_diagnostics as _run_diag
                print(f"\n{'=' * 60}")
                print(f"  DIAGNOSTICS (vs KS4-FULL)")
                print(f"{'=' * 60}")
                _run_diag(ks4_comp, clust_results, positions)
                if channel_regions is not None:
                    _print_brain_region_diagnostic(
                        ks4_comp, clust_results, channel_regions,
                        output_dir=output_dir)
    except Exception as e:
        print(f"\n[KS4-FULL] Skipped: {e}")
        import traceback
        traceback.print_exc()

    # ======== VALIDATION FILTERING + DECODING ========
    if getattr(config, 'DECODE_ACC_THRESHOLD', None) is not None:
        try:
            _run_validation_and_decoding(
                clust_results, comp,
                one, eid, probe, positions, raw_ind,
                channel_regions, output_dir,
                ks4_full=ks4_full)
        except Exception as e:
            print(f"\n[DECODE] Failed: {e}")
            import traceback
            traceback.print_exc()

    close_streamer()
    stop_log()
    return clust_results


# ==============================================================
# LOCAL (Open Ephys) pipeline
# ==============================================================

def _run_local_reference_comparison(clust_results, positions, output_dir):
    """Compare pipeline results against local Kilosort reference (st0.mat)."""
    ref_path = config.LOCAL_REFERENCE_PATH
    if not ref_path or not os.path.exists(ref_path):
        print("[LOCAL] No reference path configured, skipping comparison")
        return None, {}

    try:
        from comparison import (filter_reference, match_units,
                                compare_firing_rates, save_comparison,
                                _get_pipeline_time_window)

        ref = load_local_reference(ref_path, sample_rate=config.SAMPLE_RATE)
        if ref is None:
            return None, {}

        print(f"\n{'='*50}")
        print(f"  REFERENCE COMPARISON (local KS)")
        print(f"{'='*50}")

        t_start, t_end = _get_pipeline_time_window(clust_results)
        ref_filtered = filter_reference(ref, t_start, t_end)

        # Disable spatial constraint when reference has no real channel info
        # (load_local_reference sets all spike_channels to 0)
        has_real_channels = not np.all(ref_filtered['spike_channels'] == 0)
        matches, summary = match_units(
            clust_results, ref_filtered,
            channel_positions=positions if has_real_channels else None)

        rate_comp = None
        if positions is not None:
            rate_comp = compare_firing_rates(
                clust_results, ref_filtered, positions)

        comp = {
            'ref_filtered': ref_filtered,
            'matches': matches,
            'summary': summary,
            'rate_comparison': rate_comp,
            't_start': t_start,
            't_end': t_end,
        }

        print(f"\n  — Comparison Summary —")
        print(f"    Reference sorter: {ref['sorter']}")
        print(f"    Time window:      [{t_start:.2f}, {t_end:.2f}] s")
        print(f"    Pipeline:         {summary['n_pipe_clusters']} clusters")
        print(f"    Reference:        {summary['n_ref_clusters']} clusters "
              f"({summary['n_ref_good']} good)")
        print(f"    Matched pairs:    {summary['n_matched_pairs']}")
        print(f"    Global recall:    {summary['global_recall']:.4f}")
        print(f"    Mean accuracy:    {summary['mean_accuracy']:.3f}")

        save_comparison(comp, os.path.join(output_dir, 'comparison'))
        return comp, {}
    except Exception as e:
        print(f"[LOCAL] Reference comparison failed: {e}")
        import traceback; traceback.print_exc()
        return None, {}


def _run_local_decoding(clust_results, positions, output_dir, ks4_full=None):
    """Compare local KS4 decoding against our streaming output."""
    ttl_dir = config.LOCAL_TTL_DIR
    if not ttl_dir:
        ttl_dir = find_ttl_dir(config.LOCAL_RECORDING_DIR)
    if not ttl_dir or not os.path.isdir(ttl_dir):
        print("[LOCAL] No TTL directory found, skipping decoding")
        return

    trials = load_ttl_events(ttl_dir)
    if trials is None or trials['n_trials'] < 10:
        print("[LOCAL] Too few trials from TTL events, skipping decoding")
        return

    try:
        from decode_task import (compute_firing_rates, decode_logistic,
                                 get_choice_event_times)

        print(f"\n{'='*50}")
        print(f"  NEURAL DECODING (local: streaming vs KS4)")
        print(f"{'='*50}")
        print(f"  Trials: {trials['n_trials']}, "
              f"valid: {trials['valid'].sum()}")

        cali_end = config.CALIBRATION_DURATION_S
        times_s = (clust_results['times'].astype(np.float64)
                   / config.SAMPLE_RATE + cali_end)
        labels = clust_results['labels']
        is_good = clust_results.get('is_good', None)
        all_active = sorted(set(labels[labels >= 0]))
        good_ids = [cid for cid in all_active
                    if is_good is not None and cid < len(is_good)
                    and is_good[cid]]

        if not good_ids:
            print("[LOCAL] No good clusters for decoding")
            return

        ref_spike_data = None
        ref_cluster_ids = []
        ref_label = None

        ref_path = config.LOCAL_REFERENCE_PATH
        if ref_path and os.path.exists(ref_path):
            ref = load_local_reference(ref_path, sample_rate=config.SAMPLE_RATE)
            if ref is not None:
                ref_spike_data = {
                    'times': ref['spike_times'],
                    'labels': ref['spike_clusters'],
                }
                ref_cluster_ids = sorted(int(cid) for cid in ref['cluster_ids'])
                ref_label = 'ks4_local'
        elif ks4_full is not None:
            sr = float(config.SAMPLE_RATE)
            ref_times = ks4_full['all_times_raw'].astype(np.float64) / sr
            ref_labels = ks4_full['all_labels']
            is_ref = ks4_full.get('is_ref', None)
            if is_ref is not None:
                ref_cluster_ids = sorted(
                    int(i) for i in np.unique(ref_labels)
                    if i < len(is_ref) and is_ref[i]
                )
            else:
                ref_cluster_ids = sorted(int(i) for i in np.unique(ref_labels))
            ref_spike_data = {
                'times': ref_times,
                'labels': ref_labels,
            }
            ref_label = 'ks4_full'

        if ref_spike_data is None or len(ref_cluster_ids) < 3:
            print("[LOCAL] No usable KS4 reference for local decoding, skipping")
            return

        valid_base = (trials['valid'] &
                      (trials['stim_on_times'] >= cali_end) &
                      np.isfinite(trials['stim_on_times']) &
                      np.isfinite(trials['feedback_times']))
        feedback_idx = np.where(valid_base)[0]
        choice_event_times_all, choice_event_label = get_choice_event_times(
            trials, allow_stim_fallback=False)
        if choice_event_times_all is not None:
            choice_valid = valid_base & np.isfinite(choice_event_times_all)
            choice_idx = np.where(choice_valid)[0]
        else:
            choice_idx = np.array([], dtype=np.int64)

        decode_results = {}

        def _run_decode_for_source(spike_times, spike_labels, cluster_ids, label):
            if len(cluster_ids) < 3:
                print(f"  [{label}] Too few clusters ({len(cluster_ids)}), skipping")
                return

            if len(choice_idx) >= 10:
                choice_labels = trials['choice'][choice_idx]
                if len(np.unique(choice_labels)) >= 2:
                    choice_window = getattr(config, 'DECODE_CHOICE_WINDOW',
                                            (-0.1, 0.0))
                    print(f"\n  [{label}] Choice ({len(cluster_ids)} clusters, "
                          f"{len(choice_idx)} trials, align={choice_event_label}, "
                          f"window={choice_window})")
                    rates = compute_firing_rates(
                        spike_times, spike_labels, cluster_ids,
                        choice_event_times_all[choice_idx], choice_window)
                    active_cols = rates.std(axis=0) > 1e-6
                    if active_cols.sum() >= 2:
                        result = decode_logistic(
                            rates[:, active_cols],
                            (choice_labels == 1).astype(int))
                        decode_results[f'{label}_choice'] = result
                        print(f"    Full: {result['real_acc']:.3f} "
                              f"(null: {result['null_mean']:.3f}+/-{result['null_std']:.3f}) "
                              f"p={result['p_value']:.4f}")
                    else:
                        print(f"    Choice skipped: only {active_cols.sum()} active clusters")
                else:
                    print(f"\n  [{label}] Choice skipped (labels are constant)")
            else:
                print(f"\n  [{label}] Choice skipped "
                      f"(align={choice_event_label or 'unavailable'}, "
                      f"n_trials={len(choice_idx)})")

            if len(feedback_idx) >= 10:
                feedback_labels = trials['feedbackType'][feedback_idx]
                if len(np.unique(feedback_labels)) >= 2:
                    feedback_window = getattr(config, 'DECODE_FEEDBACK_WINDOW',
                                              (0.0, 0.2))
                    print(f"  [{label}] Feedback ({len(feedback_idx)} trials, "
                          f"window={feedback_window})")
                    rates = compute_firing_rates(
                        spike_times, spike_labels, cluster_ids,
                        trials['feedback_times'][feedback_idx], feedback_window)
                    active_cols = rates.std(axis=0) > 1e-6
                    if active_cols.sum() >= 2:
                        result = decode_logistic(
                            rates[:, active_cols],
                            (feedback_labels == 1).astype(int))
                        decode_results[f'{label}_feedback'] = result
                        print(f"    Full: {result['real_acc']:.3f} "
                              f"(null: {result['null_mean']:.3f}+/-{result['null_std']:.3f}) "
                              f"p={result['p_value']:.4f}")
                    else:
                        print(f"    Feedback skipped: only {active_cols.sum()} active clusters")
                else:
                    print(f"  [{label}] Feedback skipped (labels are constant)")
            else:
                print(f"  [{label}] Feedback skipped (n_trials={len(feedback_idx)})")

        _run_decode_for_source(times_s, labels, good_ids, 'streaming')
        _run_decode_for_source(ref_spike_data['times'], ref_spike_data['labels'],
                               ref_cluster_ids, ref_label)

        if decode_results:
            quant_bits = list(getattr(config, 'DECODE_QUANT_BITS', [2, 4, 6, 8]))
            print(f"\n{'=' * 70}")
            print(f"  LOCAL DECODING SUMMARY")
            print(f"{'=' * 70}")
            qb_headers = ''.join(f" {b:>4d}b" for b in quant_bits)
            print(f"  {'Condition':<18s} {'Task':>10s} {'Full':>7s}{qb_headers} "
                  f"{'Null':>12s} {'p':>8s} {'N_tr':>6s}")
            print(f"  {chr(9472) * (72 + 6 * len(quant_bits))}")
            for key, res in sorted(decode_results.items()):
                cond, task = key.rsplit('_', 1)
                qa = res.get('quant_accs', {})
                qb_vals = ''.join(f" {qa.get(b, 0.5):>5.3f}" for b in quant_bits)
                print(f"  {cond:<18s} {task:>10s} {res['real_acc']:>7.3f}{qb_vals} "
                      f"{res['null_mean']:>5.3f}+/-{res['null_std']:.3f} "
                      f"{res['p_value']:>8.4f} {res['n_trials']:>6d}")

    except Exception as e:
        print(f"[LOCAL] Decoding failed: {e}")
        import traceback; traceback.print_exc()


@_with_pipeline_cleanups
def run_local(output_dir="results_local", duration_s=None, stop_event=None,
              calibration_only=False, hardware_export=False):
    """Run spike sorting on a local Open Ephys Neuropixels recording.

    Parameters
    ----------
    output_dir : str
        Directory for results.
    duration_s : float or None
        Limit streaming duration (seconds). None = process all data.
    stop_event : threading.Event or None
        If set, the streaming loop will exit early.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    tee = start_logging(output_dir)
    stop_log = _register_pipeline_cleanup(lambda: stop_logging(tee))
    reset_quantization_diagnostics()
    _begin_quant_range_calibration()

    print("=" * 60)
    print("  SPIKE SORTING -- LOCAL RECORDING (Open Ephys)")
    print("=" * 60)

    recording_dir = config.LOCAL_RECORDING_DIR
    if not recording_dir or not os.path.isdir(recording_dir):
        raise ValueError(
            f"LOCAL_RECORDING_DIR is not set or does not exist: "
            f"'{recording_dir}'")
    print(f"[LOCAL] Recording directory: {recording_dir}")

    # Open streamer FIRST to auto-detect parameters from actual data
    streamer = LocalDataStreamer(recording_dir, config.LOCAL_AP_FOLDER or None)
    streamer.open()
    close_streamer = _register_pipeline_cleanup(streamer.close)

    print(f"\n[LOCAL] Auto-detecting recording parameters from data...")
    preproc_state = _sync_config_from_streamer(streamer)

    # Now print config (with correct values from data)
    params = collect_config_params()
    print(f"\n[CONFIG] {len(params)} parameters loaded")
    print(f"\n{chr(9472) * 60}")
    print(f"  FULL CONFIGURATION")
    print(f"{chr(9472) * 60}")
    for k, v in sorted(params.items()):
        print(f"  {k:40s} = {v}")
    print(f"{chr(9472) * 60}\n")

    # Channel geometry
    ks_path = config.LOCAL_KS_SETTINGS_PATH
    positions, raw_ind = load_local_channel_geometry(recording_dir, ks_path)

    # No brain regions for local data
    channel_regions = None

    # ======== CALIBRATION PHASE ========

    if stop_event is not None and stop_event.is_set():
        print("[MAIN] Stop requested before calibration"); close_streamer(); stop_log(); return None

    source_info = {
        "recording_dir": recording_dir,
        "local_ap_folder": config.LOCAL_AP_FOLDER,
        "local_ks_settings_path": config.LOCAL_KS_SETTINGS_PATH,
    }
    final_fit, range_payload, n_total_calib = _run_calibration_phase(
        "LOCAL", "LOCAL DATA", output_dir, streamer, positions, source_info,
        stop_event=stop_event, diagnostic_required=False,
        preproc_state=preproc_state)

    if stop_event is not None and stop_event.is_set():
        print("[MAIN] Stop requested after calibration"); close_streamer(); stop_log(); return None

    pipe = final_fit["pipe"]
    coeffs = final_fit["coeffs"]
    clust = final_fit["clust"]
    wTEMP = final_fit["wTEMP"]
    sa = final_fit["sa"]
    whitened_all = final_fit["whitened_all"]

    if calibration_only:
        recording_info = {
            'source': 'Local',
            'recording_dir': recording_dir,
            'n_channels': positions.shape[0],
            'sample_rate': config.SAMPLE_RATE,
            'total_duration_s': streamer.total_duration_s,
            'calibration_duration_s': config.CALIBRATION_DURATION_S,
            'test_duration_s': 0.0,
            'n_test_chunks': 0,
            'chunk_samples': config.CHUNK_SAMPLES,
        }
        _print_summary(coeffs, pipe.groups, pipe,
                       clust_results=None,
                       recording_info=recording_info)
        _record_calibration_only_result(
            "Local", output_dir, t_start,
            recording_info=recording_info,
            range_payload=range_payload)
        close_streamer()
        stop_log()
        return range_payload

    if hardware_export or bool(getattr(config, "HARDWARE_EXPORT_ENABLE", False)):
        manifest = _run_hardware_export(
            output_dir, "LOCAL", streamer, positions, pipe, wTEMP, sa, clust,
            source_info)
        close_streamer()
        stop_log()
        return manifest

    del whitened_all

    # ======== STREAMING PHASE ========
    print(f"\n{'=' * 60}")
    print(f"  STREAMING PROCESSING")
    print(f"{'=' * 60}")

    pipe.fir_filter.reset(positions.shape[0])
    total_spikes = 0
    n_batches = 0
    cs = int(config.CHUNK_SAMPLES)

    max_batches = config.MAX_TEST_BATCHES or 999999
    print_interval = 100
    stream_t0 = time.time()

    sa_stream = _SAStreamState()
    chunk_gen = streamer.stream_chunks(duration_s=duration_s)
    while True:
        if n_batches >= max_batches:
            break
        if stop_event is not None and stop_event.is_set():
            print(f"[MAIN] Stop requested by user after {n_batches} batches")
            break
        try:
            raw_chunk = next(chunk_gen)
        except StopIteration:
            break
        except Exception as e:
            print(f"  Batch {n_batches}: READ ERROR ({e})")
            break

        whitened = pipe.process_chunk(raw_chunk)
        offset = n_batches * cs
        sa_ev, labels = sa_stream.process_chunk(
            whitened, wTEMP, sa, clust, batch_time_offset=offset)
        total_spikes += len(sa_ev['times'])
        n_batches += 1

        if n_batches % print_interval == 0 or n_batches <= 5:
            n_clusters = len(np.unique(labels)) if len(labels) > 0 else 0
            print(_format_stream_progress(
                n_batches, max_batches, total_spikes, len(sa_ev['times']),
                n_clusters, stream_t0))

    # ---- Post-processing ----
    clust_results = _run_postprocessing(clust, positions, output_dir)

    comp, _ = _run_local_reference_comparison(
        clust_results, positions, output_dir)

    # ======== KS4 FULL-RECORDING REFERENCE ========
    ks4_full = None
    try:
        from ks4_full_reference import get_ks4_full_reference, run_full_ks4_comparison
        ks4_full = get_ks4_full_reference(streamer, positions, output_dir)
        if ks4_full is not None:
            ks4_comp = run_full_ks4_comparison(
                clust_results, ks4_full, positions,
                output_dir=output_dir)
            if ks4_comp is not None:
                from diagnostics import run_all_diagnostics as _run_diag
                print(f"\n{'=' * 60}")
                print(f"  DIAGNOSTICS (vs KS4-FULL)")
                print(f"{'=' * 60}")
                _run_diag(ks4_comp, clust_results, positions)
    except Exception as e:
        print(f"\n[KS4-FULL] Skipped: {e}")
        import traceback
        traceback.print_exc()

    comp_summary = _flatten_comparison_summary(comp)

    elapsed = time.time() - t_start
    print(f"\n[MAIN] Done: {total_spikes} spikes, {n_batches} batches, "
          f"{elapsed:.1f}s")

    recording_info = {
        'source': 'Local',
        'recording_dir': recording_dir,
        'n_channels': positions.shape[0],
        'sample_rate': config.SAMPLE_RATE,
        'total_duration_s': streamer.total_duration_s,
        'calibration_duration_s': config.CALIBRATION_DURATION_S,
        'test_duration_s': n_batches * config.CHUNK_SAMPLES / config.SAMPLE_RATE,
        'n_test_chunks': n_batches,
        'chunk_samples': config.CHUNK_SAMPLES,
    }

    _print_summary(coeffs, pipe.groups, pipe,
                   clust_results=clust_results,
                   recording_info=recording_info,
                   comparison_summary=comp_summary)

    params = collect_config_params()
    results_dict = collect_results(clust_results=clust_results,
                                   recording_info=recording_info,
                                   elapsed_s=elapsed,
                                   comparison_summary=comp_summary)
    dm = clust_results.get('drift_metrics')
    if dm:
        results_dict.update({
            'drift_total_um': dm['total_drift_um'],
            'drift_max_step_um': dm['max_drift_um'],
            'drift_rate_um_per_min': dm['drift_rate_um_per_min'],
            'drift_n_events': dm['n_drift_events'],
        })
    xlsx_path = os.path.join(output_dir, 'results_log.xlsx')
    append_to_xlsx(params, results_dict, xlsx_path=xlsx_path)

    # ======== DECODING ========
    if getattr(config, 'DECODE_ACC_THRESHOLD', None) is not None:
        try:
            _run_local_decoding(clust_results, positions, output_dir,
                                ks4_full=ks4_full)
        except Exception as e:
            print(f"\n[DECODE] Failed: {e}")
            import traceback
            traceback.print_exc()

    close_streamer()
    stop_log()
    return clust_results


# ==============================================================
# Summary
# ==============================================================

def _print_summary(coeffs, groups, pipe, events=None,
                   sa_events=None, clust_results=None,
                   recording_info=None, comparison_summary=None):
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")

    if recording_info:
        ri = recording_info
        print(f"\n  -- Recording --")
        print(f"    Source:         {ri.get('source', '?')}")
        if ri.get('session_eid'):
            print(f"    Session EID:   {ri['session_eid']}")
        print(f"    Channels:      {ri.get('n_channels', '?')}")
        print(f"    Duration:      {ri.get('total_duration_s', 0):.1f}s total, "
              f"{ri.get('calibration_duration_s', 0):.1f}s calib, "
              f"{ri.get('test_duration_s', 0):.1f}s test")
        print(f"    Test chunks:   {ri.get('n_test_chunks', 0)}")

    n_out = pipe.whitening_processor.n_output_channels
    print(f"\n  -- Preprocessing --")
    print(f"    FIR taps:       {len(coeffs)}  "
          f"(group delay = {config.FIR_GROUP_DELAY} samples "
          f"= {config.FIR_GROUP_DELAY / config.SAMPLE_RATE * 1000:.2f} ms)")
    print(f"    Precision:      FIR={config.PRECISION_FIR}b, "
          f"Whiten={config.PRECISION_WHITEN}b, "
          f"Match={config.PRECISION_MATCH}b, "
          f"SA={config.PRECISION_SA}b, "
          f"Assign={config.PRECISION_ASSIGN}b, "
          f"ADC={config.PRECISION_ADC}b, DAC={config.PRECISION_DAC}b")
    print(f"    Quant backend:  {getattr(config, 'QUANTIZATION_BACKEND', 'fake')}")
    print(f"    Groups:         {len(groups)} (mode={config.WHITEN_MODE})")
    print(f"    Output ch:      {n_out}/{config.N_CHANNELS}")

    quant_lines = format_quantization_summary()
    if quant_lines:
        print(f"\n  -- Quantization Diagnostics --")
        for line in quant_lines:
            print(f"    {line}")

    if clust_results is not None:
        n_total = len(clust_results['times'])
        n_active = clust_results['n_active']
        clust_mode = clust_results.get('mode', '?')
        n_merges = clust_results.get('n_merges', 0)
        pre_n = clust_results.get('pre_merge_n_clusters', 0)

        print(f"\n  -- Clustering ({clust_mode}) --")
        print(f"    Total spikes: {n_total}")
        print(f"    Clusters:     {n_active} active")
        if n_merges > 0:
            print(f"    Merges:       {n_merges} ({pre_n} -> {n_active})")
        if 'is_good' in clust_results:
            n_good = clust_results['is_good'].sum()
            print(f"    Good units:   {n_good}/{n_active}")

    if comparison_summary:
        print(f"\n  -- Reference Comparison --")
        for key, label in [
            ('n_pipe_clusters', 'Pipeline clusters'),
            ('n_ref_clusters', 'Reference clusters'),
            ('n_matched_pairs', 'Matched pairs'),
            ('global_recall', 'Global recall'),
            ('good_global_recall', 'Good-unit recall'),
            ('mean_accuracy', 'Mean accuracy'),
            ('mean_precision', 'Mean precision'),
        ]:
            if key in comparison_summary:
                val = comparison_summary[key]
                if isinstance(val, (float, np.floating)):
                    print(f"    {label}: {val:.4f}")
                else:
                    print(f"    {label}: {val}")

    print("=" * 60)


def _main_cli(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None,
                        choices=['ibl', 'local'],
                        help='Data source (overrides config.DATA_SOURCE)')
    parser.add_argument('--config-json', default=None,
                        help='Path to a JSON config snapshot to apply before running')
    parser.add_argument('--eid', default=None,
                        help='IBL session EID (overrides config.IBL_SESSION_EID)')
    parser.add_argument('--probe', default=None,
                        help='IBL probe label (overrides config.IBL_PROBE_LABEL)')
    parser.add_argument('--region', default='VISp')
    parser.add_argument('--dir', default=None,
                        help='Local recording directory')
    parser.add_argument('--output', default=None,
                        help='Output directory (defaults to results_ibl or results_local)')
    parser.add_argument('--calibration-only', action='store_true',
                        help='Run calibration and quantization range plots only; skip streaming')
    parser.add_argument('--hardware-export', action='store_true',
                        help='Run calibration, export one hardware golden batch, then skip streaming')
    args = parser.parse_args(argv)

    source = None
    output_dir = None
    try:
        if args.config_json:
            apply_config_snapshot(load_config_snapshot(args.config_json))
        if args.calibration_only and args.hardware_export:
            raise ValueError(
                "Use either --calibration-only or --hardware-export, not both")

        source = args.source or config.DATA_SOURCE
        if args.eid:
            config.IBL_SESSION_EID = args.eid
        if args.probe:
            config.IBL_PROBE_LABEL = args.probe
        if args.dir:
            config.LOCAL_RECORDING_DIR = args.dir
            source = 'local'

        if args.output:
            output_dir = args.output
        elif args.hardware_export:
            output_dir = ("results_local_export" if source == 'local'
                          else "results_ibl_export")
        elif args.calibration_only:
            output_dir = ("results_local_cali" if source == 'local'
                          else "results_ibl_cali")
        else:
            output_dir = "results_local" if source == 'local' else "results_ibl"
        if source == 'local':
            run_local(output_dir=output_dir,
                      calibration_only=args.calibration_only,
                      hardware_export=args.hardware_export)
        else:
            run_ibl(output_dir=output_dir, brain_region=args.region,
                    calibration_only=args.calibration_only,
                    hardware_export=args.hardware_export)
        return 0
    except Exception:
        report = format_error_report(
            "CLI pipeline crash",
            metadata={
                "Source": source or "",
                "Output directory": output_dir or "",
            },
        )
        try:
            paths = write_crash_report(report, output_dir=output_dir)
        except Exception as write_err:
            paths = {"global_path": None, "global_archive_path": None,
                     "output_path": None, "output_archive_path": None}
            report = report.rstrip() + f"\nCrash-report write failure: {write_err}\n"

        print(report, file=sys.stderr)
        if paths.get("global_path"):
            print(f"Crash log: {paths['global_path']}", file=sys.stderr)
        if paths.get("global_archive_path"):
            print(f"Crash archive: {paths['global_archive_path']}", file=sys.stderr)
        if paths.get("output_path"):
            print(f"Output copy: {paths['output_path']}", file=sys.stderr)
        if paths.get("output_archive_path"):
            print(f"Output archive: {paths['output_archive_path']}", file=sys.stderr)
        for err in paths.get("write_errors", []):
            print(f"Crash report warning: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_cli())
