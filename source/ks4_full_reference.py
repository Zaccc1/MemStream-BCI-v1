"""
KS4 Full-Recording Reference
==============================
Runs KS4 on the entire recording (not just 120s calibration) and caches results.
Used as a reference to evaluate pipeline streaming fidelity.

Usage in main.py (after streaming, before/during comparison):

    from ks4_full_reference import get_ks4_full_reference
    ks4_full = get_ks4_full_reference(streamer, channel_positions, output_dir)
    # ks4_full has: all_times, all_times_raw, all_channels, all_labels, is_ref, summary

Config:
    KS4_FULL_ENABLE = True          # enable full-recording KS4
    KS4_FULL_CACHE_SUBDIR = 'ks4_cache_full'
    KS4_FULL_DEVICE = 'auto'        # auto / cuda / cpu
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import config


# ==============================================================
# Cache
# ==============================================================

def _full_cache_path(output_dir, total_samples, n_channels):
    cache_dir = os.path.join(output_dir,
                             getattr(config, 'KS4_FULL_CACHE_SUBDIR', 'ks4_cache_full'))
    os.makedirs(cache_dir, exist_ok=True)
    duration_s = total_samples / config.SAMPLE_RATE
    fname = f'ks4_full_{duration_s:.0f}s_{n_channels}ch.npz'
    return os.path.join(cache_dir, fname)


def _save_full_cache(path, result):
    save_dict = {
        'all_times_raw': np.asarray(result['all_times_raw'], dtype=np.int64),
        'all_channels': np.asarray(result['all_channels'], dtype=np.int32),
        'all_labels': np.asarray(result['all_labels'], dtype=np.int32),
    }
    if 'is_ref' in result and result['is_ref'] is not None:
        save_dict['is_ref'] = np.asarray(result['is_ref'], dtype=bool)
    if 'spike_xy' in result and result['spike_xy'] is not None:
        save_dict['spike_xy'] = np.asarray(result['spike_xy'], dtype=np.float32)
    if 'summary' in result and isinstance(result['summary'], dict):
        save_dict['_summary_json'] = np.array([json.dumps(result['summary'])])
    np.savez_compressed(path, **save_dict)
    n_spikes = len(save_dict['all_times_raw'])
    n_clusters = int(np.unique(save_dict['all_labels']).size)
    print(f'[KS4-FULL-CACHE] Saved: {n_spikes} spikes, {n_clusters} clusters -> {path}')


def _load_full_cache(path):
    data = np.load(path, allow_pickle=False)
    result = {
        'backend': 'official_ks4_full_cached',
        'all_times_raw': data['all_times_raw'],
        'all_channels': data['all_channels'],
        'all_labels': data['all_labels'],
    }
    if 'is_ref' in data:
        result['is_ref'] = data['is_ref']
    if 'spike_xy' in data:
        result['spike_xy'] = data['spike_xy']
    if '_summary_json' in data:
        result['summary'] = json.loads(str(data['_summary_json'][0]))
    else:
        n_spikes = len(result['all_times_raw'])
        n_clusters = int(np.unique(result['all_labels']).size)
        n_good = int(result['is_ref'].sum()) if 'is_ref' in result else 0
        result['summary'] = {
            'n_spikes': n_spikes, 'n_clusters': n_clusters, 'n_good_units': n_good
        }
    n_spikes = len(result['all_times_raw'])
    n_clusters = int(np.unique(result['all_labels']).size)
    n_good = int(result['is_ref'].sum()) if 'is_ref' in result else 0
    print(f'[KS4-FULL-CACHE] Loaded: {n_spikes} spikes, {n_clusters} clusters, '
          f'{n_good} good <- {path}')
    return result


# ==============================================================
# Write full recording to int16 binary (chunked, memory-safe)
# ==============================================================

def _write_full_binary(streamer, bin_path, n_channels):
    """Write entire recording to int16 binary in chunks."""
    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = int(streamer.total_duration_s * config.SAMPLE_RATE)
    chunk_s = 30  # 30 seconds per chunk
    chunk_samples = int(chunk_s * config.SAMPLE_RATE)
    # Use streamer's own bit_volts for correct round-trip quantization
    if hasattr(streamer, '_bit_volts') and streamer._bit_volts is not None:
        uv_per_bit = float(streamer._bit_volts)
    else:
        uv_per_bit = float(config.UV_PER_BIT)

    print(f'[KS4-FULL] Writing full recording to {bin_path}')
    print(f'[KS4-FULL] Total: {total_samples} samples ({total_samples/config.SAMPLE_RATE:.1f}s), '
          f'{n_channels} channels')

    with open(bin_path, 'wb') as f:
        written = 0
        t0 = time.time()
        while written < total_samples:
            n = min(chunk_samples, total_samples - written)
            # Read chunk in µV
            raw = streamer._read(written, n)
            data = streamer._to_uv(raw)
            # Convert back to int16 for KS4
            i16 = np.clip(np.round(data / uv_per_bit), -32768, 32767).astype(np.int16)
            i16.tofile(f)
            written += n
            elapsed = time.time() - t0
            if written == n or written % (chunk_samples * 10) == 0 or written >= total_samples:
                pct = written / total_samples * 100
                print(f'  {pct:5.1f}% ({written/config.SAMPLE_RATE:.0f}s / '
                      f'{total_samples/config.SAMPLE_RATE:.0f}s) [{elapsed:.1f}s]')

    print(f'[KS4-FULL] Binary written: {bin_path} '
          f'({bin_path.stat().st_size / 1e9:.1f} GB)')
    return bin_path


# ==============================================================
# Run KS4 on full recording
# ==============================================================

def _run_ks4_full(streamer, channel_positions, output_dir):
    """Run official KS4 on the full recording."""
    from official_ks4_bridge import (
        import_official_kilosort, make_probe_dict, _pick_nearest_channels,
        resolve_ks4_device,
    )

    kilosort_pkg, rk_mod, pp_mod = import_official_kilosort()
    run_kilosort = rk_mod.run_kilosort
    compute_spike_positions = pp_mod.compute_spike_positions
    remove_duplicates = pp_mod.remove_duplicates

    n_channels = channel_positions.shape[0]
    total_samples = int(streamer.total_duration_s * config.SAMPLE_RATE)
    duration_s = total_samples / config.SAMPLE_RATE

    ks4_dir = os.path.join(output_dir,
                           getattr(config, 'KS4_FULL_CACHE_SUBDIR', 'ks4_cache_full'))
    os.makedirs(ks4_dir, exist_ok=True)
    bin_path = Path(ks4_dir) / 'full_recording_ap.bin'

    # Write binary (skip if already exists and correct size)
    expected_size = total_samples * n_channels * 2  # int16
    if bin_path.exists() and bin_path.stat().st_size == expected_size:
        print(f'[KS4-FULL] Binary already exists: {bin_path}')
    else:
        _write_full_binary(streamer, bin_path, n_channels)

    # Build probe and settings
    probe = make_probe_dict(channel_positions)
    device_str = getattr(config, 'KS4_FULL_DEVICE',
                         getattr(config, 'KS4_DEVICE', 'auto'))
    device = resolve_ks4_device(device_str)

    settings = {
        'n_chan_bin': n_channels,
        'fs': float(config.SAMPLE_RATE),
        'batch_size': int(getattr(config, 'KS4_BATCH_SIZE', 60000)),
        'nblocks': int(getattr(config, 'KS4_NBLOCKS', 1)),
        'nt': int(getattr(config, 'KS4_NT', 61)),
        'n_pcs': int(getattr(config, 'KS4_N_PCS', 6)),
        'Th_universal': float(getattr(config, 'KS4_TH_UNIVERSAL', 10.0)),
        'Th_learned': float(getattr(config, 'KS4_TH_LEARNED', 8.0)),
        'tmin': 0.0,
        'tmax': float(duration_s),
        'artifact_threshold': float(getattr(config, 'KS4_ARTIFACT_THRESHOLD', float('inf'))),
        'templates_from_data': True,
    }

    print(f'\n{"=" * 60}')
    print(f'  KS4 FULL RECORDING ({duration_s:.1f}s)')
    print(f'{"=" * 60}')
    print(f'[KS4-FULL] Binary: {bin_path}')
    print(f'[KS4-FULL] Device: {device}')
    print(f'[KS4-FULL] Channels: {n_channels}')

    t0 = time.time()
    ks4_out = run_kilosort(
        settings=settings,
        probe=probe,
        filename=bin_path,
        results_dir=ks4_dir,
        data_dtype='int16',
        do_CAR=bool(getattr(config, 'KS4_DO_CAR', True)),
        invert_sign=bool(getattr(config, 'KS4_INVERT_SIGN', False)),
        device=device,
        save_extra_vars=False,
    )

    # Parse output (same logic as official_ks4_bridge)
    if len(ks4_out) == 9:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = ks4_out
    elif len(ks4_out) == 8:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = ks4_out
        kept_spikes = None
    else:
        raise ValueError(f'Unexpected run_kilosort return length: {len(ks4_out)}')

    st = np.asarray(st)
    clu = np.asarray(clu)

    def _slice(x, idx):
        if x is None:
            return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                idx_t = torch.as_tensor(idx, dtype=torch.bool if idx.dtype == bool else torch.long,
                                        device=x.device)
                return x[idx_t]
        except Exception:
            pass
        return x[idx]

    # Apply kept_spikes mask
    if kept_spikes is not None:
        kept_spikes = np.asarray(kept_spikes, dtype=bool)
        st_kept = st[kept_spikes]
        clu_kept = clu[kept_spikes]
        tF_kept = _slice(tF, kept_spikes) if tF is not None and len(tF) == len(clu) else tF
    else:
        st_kept = st
        clu_kept = clu
        tF_kept = tF

    # Remove duplicates
    spike_times_raw = st_kept[:, 0].astype(np.int64)
    spike_labels = np.asarray(clu_kept, dtype=np.int32)
    dup_dt = int(ops['settings'].get('duplicate_spike_bins', 15)) if isinstance(ops, dict) else 15
    spike_times_raw, spike_labels, keep = remove_duplicates(spike_times_raw, spike_labels, dt=dup_dt)
    st_kept = st_kept[keep]
    if tF_kept is not None and len(tF_kept) == len(keep):
        tF_kept = _slice(tF_kept, keep)

    # Compute positions
    try:
        import torch
        if tF_kept is not None and not isinstance(tF_kept, torch.Tensor):
            tF_kept = torch.as_tensor(tF_kept)
    except Exception:
        pass

    xs, ys = compute_spike_positions(st_kept, tF_kept, ops)
    spike_xy = np.column_stack([xs, ys]).astype(np.float32)

    # Nearest channel
    from scipy.spatial.distance import cdist
    d = cdist(spike_xy, channel_positions.astype(np.float32))
    spike_channels = np.argmin(d, axis=1).astype(np.int32)

    n_units = int(np.unique(spike_labels).size)
    n_good = int(np.asarray(is_ref).sum()) if is_ref is not None else 0
    elapsed = time.time() - t0

    print(f'[KS4-FULL] Done: {len(spike_times_raw)} spikes, {n_units} clusters, '
          f'{n_good} good, {elapsed:.1f}s')

    result = {
        'backend': 'official_ks4_full',
        'all_times_raw': spike_times_raw,
        'all_channels': spike_channels,
        'all_labels': spike_labels,
        'is_ref': np.asarray(is_ref) if is_ref is not None else None,
        'spike_xy': spike_xy,
        'summary': {
            'n_spikes': int(len(spike_times_raw)),
            'n_clusters': n_units,
            'n_good_units': n_good,
            'duration_s': duration_s,
            'elapsed_s': elapsed,
        },
    }

    # Clean up binary if not needed
    if not getattr(config, 'KS4_FULL_KEEP_BINARY', False):
        try:
            os.remove(bin_path)
            print(f'[KS4-FULL] Removed binary: {bin_path}')
        except OSError:
            pass

    return result


# ==============================================================
# Main entry point (with caching)
# ==============================================================

def get_ks4_full_reference(streamer, channel_positions, output_dir):
    """Run KS4 on full recording, with caching.

    Returns dict with:
        all_times_raw : (N,) int64 — spike times in raw sample indices
        all_channels  : (N,) int32 — peak channel per spike
        all_labels    : (N,) int32 — cluster labels
        is_ref        : (n_clusters,) bool — good unit flag (optional)
        summary       : dict — n_spikes, n_clusters, n_good_units
    """
    if not getattr(config, 'KS4_FULL_ENABLE', False):
        print('[KS4-FULL] Disabled (set KS4_FULL_ENABLE=True to enable)')
        return None

    total_samples = int(streamer.total_duration_s * config.SAMPLE_RATE)
    n_channels = channel_positions.shape[0]
    cache_path = _full_cache_path(output_dir, total_samples, n_channels)

    if os.path.exists(cache_path):
        return _load_full_cache(cache_path)

    result = _run_ks4_full(streamer, channel_positions, output_dir)
    _save_full_cache(cache_path, result)
    return result


# ==============================================================
# Comparison helper
# ==============================================================

def compare_ks4_full_vs_iblsorter(ks4_full, one, eid, probe, channel_positions,
                                   raw_ind=None, t_start=None, t_end=None,
                                   output_dir=None):
    """Compare KS4 full-recording output vs iblsorter on a time window.

    Uses window_compare logic for matching.
    """
    from window_compare import _match_units_absolute, _print_summary
    from comparison import load_reference, filter_reference

    ref = load_reference(one, eid, probe, raw_ind=raw_ind)
    if ref is None:
        print('[KS4-FULL] Cannot load reference')
        return None

    sr = float(config.SAMPLE_RATE)
    ks_times_s = ks4_full['all_times_raw'].astype(np.float64) / sr

    if t_start is None:
        t_start = 0.0
    if t_end is None:
        t_end = float(ks_times_s.max()) + 1.0

    ref_filtered = filter_reference(ref, t_start, t_end)

    mask = (ks_times_s >= t_start) & (ks_times_s <= t_end)
    matches, summary = _match_units_absolute(
        ks_times_s[mask],
        ks4_full['all_labels'][mask],
        ks4_full['all_channels'][mask],
        ref_filtered,
        channel_positions=channel_positions,
        max_dt_ms=0.5,
        max_dist_um=100.0,
    )

    comp = {'reference': ref_filtered, 'matches': matches, 'summary': summary}
    _print_summary(f'KS4-FULL vs IBL [{t_start:.0f}-{t_end:.0f}s]',
                   t_start, t_end, comp)

    if output_dir:
        txt_path = os.path.join(output_dir,
                                f'compare_ks4_full_{int(t_start)}_{int(t_end)}s.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'summary': summary, 'top_matches': matches[:50]}, indent=2))
        print(f'[SAVE] {txt_path}')

    return comp


def compare_pipeline_vs_ks4_full(clust_results, ks4_full, channel_positions,
                                  t_start=None, t_end=None, output_dir=None):
    """Compare pipeline streaming output vs KS4 full-recording reference.

    This is the key metric: how much of KS4-full does the pipeline preserve?
    """
    from window_compare import _match_units_absolute, _print_summary

    sr = float(config.SAMPLE_RATE)

    # Pipeline times -> absolute seconds
    pipe_times_s = clust_results['times'].astype(np.float64) / sr
    pipe_times_s += config.CALIBRATION_DURATION_S
    pipe_labels = clust_results['labels']
    pipe_channels = clust_results['channels']

    # KS4-full as reference
    ks_times_s = ks4_full['all_times_raw'].astype(np.float64) / sr
    ks_labels = ks4_full['all_labels']
    ks_channels = ks4_full['all_channels']

    if t_start is None:
        t_start = config.CALIBRATION_DURATION_S
    if t_end is None:
        t_end = float(pipe_times_s.max()) + 1.0

    # Build pseudo-reference dict (same format as iblsorter reference)
    ks_mask = (ks_times_s >= t_start) & (ks_times_s <= t_end)
    n_clusters_ks = int(np.unique(ks_labels[ks_mask]).size) if ks_mask.any() else 0

    # Determine good units from is_ref
    is_ref = ks4_full.get('is_ref', None)
    unique_labels = np.unique(ks_labels[ks_mask]) if ks_mask.any() else np.array([])
    max_label = int(unique_labels.max()) + 1 if len(unique_labels) > 0 else 0
    cluster_labels = np.zeros(max_label, dtype=np.int32)  # 0 = not good
    if is_ref is not None:
        for cid in unique_labels:
            if cid < len(is_ref) and is_ref[cid]:
                cluster_labels[cid] = 1  # good

    n_good = int((cluster_labels == 1).sum())

    ks_ref = {
        'spike_times': ks_times_s[ks_mask],
        'spike_clusters': ks_labels[ks_mask],
        'spike_channels': ks_channels[ks_mask],
        'cluster_ids': unique_labels,
        'cluster_labels': cluster_labels,
        'n_good': n_good,
    }

    # Filter pipeline to same window
    pipe_mask = (pipe_times_s >= t_start) & (pipe_times_s <= t_end)
    matches, summary = _match_units_absolute(
        pipe_times_s[pipe_mask],
        pipe_labels[pipe_mask],
        pipe_channels[pipe_mask],
        ks_ref,
        channel_positions=channel_positions,
        max_dt_ms=0.5,
        max_dist_um=100.0,
    )

    comp = {'reference': ks_ref, 'matches': matches, 'summary': summary}
    _print_summary(f'PIPELINE vs KS4-FULL [{t_start:.0f}-{t_end:.0f}s]',
                   t_start, t_end, comp)

    if output_dir:
        txt_path = os.path.join(output_dir,
                                f'compare_pipeline_vs_ks4_full_{int(t_start)}_{int(t_end)}s.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'summary': summary, 'top_matches': matches[:50]}, indent=2))
        print(f'[SAVE] {txt_path}')

    return comp

# ==============================================================
# Full comparison (same format as iblsorter comparison)
# ==============================================================

def _build_ks4_full_ref_dict(ks4_full, t_start, t_end):
    """Build a reference dict from KS4-full in the same format as iblsorter reference."""
    sr = float(config.SAMPLE_RATE)
    ks_times_s = ks4_full['all_times_raw'].astype(np.float64) / sr
    ks_labels = ks4_full['all_labels']
    ks_channels = ks4_full['all_channels']

    mask = (ks_times_s >= t_start) & (ks_times_s <= t_end)
    times_f = ks_times_s[mask]
    labels_f = ks_labels[mask]
    channels_f = ks_channels[mask]

    unique_ids = np.unique(labels_f)
    max_label = int(unique_ids.max()) + 1 if len(unique_ids) > 0 else 0

    is_ref = ks4_full.get('is_ref', None)
    cluster_labels = np.zeros(max_label, dtype=np.int32)
    if is_ref is not None:
        for cid in unique_ids:
            if cid < len(is_ref) and is_ref[cid]:
                cluster_labels[cid] = 1

    cluster_channels = np.zeros(max_label, dtype=np.int32)
    for cid in unique_ids:
        ch = channels_f[labels_f == cid]
        if len(ch) > 0:
            cluster_channels[cid] = int(np.bincount(ch).argmax())

    n_good = int((cluster_labels == 1).sum())

    return {
        'spike_times': times_f,
        'spike_clusters': labels_f,
        'spike_channels': channels_f,
        'cluster_ids': unique_ids,
        'cluster_labels': cluster_labels,
        'cluster_channels': cluster_channels,
        'n_good': n_good,
        'sorter': 'kilosort4_full',
    }


def _get_pipeline_time_window(clust_results):
    """Get the absolute time window of pipeline output."""
    sr = float(config.SAMPLE_RATE)
    times_s = clust_results['times'].astype(np.float64) / sr
    times_s += config.CALIBRATION_DURATION_S
    return float(times_s.min()), float(times_s.max())


def run_full_ks4_comparison(clust_results, ks4_full, channel_positions,
                             output_dir=None):
    """Full comparison of pipeline vs KS4-full, same format as iblsorter comparison.

    Returns a comp dict compatible with run_all_diagnostics().
    """
    from comparison import match_units, compare_firing_rates, save_comparison

    t_start, t_end = _get_pipeline_time_window(clust_results)
    ref_filtered = _build_ks4_full_ref_dict(ks4_full, t_start, t_end)

    print(f"\n{'=' * 50}")
    print(f"  REFERENCE COMPARISON (vs KS4-FULL)")
    print(f"{'=' * 50}")

    matches, summary = match_units(clust_results, ref_filtered,
                                    channel_positions=channel_positions)

    rate_comp = None
    if channel_positions is not None:
        try:
            rate_comp = compare_firing_rates(
                clust_results, ref_filtered, channel_positions)
        except Exception:
            pass

    print(f"\n  — Comparison Summary (vs KS4-FULL) —")
    print(f"    Reference sorter: kilosort4_full")
    print(f"    Time window:      [{t_start:.2f}, {t_end:.2f}] s "
          f"({t_end - t_start:.1f} s)")
    print(f"    Pipeline:         {summary['n_pipe_clusters']} clusters, "
          f"{len(clust_results['labels'])} spikes")
    print(f"    Reference:        {summary['n_ref_clusters']} clusters "
          f"({summary['n_ref_good']} good), "
          f"{len(ref_filtered['spike_times'])} spikes")
    print(f"    Matched pairs:    {summary['n_matched_pairs']}")
    print(f"    Global recall:    {summary['global_recall']:.4f} "
          f"({summary['total_TP']}/{summary['total_ref_spikes']})")
    print(f"    Good-unit recall: {summary['good_global_recall']:.4f} "
          f"({summary['good_TP']}/{summary['good_ref_total']})")
    print(f"    Mean accuracy:    {summary['mean_accuracy']:.3f} "
          f"(all), {summary['mean_accuracy_good']:.3f} (good units)")
    print(f"    High accuracy:    {summary['n_high_accuracy']} pairs > 0.8")
    if rate_comp:
        print(f"    Rate correlation:  {rate_comp['correlation']:.3f}")
        print(f"    Rate ratio:        {rate_comp['ratio']:.2f} (pipe/ref)")

    if matches:
        print(f"\n    Top matched units:")
        for m in matches[:10]:
            good_str = " [GOOD]" if m['ref_is_good'] else ""
            print(f"      pipe={m['pipe_id']:3d} <-> "
                  f"ks4={m['ref_id']:3d}{good_str}: "
                  f"acc={m['accuracy']:.3f}, "
                  f"matched={m['n_matched']}/{m['n_ref']} "
                  f"(prec={m['precision']:.2f}, rec={m['recall']:.2f})")

    good_matches = [m for m in matches if m['ref_is_good']]
    matched_ref_ids = set(m['ref_id'] for m in matches)
    all_good_ref = [k for k in ref_filtered['cluster_ids']
                    if k < len(ref_filtered['cluster_labels'])
                    and ref_filtered['cluster_labels'][k] == 1]
    unmatched_good = [k for k in all_good_ref if k not in matched_ref_ids]

    if all_good_ref:
        print(f"\n    === KS4-FULL GOOD UNITS ({len(all_good_ref)} total) ===")
        print(f"      {'ks4':>5s} {'pipe':>5s}  {'n_ref':>5s} {'matched':>7s} "
              f"{'prec':>5s} {'rec':>5s} {'acc':>5s} {'peak_ch':>7s}")
        for m in sorted(good_matches, key=lambda m: -m['accuracy']):
            ch = ref_filtered['cluster_channels'][m['ref_id']] \
                if m['ref_id'] < len(ref_filtered['cluster_channels']) else -1
            print(f"      {m['ref_id']:5d} {m['pipe_id']:5d}  {m['n_ref']:5d} "
                  f"{m['n_matched']:7d} {m['precision']:.3f} "
                  f"{m['recall']:.3f} {m['accuracy']:.3f} {ch:>7d}")
        for rk in sorted(unmatched_good):
            n_ref = int((ref_filtered['spike_clusters'] == rk).sum())
            ch = ref_filtered['cluster_channels'][rk] \
                if rk < len(ref_filtered['cluster_channels']) else -1
            print(f"      {rk:5d}     -  {n_ref:5d}       -"
                  f"     -     -     - {ch:>7d}  (unmatched)")
        print(f"    ---")
        print(f"    Good units matched: {len(good_matches)}/{len(all_good_ref)}")
        if good_matches:
            ga = [m['accuracy'] for m in good_matches]
            print(f"    Accuracy:  mean={np.mean(ga):.3f}, "
                  f"median={np.median(ga):.3f}, "
                  f">0.5: {sum(1 for a in ga if a > 0.5)}, "
                  f">0.8: {sum(1 for a in ga if a > 0.8)}")
        print(f"    === END KS4-FULL GOOD UNITS ===")

    comp = {
        'ref_full': ref_filtered,
        'ref_filtered': ref_filtered,
        'matches': matches,
        'summary': summary,
        'rate_comparison': rate_comp,
    }

    if output_dir:
        save_comparison(comp, os.path.join(output_dir, 'comparison_vs_ks4_full'))

    return comp
