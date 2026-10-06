"""
Alignment Diagnostic — Three methods to fix calibration/streaming feature mismatch
====================================================================================

Run after calibration, before streaming.  Uses the calibration data itself
to measure how well each method aligns streaming-equivalent features with
calibration cluster centers.

Usage (add to main.py after clust.calibrate):
    from alignment_diagnostic import run_alignment_diagnostic
    run_alignment_diagnostic(whitened_all, clust, wTEMP, sa, positions, output_dir)

Outputs a text report and saves to alignment_diagnostic.txt.
"""

import numpy as np
import time
import os
from scipy.spatial.distance import cdist

import config
from template_matching import compute_score_matrix
from spike_clustering import extract_temporal_features


# ================================================================
# Shared: Streaming-equivalent detection on calibration data
# ================================================================

def _detect_streaming_on_calibdata(whitened_data, wTEMP, sa, chunk_size=None):
    """
    Run the quantized streaming detection pipeline on calibration data.
    Processes chunk-by-chunk with overlap, exactly matching streaming behavior.
    Returns event dict with times/channels/amplitudes/templates/scales.
    """
    cs = chunk_size or config.CHUNK_SAMPLES
    overlap_needed = config.MATCH_NT - 1
    T_total, n_ch = whitened_data.shape

    all_ev = {k: [] for k in
              ['times', 'channels', 'amplitudes', 'templates', 'scales']}
    prev_tail = None
    offset = 0
    n_chunks = 0

    for start in range(0, T_total, cs):
        end = min(start + cs, T_total)
        chunk = whitened_data[start:end]
        if chunk.shape[0] < config.MATCH_NT:
            break

        # Prepend overlap from previous chunk
        if prev_tail is not None:
            extended = np.vstack([prev_tail, chunk])
            ov_len = prev_tail.shape[0]
        else:
            extended = chunk
            ov_len = 0

        B, _ = compute_score_matrix(extended, wTEMP,
                                    precision_bits=config.PRECISION_MATCH)
        ev = sa.detect(B, stride=config.MATCH_STRIDE, verbose=False)
        n_det = len(ev['times'])

        # Remove detections in overlap zone
        if n_det > 0 and ov_len > 0:
            mask = ev['times'] >= ov_len
            for k in ev:
                if hasattr(ev[k], '__len__') and len(ev[k]) == n_det:
                    ev[k] = ev[k][mask]
            n_det = len(ev['times'])

        # Convert to global time
        if n_det > 0:
            ev['times'] = ev['times'] - ov_len + start
            for k in all_ev:
                if k in ev and len(ev[k]) > 0:
                    all_ev[k].append(ev[k])

        prev_tail = chunk[-overlap_needed:].copy()
        n_chunks += 1

    result = {}
    for k in all_ev:
        if all_ev[k]:
            result[k] = np.concatenate(all_ev[k])
        else:
            dt = np.int64 if k == 'times' else (
                np.float32 if k == 'amplitudes' else np.int32)
            result[k] = np.array([], dtype=dt)

    print(f"[DIAG] Streaming detection: {len(result['times'])} spikes "
          f"from {n_chunks} chunks")
    return result


# ================================================================
# Shared: Match streaming spikes to calibration spikes
# ================================================================

def _match_spikes(stream_times, stream_channels,
                  cali_times, cali_channels, max_dt=3):
    """
    For each streaming spike, find the nearest calibration spike
    (within ±max_dt samples, same or ±1 channel).
    Returns (n_stream,) int64 array: index into cali arrays, or -1.
    """
    n_stream = len(stream_times)
    matched = np.full(n_stream, -1, dtype=np.int64)

    if len(cali_times) == 0 or n_stream == 0:
        return matched

    order = np.argsort(cali_times)
    ct = cali_times[order]
    cc = cali_channels[order]

    for i in range(n_stream):
        t = stream_times[i]
        ch = stream_channels[i]
        lo = np.searchsorted(ct, t - max_dt, side='left')
        hi = np.searchsorted(ct, t + max_dt, side='right')
        if lo >= hi:
            continue

        cands = np.arange(lo, hi)
        dt_abs = np.abs(ct[cands] - t)

        # Prefer same channel
        same_ch = cc[cands] == ch
        if same_ch.any():
            sub = cands[same_ch]
            best = sub[np.argmin(np.abs(ct[sub] - t))]
            matched[i] = order[best]
        else:
            close_ch = np.abs(cc[cands].astype(np.int64) - ch) <= 1
            if close_ch.any():
                sub = cands[close_ch]
                best = sub[np.argmin(np.abs(ct[sub] - t))]
                matched[i] = order[best]

    return matched


# ================================================================
# Robust Feature Extraction (Method 2)
# ================================================================

def extract_robust_features(whitened_data, spike_times, spike_channels,
                            nearby_idx, hw=30):
    """
    Quantization-robust features: spatial amplitude profile + Peak-FSDE.

    Per spike:
      [0:N]    max |amplitude| on each of N nearby channels
      [N:N+4]  FSDE: FD_max, FD_min, SD_max, SD_min on peak channel
      [N+4]    signed peak amplitude on peak channel
    Total: N+5 dimensions (default N=10 -> 15d)
    """
    T_total, n_ch = whitened_data.shape
    n_spikes = len(spike_times)
    n_nearby = nearby_idx.shape[1]
    n_feat = n_nearby + 5

    features = np.zeros((n_spikes, n_feat), dtype=np.float32)
    times = np.asarray(spike_times, dtype=np.int64)
    channels = np.asarray(spike_channels, dtype=np.int64)

    for i in range(n_spikes):
        t = int(times[i])
        ch = int(channels[i])
        t0 = max(0, t - hw)
        t1 = min(T_total, t + hw + 1)
        if t1 - t0 < 5:
            continue

        chs = nearby_idx[ch]

        # Spatial amplitude profile
        for j, c in enumerate(chs):
            features[i, j] = np.max(np.abs(whitened_data[t0:t1, c]))

        # FSDE on peak channel
        sig = whitened_data[t0:t1, ch]
        d1 = np.diff(sig.astype(np.float64)).astype(np.float32)
        features[i, n_nearby] = np.max(d1)
        features[i, n_nearby + 1] = np.min(d1)
        if len(d1) > 1:
            d2 = np.diff(d1.astype(np.float64)).astype(np.float32)
            features[i, n_nearby + 2] = np.max(d2)
            features[i, n_nearby + 3] = np.min(d2)

        # Signed peak
        if 0 <= t < T_total:
            features[i, n_nearby + 4] = whitened_data[t, ch]

    return features


# ================================================================
# Evaluation helpers
# ================================================================

def _global_assign(features, centers):
    """Simple global nearest-center assignment. Returns labels, distances."""
    if len(features) == 0 or len(centers) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # Chunked to avoid memory explosion
    n = len(features)
    labels = np.zeros(n, dtype=np.int32)
    dists = np.zeros(n, dtype=np.float32)
    chunk = 5000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = cdist(features[s:e], centers, metric='sqeuclidean')
        lab = np.argmin(d2, axis=1)
        labels[s:e] = lab
        dists[s:e] = np.sqrt(d2[np.arange(e - s), lab])
    return labels, dists


def _regional_assign(features, spike_channels, ch_to_region,
                     region_centers, region_offsets):
    """Regional nearest-center assignment (matches streaming behavior)."""
    n_spikes = len(spike_channels)
    labels = np.zeros(n_spikes, dtype=np.int32)
    dists = np.zeros(n_spikes, dtype=np.float32)
    region_ids = ch_to_region[spike_channels]

    for r_id in np.unique(region_ids):
        mask = region_ids == r_id
        if mask.sum() == 0 or r_id not in region_centers:
            continue
        C = region_centers[r_id]
        if len(C) == 0:
            continue
        X = features[mask]
        d2 = cdist(X, C, metric='sqeuclidean')
        loc = np.argmin(d2, axis=1)
        labels[mask] = region_offsets[r_id] + loc
        dists[mask] = np.sqrt(d2[np.arange(len(loc)), loc])

    return labels, dists


def _count_good_units(spike_times_samples, labels, n_clusters,
                      sample_rate, min_spikes=20):
    """Count clusters with refractory ACG (good units)."""
    from merging import check_CCG
    spike_sec = spike_times_samples.astype(np.float64) / sample_rate
    n_good = 0
    n_checked = 0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() < min_spikes:
            continue
        n_checked += 1
        st = spike_sec[mask]
        is_ref, _, _ = check_CCG(st)
        if is_ref:
            n_good += 1
    return n_good, n_checked


def _evaluate(name, stream_times, stream_channels, stream_labels, stream_dists,
              n_clusters, matched_idx, cali_labels,
              sample_rate=None):
    """Compute evaluation metrics for one method."""
    fs = sample_rate or config.SAMPLE_RATE
    n_stream = len(stream_times)
    result = {'name': name}

    if n_stream == 0:
        result['n_spikes'] = 0
        return result

    result['n_spikes'] = n_stream

    # -- Distance stats --
    valid_d = stream_dists[stream_dists > 0]
    if len(valid_d) > 0:
        result['dist_mean'] = float(np.mean(valid_d))
        result['dist_median'] = float(np.median(valid_d))
        result['dist_std'] = float(np.std(valid_d))
        result['dist_p90'] = float(np.percentile(valid_d, 90))
    else:
        result['dist_mean'] = 0
        result['dist_median'] = 0

    # -- Cluster utilization --
    counts = np.bincount(stream_labels[stream_labels >= 0],
                         minlength=n_clusters)
    result['clusters_used_1'] = int((counts >= 1).sum())
    result['clusters_used_5'] = int((counts >= 5).sum())
    result['clusters_total'] = n_clusters
    result['utilization_pct'] = round(
        (counts >= 5).sum() / max(n_clusters, 1) * 100, 1)

    # -- Label preservation (for matched spikes) --
    has_match = matched_idx >= 0
    n_matched = has_match.sum()
    result['n_matched'] = int(n_matched)
    if n_matched > 0:
        cali_lab = cali_labels[matched_idx[has_match]]
        stream_lab = stream_labels[has_match]
        agree = (cali_lab == stream_lab).sum()
        result['label_preservation'] = round(agree / n_matched * 100, 1)
    else:
        result['label_preservation'] = 0.0

    # -- Good units (ACG refractory) --
    try:
        n_good, n_checked = _count_good_units(
            stream_times, stream_labels, n_clusters, fs)
        result['good_units'] = n_good
        result['checked_units'] = n_checked
    except Exception as e:
        result['good_units'] = -1
        result['checked_units'] = -1
        result['good_unit_error'] = str(e)

    # -- Cluster size distribution --
    active_counts = counts[counts >= 5]
    if len(active_counts) > 0:
        result['cluster_size_median'] = int(np.median(active_counts))
        result['cluster_size_max'] = int(np.max(active_counts))
    else:
        result['cluster_size_median'] = 0
        result['cluster_size_max'] = 0

    return result


# ================================================================
# Method 1: Recalibrate centers in quantized space
# ================================================================

def method1_recalibrate(whitened_data, cali_result, temporal_pca,
                        nearby_idx, positions,
                        stream_events, matched_idx,
                        lda_mean=None, lda_scalings_q=None, lda_mode='off',
                        direct_nearby_idx=None, lda_per_region=None):
    """
    Inherit calibration labels for matched streaming spikes,
    recompute cluster centers from streaming features.
    """
    print("\n" + "=" * 60)
    print("  METHOD 1: Recalibrate Centers in Quantized Space")
    print("=" * 60)

    cali_labels = cali_result['all_labels']
    n_clusters = cali_labels.max() + 1

    if lda_mode == 'direct' and lda_mean is not None and lda_scalings_q is not None:
        from spike_clustering import _apply_lda, extract_raw_waveform_features
        snippet_len = 2 * config.CLUST_SNIPPET_HW + 1
        raw_feat = extract_raw_waveform_features(
            whitened_data, stream_events['times'], stream_events['channels'],
            positions, snippet_len=snippet_len,
            nearby_idx=direct_nearby_idx)
        stream_feat = _apply_lda(raw_feat, lda_mean, lda_scalings_q)
    elif lda_mode == 'per_region' and lda_per_region:
        from spike_clustering import _apply_lda, compute_channel_regions
        temporal_pca.set_full_precision(False)
        pca_feat = extract_temporal_features(
            whitened_data, stream_events['times'], stream_events['channels'],
            positions, temporal_pca, nearby_idx=nearby_idx)
        _, ch_to_region = compute_channel_regions(positions)
        region_ids = ch_to_region[stream_events['channels']]
        n_lda = max(s['scalings_q'].shape[1] for s in lda_per_region.values())
        stream_feat = np.zeros((len(pca_feat), n_lda), dtype=np.float32)
        for r_id, lda_state in lda_per_region.items():
            mask = region_ids == r_id
            if mask.any():
                proj = _apply_lda(
                    pca_feat[mask], lda_state['mean'], lda_state['scalings_q'])
                actual_n = proj.shape[1]
                stream_feat[mask, :actual_n] = proj
    else:
        temporal_pca.set_full_precision(False)
        stream_feat = extract_temporal_features(
            whitened_data, stream_events['times'], stream_events['channels'],
            positions, temporal_pca, nearby_idx=nearby_idx)
        if lda_mode == 'post_pca' and lda_mean is not None and lda_scalings_q is not None:
            from spike_clustering import _apply_lda
            stream_feat = _apply_lda(stream_feat, lda_mean, lda_scalings_q)

    n_stream = len(stream_events['times'])
    has_match = matched_idx >= 0
    n_matched = has_match.sum()
    print(f"  Matched: {n_matched}/{n_stream} "
          f"({n_matched / max(n_stream, 1) * 100:.1f}%)")

    # Inherit labels
    inherited = np.full(n_stream, -1, dtype=np.int32)
    inherited[has_match] = cali_labels[matched_idx[has_match]]

    # Recompute centers from streaming features
    n_feat = stream_feat.shape[1]
    new_centers = np.zeros((n_clusters, n_feat), dtype=np.float32)
    new_counts = np.zeros(n_clusters, dtype=np.int64)
    for cid in range(n_clusters):
        mask = inherited == cid
        if mask.sum() > 0:
            new_centers[cid] = stream_feat[mask].mean(axis=0)
            new_counts[cid] = mask.sum()

    # Only keep populated centers to avoid ghost-center assignment
    populated_mask = new_counts > 0
    populated_ids = np.where(populated_mask)[0]
    n_populated = len(populated_ids)
    print(f"  Centers recomputed: {n_populated}/{n_clusters} populated")

    if n_populated == 0:
        labels = np.full(n_stream, -1, dtype=np.int32)
        dists = np.full(n_stream, np.inf, dtype=np.float32)
        return labels, dists, new_centers, stream_feat

    # Assign to populated centers only, then map back to original IDs
    populated_centers = new_centers[populated_ids]
    raw_labels, dists = _global_assign(stream_feat, populated_centers)
    labels = populated_ids[raw_labels]  # map back to original cluster IDs

    return labels, dists, new_centers, stream_feat


# ================================================================
# Method 2: Robust features (spatial amplitude + FSDE)
# ================================================================

def method2_robust(whitened_data, cali_result, nearby_idx):
    """
    Extract 15d robust features for both calibration and streaming spikes.
    Compute centers from calibration labels + robust features.
    """
    print("\n" + "=" * 60)
    print("  METHOD 2: Robust Features (Spatial + FSDE)")
    print("=" * 60)

    cali_times = cali_result['all_times']
    cali_channels = cali_result['all_channels']
    cali_labels = cali_result['all_labels']
    n_clusters = cali_labels.max() + 1

    # Extract robust features for calibration spikes
    print("  Extracting robust features for calibration spikes...")
    t0 = time.time()
    cali_robust = extract_robust_features(
        whitened_data, cali_times, cali_channels, nearby_idx)
    print(f"  Done: {cali_robust.shape} in {time.time() - t0:.1f}s")

    # Normalize features (z-score per dimension)
    feat_mean = cali_robust.mean(axis=0)
    feat_std = cali_robust.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    cali_robust_norm = (cali_robust - feat_mean) / feat_std

    # Compute centers from calibration labels
    n_feat = cali_robust_norm.shape[1]
    centers = np.zeros((n_clusters, n_feat), dtype=np.float32)
    populated_mask = np.zeros(n_clusters, dtype=bool)
    for cid in range(n_clusters):
        mask = cali_labels == cid
        if mask.sum() > 0:
            centers[cid] = cali_robust_norm[mask].mean(axis=0)
            populated_mask[cid] = True

    n_populated = populated_mask.sum()
    print(f"  Centers: {n_populated}/{n_clusters} populated")

    return centers, feat_mean, feat_std, populated_mask


def method2_assign(whitened_data, stream_events, nearby_idx,
                   centers, feat_mean, feat_std, populated_mask=None):
    """Assign streaming spikes using robust features."""
    print("  Extracting robust features for streaming spikes...")
    t0 = time.time()
    stream_robust = extract_robust_features(
        whitened_data,
        stream_events['times'], stream_events['channels'],
        nearby_idx)
    stream_robust_norm = (stream_robust - feat_mean) / feat_std
    print(f"  Done: {stream_robust.shape} in {time.time() - t0:.1f}s")

    # Only assign to populated centers
    if populated_mask is not None and not populated_mask.all():
        populated_ids = np.where(populated_mask)[0]
        if len(populated_ids) == 0:
            n = len(stream_events['times'])
            return (np.full(n, -1, dtype=np.int32),
                    np.full(n, np.inf, dtype=np.float32),
                    stream_robust_norm)
        raw_labels, dists = _global_assign(stream_robust_norm,
                                           centers[populated_ids])
        labels = populated_ids[raw_labels]
    else:
        labels, dists = _global_assign(stream_robust_norm, centers)

    return labels, dists, stream_robust_norm


# ================================================================
# Method 3: Linear alignment transform
# ================================================================

def method3_alignment(whitened_data, cali_result, temporal_pca,
                      nearby_idx, positions,
                      stream_events, matched_idx, stream_feat_quant):
    """
    Learn affine transform from full-precision → quantized features.
    Apply to calibration centers to get aligned centers.
    """
    print("\n" + "=" * 60)
    print("  METHOD 3: Linear Alignment Transform")
    print("=" * 60)

    cali_times = cali_result['all_times']
    cali_channels = cali_result['all_channels']
    cali_features_full = cali_result['features']  # full-precision 60d

    has_match = matched_idx >= 0
    n_matched = has_match.sum()
    print(f"  Matched pairs for alignment: {n_matched}")

    if n_matched < 100:
        print("  ERROR: Too few matched pairs, skipping")
        return None, None, None, None

    # F = full-precision features of matched calibration spikes
    F = cali_features_full[matched_idx[has_match]]
    # G = quantized features of corresponding streaming spikes
    G = stream_feat_quant[has_match]

    # Subsample if too many (for speed)
    max_pairs = 50000
    if len(F) > max_pairs:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(len(F), max_pairs, replace=False)
        F = F[idx]
        G = G[idx]

    # Feature-space statistics before alignment
    cos_before = np.sum(F * G, axis=1) / (
        np.linalg.norm(F, axis=1) * np.linalg.norm(G, axis=1) + 1e-12)
    print(f"  Cosine similarity BEFORE alignment: "
          f"mean={cos_before.mean():.4f}, "
          f"median={np.median(cos_before):.4f}")

    # Solve G ≈ F @ W^T + b via augmented least squares
    F_aug = np.hstack([F, np.ones((len(F), 1), dtype=np.float32)])
    P, residuals, rank, sv = np.linalg.lstsq(F_aug, G, rcond=None)
    W = P[:-1].T.astype(np.float32)   # (60, 60)
    b = P[-1].astype(np.float32)      # (60,)

    # Check alignment quality
    G_pred = F @ W.T + b
    residual = G - G_pred
    rmse = np.sqrt((residual ** 2).mean())
    cos_after = np.sum(G_pred * G, axis=1) / (
        np.linalg.norm(G_pred, axis=1) * np.linalg.norm(G, axis=1) + 1e-12)
    print(f"  RMSE: {rmse:.4f} (feature std: {G.std():.4f}, "
          f"ratio: {rmse / max(G.std(), 1e-8):.3f})")
    print(f"  Cosine similarity AFTER alignment: "
          f"mean={cos_after.mean():.4f}, "
          f"median={np.median(cos_after):.4f}")
    print(f"  Transform rank: {rank}, "
          f"SV range: [{sv.min():.2f}, {sv.max():.2f}]")

    # Transform calibration centers to quantized space
    centers_full = cali_result['centers_60d']
    centers_aligned = (centers_full @ W.T + b).astype(np.float32)

    # Assign all streaming spikes to aligned centers
    labels, dists = _global_assign(stream_feat_quant, centers_aligned)

    return labels, dists, centers_aligned, {'W': W, 'b': b,
                                             'rmse': rmse,
                                             'cos_before': cos_before,
                                             'cos_after': cos_after}


# ================================================================
# Main diagnostic entry point
# ================================================================

def run_alignment_diagnostic(whitened_data, clust, wTEMP, sa, positions,
                             output_dir="results"):
    """
    Run all three alignment methods and print comparison.

    Parameters
    ----------
    whitened_data : (T, n_ch) float32 — calibration data (whitened)
    clust : SpikeClustering — calibrated pipeline
    wTEMP : (n_templates, nt) float32
    sa : SpatialAggregator
    positions : (n_ch, 2) float32
    output_dir : str — for saving report

    Returns
    -------
    report : dict with all metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'alignment_diagnostic.txt')
    t_total_start = time.time()

    cali_result = clust._cali_result
    temporal_pca = clust.temporal_pca
    nearby_idx = clust._nearby_idx
    ch_to_region = clust.assigner.ch_to_region

    cali_times = cali_result['all_times']
    cali_channels = cali_result['all_channels']
    cali_labels = cali_result['all_labels']
    cali_centers = cali_result['centers_60d']
    n_clusters = cali_labels.max() + 1

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  ALIGNMENT DIAGNOSTIC")
    log("=" * 70)
    log(f"  Calibration data: {whitened_data.shape[0]} samples "
        f"({whitened_data.shape[0] / config.SAMPLE_RATE:.1f}s), "
        f"{whitened_data.shape[1]} channels")
    log(f"  Calibration spikes: {len(cali_times)}")
    log(f"  Calibration clusters: {n_clusters}")
    log(f"  Precisions: Match={config.PRECISION_MATCH}b, "
        f"SA={config.PRECISION_SA}b, "
        f"PCA={config.PRECISION_PCA}b, "
        f"Assign={config.PRECISION_ASSIGN}b")

    # ==== Phase 0: Streaming detection on calibration data ====
    log(f"\n{'─' * 70}")
    log("  PHASE 0: Streaming-equivalent detection on calibration data")
    log(f"{'─' * 70}")
    t0 = time.time()
    stream_ev = cali_result.get('_stream_ev') if isinstance(cali_result, dict) else None
    if isinstance(stream_ev, dict) and 'times' in stream_ev and 'channels' in stream_ev:
        stream_ev = {k: np.asarray(v) for k, v in stream_ev.items()}
        if len(stream_ev['times']) and int(np.max(stream_ev['times'])) >= whitened_data.shape[0]:
            stream_ev = None
    if stream_ev is None:
        stream_ev = _detect_streaming_on_calibdata(whitened_data, wTEMP, sa)
    else:
        log("  Using cached streaming detection from calibration")
    n_stream = len(stream_ev['times'])
    log(f"  Streaming detected: {n_stream} spikes ({time.time() - t0:.1f}s)")
    log(f"  Detection ratio: {n_stream / max(len(cali_times), 1) * 100:.1f}% "
        f"of calibration spike count")

    if n_stream == 0:
        log("\n  FATAL: No streaming spikes detected. Cannot continue.")
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        return {'error': 'no_streaming_spikes'}

    # ==== Phase 1: Match streaming ↔ calibration spikes ====
    log(f"\n{'─' * 70}")
    log("  PHASE 1: Spike matching (streaming ↔ calibration)")
    log(f"{'─' * 70}")
    t0 = time.time()
    matched_idx = _match_spikes(
        stream_ev['times'], stream_ev['channels'],
        cali_times, cali_channels, max_dt=5)
    has_match = matched_idx >= 0
    n_matched = has_match.sum()
    match_rate = n_matched / max(n_stream, 1) * 100
    log(f"  Matched: {n_matched}/{n_stream} ({match_rate:.1f}%)")
    # Check how many unique calibration spikes were matched
    n_unique_cali = len(np.unique(matched_idx[has_match]))
    log(f"  Unique calibration spikes matched: {n_unique_cali}/{len(cali_times)} "
        f"({n_unique_cali / max(len(cali_times), 1) * 100:.1f}%)")

    # Channel offset distribution
    if n_matched > 0:
        ch_stream = stream_ev['channels'][has_match]
        ch_cali = cali_channels[matched_idx[has_match]]
        ch_offset = ch_stream.astype(np.int64) - ch_cali.astype(np.int64)
        log(f"  Channel offset: same={( ch_offset == 0).sum()} "
            f"(±1={( np.abs(ch_offset) <= 1).sum()})")
        t_stream = stream_ev['times'][has_match]
        t_cali = cali_times[matched_idx[has_match]]
        dt = t_stream.astype(np.int64) - t_cali.astype(np.int64)
        log(f"  Time offset (samples): mean={dt.mean():.2f}, "
            f"std={dt.std():.2f}, "
            f"|dt|<=1: {(np.abs(dt) <= 1).sum()}, "
            f"|dt|<=3: {(np.abs(dt) <= 3).sum()}")
    log(f"  ({time.time() - t0:.1f}s)")

    # ==== Phase 2: Extract quantized 60d features (shared by M0, M1, M3) ====
    log(f"\n{'─' * 70}")
    log("  PHASE 2: Extract streaming-equivalent 60d features")
    log(f"{'─' * 70}")
    t0 = time.time()
    temporal_pca.set_full_precision(False)
    stream_feat_q = extract_temporal_features(
        whitened_data, stream_ev['times'], stream_ev['channels'],
        positions, temporal_pca, nearby_idx=nearby_idx)
    log(f"  Features: {stream_feat_q.shape} ({time.time() - t0:.1f}s)")

    # Apply LDA projection if enabled (must match cali_result['features'] space)
    _lda_mean = getattr(clust, 'lda_mean_', None)
    _lda_scalings_q = getattr(clust, 'lda_scalings_q_', None)
    _lda_enabled = getattr(clust, 'lda_enabled', False)
    _lda_mode = getattr(clust, 'lda_mode', 'off')
    _lda_per_region = getattr(clust, 'lda_per_region_', None)
    if _lda_enabled:
        from spike_clustering import _apply_lda, extract_raw_waveform_features
        if _lda_mode == 'direct' and _lda_scalings_q is not None:
            snippet_len = 2 * config.CLUST_SNIPPET_HW + 1
            _direct_nearby_idx = getattr(clust, '_direct_nearby_idx', None)
            raw_stream = extract_raw_waveform_features(
                whitened_data, stream_ev['times'], stream_ev['channels'],
                positions, snippet_len=snippet_len,
                nearby_idx=_direct_nearby_idx)
            stream_feat_q = _apply_lda(raw_stream, _lda_mean, _lda_scalings_q)
            log(f"  Direct LDA projected: {raw_stream.shape[1]}d -> {stream_feat_q.shape[1]}d")
        elif _lda_mode == 'per_region' and _lda_per_region:
            ch_to_region = clust.assigner.ch_to_region
            region_ids = ch_to_region[stream_ev['channels']]
            # Find max LDA output dim across regions
            n_lda = max(s['scalings_q'].shape[1] for s in _lda_per_region.values())
            stream_feat_pr = np.zeros((len(stream_feat_q), n_lda), dtype=np.float32)
            for r_id, lda_state in _lda_per_region.items():
                mask = region_ids == r_id
                if mask.any():
                    proj = _apply_lda(
                        stream_feat_q[mask], lda_state['mean'], lda_state['scalings_q'])
                    actual_n = proj.shape[1]
                    stream_feat_pr[mask, :actual_n] = proj
            stream_feat_q = stream_feat_pr
            log(f"  Per-region LDA projected: 60d -> {n_lda}d")
        elif _lda_mode == 'post_pca' and _lda_scalings_q is not None:
            stream_feat_q = _apply_lda(stream_feat_q, _lda_mean, _lda_scalings_q)
            log(f"  LDA projected: {stream_feat_q.shape[1]}d")

    # Feature-space alignment diagnostic
    if n_matched > 100:
        cali_feat_full = cali_result['features']
        F = cali_feat_full[matched_idx[has_match]]
        G = stream_feat_q[has_match]
        # Per-dimension shift
        dim_shift = np.mean(G - F, axis=0)
        dim_shift_norm = np.linalg.norm(dim_shift)
        feat_norm = np.linalg.norm(F, axis=1).mean()
        log(f"  Mean feature shift (cali→stream): L2={dim_shift_norm:.4f} "
            f"(vs avg feature norm {feat_norm:.4f}, "
            f"ratio={dim_shift_norm / max(feat_norm, 1e-8):.3f})")
        cos = np.sum(F * G, axis=1) / (
            np.linalg.norm(F, axis=1) * np.linalg.norm(G, axis=1) + 1e-12)
        log(f"  Cosine similarity (matched pairs): "
            f"mean={cos.mean():.4f}, median={np.median(cos):.4f}, "
            f"<0.9: {(cos < 0.9).sum()}, <0.5: {(cos < 0.5).sum()}")

    m1_only = bool(getattr(config, 'DIAG_M1_ONLY', False))

    # ==== BASELINE (Method 0) ====
    m0 = None
    if not m1_only:
        log(f"\n{'─' * 70}")
        log("  METHOD 0: BASELINE (original calibration centers, quantized features)")
        log(f"{'─' * 70}")
        t0 = time.time()
        m0_labels, m0_dists = _global_assign(stream_feat_q, cali_centers)
        m0 = _evaluate("M0_baseline", stream_ev['times'], stream_ev['channels'],
                       m0_labels, m0_dists, n_clusters, matched_idx, cali_labels)
        m0['time_s'] = time.time() - t0
        _log_result(log, m0)

    # ==== METHOD 1: Recalibrate centers ====
    t0 = time.time()
    _direct_nearby_idx = getattr(clust, '_direct_nearby_idx', None)
    m1_labels, m1_dists, m1_centers, _ = method1_recalibrate(
        whitened_data, cali_result, temporal_pca, nearby_idx, positions,
        stream_ev, matched_idx,
        lda_mean=_lda_mean if _lda_enabled else None,
        lda_scalings_q=_lda_scalings_q if _lda_enabled else None,
        lda_mode=_lda_mode,
        direct_nearby_idx=_direct_nearby_idx,
        lda_per_region=_lda_per_region if _lda_enabled else None)
    m1 = _evaluate("M1_recalibrate", stream_ev['times'], stream_ev['channels'],
                   m1_labels, m1_dists, n_clusters, matched_idx, cali_labels)
    m1['time_s'] = time.time() - t0
    _log_result(log, m1)

    # ==== METHOD 2 & 3 (optional) ====
    m2, m3 = None, None
    m2_centers, m2_mean, m2_std, m3_centers, m3_info = None, None, None, None, None
    if not m1_only:
        t0 = time.time()
        m2_centers, m2_mean, m2_std, m2_pop = method2_robust(
            whitened_data, cali_result, nearby_idx)
        m2_labels, m2_dists, m2_feat = method2_assign(
            whitened_data, stream_ev, nearby_idx,
            m2_centers, m2_mean, m2_std, populated_mask=m2_pop)
        m2 = _evaluate("M2_robust", stream_ev['times'], stream_ev['channels'],
                       m2_labels, m2_dists, n_clusters, matched_idx, cali_labels)
        m2['time_s'] = time.time() - t0
        _log_result(log, m2)

        t0 = time.time()
        m3_labels, m3_dists, m3_centers, m3_info = method3_alignment(
            whitened_data, cali_result, temporal_pca, nearby_idx, positions,
            stream_ev, matched_idx, stream_feat_q)
        if m3_labels is not None:
            m3 = _evaluate("M3_alignment", stream_ev['times'], stream_ev['channels'],
                           m3_labels, m3_dists, n_clusters, matched_idx, cali_labels)
        else:
            m3 = {'name': 'M3_alignment', 'n_spikes': 0,
                   'label_preservation': 0, 'good_units': -1}
        m3['time_s'] = time.time() - t0
        _log_result(log, m3)

    # ==== COMPARISON TABLE ====
    log(f"\n{'=' * 70}")
    log("  COMPARISON SUMMARY")
    log(f"{'=' * 70}")
    methods = [m for m in [m0, m1, m2, m3] if m is not None]
    names = [m.get('name', '?') for m in methods]
    header = f"{'Metric':<30}" + "".join(f" {n:>14}" for n in names)
    log(header)
    log("─" * len(header))
    for key in ['label_preservation', 'good_units', 'checked_units',
                'clusters_used_5', 'utilization_pct',
                'dist_mean', 'dist_median', 'dist_p90',
                'cluster_size_median', 'time_s']:
        vals = []
        for m in methods:
            v = m.get(key, '—')
            if isinstance(v, float):
                vals.append(f"{v:>14.2f}")
            elif isinstance(v, int):
                vals.append(f"{v:>14d}")
            else:
                vals.append(f"{str(v):>14}")
        log(f"{key:<30}" + "".join(vals))

    log(f"\n  Total diagnostic time: {time.time() - t_total_start:.1f}s")
    log(f"  Detection overlap: {n_stream} streaming vs {len(cali_times)} calibration")
    log(f"  Match rate: {match_rate:.1f}%")

    # ==== INTERPRETATION ====
    log(f"\n{'=' * 70}")
    log("  INTERPRETATION GUIDE")
    log(f"{'=' * 70}")
    log("  label_preservation: % matched spikes assigned to same cluster as cali")
    log("     >80% = feature alignment is good; <50% = serious mismatch")
    log("  good_units: clusters meeting the refractory ACG criterion")
    log("     Higher = better; the ultimate metric")
    log("  utilization_pct: % of calibration clusters receiving ≥5 spikes")
    log("     Low = many ghost clusters; streaming misses entire neuron types")
    log("  dist_mean: average Euclidean distance to assigned center")
    log("     Lower = tighter clusters; compare across methods")
    log(f"{'=' * 70}")

    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"\n  Report saved to: {report_path}")

    return {
        'baseline': m0, 'method1': m1, 'method2': m2, 'method3': m3,
        'n_stream': n_stream, 'n_cali': len(cali_times),
        'match_rate': match_rate,
        'm1_centers': m1_centers,
        'm2_centers': m2_centers, 'm2_mean': m2_mean, 'm2_std': m2_std,
        'm3_centers': m3_centers, 'm3_info': m3_info,
    }


def _log_result(log, m):
    """Print evaluation result for one method."""
    name = m.get('name', '?')
    log(f"\n  Results for {name}:")
    log(f"    Label preservation: {m.get('label_preservation', '?')}%")
    log(f"    Good units: {m.get('good_units', '?')}/{m.get('checked_units', '?')}")
    log(f"    Cluster utilization (≥5): {m.get('clusters_used_5', '?')}"
        f"/{m.get('clusters_total', '?')} "
        f"({m.get('utilization_pct', '?')}%)")
    log(f"    Distance to center: mean={m.get('dist_mean', '?'):.2f}, "
        f"median={m.get('dist_median', '?'):.2f}")
    if 'dist_p90' in m:
        log(f"    Distance P90: {m['dist_p90']:.2f}")


# ================================================================
# Standalone entry point (for testing)
# ================================================================

if __name__ == "__main__":
    print("This module is meant to be called from main.py.")
    print("Add the following after clust.calibrate(...):")
    print()
    print("    from alignment_diagnostic import run_alignment_diagnostic")
    print("    diag = run_alignment_diagnostic(")
    print("        whitened_all, clust, wTEMP, sa, positions, output_dir)")
    print()
