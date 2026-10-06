"""
Diagnostics Module
===================
Extracted from main.py — all reference-comparison diagnostic functions.
Enable/disable via config flags:
  DIAG_DETECTION_AUDIT, DIAG_FN_ANALYSIS,
  DIAG_FEATURE_SEPARABILITY, DIAG_LABEL_ANALYSIS
"""

import numpy as np
import config


# ==============================================================
# Helper: convert pipeline sample times → absolute seconds
# ==============================================================

def pipe_times_to_seconds(clust_results):
    """Convert FIR-corrected sample times to absolute seconds.

    Pipeline times are sample indices relative to the start of the
    streaming phase (after calibration).  They are already corrected
    for FIR group delay in get_results().
    """
    times_samples = clust_results['times']
    return times_samples.astype(np.float64) / config.SAMPLE_RATE \
        + config.CALIBRATION_DURATION_S


# ==============================================================
# 1. Detection Audit  (recall / precision vs reference)
# ==============================================================

def run_detection_audit(comp, clust_results, positions):
    """Measure what fraction of reference spikes the pipeline detected.

    Uses time + spatial constraints with multiple spatial thresholds.

    Returns
    -------
    audit : dict  —  keyed by spatial threshold, contains recall, etc.
    """
    print(f"\n{'=' * 60}")
    print(f"  DETECTION AUDIT: ref→pipe recall (time + space)")
    print(f"{'=' * 60}")

    ref_f = comp['ref_filtered']
    ref_times = ref_f['spike_times']
    ref_clusters = ref_f['spike_clusters']
    ref_channels = ref_f['spike_channels']

    pipe_times_sec = pipe_times_to_seconds(clust_results)
    pipe_channels = clust_results['channels']

    print(f"  Ref spikes:  {len(ref_times)}")
    print(f"  Pipe spikes: {len(pipe_times_sec)}")

    from scipy.spatial import cKDTree
    tree_pipe = cKDTree(pipe_times_sec.reshape(-1, 1))
    max_dt = 0.0005  # 0.5 ms

    ref_neighbors = tree_pipe.query_ball_point(
        ref_times.reshape(-1, 1), r=max_dt)

    # For each ref spike find best pipe neighbour (by spatial distance)
    ref_best_dist = np.full(len(ref_times), np.inf)
    ref_best_pipe_idx = np.full(len(ref_times), -1, dtype=np.int64)

    for i, neighbors in enumerate(ref_neighbors):
        if len(neighbors) == 0:
            continue
        ref_ch = int(ref_channels[i])
        ref_pos = positions[ref_ch]
        for j in neighbors:
            d = np.sqrt(((positions[int(pipe_channels[j])] - ref_pos) ** 2).sum())
            if d < ref_best_dist[i]:
                ref_best_dist[i] = d
                ref_best_pipe_idx[i] = j

    spatial_thresholds = [50, 100, 150]
    audit = {}

    for sp_thresh in spatial_thresholds:
        ref_detected = ref_best_dist < sp_thresh
        n_ref = len(ref_times)
        n_det = ref_detected.sum()
        recall = n_det / n_ref if n_ref > 0 else 0

        # Per-neuron recall
        unique_ids = np.unique(ref_clusters)
        neuron_recalls = {}
        for nid in unique_ids:
            mask_n = ref_clusters == nid
            n_total_n = mask_n.sum()
            n_det_n = (ref_detected & mask_n).sum()
            neuron_recalls[int(nid)] = (n_det_n, n_total_n,
                                         n_det_n / n_total_n if n_total_n else 0)
        recalls = np.array([v[2] for v in neuron_recalls.values()])

        pipe_used = ref_best_pipe_idx[ref_detected]
        n_dup = len(pipe_used) - len(np.unique(pipe_used))

        print(f"\n  --- Spatial threshold: {sp_thresh} μm ---")
        print(f"  Overall recall: {n_det}/{n_ref} = {recall:.3f}")
        print(f"  Duplicate assignments: {n_dup}")
        print(f"  Per-neuron recall (n={len(unique_ids)}): "
              f"mean={recalls.mean():.3f}, median={np.median(recalls):.3f}")
        print(f"    recall=0: {(recalls == 0).sum()},  "
              f"<0.5: {(recalls < 0.5).sum()},  "
              f">=0.8: {(recalls >= 0.8).sum()},  "
              f">=0.95: {(recalls >= 0.95).sum()}")

        # Breakdown by quality label
        ref_labels = ref_f['cluster_labels']
        neuron_label_vals = np.array([ref_labels[int(nid)]
                                      for nid in unique_ids])
        for lval, lname in [(1, 'good'), (2, 'mua'), (0, 'noise')]:
            lmask = neuron_label_vals == lval
            if lmask.sum() == 0:
                continue
            lr = recalls[lmask]
            print(f"  {lname:5s} (n={lmask.sum():3d}): "
                  f"mean={lr.mean():.3f}, median={np.median(lr):.3f}, "
                  f">=0.8: {(lr >= 0.8).sum()}")

        audit[sp_thresh] = {
            'recall': recall,
            'per_neuron_recalls': neuron_recalls,
            'n_dup': n_dup,
        }

    # Sensitivity check
    r_vals = [audit[t]['recall'] for t in spatial_thresholds]
    spread = max(r_vals) - min(r_vals)
    print(f"\n  Sensitivity: "
          + ", ".join(f"@{t}μm={r:.3f}" for t, r in zip(spatial_thresholds, r_vals)))
    if spread < 0.05:
        print(f"  → Robust (spread={spread:.3f})")
    else:
        print(f"  → Sensitive (spread={spread:.3f}) — check channel alignment")

    # --- Precision (forward direction) at 100 μm ---
    tree_ref = cKDTree(ref_times.reshape(-1, 1))
    pipe_neighbors = tree_ref.query_ball_point(
        pipe_times_sec.reshape(-1, 1), r=max_dt)
    pipe_has_ref = np.zeros(len(pipe_times_sec), dtype=bool)
    for i, neighbors in enumerate(pipe_neighbors):
        for j in neighbors:
            d = np.sqrt(((positions[int(ref_channels[j])]
                          - positions[int(pipe_channels[i])]) ** 2).sum())
            if d < 100:
                pipe_has_ref[i] = True
                break
    n_matched = pipe_has_ref.sum()
    n_total = len(pipe_times_sec)
    precision = n_matched / n_total if n_total else 0
    print(f"\n  Detection precision @100μm: "
          f"{n_matched}/{n_total} = {precision:.3f}")
    print(f"  False positives: {n_total - n_matched} "
          f"({(n_total - n_matched) / max(n_total, 1) * 100:.1f}%)")
    print("=" * 60)

    return audit


# ==============================================================
# 2. False-Negative Root-Cause Analysis
# ==============================================================

def run_fn_analysis(comp, clust_results, positions):
    """Classify why each reference spike was missed.

    Categories:
      sub_threshold — no pipeline detection within 2 ms on any channel
      channel_displaced — detection within 0.5 ms but channel >100 μm away
      time_displaced — detection on nearby channel but Δt 0.5–2 ms
    Reports breakdown by ref quality and amplitude.

    Returns
    -------
    fn_info : dict
    """
    print(f"\n{'=' * 60}")
    print(f"  FALSE-NEGATIVE ANALYSIS")
    print(f"{'=' * 60}")

    ref_f = comp['ref_filtered']
    ref_times = ref_f['spike_times']
    ref_clusters = ref_f['spike_clusters']
    ref_channels = ref_f['spike_channels']
    ref_labels = ref_f['cluster_labels']

    pipe_times_sec = pipe_times_to_seconds(clust_results)
    pipe_channels = clust_results['channels']

    from scipy.spatial import cKDTree

    # Generous 2 ms window
    max_dt_wide = 0.002
    max_dt_strict = 0.0005
    max_dist = 100.0  # μm

    tree_pipe = cKDTree(pipe_times_sec.reshape(-1, 1))
    neighbors_wide = tree_pipe.query_ball_point(
        ref_times.reshape(-1, 1), r=max_dt_wide)
    neighbors_strict = tree_pipe.query_ball_point(
        ref_times.reshape(-1, 1), r=max_dt_strict)

    # Classify each ref spike
    N = len(ref_times)
    # 0=true_match, 1=channel_displaced, 2=time_displaced, 3=sub_threshold
    reason = np.full(N, 3, dtype=np.int32)

    for i in range(N):
        ref_ch = int(ref_channels[i])
        ref_pos = positions[ref_ch]

        # Check strict window first (0.5 ms)
        if len(neighbors_strict[i]) > 0:
            dists = [np.sqrt(((positions[int(pipe_channels[j])]
                                - ref_pos) ** 2).sum())
                     for j in neighbors_strict[i]]
            min_d = min(dists)
            if min_d < max_dist:
                reason[i] = 0  # true match
            else:
                reason[i] = 1  # channel displaced
        elif len(neighbors_wide[i]) > 0:
            # Something within 2 ms — check if spatially close
            dists = [np.sqrt(((positions[int(pipe_channels[j])]
                                - ref_pos) ** 2).sum())
                     for j in neighbors_wide[i]]
            min_d = min(dists)
            if min_d < max_dist:
                reason[i] = 2  # time displaced (0.5–2 ms)
            else:
                reason[i] = 1  # channel displaced (even in wide window)
        # else: stays 3 = sub_threshold

    labels_map = {0: 'true_match', 1: 'ch_displaced',
                  2: 'time_displaced', 3: 'sub_threshold'}
    counts = {v: int((reason == k).sum()) for k, v in labels_map.items()}
    fn_total = N - counts['true_match']

    print(f"  Total ref spikes: {N}")
    print(f"  True matches:     {counts['true_match']} "
          f"({counts['true_match'] / N * 100:.1f}%)")
    print(f"  False negatives:  {fn_total} ({fn_total / N * 100:.1f}%)")
    if fn_total > 0:
        print(f"    sub-threshold:    {counts['sub_threshold']} "
              f"({counts['sub_threshold'] / fn_total * 100:.1f}% of FN)")
        print(f"    channel-displaced: {counts['ch_displaced']} "
              f"({counts['ch_displaced'] / fn_total * 100:.1f}% of FN)")
        print(f"    time-displaced:    {counts['time_displaced']} "
              f"({counts['time_displaced'] / fn_total * 100:.1f}% of FN)")

    # Breakdown by quality
    fn_mask = reason > 0
    for lval, lname in [(1, 'good'), (2, 'mua'), (0, 'noise')]:
        unique_ids = np.unique(ref_clusters)
        ids_of_label = [nid for nid in unique_ids
                        if nid < len(ref_labels) and ref_labels[nid] == lval]
        if not ids_of_label:
            continue
        q_mask = np.isin(ref_clusters, ids_of_label)
        q_fn = fn_mask & q_mask
        n_q = q_mask.sum()
        n_fn_q = q_fn.sum()
        if n_q == 0:
            continue
        sub = (reason == 3) & q_mask
        ch_d = (reason == 1) & q_mask
        tm_d = (reason == 2) & q_mask
        print(f"\n  {lname} units (n_spikes={n_q}, FN={n_fn_q}, "
              f"FN rate={n_fn_q / n_q * 100:.1f}%):")
        if n_fn_q > 0:
            print(f"    sub-threshold: {sub.sum()} ({sub.sum() / n_fn_q * 100:.0f}%)")
            print(f"    ch-displaced:  {ch_d.sum()} ({ch_d.sum() / n_fn_q * 100:.0f}%)")
            print(f"    time-displaced: {tm_d.sum()} ({tm_d.sum() / n_fn_q * 100:.0f}%)")

    # Breakdown by amplitude tercile (if available)
    cluster_amps = ref_f.get('cluster_amplitudes', None)
    if cluster_amps is not None:
        unique_ids = np.unique(ref_clusters)
        neuron_amps = np.array([cluster_amps[int(nid)]
                                if int(nid) < len(cluster_amps) else 0
                                for nid in unique_ids])
        valid_amp = (neuron_amps > 0) & np.isfinite(neuron_amps)
        if valid_amp.sum() > 10:
            t33 = np.percentile(neuron_amps[valid_amp], 33.3)
            t66 = np.percentile(neuron_amps[valid_amp], 66.7)
            print(f"\n  By amplitude tercile (thresholds: {t33:.2e}, {t66:.2e}):")

            # Map spike-level amplitudes
            spike_amp = np.array([cluster_amps[int(c)]
                                  if int(c) < len(cluster_amps) else 0
                                  for c in ref_clusters])
            for alabel, lo, hi in [('Low', 0, t33), ('Mid', t33, t66),
                                     ('High', t66, 1e30)]:
                amask = (spike_amp > lo) & (spike_amp <= hi) & (spike_amp > 0)
                n_a = amask.sum()
                if n_a == 0:
                    continue
                fn_a = (fn_mask & amask).sum()
                sub_a = ((reason == 3) & amask).sum()
                print(f"    {alabel:4s} (n={n_a:5d}): FN={fn_a} "
                      f"({fn_a / n_a * 100:.1f}%), "
                      f"sub-threshold={sub_a}")

    print("=" * 60)

    return {
        'reason': reason,
        'labels_map': labels_map,
        'counts': counts,
        'fn_total': fn_total,
    }


# ==============================================================
# 3. Feature Separability A/B Tests
# ==============================================================

def _compute_ratio(feat_arr, ch_arr, ref_arr, label):
    """Compute per-channel inter/intra distance ratio (silently)."""
    unique_ch = np.unique(ch_arr)
    all_ratios = []
    results = []
    rng = np.random.default_rng(config.RANDOM_SEED)

    for ch in unique_ch:
        m = ch_arr == ch
        if m.sum() < 20:
            continue
        cf = feat_arr[m]
        cr = ref_arr[m]
        rids, rcnts = np.unique(cr, return_counts=True)
        ok = rcnts >= 5
        if ok.sum() < 2:
            continue
        rids = rids[ok]

        intra = []
        for rid in rids[:10]:
            idx = np.where(cr == rid)[0]
            if len(idx) < 2:
                continue
            np_pairs = min(200, len(idx) * (len(idx) - 1) // 2)
            for _ in range(np_pairs):
                a, b = rng.choice(len(idx), 2, replace=False)
                intra.append(np.linalg.norm(cf[idx[a]] - cf[idx[b]]))

        inter = []
        for _ in range(min(500, len(rids) * (len(rids) - 1))):
            r1, r2 = rng.choice(rids, 2, replace=False)
            i1 = rng.choice(np.where(cr == r1)[0])
            i2 = rng.choice(np.where(cr == r2)[0])
            inter.append(np.linalg.norm(cf[i1] - cf[i2]))

        if intra and inter:
            mi, me = np.mean(intra), np.mean(inter)
            r = me / mi if mi > 1e-10 else float('inf')
            all_ratios.append(r)
            results.append((ch, m.sum(), len(rids), mi, me, r))

    if all_ratios:
        arr = np.array(all_ratios)
        print(f"  {label}: median={np.median(arr):.3f}, "
              f"mean={np.mean(arr):.3f}, "
              f">1.5: {(arr > 1.5).sum()}, "
              f">2.0: {(arr > 2.0).sum()}")
        return np.median(arr)
    else:
        print(f"  {label}: insufficient data")
        return None


def run_feature_separability(comp, clust_results, positions):
    """Run A/B tests on different feature representations.

    Returns dict of {test_name: median_ratio}.
    """
    print(f"\n{'=' * 60}")
    print(f"  FEATURE SEPARABILITY TESTS")
    print(f"{'=' * 60}")

    ref_f = comp['ref_filtered']
    ref_times = ref_f['spike_times']
    ref_clusters = ref_f['spike_clusters']

    pipe_times_sec = pipe_times_to_seconds(clust_results)
    pipe_labels = clust_results['labels']
    pipe_channels = clust_results['channels']
    pipe_features = clust_results.get('features', None)

    # Time alignment check
    print(f"  Ref  time: [{ref_times.min():.3f}, {ref_times.max():.3f}] s")
    print(f"  Pipe time: [{pipe_times_sec.min():.3f}, {pipe_times_sec.max():.3f}] s")

    if pipe_features is None:
        print("  WARNING: features not stored, skipping")
        return {}

    # Match pipeline spikes to ref
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_times.reshape(-1, 1))
    dists, indices = tree.query(pipe_times_sec.reshape(-1, 1))
    dists, indices = dists.ravel(), indices.ravel()

    matched = dists < 0.0005
    ref_label_for_pipe = np.full(len(pipe_times_sec), -1, dtype=np.int64)
    ref_label_for_pipe[matched] = ref_clusters[indices[matched]]

    valid = ref_label_for_pipe >= 0
    assigned = pipe_labels >= 0
    usable = valid & assigned
    n_usable = usable.sum()
    print(f"  Matched: {matched.sum()}/{len(pipe_times_sec)}, "
          f"usable (valid+assigned): {n_usable}")

    if n_usable < 100:
        print("  Too few usable spikes, skipping")
        return {}

    r_lab = ref_label_for_pipe[usable]
    ch_valid = pipe_channels[usable]
    feat_raw = pipe_features[usable]

    n_nearby = config.CLUST_N_NEARBY
    n_pc = feat_raw.shape[1] // n_nearby

    # Baseline: raw features
    ratio_raw = _compute_ratio(feat_raw, ch_valid, r_lab,
                               "Baseline (raw 60d)")

    # Test C: spatial amplitude profile
    feat_reshaped = feat_raw.reshape(-1, n_nearby, n_pc)
    feat_spatial = np.linalg.norm(feat_reshaped, axis=2)
    ratio_c = _compute_ratio(feat_spatial, ch_valid, r_lab,
                             "Test C (PCA spatial 10d)")

    # Test E: raw channel amplitudes (if available)
    spatial_amps = clust_results.get('spatial_amps', None)
    ratio_e = None
    if spatial_amps is not None:
        sa_valid = np.abs(spatial_amps[usable])
        ratio_e = _compute_ratio(sa_valid, ch_valid, r_lab,
                                 "Test E (raw spatial 10d)")

    # Test G: spatial + peak-channel PCA
    feat_peak_pca = feat_raw[:, :n_pc]
    feat_g = np.hstack([feat_spatial, feat_peak_pca])
    ratio_g = _compute_ratio(feat_g, ch_valid, r_lab,
                             "Test G (spatial+peakPC 16d)")

    results = {
        'baseline_60d': ratio_raw,
        'spatial_10d': ratio_c,
        'raw_spatial_10d': ratio_e,
        'spatial_peakPC_16d': ratio_g,
    }

    # Summary
    print(f"\n  Best ratio: ", end="")
    best = max(((k, v) for k, v in results.items() if v is not None),
               key=lambda x: x[1], default=None)
    if best:
        print(f"{best[0]} = {best[1]:.3f}")
    else:
        print("N/A")
    print("=" * 60)
    return results


# ==============================================================
# 4. Label Purity / Fragmentation Analysis
# ==============================================================

def run_label_analysis(comp, clust_results, positions):
    """Purity (forward) and fragmentation (reverse) analysis.

    Returns dict with summary statistics.
    """
    print(f"\n{'=' * 60}")
    print(f"  LABEL PURITY / FRAGMENTATION ANALYSIS")
    print(f"{'=' * 60}")

    ref_f = comp['ref_filtered']
    ref_times = ref_f['spike_times']
    ref_clusters = ref_f['spike_clusters']

    pipe_times_sec = pipe_times_to_seconds(clust_results)
    pipe_labels = clust_results['labels']

    from scipy.spatial import cKDTree
    tree = cKDTree(ref_times.reshape(-1, 1))
    dists, indices = tree.query(pipe_times_sec.reshape(-1, 1))
    dists, indices = dists.ravel(), indices.ravel()
    matched = dists < 0.0005

    ref_label_for_pipe = np.full(len(pipe_times_sec), -1, dtype=np.int64)
    ref_label_for_pipe[matched] = ref_clusters[indices[matched]]

    valid = ref_label_for_pipe >= 0
    assigned = pipe_labels >= 0
    usable = valid & assigned

    if usable.sum() < 50:
        print("  Too few usable spikes, skipping")
        return {}

    p_lab = pipe_labels[usable]
    r_lab = ref_label_for_pipe[usable]

    # --- Forward: purity per pipe cluster ---
    print(f"\n  Forward: pipe cluster purity")
    print(f"  {'pipe':>6} {'total':>6} {'matched':>7} "
          f"{'n_ref':>5} {'purity':>7}")

    sizes = [(cid, (p_lab == cid).sum()) for cid in np.unique(p_lab)
             if (p_lab == cid).sum() >= 5]
    sizes.sort(key=lambda x: -x[1])

    purities = []
    for cid, _ in sizes[:15]:
        mask = p_lab == cid
        refs = r_lab[mask]
        ref_hist = np.bincount(refs)
        n_ref = (ref_hist > 0).sum()
        top_count = ref_hist.max()
        total_matched = mask.sum()
        total_pipe = (pipe_labels == cid).sum()
        purity = top_count / total_matched
        purities.append(purity)
        print(f"  {cid:>6} {total_pipe:>6} {total_matched:>7} "
              f"{n_ref:>5} {purity:>6.1%}")

    # --- Reverse: fragmentation per ref neuron ---
    print(f"\n  Reverse: fragmentation per ref neuron")
    print(f"  {'ref':>6} {'n_spike':>7} {'n_pipe':>6} {'concent':>8}")

    ref_sizes = [(rid, (r_lab == rid).sum()) for rid in np.unique(r_lab)
                 if (r_lab == rid).sum() >= 5]
    ref_sizes.sort(key=lambda x: -x[1])

    concentrations = []
    for rid, _ in ref_sizes[:15]:
        mask = r_lab == rid
        pipes = p_lab[mask]
        pipe_hist = np.bincount(pipes)
        n_pipe = (pipe_hist > 0).sum()
        top_count = pipe_hist.max()
        total = mask.sum()
        conc = top_count / total
        concentrations.append(conc)
        print(f"  {rid:>6} {total:>7} {n_pipe:>6} {conc:>7.1%}")

    # Summary
    purities = np.array(purities) if purities else np.array([])
    concentrations = np.array(concentrations) if concentrations else np.array([])
    summary = {}
    if len(purities) > 0:
        summary['median_purity'] = float(np.median(purities))
        summary['purity_gt50'] = int((purities > 0.5).sum())
        summary['purity_gt80'] = int((purities > 0.8).sum())
        print(f"\n  Purity:  median={np.median(purities):.1%}, "
              f">50%: {(purities > 0.5).sum()}/{len(purities)}, "
              f">80%: {(purities > 0.8).sum()}/{len(purities)}")
    if len(concentrations) > 0:
        summary['median_concentration'] = float(np.median(concentrations))
        print(f"  Concentration: median={np.median(concentrations):.1%}, "
              f">50%: {(concentrations > 0.5).sum()}/{len(concentrations)}, "
              f">80%: {(concentrations > 0.8).sum()}/{len(concentrations)}")
    print("=" * 60)
    return summary


# ==============================================================
# 5. Candidate Proximity Analysis
# ==============================================================

def run_candidate_proximity(comp, clust_results, positions):
    """For each reference spike, count pipeline candidates within ±0.5 ms
    and their spatial distance distribution at 50/100/150 μm.

    Answers: "Is the problem too few temporal candidates, or spatial
    displacement of those candidates?"

    Returns
    -------
    proximity : dict with candidate count and spatial distributions
    """
    print(f"\n{'=' * 60}")
    print(f"  CANDIDATE PROXIMITY ANALYSIS")
    print(f"{'=' * 60}")

    ref_f = comp['ref_filtered']
    ref_times = ref_f['spike_times']
    ref_channels = ref_f['spike_channels']
    ref_clusters = ref_f['spike_clusters']
    ref_labels = ref_f['cluster_labels']

    pipe_times_sec = pipe_times_to_seconds(clust_results)
    pipe_channels = clust_results['channels']

    n_ref = len(ref_times)
    n_pipe = len(pipe_times_sec)
    print(f"  Ref spikes:  {n_ref}")
    print(f"  Pipe spikes: {n_pipe}")

    from scipy.spatial import cKDTree
    max_dt = 0.0005  # 0.5 ms

    tree_pipe = cKDTree(pipe_times_sec.reshape(-1, 1))
    ref_neighbors = tree_pipe.query_ball_point(
        ref_times.reshape(-1, 1), r=max_dt)

    # --- Per ref spike: count temporal candidates ---
    n_candidates = np.array([len(nb) for nb in ref_neighbors], dtype=np.int32)

    print(f"\n  Temporal candidates per ref spike (within ±0.5 ms):")
    print(f"    mean={n_candidates.mean():.2f}, "
          f"median={np.median(n_candidates):.1f}, "
          f"max={n_candidates.max()}")
    print(f"    0 candidates: {(n_candidates == 0).sum()} "
          f"({(n_candidates == 0).sum() / max(n_ref, 1) * 100:.1f}%)")
    print(f"    1 candidate:  {(n_candidates == 1).sum()} "
          f"({(n_candidates == 1).sum() / max(n_ref, 1) * 100:.1f}%)")
    print(f"    2-5:          {((n_candidates >= 2) & (n_candidates <= 5)).sum()}")
    print(f"    6-10:         {((n_candidates >= 6) & (n_candidates <= 10)).sum()}")
    print(f"    >10:          {(n_candidates > 10).sum()}")

    # --- Spatial distance distribution of candidates ---
    spatial_thresholds = [50, 100, 150]
    # For each ref spike that has ≥1 candidate, compute min distance and
    # count candidates within each threshold
    has_cand = n_candidates > 0
    n_with_cand = has_cand.sum()

    # Per-spike: count within each spatial threshold, and min distance
    cand_within = {th: np.zeros(n_ref, dtype=np.int32)
                   for th in spatial_thresholds}
    min_dist = np.full(n_ref, np.inf, dtype=np.float32)

    for i in range(n_ref):
        if n_candidates[i] == 0:
            continue
        ref_pos = positions[int(ref_channels[i])]
        for j in ref_neighbors[i]:
            d = np.sqrt(((positions[int(pipe_channels[j])] - ref_pos) ** 2).sum())
            if d < min_dist[i]:
                min_dist[i] = d
            for th in spatial_thresholds:
                if d < th:
                    cand_within[th][i] += 1

    print(f"\n  Of {n_with_cand} ref spikes with ≥1 temporal candidate:")
    print(f"    Min spatial distance: "
          f"mean={min_dist[has_cand].mean():.1f} μm, "
          f"median={np.median(min_dist[has_cand]):.1f} μm")

    for th in spatial_thresholds:
        counts = cand_within[th][has_cand]
        has_any = (counts > 0).sum()
        mean_c = counts[counts > 0].mean() if has_any > 0 else 0
        print(f"\n  --- Within {th} μm ---")
        print(f"    Ref spikes with ≥1 candidate within {th}μm: "
              f"{has_any}/{n_with_cand} "
              f"({has_any / max(n_with_cand, 1) * 100:.1f}%)")
        print(f"    Mean candidates (when >0): {mean_c:.2f}")
        print(f"    0 within {th}μm (but has temporal cand): "
              f"{n_with_cand - has_any}")

    # --- Breakdown by ref quality ---
    print(f"\n  Breakdown by ref unit quality:")
    for lval, lname in [(1, 'good'), (2, 'mua'), (0, 'noise')]:
        q_mask = np.zeros(n_ref, dtype=bool)
        for i in range(n_ref):
            cid = ref_clusters[i]
            if cid < len(ref_labels) and ref_labels[cid] == lval:
                q_mask[i] = True
        n_q = q_mask.sum()
        if n_q == 0:
            continue

        no_cand = (n_candidates[q_mask] == 0).sum()
        has_cand_q = n_candidates[q_mask] > 0
        n_has = has_cand_q.sum()
        mean_cand = n_candidates[q_mask].mean()

        within_100 = (cand_within[100][q_mask] > 0).sum()

        print(f"    {lname:5s} (n={n_q}): "
              f"no_cand={no_cand} ({no_cand / n_q * 100:.1f}%), "
              f"mean_cand={mean_cand:.2f}, "
              f"within_100μm={within_100}/{n_has} "
              f"({within_100 / max(n_has, 1) * 100:.1f}%)")

    # --- Key summary ---
    no_temporal = (n_candidates == 0).sum()
    has_temporal_no_spatial = (has_cand & (cand_within[100] == 0)).sum()
    has_both = (cand_within[100] > 0).sum()
    print(f"\n  === SUMMARY ===")
    print(f"    No temporal candidate (±0.5ms):     "
          f"{no_temporal}/{n_ref} ({no_temporal / n_ref * 100:.1f}%)")
    print(f"    Has temporal but no spatial (<100μm): "
          f"{has_temporal_no_spatial}/{n_ref} "
          f"({has_temporal_no_spatial / n_ref * 100:.1f}%)")
    print(f"    Has both temporal + spatial:          "
          f"{has_both}/{n_ref} ({has_both / n_ref * 100:.1f}%)")
    print("=" * 60)

    return {
        'n_candidates': n_candidates,
        'min_dist': min_dist,
        'cand_within': cand_within,
        'no_temporal_pct': no_temporal / max(n_ref, 1),
        'has_temporal_no_spatial_pct': has_temporal_no_spatial / max(n_ref, 1),
        'has_both_pct': has_both / max(n_ref, 1),
    }


# ==============================================================
# Top-level dispatcher
# ==============================================================

def run_all_diagnostics(comp, clust_results, positions):
    """Run all enabled diagnostic analyses.

    Checks config.DIAG_* flags and calls the corresponding functions.
    Returns a dict collecting all results for downstream use.
    """
    results = {}

    if getattr(config, 'DIAG_DETECTION_AUDIT', False):
        try:
            results['detection_audit'] = run_detection_audit(
                comp, clust_results, positions)
        except Exception as e:
            print(f"[DIAG] Detection audit failed: {e}")

    if getattr(config, 'DIAG_FN_ANALYSIS', False):
        try:
            results['fn_analysis'] = run_fn_analysis(
                comp, clust_results, positions)
        except Exception as e:
            print(f"[DIAG] FN analysis failed: {e}")

    if getattr(config, 'DIAG_CANDIDATE_PROXIMITY', False):
        try:
            results['candidate_proximity'] = run_candidate_proximity(
                comp, clust_results, positions)
        except Exception as e:
            print(f"[DIAG] Candidate proximity failed: {e}")

    if getattr(config, 'DIAG_FEATURE_SEPARABILITY', False):
        try:
            results['feature_separability'] = run_feature_separability(
                comp, clust_results, positions)
        except Exception as e:
            print(f"[DIAG] Feature separability failed: {e}")

    if getattr(config, 'DIAG_LABEL_ANALYSIS', False):
        try:
            results['label_analysis'] = run_label_analysis(
                comp, clust_results, positions)
        except Exception as e:
            print(f"[DIAG] Label analysis failed: {e}")

    return results
