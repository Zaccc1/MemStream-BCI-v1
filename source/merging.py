"""
Cluster Merging Module
=======================
Post-hoc merging of over-split clusters, run after all streaming batches.

Strategy (adapted from Kilosort4):
  1. Cluster similarity: cosine similarity on cluster center vectors + spatial
     proximity penalty
  2. CCG verification: cross-correlogram refractory period check (direct port
     from KS4's CCG logic, numba-accelerated when available)
  3. Merge execution: remap labels, weighted-average centers, re-index

This is a purely digital post-processing step (no crossbar VMM).

Usage (inserted in main.py after streaming phase):
    from merging import merge_clusters
    clust_results = clust.get_results()
    merged = merge_clusters(clust_results, channel_positions=positions)
"""

import numpy as np
import math

import config


# ==============================================================
# CCG Computation (ported from KS4 CCG.py)
# ==============================================================

try:
    from numba import njit

    @njit
    def _compute_ccg_numba(st1, st2, tbin, nbins):
        """Numba-accelerated CCG kernel (direct port from KS4)."""
        st1 = np.sort(st1)
        st2 = np.sort(st2)
        dt = nbins * tbin
        K = np.zeros(2 * nbins + 1)
        ilow = 0
        ihigh = 0
        j = 0
        while j < len(st2):
            while ihigh < len(st1) and st1[ihigh] <= st2[j] + dt:
                ihigh += 1
            while ilow < len(st1) and st1[ilow] <= st2[j] - dt:
                ilow += 1
            if ilow >= len(st1):
                break
            if st1[ilow] > st2[j] + dt:
                j += 1
                continue
            for k in range(ilow, ihigh):
                ibin = int(np.round((st2[j] - st1[k]) / tbin))
                K[ibin + nbins] += 1
            j += 1
        return K

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


def compute_CCG(st1, st2, tbin=None, nbins=None):
    """
    Compute cross-correlogram between two spike trains.

    Parameters
    ----------
    st1, st2 : 1D float64 arrays -- spike times in SECONDS
    tbin : float -- time bin size in seconds (default: config.CCG_TBIN)
    nbins : int -- number of bins on each side (default: config.CCG_NBINS)

    Returns
    -------
    K : (2*nbins+1,) float64 -- cross-correlogram counts
    T : float -- total recording duration in seconds
    """
    tbin = tbin if tbin is not None else config.CCG_TBIN
    nbins = nbins if nbins is not None else config.CCG_NBINS
    st1 = np.sort(np.asarray(st1, dtype=np.float64))
    st2 = np.sort(np.asarray(st2, dtype=np.float64))
    if len(st1) < 2 or len(st2) < 2:
        return np.zeros(2 * nbins + 1, dtype=np.float64), 0.0
    T = max(st1.max(), st2.max()) - min(st1.min(), st2.min())

    if _HAS_NUMBA:
        K = _compute_ccg_numba(st1, st2, tbin, nbins)
    else:
        # Pure NumPy fallback
        dt = nbins * tbin
        K = np.zeros(2 * nbins + 1)
        ilow = 0
        ihigh = 0
        for j in range(len(st2)):
            while ihigh < len(st1) and st1[ihigh] <= st2[j] + dt:
                ihigh += 1
            while ilow < len(st1) and st1[ilow] <= st2[j] - dt:
                ilow += 1
            if ilow >= len(st1):
                break
            if st1[ilow] > st2[j] + dt:
                continue
            for k in range(ilow, ihigh):
                ibin = int(round((st2[j] - st1[k]) / tbin))
                K[ibin + nbins] += 1
    return K, T


def CCG_metrics(st1, st2, K, T, nbins=None, tbin=None):
    """
    Compute refractory period violation metrics from CCG (KS4 logic).

    Returns
    -------
    Q12 : float -- contamination ratio (low = good refractory dip)
    R12 : float -- statistical significance of refractory dip
    R00 : float -- baseline firing rate estimate
    """
    nbins = nbins if nbins is not None else config.CCG_NBINS
    tbin = tbin if tbin is not None else config.CCG_TBIN
    irange1 = np.concatenate([
        np.arange(1, nbins // 2),
        np.arange(3 * nbins // 2, 2 * nbins)
    ])
    irange2 = np.arange(nbins - 50, nbins - 10)
    irange3 = np.arange(nbins + 10, nbins + 50)

    denom = tbin * len(st1) * len(st2) / T if T > 0 else 0

    Q00 = K[irange1].sum() / (len(irange1) * denom) if denom > 0 else 0
    Q1 = K[irange2].sum() / (len(irange2) * denom) if denom > 0 else 0
    Q2 = K[irange3].sum() / (len(irange3) * denom) if denom > 0 else 0
    Q01 = max(Q1, Q2)

    R00 = max(K[irange2].mean(), K[irange3].mean())
    R00 = max(R00, K[irange1].mean())

    a = K[nbins]
    K[nbins] = 0

    Qi = np.zeros(10)
    Ri = np.zeros(10)
    for i in range(1, 11):
        irange = np.arange(nbins - i, nbins + i + 1)
        Qi0 = K[irange].sum() / (2 * i * denom) if denom > 0 else 0
        Qi[i - 1] = Qi0

        n = K[irange].sum() / 2
        lam = R00 * i
        p = 0.5 * (1 + math.erf((n - lam) / (1e-10 + (2 * lam) ** 0.5)))
        Ri[i - 1] = p

    K[nbins] = a
    Q12 = np.min(Qi) / (1e-10 + max(Q00, Q01))
    R12 = np.min(Ri)

    return Q12, R12, R00


def check_CCG(st1, st2=None, nbins=None, tbin=None,
              acg_threshold=None, ccg_threshold=None):
    """
    Check whether two spike trains share a refractory period.

    Parameters
    ----------
    st1 : array -- spike times in seconds
    st2 : array or None -- if None, computes ACG (auto-correlogram)
    acg_threshold : float -- Q12 threshold for ACG (single unit quality)
    ccg_threshold : float -- Q12 threshold for CCG (merge criterion)

    Returns
    -------
    is_refractory : bool -- True if single unit has good refractory period
    cross_refractory : bool -- True if pair shares refractory period (-> merge)
    Q12 : float -- contamination metric
    """
    nbins = nbins if nbins is not None else config.CCG_NBINS
    tbin = tbin if tbin is not None else config.CCG_TBIN
    acg_th = acg_threshold if acg_threshold is not None \
        else config.MERGE_ACG_THRESHOLD
    ccg_th = ccg_threshold if ccg_threshold is not None \
        else config.MERGE_CCG_THRESHOLD

    if st2 is None:
        st2 = np.asarray(st1, dtype=np.float64).copy()
    else:
        st2 = np.asarray(st2, dtype=np.float64)
    st1 = np.asarray(st1, dtype=np.float64)

    if len(st1) < 2 or len(st2) < 2:
        return False, False, 1.0

    K, T = compute_CCG(st1, st2, nbins=nbins, tbin=tbin)

    if T <= 0:
        return False, False, 1.0

    Q12, R12, R00 = CCG_metrics(st1, st2, K, T, nbins=nbins, tbin=tbin)

    is_refractory = Q12 < acg_th and R12 < config.CCG_R12_ACG_THRESHOLD
    cross_refractory = Q12 < ccg_th and R12 < config.CCG_R12_CCG_THRESHOLD

    return is_refractory, cross_refractory, Q12


# ==============================================================
# Cluster Similarity (cosine on center vectors + spatial penalty)
# ==============================================================

def compute_cluster_similarity(centers, counts, channel_positions=None,
                               peak_channels=None):
    """
    Compute pairwise similarity between cluster centers.

    Uses cosine similarity on feature-space cluster center vectors.
    Optionally applies spatial distance penalty based on peak channel positions.

    Parameters
    ----------
    centers : (K, D) float32 -- regional assignment center vectors
    counts : (K,) int64 -- spike counts per cluster
    channel_positions : (n_ch, 2) or None -- for spatial distance penalty
    peak_channels : (K,) int or None -- peak channel per cluster

    Returns
    -------
    similarity : (K, K) float32 -- pairwise similarity in [0, 1]
    """
    K, D = centers.shape

    # Cosine similarity
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = centers / norms
    cosine_sim = normed @ normed.T  # (K, K), range [-1, 1]

    # Absolute value (templates can be sign-flipped)
    similarity = np.abs(cosine_sim).astype(np.float32)

    # Spatial penalty: Gaussian decay with distance between peak channels
    if channel_positions is not None and peak_channels is not None:
        pos = channel_positions[peak_channels]  # (K, 2)
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        dist = np.sqrt(dx ** 2 + dy ** 2)
        sigma = config.MERGE_SPATIAL_SIGMA
        spatial_weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        similarity *= spatial_weight.astype(np.float32)

    # Zero out self-similarity and inactive clusters
    np.fill_diagonal(similarity, 0)
    inactive = counts < config.CLUST_MIN_COUNT
    similarity[inactive, :] = 0
    similarity[:, inactive] = 0

    return similarity


def _estimate_peak_channels(labels, spike_channels, n_clusters):
    """For each cluster, find the most common spike channel."""
    peak_ch = np.zeros(n_clusters, dtype=np.int32)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            ch = spike_channels[mask]
            peak_ch[k] = np.bincount(ch).argmax()
    return peak_ch


# ==============================================================
# Main Merge Function
# ==============================================================

def merge_clusters(clust_results, channel_positions=None,
                   sample_rate=None,
                   similarity_threshold=None,
                   acg_threshold=None, ccg_threshold=None,
                   verbose=True):
    """
    Post-hoc cluster merging: center similarity + CCG verification.

    Run once after all streaming batches are processed.

    Algorithm (following KS4):
      1. Sort clusters by spike count (largest first)
      2. For each cluster, find candidates with high center similarity
      3. For candidate pairs, compute CCG to verify shared refractory period
      4. If cross-refractory -> merge (same neuron)
      5. Weighted-average centers, remap labels, re-index

    Parameters
    ----------
    clust_results : dict from SpikeClustering.get_results()
    channel_positions : (n_ch, 2) or None
    sample_rate : float (default: config.SAMPLE_RATE)
    similarity_threshold : float (default: config.MERGE_SIMILARITY_THRESHOLD)
    acg_threshold : float (default: config.MERGE_ACG_THRESHOLD)
    ccg_threshold : float (default: config.MERGE_CCG_THRESHOLD)
    verbose : bool

    Returns
    -------
    merged_results : dict -- same structure plus 'merge_map', 'n_merges',
                     'pre_merge_n_clusters'
    """
    fs = sample_rate or config.SAMPLE_RATE
    sim_th = similarity_threshold if similarity_threshold is not None \
        else config.MERGE_SIMILARITY_THRESHOLD
    acg_th = acg_threshold if acg_threshold is not None \
        else config.MERGE_ACG_THRESHOLD
    ccg_th = ccg_threshold if ccg_threshold is not None \
        else config.MERGE_CCG_THRESHOLD

    times = clust_results['times'].copy()
    channels = clust_results['channels'].copy()
    labels = clust_results['labels'].copy()
    centers = clust_results['centers'].copy()
    counts = clust_results['counts'].copy()

    n_clusters = len(counts)
    n_spikes = len(labels)

    if n_spikes == 0 or n_clusters < 2:
        clust_results['merge_map'] = {}
        clust_results['n_merges'] = 0
        clust_results['pre_merge_n_clusters'] = clust_results.get('n_active', 0)
        return clust_results

    # Estimate peak channel per cluster
    peak_channels = _estimate_peak_channels(labels, channels, n_clusters)

    # Compute pairwise similarity
    similarity = compute_cluster_similarity(
        centers, counts, channel_positions, peak_channels
    )

    # Convert spike times to seconds for CCG
    spike_times_sec = times.astype(np.float64) / fs

    # Greedy merge (KS4 strategy):
    #   - Sort clusters by spike count, largest first.
    #   - Each cluster kk tries to absorb similar neighbours one at a time.
    #   - After each successful absorption, recompute similarity for kk and
    #     try again (t does NOT advance).  This lets one cluster snowball.
    #   - Once kk cannot absorb anything more, advance to the next cluster.
    #   - A cluster is only ever an *absorber* once (when its turn comes).
    #     Once skipped, it never comes back — no multi-pass.
    is_merged = np.zeros(n_clusters, dtype=bool)
    merge_into = np.arange(n_clusters, dtype=np.int32)
    n_merges = 0

    active = np.where(counts >= config.CLUST_MIN_COUNT)[0]
    pre_merge_n = len(active)
    sort_order = active[np.argsort(-counts[active])]

    if verbose:
        print(f"\n{'='*50}")
        print(f"  CLUSTER MERGING")
        print(f"{'='*50}")
        print(f"[MERGE] {pre_merge_n} active clusters, {n_spikes} spikes")
        print(f"[MERGE] Similarity threshold: {sim_th}, "
              f"CCG threshold: {ccg_th}")

    t = 0
    while t < len(sort_order):
        kk = sort_order[t]

        if is_merged[kk]:
            t += 1
            continue

        # Must have enough spikes
        mask_kk = labels == kk
        st_kk = spike_times_sec[mask_kk]
        if len(st_kk) < config.MERGE_MIN_SPIKES:
            t += 1
            continue

        # Only clusters with good refractory period can absorb
        is_ref_kk, _, _ = check_CCG(st_kk, acg_threshold=acg_th,
                                     ccg_threshold=ccg_th)
        if not is_ref_kk:
            t += 1
            continue

        # Recompute similarity row for kk (center may have changed
        # after previous absorptions by this same cluster)
        norms = np.linalg.norm(centers, axis=1)
        norm_kk = max(np.linalg.norm(centers[kk]), 1e-12)
        sim_kk = np.abs(centers @ centers[kk]) / (np.maximum(norms, 1e-12) * norm_kk)
        if channel_positions is not None and peak_channels is not None:
            pos = channel_positions[peak_channels]
            dist_kk = np.sqrt(((pos - pos[kk]) ** 2).sum(axis=1))
            sigma = config.MERGE_SPATIAL_SIGMA
            sim_kk *= np.exp(-dist_kk ** 2 / (2 * sigma ** 2))
        sim_kk[kk] = 0
        sim_kk[is_merged] = 0

        candidates = np.where(sim_kk > sim_th)[0]
        if len(candidates) == 0:
            t += 1
            continue

        candidates = candidates[np.argsort(-sim_kk[candidates])]

        did_merge = False
        for jj in candidates:
            if is_merged[jj] or jj == kk:
                continue

            mask_jj = labels == jj
            st_jj = spike_times_sec[mask_jj]
            if len(st_jj) < config.MERGE_MIN_SPIKES:
                continue

            _, cross_ref, Q12 = check_CCG(st_kk, st_jj,
                                           acg_threshold=acg_th,
                                           ccg_threshold=ccg_th)

            if cross_ref:
                # ABSORB jj into kk
                n_kk = counts[kk]
                n_jj = counts[jj]
                centers[kk] = (n_kk * centers[kk] + n_jj * centers[jj]) \
                    / (n_kk + n_jj)
                counts[kk] += n_jj
                counts[jj] = 0
                labels[mask_jj] = kk

                is_merged[jj] = True
                merge_into[jj] = kk
                n_merges += 1

                # Update kk spike train for subsequent CCG checks
                mask_kk = labels == kk
                st_kk = spike_times_sec[mask_kk]

                if verbose and n_merges <= 30:
                    print(f"  Merged {jj} -> {kk} "
                          f"(sim={sim_kk[jj]:.3f}, "
                          f"Q12={Q12:.3f}, "
                          f"n={n_kk}+{n_jj}={counts[kk]})")

                did_merge = True
                break  # restart similarity scan for kk with updated center

        if not did_merge:
            t += 1
            # else: stay on kk — loop back, recompute similarity, try again

    # ---- Re-index: compact cluster IDs ----
    kept = ~is_merged & (counts >= config.CLUST_MIN_COUNT)
    old_to_new = np.full(n_clusters, -1, dtype=np.int32)
    new_id = 0
    for old_id in range(n_clusters):
        if kept[old_id]:
            old_to_new[old_id] = new_id
            new_id += 1

    def resolve(j):
        if j < 0:
            return -1
        visited = set()
        while merge_into[j] != j and j not in visited:
            visited.add(j)
            j = merge_into[j]
        return j

    new_labels = np.full_like(labels, -1)
    for i in range(n_spikes):
        old = labels[i]
        if old < 0:
            continue  # preserve rejected/unassigned spikes
        if old_to_new[old] >= 0:
            new_labels[i] = old_to_new[old]
        else:
            target = resolve(old)
            if target >= 0 and old_to_new[target] >= 0:
                new_labels[i] = old_to_new[target]
            # else: stays -1 (unassigned)

    new_centers = centers[kept]
    new_counts = counts[kept]
    n_new = new_id
    new_active = np.where(new_counts >= config.CLUST_MIN_COUNT)[0]

    merge_map = {}
    for j in range(n_clusters):
        if is_merged[j]:
            merge_map[int(j)] = int(resolve(j))

    if verbose:
        n_unassigned = (new_labels < 0).sum()
        print(f"[MERGE] Done: {n_merges} merges, "
              f"{pre_merge_n} -> {n_new} active clusters")
        if n_unassigned > 0:
            print(f"[MERGE] {n_unassigned} spikes unassigned "
                  f"({n_unassigned/n_spikes*100:.1f}% from small clusters)")

    # Build output
    merged = dict(clust_results)
    merged['labels'] = new_labels
    merged['centers'] = new_centers
    merged['counts'] = new_counts
    merged['active_clusters'] = new_active
    merged['n_active'] = len(new_active)
    merged['n_clusters_total'] = n_new
    merged['merge_map'] = merge_map
    merged['n_merges'] = n_merges
    merged['pre_merge_n_clusters'] = pre_merge_n

    return merged


# ==============================================================
# Cluster Quality Metrics (ACG-based)
# ==============================================================

def compute_cluster_quality(clust_results, sample_rate=None):
    """
    Compute ACG-based quality metric for each cluster.

    Returns
    -------
    is_good : (n_clusters,) bool -- True if cluster has good refractory period
    contam_rate : (n_clusters,) float -- estimated contamination rate (Q12)
    """
    fs = sample_rate or config.SAMPLE_RATE
    times = clust_results['times']
    labels = clust_results['labels']
    n_clusters = clust_results['n_clusters_total']

    is_good = np.zeros(n_clusters, dtype=bool)
    contam_rate = np.ones(n_clusters, dtype=np.float32)

    spike_times_sec = times.astype(np.float64) / fs

    # Single pass: compute CCG and collect diagnostics
    q12_vals = []
    r12_vals = []

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() < config.MERGE_MIN_SPIKES:
            continue
        st = spike_times_sec[mask]
        K_ccg, T = compute_CCG(st, st.copy())
        if T <= 0 or len(st) < 2:
            continue
        Q12, R12, R00 = CCG_metrics(st, st, K_ccg, T)

        is_ref = Q12 < config.MERGE_ACG_THRESHOLD and R12 < config.CCG_R12_ACG_THRESHOLD
        is_good[k] = is_ref
        contam_rate[k] = Q12
        q12_vals.append(Q12)
        r12_vals.append(R12)

    n_good = is_good.sum()
    print(f"[QUALITY] {n_good}/{n_clusters} clusters with good "
          f"refractory periods")

    if r12_vals:
        r12 = np.array(r12_vals)
        q12 = np.array(q12_vals)
        print(f"[QUALITY] R12 distribution (n={len(r12)}):")
        print(f"  R12: min={r12.min():.4f}, median={np.median(r12):.4f}, "
              f"max={r12.max():.4f}")
        print(f"  R12<0.05: {(r12<0.05).sum()}, "
              f"R12<0.2: {(r12<0.2).sum()}, "
              f"R12<0.5: {(r12<0.5).sum()}, "
              f"R12>=0.5: {(r12>=0.5).sum()}")
        print(f"  Q12: min={q12.min():.4f}, median={np.median(q12):.4f}, "
              f"max={q12.max():.4f}")
        print(f"  Q12<0.2: {(q12<0.2).sum()}, "
              f"Q12<0.5: {(q12<0.5).sum()}, "
              f"Q12>=0.5: {(q12>=0.5).sum()}")

    return is_good, contam_rate
