"""
Spike Clustering Module
========================
Pipeline:
  Calibration:
    1. Temporal PCA basis (6x61 crossbar, shared by all channels)
    2. Official KS4 labels -> Version B centers (streaming-visible spikes only)
    3. Optional LDA projection (post_pca / direct / per_region)

  Streaming (per batch):
    1. Extract features via PCA/LDA crossbar(s)
    2. Assign each spike to nearest center in its region via distance crossbar
    3. Optional EMA center update

Signal quantization (DAC/ADC) at every crossbar boundary.
"""

import numpy as np
import os
from scipy.spatial.distance import cdist

import config
from preprocessing import (
    quantize_weights, quantize_signal, quantized_vmm,
    using_int_accum_quantization, prepare_quantized_weights,
)
from official_ks4_bridge import run_official_ks4_calibration

_LDA_WEIGHT_CACHE = {}


def _lda_weight_cache_key(lda_scalings, precision_bits):
    arr = np.asarray(lda_scalings)
    return (id(arr), arr.shape, arr.dtype.str, int(precision_bits))


def _prepared_lda_weights(lda_scalings, precision_bits):
    """Cache LDA projection weights for repeated batch projections."""
    key = _lda_weight_cache_key(lda_scalings, precision_bits)
    cached = _LDA_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = np.asarray(lda_scalings, dtype=np.float32).T
    prepared = prepare_quantized_weights(weights, int(precision_bits))
    if len(_LDA_WEIGHT_CACHE) >= 32:
        _LDA_WEIGHT_CACHE.clear()
    _LDA_WEIGHT_CACHE[key] = prepared
    return prepared


# ==============================================================
# Waveform Collection (for temporal PCA calibration)
# ==============================================================

def _collect_single_channel_waveforms(whitened_data, spike_times, spike_channels,
                                       channel_positions, n_nearby=None, nt=None):
    """Collect single-channel clips from peak + nearby channels for PCA training."""
    n_nearby_collect = min(n_nearby or config.CLUST_WAVEFORM_COLLECT_N_NEARBY,
                           config.CLUST_N_NEARBY)
    hw = nt or config.CLUST_SNIPPET_HW
    snippet_len = 2 * hw + 1
    T_total, n_ch = whitened_data.shape

    dist_matrix = cdist(channel_positions, channel_positions)
    nearby_idx = np.argsort(dist_matrix, axis=1)[:, :n_nearby_collect]

    clips = []
    max_clips = config.CLUST_WAVEFORM_MAX_CLIPS
    for i in range(len(spike_times)):
        t = int(spike_times[i])
        ch = int(spike_channels[i])
        t_start = t - hw
        t_end = t + hw + 1
        if t_start < 0 or t_end > T_total:
            continue
        for j in range(n_nearby_collect):
            c = nearby_idx[ch, j]
            clips.append(whitened_data[t_start:t_end, c].copy())
            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break

    waveforms = np.array(clips, dtype=np.float32) if clips else \
        np.zeros((0, snippet_len), dtype=np.float32)
    print(f"[TEMPORAL] Collected {len(waveforms)} single-channel waveforms "
          f"({snippet_len} points each)")
    return waveforms


def _clean_waveforms_kmeans(waveforms, k=None, min_count=None):
    """K-Means on normalized waveforms to filter noise clips."""
    k = k or config.CLUST_KMEANS_CLEAN_K
    min_count = min_count or config.CLUST_KMEANS_CLEAN_MIN

    N = len(waveforms)
    if N < k * 2:
        return waveforms, 0

    norms = np.linalg.norm(waveforms, axis=1, keepdims=True)
    valid = norms.squeeze() > 1e-6
    wf_valid = waveforms[valid]
    norms_valid = norms[valid]
    normalised = wf_valid / norms_valid

    k = min(k, len(normalised) // 2)
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(normalised), k, replace=False)
    centers = normalised[idx].copy()

    for iteration in range(config.KMEANS_CLEAN_MAX_ITER):
        dists = cdist(normalised, centers, 'sqeuclidean')
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                m = normalised[mask].mean(axis=0)
                norm = np.linalg.norm(m)
                new_centers[j] = m / norm if norm > 1e-12 else centers[j]
            else:
                new_centers[j] = centers[j]
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < config.KMEANS_CONVERGENCE_TOL:
            break

    counts = np.bincount(labels, minlength=k)
    good_clusters = np.where(counts >= min_count)[0]
    keep_mask = np.isin(labels, good_clusters)
    clean = wf_valid[keep_mask]
    n_removed = len(wf_valid) - len(clean)

    print(f"[CLEAN] K-Means k={k}: kept {len(clean)}/{len(wf_valid)} "
          f"waveforms, removed {n_removed}")
    return clean, n_removed


# ==============================================================
# Temporal PCA Projector
# ==============================================================

class TemporalPCAProjector:
    """Shared PCA basis on single-channel waveform time dimension.
    Hardware: one 6x61 crossbar, shared by all channels."""

    def __init__(self, n_components=None, precision_bits=None):
        self.n_components = n_components or config.CLUST_TEMPORAL_PCA_NC
        self.precision_bits = precision_bits or config.PRECISION_PCA
        self.basis = None
        self.basis_full = None
        self.basis_prepared = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.is_calibrated = False

    def set_external_basis(self, basis, explained_variance_ratio=None):
        """Use externally computed PCA basis (e.g. wPCA from template extraction)."""
        self.basis_full = basis.astype(np.float32)
        self.basis = quantize_weights(self.basis_full, self.precision_bits)
        self.basis_prepared = prepare_quantized_weights(
            self.basis, self.precision_bits)
        self.n_components = basis.shape[0]
        snippet_len = basis.shape[1]
        self.mean_ = np.zeros(snippet_len, dtype=np.float32)
        self.explained_variance_ratio_ = None if explained_variance_ratio is None \
            else np.asarray(explained_variance_ratio, dtype=np.float32)
        self.is_calibrated = True

        print(f"[TPCA] Using external basis (wPCA): ({self.n_components}, {snippet_len}), "
              f"{self.precision_bits}-bit weights")
        if self.precision_bits < 32:
            err = np.abs(self.basis - self.basis_full).max()
            print(f"[TPCA] Basis weight quant error: {err:.4e}")

    def calibrate(self, clean_waveforms):
        """Compute temporal PCA basis from clean waveforms via SVD."""
        N, snippet_len = clean_waveforms.shape
        nc = min(self.n_components, snippet_len, N)

        self.mean_ = clean_waveforms.mean(axis=0).astype(np.float32)
        centered = clean_waveforms - self.mean_

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.basis_full = Vt[:nc].astype(np.float32)
        self.basis = quantize_weights(self.basis_full, self.precision_bits)
        self.basis_prepared = prepare_quantized_weights(
            self.basis, self.precision_bits)
        total_var = (S ** 2).sum()
        self.explained_variance_ratio_ = \
            (S[:nc] ** 2 / max(total_var, 1e-12)).astype(np.float32)
        self.n_components = nc
        self.is_calibrated = True

        cumvar = np.cumsum(self.explained_variance_ratio_)
        print(f"[TPCA] Calibrated: {N} waveforms, {snippet_len}d -> {nc} PCs, "
              f"{self.precision_bits}-bit weights, var={cumvar[-1]*100:.1f}%")

    def project_channel(self, waveform):
        """Project single-channel waveform: (snippet_len,) -> (n_components,)"""
        centered = waveform - self.mean_
        if using_int_accum_quantization():
            basis = self.basis_prepared if self.basis_prepared is not None \
                else self.basis
            coeffs = quantized_vmm(
                centered[np.newaxis, :], basis,
                config.PRECISION_DAC, self.precision_bits,
                op_name="TPCA").ravel()
        else:
            coeffs = (centered @ self.basis.T).astype(np.float32)
        adc_name = "TPCA.adc" if using_int_accum_quantization() else None
        return quantize_signal(coeffs, config.PRECISION_ADC, op_name=adc_name)

    def set_full_precision(self, enable=True):
        """Switch between full-precision (calibration) and quantized (streaming)."""
        if enable and self.basis_full is not None:
            self.basis = self.basis_full
        elif not enable and self.basis_full is not None:
            self.basis = quantize_weights(self.basis_full, self.precision_bits)
        if self.basis is not None:
            self.basis_prepared = prepare_quantized_weights(
                self.basis, self.precision_bits)

    def project_multichannel(self, whitened_chunk, spike_time, spike_channel,
                              nearby_idx):
        """Project one spike: 10 channels x 6 PCs = 60d feature vector."""
        hw = (self.basis.shape[1] - 1) // 2
        n_nearby = nearby_idx.shape[1]
        T_total = whitened_chunk.shape[0]
        snippet_len = self.basis.shape[1]

        t_start = spike_time - hw
        t_end = spike_time + hw + 1
        channels = nearby_idx[spike_channel]
        coeffs = np.zeros((n_nearby, self.n_components), dtype=np.float32)

        for j, ch in enumerate(channels):
            if t_start < 0 or t_end > T_total:
                t_s = max(0, t_start)
                t_e = min(T_total, t_end)
                wf = np.zeros(snippet_len, dtype=np.float32)
                wf[t_s - t_start: t_s - t_start + (t_e - t_s)] = \
                    whitened_chunk[t_s:t_e, ch]
            else:
                wf = whitened_chunk[t_start:t_end, ch]
            coeffs[j] = self.project_channel(wf)

        return coeffs.ravel()


# ==============================================================
# Feature Extraction
# ==============================================================

def extract_temporal_features(whitened_chunk, spike_times, spike_channels,
                               channel_positions, temporal_pca,
                               n_nearby=None, nearby_idx=None):
    """Extract temporal PCA features for a batch of spikes (vectorized).
    Each spike: 10 nearby channels x 6 temporal PCs = 60d.

    Signal path: DAC(waveform snippets) -> weight-quantized PCA VMM -> ADC(coefficients)
    """
    n_nearby = n_nearby or config.CLUST_N_NEARBY
    n_spikes = len(spike_times)
    nc = temporal_pca.n_components
    nt = temporal_pca.basis.shape[1]
    hw = (nt - 1) // 2
    T_total, n_ch = whitened_chunk.shape

    if nearby_idx is None:
        dist_matrix = cdist(channel_positions, channel_positions)
        nearby_idx = np.argsort(dist_matrix, axis=1)[:, :n_nearby]

    if n_spikes == 0:
        return np.zeros((0, n_nearby * nc), dtype=np.float32)

    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_channels = np.asarray(spike_channels, dtype=np.int64)

    channels_per_spike = nearby_idx[spike_channels]
    snippets = np.zeros((n_spikes, n_nearby, nt), dtype=np.float32)

    t_starts = spike_times - hw
    t_ends = spike_times + hw + 1

    valid = (t_starts >= 0) & (t_ends <= T_total)
    valid_idx = np.nonzero(valid)[0]
    edge_idx = np.nonzero(~valid)[0]

    if len(valid_idx) > 0:
        time_idx = t_starts[valid_idx, None] + np.arange(nt)[None, :]
        v_chs = channels_per_spike[valid_idx]
        for j in range(n_nearby):
            snippets[valid_idx, j, :] = whitened_chunk[time_idx, v_chs[:, j:j+1]]

    for i in edge_idx:
        ts = max(0, int(t_starts[i]))
        te = min(T_total, int(t_ends[i]))
        offset = ts - int(t_starts[i])
        for j, ch in enumerate(channels_per_spike[i]):
            snippets[i, j, offset:offset + (te - ts)] = \
                whitened_chunk[ts:te, ch]

    # PCA crossbar: DAC(snippets) -> weight-quantized basis -> ADC(coefficients)
    basis = temporal_pca.basis
    basis_vmm = temporal_pca.basis_prepared \
        if getattr(temporal_pca, 'basis_prepared', None) is not None else basis
    mean = temporal_pca.mean_
    flat = snippets.reshape(-1, nt)
    centered = flat - mean
    if using_int_accum_quantization():
        coeffs = quantized_vmm(
            centered, basis_vmm, config.PRECISION_DAC,
            temporal_pca.precision_bits,
            op_name="TPCA")
    else:
        # DAC quantization on crossbar input
        flat_dac = quantize_signal(centered, config.PRECISION_DAC)
        coeffs = flat_dac @ basis.T
    # ADC quantization on crossbar output
    adc_name = "TPCA.adc" if using_int_accum_quantization() else None
    coeffs = quantize_signal(coeffs, config.PRECISION_ADC, op_name=adc_name)

    features = coeffs.reshape(n_spikes, n_nearby * nc).astype(np.float32)
    return features


def extract_raw_waveform_features(whitened_chunk, spike_times, spike_channels,
                                   channel_positions, snippet_len=61,
                                   n_nearby=None, nearby_idx=None):
    """Extract raw flattened waveform features (no PCA).
    Each spike: n_nearby channels x snippet_len samples, flattened.
    Returns: (n_spikes, n_nearby * snippet_len) float32
    """
    n_nearby = n_nearby or config.CLUST_LDA_DIRECT_N_CHANNELS
    n_spikes = len(spike_times)
    nt = snippet_len
    hw = (nt - 1) // 2
    T_total, n_ch = whitened_chunk.shape

    if nearby_idx is None:
        dist_matrix = cdist(channel_positions, channel_positions)
        nearby_idx = np.argsort(dist_matrix, axis=1)[:, :n_nearby]

    if n_spikes == 0:
        return np.zeros((0, n_nearby * nt), dtype=np.float32)

    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_channels = np.asarray(spike_channels, dtype=np.int64)

    channels_per_spike = nearby_idx[spike_channels]
    snippets = np.zeros((n_spikes, n_nearby, nt), dtype=np.float32)

    t_starts = spike_times - hw
    t_ends = spike_times + hw + 1

    valid = (t_starts >= 0) & (t_ends <= T_total)
    valid_idx = np.nonzero(valid)[0]
    edge_idx = np.nonzero(~valid)[0]

    if len(valid_idx) > 0:
        time_idx = t_starts[valid_idx, None] + np.arange(nt)[None, :]
        v_chs = channels_per_spike[valid_idx]
        for j in range(n_nearby):
            snippets[valid_idx, j, :] = whitened_chunk[time_idx, v_chs[:, j:j+1]]

    for i in edge_idx:
        ts = max(0, int(t_starts[i]))
        te = min(T_total, int(t_ends[i]))
        offset = ts - int(t_starts[i])
        for j, ch in enumerate(channels_per_spike[i]):
            snippets[i, j, offset:offset + (te - ts)] = \
                whitened_chunk[ts:te, ch]

    return snippets.reshape(n_spikes, n_nearby * nt).astype(np.float32)


# ==============================================================
# Channel Region Partitioning
# ==============================================================

def compute_channel_regions(channel_positions, region_size=None, overlap=None):
    """Partition channels into overlapping regions by depth."""
    rs = region_size or config.CLUST_REGION_SIZE
    ov = overlap or config.CLUST_REGION_OVERLAP
    n_ch = channel_positions.shape[0]
    depths = channel_positions[:, 1]
    sort_idx = np.argsort(depths)

    stride = rs - ov
    regions = []
    start = 0
    while start < n_ch:
        end = min(start + rs, n_ch)
        regions.append(sort_idx[start:end].copy())
        if end >= n_ch:
            break
        start += stride

    ch_to_region = np.zeros(n_ch, dtype=np.int32)
    for r_id, region in enumerate(regions):
        for ch in region:
            ch_to_region[ch] = r_id

    print(f"[REGION] {len(regions)} regions, size={rs}, overlap={ov}")
    return regions, ch_to_region


# ==============================================================
# Regional Center Assigner
# ==============================================================

class RegionalCenterAssigner:
    """Assigns spikes to nearest cluster center within their spatial region.

    Signal path for distance computation:
      DAC(features) -> weight-quantized dot product with centers -> ADC(distances)

    Reject gate and margin check are pure digital.
    """

    def __init__(self, channel_positions, region_size=None, overlap=None,
                 precision_bits=None):
        self.precision_bits = precision_bits or config.PRECISION_ASSIGN
        self.regions, self.ch_to_region = compute_channel_regions(
            channel_positions, region_size=region_size, overlap=overlap
        )
        self.n_regions = len(self.regions)

        self.k_per_region = []
        self.region_offsets = []
        self.n_clusters_total = 0

        self.centers_ = {}
        self.centers_full_ = {}
        self.centers_prepared_ = {}
        self.centers_scale_ = {}  # frozen quantization scale per region
        self.counts_ = {}
        self.ema_counts_ = {}
        self.accept_radii_ = {}
        self.update_radii_ = {}
        self.is_calibrated = False

    def _recompute_offsets(self):
        self.region_offsets = []
        offset = 0
        for k in self.k_per_region:
            self.region_offsets.append(offset)
            offset += k
        self.n_clusters_total = offset

    def set_precomputed_centers(self, region_centers, region_counts,
                                 precision_bits=None,
                                 region_accept_radii=None,
                                 region_update_radii=None):
        """Load per-region centers from calibration."""
        prec = precision_bits or self.precision_bits
        self.k_per_region = []
        n_feat = next((v.shape[1] for v in region_centers.values() if v.shape[0] > 0), 60)
        region_accept_radii = region_accept_radii or {}
        region_update_radii = region_update_radii or {}
        for r_id in range(self.n_regions):
            if r_id in region_centers and region_centers[r_id].shape[0] > 0:
                C = region_centers[r_id].astype(np.float32)
                K_r = C.shape[0]
                base_counts = region_counts.get(r_id, np.zeros(K_r, dtype=np.int64)).copy()
                self.centers_full_[r_id] = C.copy()
                self.centers_[r_id], self.centers_scale_[r_id] = \
                    quantize_weights(C, prec, return_scale=True)
                self.centers_prepared_[r_id] = prepare_quantized_weights(
                    self.centers_[r_id], prec)
                self.counts_[r_id] = base_counts.copy()
                self.ema_counts_[r_id] = base_counts.copy()
                a = np.asarray(region_accept_radii.get(r_id, np.full(K_r, np.inf, dtype=np.float32)), dtype=np.float32)
                u = np.asarray(region_update_radii.get(r_id, np.full(K_r, np.inf, dtype=np.float32)), dtype=np.float32)
                if len(a) != K_r:
                    a = np.full(K_r, np.inf, dtype=np.float32)
                if len(u) != K_r:
                    u = np.full(K_r, np.inf, dtype=np.float32)
                self.accept_radii_[r_id] = a
                self.update_radii_[r_id] = np.minimum(u, a)
            else:
                K_r = 0
                self.centers_full_[r_id] = np.zeros((0, n_feat), dtype=np.float32)
                self.centers_[r_id] = np.zeros((0, n_feat), dtype=np.float32)
                self.centers_prepared_[r_id] = prepare_quantized_weights(
                    self.centers_[r_id], prec)
                self.counts_[r_id] = np.zeros(0, dtype=np.int64)
                self.ema_counts_[r_id] = np.zeros(0, dtype=np.int64)
                self.accept_radii_[r_id] = np.zeros(0, dtype=np.float32)
                self.update_radii_[r_id] = np.zeros(0, dtype=np.float32)
            self.k_per_region.append(K_r)

        self._recompute_offsets()
        self.is_calibrated = True

        if prec < 32:
            errs = [np.abs(self.centers_[r] - self.centers_full_[r]).max()
                    for r in range(self.n_regions)
                    if self.centers_full_[r].size > 0]
            max_err = max(errs) if errs else 0.0
            print(f"[ASSIGN] Centers quantized to {prec}-bit weights, "
                  f"max error: {max_err:.4e}")

        k_max = max(self.k_per_region)
        k_min = min(self.k_per_region)
        accept_vals = np.concatenate([v for v in self.accept_radii_.values() if len(v) > 0]) \
            if any(len(v) > 0 for v in self.accept_radii_.values()) else np.array([], dtype=np.float32)
        update_vals = np.concatenate([v for v in self.update_radii_.values() if len(v) > 0]) \
            if any(len(v) > 0 for v in self.update_radii_.values()) else np.array([], dtype=np.float32)
        print(f"[ASSIGN] Loaded: {self.n_regions} regions, "
              f"K/region: {k_min}-{k_max}, total={self.n_clusters_total}")
        if len(accept_vals) > 0:
            print(f"[ASSIGN] Accept radii: median={np.median(accept_vals):.3f}, p90={np.quantile(accept_vals, 0.90):.3f}")
        if len(update_vals) > 0:
            print(f"[ASSIGN] Update radii: median={np.median(update_vals):.3f}, p90={np.quantile(update_vals, 0.90):.3f}")

    def assign(self, features, spike_channels):
        """Assign spikes to nearest center in their region.

        Distance crossbar: DAC(features) -> weight-quantized dot(X, C^T) -> ADC
        Reject gate and margin check are pure digital.

        Returns: global_labels, distances, update_mask
        """
        n_spikes = len(spike_channels)
        global_labels = np.full(n_spikes, -1, dtype=np.int32)
        distances = np.full(n_spikes, np.inf, dtype=np.float32)
        update_mask = np.zeros(n_spikes, dtype=bool)
        region_ids = self.ch_to_region[spike_channels]

        use_reject = bool(getattr(config, 'CLUST_ASSIGN_REJECT_ENABLE', False))
        assign_margin_min = float(getattr(config, 'CLUST_ASSIGN_MARGIN_MIN', 1.0))
        update_highconf_only = bool(getattr(config, 'CLUST_UPDATE_HIGHCONF_ONLY', True))
        update_margin_min = float(getattr(config, 'CLUST_UPDATE_MARGIN_MIN', 1.0))

        for r_id in range(self.n_regions):
            mask = region_ids == r_id
            if mask.sum() == 0:
                continue
            C = self.centers_[r_id]
            if C.shape[0] == 0:
                continue
            C_vmm = self.centers_prepared_.get(r_id, C)
            X = features[mask]

            if using_int_accum_quantization():
                X_dac = quantize_signal(
                    X, config.PRECISION_DAC, op_name="ASSIGN.input")
                xc = quantized_vmm(
                    X, C_vmm, config.PRECISION_DAC, self.precision_bits,
                    op_name="ASSIGN")
            else:
                # DAC quantization on crossbar input
                X_dac = quantize_signal(X, config.PRECISION_DAC)
                # Crossbar dot product: X @ C^T (C weights already quantized)
                xc = X_dac @ C.T
            # ADC quantization on crossbar output
            adc_name = "ASSIGN.adc" if using_int_accum_quantization() else None
            xc = quantize_signal(xc, config.PRECISION_ADC, op_name=adc_name)

            # Distance computation (digital)
            x_sq = (X_dac ** 2).sum(axis=1, keepdims=True)
            c_sq = (C ** 2).sum(axis=1, keepdims=True).T
            dist_sq = np.maximum(x_sq - 2 * xc + c_sq, 0)

            local_labels = np.argmin(dist_sq, axis=1)
            min_dist_sq = dist_sq[np.arange(len(local_labels)), local_labels]
            min_dist = np.sqrt(min_dist_sq).astype(np.float32)

            if C.shape[0] > 1:
                second_sq = np.partition(dist_sq, 1, axis=1)[:, 1]
                second_dist = np.sqrt(second_sq).astype(np.float32)
                ratios = second_dist / np.maximum(min_dist, 1e-12)
            else:
                ratios = np.full(len(local_labels), np.inf, dtype=np.float32)

            accept = np.ones(len(local_labels), dtype=bool)
            if use_reject:
                local_accept_r = self.accept_radii_[r_id][local_labels]
                accept &= min_dist <= local_accept_r
                if assign_margin_min > 1.0:
                    accept &= ratios >= assign_margin_min

            idx = np.where(mask)[0]
            if np.any(accept):
                accepted_idx = idx[accept]
                accepted_local = local_labels[accept]
                global_labels[accepted_idx] = self.region_offsets[r_id] + accepted_local
                distances[accepted_idx] = min_dist[accept]

                if update_highconf_only:
                    local_update_r = self.update_radii_[r_id][accepted_local]
                    upd = (min_dist[accept] <= local_update_r)
                    if update_margin_min > 1.0:
                        upd &= ratios[accept] >= update_margin_min
                    update_mask[accepted_idx] = upd
                else:
                    update_mask[accepted_idx] = True

        return global_labels, distances, update_mask

    def update_centers(self, features, spike_channels, global_labels, update_mask=None):
        """Register accepted assignments and optionally EMA-update centers."""
        if update_mask is None:
            update_mask = np.ones(len(global_labels), dtype=bool)
        region_ids = self.ch_to_region[spike_channels]
        enable_updates = bool(getattr(config, 'CLUST_ENABLE_CENTER_UPDATES', True))

        for r_id in range(self.n_regions):
            mask = region_ids == r_id
            if mask.sum() == 0:
                continue
            X = features[mask]
            gl = global_labels[mask]
            valid = gl >= self.region_offsets[r_id]
            valid &= gl < (self.region_offsets[r_id] + self.k_per_region[r_id])
            if not np.any(valid):
                continue
            local_labels_all = gl[valid] - self.region_offsets[r_id]
            cnt_all = np.bincount(local_labels_all, minlength=self.k_per_region[r_id]).astype(np.int64)
            self.counts_[r_id][:len(cnt_all)] += cnt_all

            if not enable_updates:
                continue

            upd_local = update_mask[mask][valid]
            if not np.any(upd_local):
                continue
            X_upd = X[valid][upd_local]
            local_labels = local_labels_all[upd_local]
            K_r = self.k_per_region[r_id]

            for k in range(K_r):
                m = local_labels == k
                n_new = int(m.sum())
                if n_new == 0:
                    continue
                new_mean = X_upd[m].mean(axis=0)
                n_old = int(self.ema_counts_[r_id][k])
                lr = n_new / (n_old + n_new + 1e-12)
                self.centers_full_[r_id][k] = (
                    (1 - lr) * self.centers_full_[r_id][k] + lr * new_mean
                )
                self.ema_counts_[r_id][k] += n_new

            self.centers_[r_id] = quantize_weights(
                self.centers_full_[r_id], self.precision_bits,
                scale=self.centers_scale_.get(r_id))
            self.centers_prepared_[r_id] = prepare_quantized_weights(
                self.centers_[r_id], self.precision_bits)

    def get_all_counts(self):
        counts = np.zeros(self.n_clusters_total, dtype=np.int64)
        for r_id in range(self.n_regions):
            start = self.region_offsets[r_id]
            K_r = self.k_per_region[r_id]
            counts[start:start + K_r] = self.counts_[r_id]
        return counts

    def get_all_centers(self):
        if not self.centers_:
            return np.array([])
        n_feat = next(
            (v.shape[1] for v in self.centers_.values() if v.shape[0] > 0), 0)
        if n_feat == 0:
            return np.array([])
        centers = np.zeros((self.n_clusters_total, n_feat), dtype=np.float32)
        for r_id in range(self.n_regions):
            start = self.region_offsets[r_id]
            K_r = self.k_per_region[r_id]
            centers[start:start + K_r] = self.centers_[r_id]
        return centers

    def get_active_clusters(self, min_count=None):
        mc = min_count or config.CLUST_MIN_COUNT
        counts = self.get_all_counts()
        return np.where(counts >= mc)[0]


# ==============================================================
# Helper functions
# ==============================================================

def _compute_centers_from_labeled_features(features, labels):
    labels = np.asarray(labels, dtype=np.int32)
    if len(labels) == 0 or len(features) == 0:
        return np.full(len(labels), -1, dtype=np.int32), np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), {}
    valid = labels >= 0
    if not valid.any():
        return np.full_like(labels, -1), np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64), {}
    uniq = np.unique(labels[valid])
    old_to_new = {int(old): i for i, old in enumerate(uniq)}
    new_labels = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        new_labels[labels == old] = new
    K = len(uniq)
    D = features.shape[1]
    centers = np.zeros((K, D), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = new_labels == k
        counts[k] = int(m.sum())
        if counts[k] > 0:
            centers[k] = features[m].mean(axis=0)
    return new_labels, centers, counts, old_to_new


def _summarize_region_payload(region_centers, region_counts):
    return {
        'total_centers': int(sum(v.shape[0] for v in region_centers.values())),
        'total_spikes': int(sum(v.sum() for v in region_counts.values())),
        'nonempty_regions': int(sum(v.shape[0] > 0 for v in region_centers.values())),
    }


def _compute_cluster_distance_thresholds(features, labels, centers):
    """Estimate per-cluster accept/update radii from calibration features."""
    K = centers.shape[0]
    accept = np.full(K, np.inf, dtype=np.float32)
    update = np.full(K, np.inf, dtype=np.float32)
    if K == 0 or len(features) == 0 or len(labels) == 0:
        return accept, update

    labels = np.asarray(labels, dtype=np.int32)
    min_samples = int(getattr(config, 'CLUST_ASSIGN_REJECT_MIN_SAMPLES', 20))
    aq = float(getattr(config, 'CLUST_ASSIGN_REJECT_QUANTILE', 0.99))
    uq = float(getattr(config, 'CLUST_UPDATE_QUANTILE', 0.90))
    ascale = float(getattr(config, 'CLUST_ASSIGN_REJECT_SCALE', 1.10))
    uscale = float(getattr(config, 'CLUST_UPDATE_SCALE', 1.00))

    pooled = []
    per_cluster = {}
    for k in range(K):
        m = labels == k
        if not np.any(m):
            per_cluster[k] = np.array([], dtype=np.float32)
            continue
        d = np.linalg.norm(features[m] - centers[k], axis=1).astype(np.float32)
        per_cluster[k] = d
        if len(d) > 0:
            pooled.append(d)

    if pooled:
        pooled = np.concatenate(pooled)
        fallback_accept = max(float(np.quantile(pooled, aq)) * ascale, 1e-6)
        fallback_update = max(float(np.quantile(pooled, uq)) * uscale, 1e-6)
    else:
        fallback_accept = np.inf
        fallback_update = np.inf

    for k in range(K):
        d = per_cluster[k]
        if len(d) == 0:
            continue
        if len(d) < min_samples:
            accept[k] = fallback_accept
            update[k] = min(fallback_update, accept[k])
            continue
        a = max(float(np.quantile(d, aq)) * ascale, 1e-6)
        u = max(float(np.quantile(d, uq)) * uscale, 1e-6)
        accept[k] = a
        update[k] = min(u, a)

    return accept, update


def _group_cluster_values_by_region(values, cluster_to_region, n_regions, dtype=np.float32):
    """Map a per-cluster vector to per-region arrays."""
    region_values = {}
    for r_id in range(n_regions):
        cids_in_region = [c for c, r in cluster_to_region.items() if r == r_id]
        if cids_in_region:
            region_values[r_id] = np.asarray(values[cids_in_region], dtype=dtype).copy()
        else:
            region_values[r_id] = np.zeros(0, dtype=dtype)
    return region_values


# ==============================================================
# KS4 Result Cache
# ==============================================================

def _ks4_cache_path(output_dir, calibration_samples, n_channels):
    """Deterministic cache filename from calibration data shape."""
    cache_dir = os.path.join(output_dir,
                             getattr(config, 'KS4_CACHE_SUBDIR', 'ks4_cache'))
    os.makedirs(cache_dir, exist_ok=True)
    duration_s = calibration_samples / config.SAMPLE_RATE
    fname = f'ks4_labels_{duration_s:.0f}s_{n_channels}ch.npz'
    return os.path.join(cache_dir, fname)


def _save_ks4_cache(path, cali_result):
    """Save minimal KS4 output for downstream center computation."""
    import json
    save_dict = {
        'all_times': np.asarray(cali_result['all_times'], dtype=np.int64),
        'all_times_raw': np.asarray(cali_result['all_times_raw'], dtype=np.int64),
        'all_channels': np.asarray(cali_result['all_channels'], dtype=np.int32),
        'all_labels': np.asarray(cali_result['all_labels'], dtype=np.int32),
    }
    if 'spike_xy' in cali_result and cali_result['spike_xy'] is not None:
        save_dict['spike_xy'] = np.asarray(cali_result['spike_xy'], dtype=np.float32)
    if 'summary' in cali_result and isinstance(cali_result['summary'], dict):
        save_dict['_summary_json'] = np.array([json.dumps(cali_result['summary'])])
    np.savez_compressed(path, **save_dict)
    n_spikes = len(save_dict['all_times'])
    n_clusters = int(np.unique(save_dict['all_labels']).size)
    print(f'[KS4-CACHE] Saved: {n_spikes} spikes, {n_clusters} clusters -> {path}')


def _load_ks4_cache(path):
    """Load cached KS4 spike labels."""
    import json
    data = np.load(path, allow_pickle=False)
    cali_result = {
        'backend': 'official_ks4_cached',
        'all_times': data['all_times'],
        'all_times_raw': data['all_times_raw'],
        'all_channels': data['all_channels'],
        'all_labels': data['all_labels'],
    }
    if 'spike_xy' in data:
        cali_result['spike_xy'] = data['spike_xy']
    if '_summary_json' in data:
        cali_result['summary'] = json.loads(str(data['_summary_json'][0]))
    else:
        n_spikes = len(cali_result['all_times'])
        n_clusters = int(np.unique(cali_result['all_labels']).size)
        cali_result['summary'] = {'n_spikes': n_spikes, 'n_clusters': n_clusters}
    n_spikes = len(cali_result['all_times'])
    n_clusters = int(np.unique(cali_result['all_labels']).size)
    print(f'[KS4-CACHE] Loaded: {n_spikes} spikes, {n_clusters} clusters <- {path}')
    return cali_result


# ==============================================================
# LDA Feature Projection
# ==============================================================

def _fit_lda_projection(features, labels, n_components=None, min_samples=10):
    """Compute LDA projection from PCA features + KS4 labels.

    Returns: (lda_mean, lda_scalings, info) or (None, None, None) if LDA cannot be computed.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    labels = np.asarray(labels, dtype=np.int32)
    valid = labels >= 0
    if valid.sum() < 2:
        return None, None, None

    X = features[valid].astype(np.float64)
    y = labels[valid]

    unique, counts = np.unique(y, return_counts=True)
    keep_classes = unique[counts >= min_samples]
    if len(keep_classes) < 2:
        print(f"[LDA] Only {len(keep_classes)} classes with >= {min_samples} samples, skipping")
        return None, None, None

    class_mask = np.isin(y, keep_classes)
    X = X[class_mask]
    y = y[class_mask]

    max_components = min(len(keep_classes) - 1, X.shape[1])
    if n_components is None or n_components > max_components:
        n_components = max_components

    print(f"[LDA] Fitting: {len(X)} spikes, {len(keep_classes)} classes, "
          f"{X.shape[1]}d -> {n_components}d")

    lda = LinearDiscriminantAnalysis(n_components=n_components, solver='svd')
    lda.fit(X, y)

    lda_mean = lda.xbar_.astype(np.float32)
    lda_scalings = lda.scalings_[:, :n_components].astype(np.float32)

    evr = lda.explained_variance_ratio_[:n_components]
    cumvar = np.cumsum(evr)
    print(f"[LDA] Explained variance ratio (top {min(5, n_components)}): "
          f"{', '.join(f'{v:.1%}' for v in evr[:5])}")
    print(f"[LDA] Cumulative: {', '.join(f'{v:.1%}' for v in cumvar[:5])}"
          f"{'...' if n_components > 5 else ''} total={cumvar[-1]:.1%}")

    # Fisher ratio evaluation
    X_proj = (X - lda_mean) @ lda_scalings
    fisher_ratios = []
    unique_y = np.unique(y)
    global_mean = X_proj.mean(axis=0)
    for cls in unique_y:
        m = y == cls
        if m.sum() < 2:
            continue
        cls_mean = X_proj[m].mean(axis=0)
        between = np.sum((cls_mean - global_mean) ** 2)
        within = np.mean(np.sum((X_proj[m] - cls_mean) ** 2, axis=1))
        if within > 1e-12:
            fisher_ratios.append(between / within)
    if fisher_ratios:
        fr = np.array(fisher_ratios)
        print(f"[LDA] Fisher ratio in LDA space: median={np.median(fr):.3f}, "
              f"mean={np.mean(fr):.3f}, >1.5: {(fr > 1.5).sum()}, >2.0: {(fr > 2.0).sum()}")

    info = {
        'n_components': n_components,
        'n_classes': len(keep_classes),
        'n_spikes': len(X),
        'explained_variance_ratio': evr,
        'fisher_ratios_median': float(np.median(fisher_ratios)) if fisher_ratios else 0,
    }
    return lda_mean, lda_scalings, info


def _apply_lda(features, lda_mean, lda_scalings, precision_bits=None):
    """Project features through LDA crossbar.

    Signal path: DAC(features - mean) -> weight-quantized projection -> ADC
    """
    centered = features - lda_mean
    if using_int_accum_quantization():
        bits = precision_bits if precision_bits is not None else \
            getattr(config, 'PRECISION_LDA', 6)
        weights = _prepared_lda_weights(lda_scalings, bits)
        proj = quantized_vmm(
            centered, weights, config.PRECISION_DAC, bits,
            op_name="LDA")
    else:
        # DAC quantization on crossbar input
        centered_dac = quantize_signal(centered, config.PRECISION_DAC)
        proj = (centered_dac @ lda_scalings).astype(np.float32)
    # ADC quantization on crossbar output
    adc_name = "LDA.adc" if using_int_accum_quantization() else None
    proj = quantize_signal(proj, config.PRECISION_ADC, op_name=adc_name)
    if precision_bits is not None and precision_bits < 32:
        proj = quantize_weights(proj, precision_bits)
    return proj


# ==============================================================
# SpikeClustering — main pipeline interface
# ==============================================================

class SpikeClustering:
    """Full spike clustering pipeline.

    Calibration: official KS4 labels -> Version B centers
    Streaming:   extract features -> nearest-center assignment -> optional EMA update
    """

    def __init__(self, channel_positions, wPCA=None, wTEMP=None,
                 pca_precision_bits=None, assign_precision_bits=None,
                 **legacy_kwargs):
        self.positions = channel_positions
        self.wPCA = wPCA
        self.wTEMP = wTEMP
        self.pca_precision_bits = pca_precision_bits or config.PRECISION_PCA
        legacy_assign_bits = legacy_kwargs.pop("kmeans_precision_bits", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected SpikeClustering arguments: {unexpected}")
        if assign_precision_bits is None and legacy_assign_bits is not None:
            assign_precision_bits = legacy_assign_bits
        self.assign_precision_bits = assign_precision_bits or config.PRECISION_ASSIGN

        self.temporal_pca = TemporalPCAProjector(
            n_components=config.CLUST_TEMPORAL_PCA_NC,
            precision_bits=self.pca_precision_bits
        )
        self.assigner = None

        self.all_times = []
        self.all_channels = []
        self.all_amplitudes = []
        self.all_templates = []
        self.all_scales = []
        self.all_labels = []
        self.all_distances = []
        self.all_features = []
        self.is_calibrated = False
        self._cali_result = None

        # LDA projection state
        self.lda_mode = getattr(config, 'CLUST_LDA_MODE', 'off')
        self.lda_mean_ = None
        self.lda_scalings_ = None
        self.lda_scalings_q_ = None
        self.lda_per_region_ = None
        self.lda_enabled = False
        self.lda_precision_bits = getattr(config, 'PRECISION_LDA', 6)

        # Precompute nearest-channel index
        if channel_positions is not None:
            n_nearby = config.CLUST_N_NEARBY
            dist_matrix = cdist(channel_positions, channel_positions)
            self._nearby_idx = np.argsort(dist_matrix, axis=1)[:, :n_nearby]
            n_direct = getattr(config, 'CLUST_LDA_DIRECT_N_CHANNELS', 5)
            self._direct_nearby_idx = np.argsort(dist_matrix, axis=1)[:, :n_direct]
        else:
            self._nearby_idx = None
            self._direct_nearby_idx = None

    def calibrate(self, whitened_chunk=None, raw_chunk=None, output_dir=None,
                  sa_events=None):
        """Run calibration: official KS4 -> Version B streaming centers.

        Version B: only spikes visible to both KS4 and the streaming detector
        are used for center computation. This ensures centers match the signal
        distribution seen during streaming.
        """
        if whitened_chunk is None:
            raise ValueError("whitened_chunk is required for calibration")
        if raw_chunk is None:
            raise ValueError("raw_chunk is required for official KS4 backend")
        if output_dir is None:
            raise ValueError("output_dir is required")

        # Temporal PCA setup
        if self.wPCA is not None:
            self.temporal_pca.set_external_basis(self.wPCA)
        else:
            if sa_events is None:
                raise ValueError("sa_events required when wPCA not provided")
            waveforms = _collect_single_channel_waveforms(
                whitened_chunk, sa_events['times'], sa_events['channels'],
                self.positions
            )
            clean_wf, _ = _clean_waveforms_kmeans(waveforms)
            if len(clean_wf) < 50:
                clean_wf = waveforms
            self.temporal_pca.calibrate(clean_wf)

        # Create assigner
        self.assigner = RegionalCenterAssigner(
            self.positions,
            region_size=config.CLUST_REGION_SIZE,
            overlap=config.CLUST_REGION_OVERLAP,
            precision_bits=self.assign_precision_bits
        )

        self.temporal_pca.set_full_precision(True)

        # --- KS4 cache logic ---
        use_cache = bool(getattr(config, 'KS4_CACHE_ENABLE', False))
        cache_path = _ks4_cache_path(output_dir,
                                     raw_chunk.shape[0], raw_chunk.shape[1]) if use_cache else None

        try:
            if use_cache and cache_path and os.path.exists(cache_path):
                cali_result = _load_ks4_cache(cache_path)
                max_t = cali_result['all_times'].max() if len(cali_result['all_times']) > 0 else 0
                if max_t >= whitened_chunk.shape[0]:
                    print(f'[KS4-CACHE] WARNING: cached max time {max_t} >= whitened buffer '
                          f'{whitened_chunk.shape[0]}, re-filtering')
                    valid = cali_result['all_times'] < whitened_chunk.shape[0]
                    for key in ('all_times', 'all_times_raw', 'all_channels', 'all_labels'):
                        if key in cali_result and len(cali_result[key]) == len(valid):
                            cali_result[key] = cali_result[key][valid]
                    if 'spike_xy' in cali_result and len(cali_result['spike_xy']) == len(valid):
                        cali_result['spike_xy'] = cali_result['spike_xy'][valid]
            else:
                cali_result = run_official_ks4_calibration(
                    raw_data=raw_chunk,
                    channel_positions=self.positions,
                    output_dir=output_dir,
                )
                valid = ((cali_result['all_times'] >= 0) &
                         (cali_result['all_times'] < whitened_chunk.shape[0]))
                if not np.all(valid):
                    n_drop = int((~valid).sum())
                    print(f"[CALI] Dropping {n_drop} KS4 spikes outside whitened buffer")
                    for key in ('all_times', 'all_times_raw', 'all_channels', 'all_labels'):
                        if key in cali_result and len(cali_result[key]) == len(valid):
                            cali_result[key] = cali_result[key][valid]
                    if 'spike_xy' in cali_result and len(cali_result['spike_xy']) == len(valid):
                        cali_result['spike_xy'] = cali_result['spike_xy'][valid]
                if use_cache and cache_path:
                    _save_ks4_cache(cache_path, cali_result)
        except Exception as e:
            raise RuntimeError(
                "Official KS4 calibration is required and failed; "
                "SA+KMeans fallback is disabled by design."
            ) from e
        finally:
            self.temporal_pca.set_full_precision(False)

        from calibration import map_clusters_to_regions

        # --- SA-based spike detection (always needed) ---
        from spatial_aggregation import SpatialAggregator
        from alignment_diagnostic import _detect_streaming_on_calibdata
        sa = SpatialAggregator(self.positions, precision_bits=config.PRECISION_SA)
        stream_ev = _detect_streaming_on_calibdata(whitened_chunk, self.wTEMP, sa)

        # --- KS4 path: Features for ALL KS4 spikes ---
        feat_all = extract_temporal_features(
            whitened_chunk, cali_result['all_times'], cali_result['all_channels'],
            self.positions, self.temporal_pca, nearby_idx=self._nearby_idx)
        labels_all, centers_all, counts_all, _ = _compute_centers_from_labeled_features(
            feat_all, cali_result['all_labels'])

        cali_result['features'] = feat_all
        cali_result['centers_60d'] = centers_all
        cali_result['counts'] = counts_all

        # Match streaming spikes to KS4 spikes
        from alignment_diagnostic import _match_spikes
        matched_idx = _match_spikes(
            stream_ev['times'], stream_ev['channels'],
            cali_result['all_times'], cali_result['all_channels'],
            max_dt=getattr(config, 'CALI_CENTER_MATCH_MAX_DT_SAMPLES', 3))
        keep = matched_idx >= 0

        feat_B = extract_temporal_features(
            whitened_chunk, stream_ev['times'][keep], stream_ev['channels'][keep],
            self.positions, self.temporal_pca, nearby_idx=self._nearby_idx)
        inherited_labels = cali_result['all_labels'][matched_idx[keep]]
        labels_B, centers_B, counts_B, _ = _compute_centers_from_labeled_features(
            feat_B, inherited_labels)
        channels_B = stream_ev['channels'][keep]
        cali_result['_stream_ev'] = {
            k: np.asarray(v).copy() for k, v in stream_ev.items()
        }

        regC_B, regN_B, cluster_to_region_B = map_clusters_to_regions(
            labels_B, channels_B, centers_B,
            self.assigner.ch_to_region, self.assigner.n_regions, self.positions.shape[0])
        accept_B, update_B = _compute_cluster_distance_thresholds(feat_B, labels_B, centers_B)
        regAccept_B = _group_cluster_values_by_region(accept_B, cluster_to_region_B, self.assigner.n_regions)
        regUpdate_B = _group_cluster_values_by_region(update_B, cluster_to_region_B, self.assigner.n_regions)

        summary_B = {
            **_summarize_region_payload(regC_B, regN_B),
            'streaming_visible_spikes': int(keep.sum()),
            'streaming_detected_spikes': int(len(stream_ev['times'])),
        }

        cali_result['region_centers'] = regC_B
        cali_result['region_counts'] = regN_B
        cali_result['cluster_to_region'] = cluster_to_region_B

        # --- LDA: supervised feature projection ---
        lda_mode = getattr(config, 'CLUST_LDA_MODE', 'off')
        self.lda_mode = lda_mode

        if lda_mode == 'post_pca':
            lda_n = getattr(config, 'CLUST_LDA_N_COMPONENTS', None)
            lda_min = int(getattr(config, 'CLUST_LDA_MIN_SAMPLES_PER_CLASS', 10))
            lda_prec = int(getattr(config, 'PRECISION_LDA', 6))
            print(f"[LDA] Mode: post_pca (60d PCA -> LDA)")
            lda_mean, lda_scalings, lda_info = _fit_lda_projection(
                feat_B, labels_B, n_components=lda_n, min_samples=lda_min)
            if lda_mean is not None:
                self.lda_mean_ = lda_mean
                self.lda_scalings_ = lda_scalings
                self.lda_enabled = True
                self.lda_precision_bits = lda_prec
                n_lda = lda_scalings.shape[1]
                print(f"[LDA] Projecting features: 60d -> {n_lda}d, "
                      f"weight quantization={lda_prec}-bit")

                lda_scalings_q = quantize_weights(lda_scalings, lda_prec)
                max_q_err = np.abs(lda_scalings - lda_scalings_q).max()
                print(f"[LDA] Scalings weight quant error: {max_q_err:.4e}")
                self.lda_scalings_q_ = lda_scalings_q

                feat_B_lda = _apply_lda(feat_B, lda_mean, lda_scalings_q)
                labels_B_r, centers_B_lda, counts_B_r, _ = \
                    _compute_centers_from_labeled_features(feat_B_lda, labels_B)
                regC_B_lda, regN_B_lda, ctr_B_lda = map_clusters_to_regions(
                    labels_B_r, channels_B, centers_B_lda,
                    self.assigner.ch_to_region, self.assigner.n_regions, self.positions.shape[0])
                acc_B_lda, upd_B_lda = _compute_cluster_distance_thresholds(
                    feat_B_lda, labels_B_r, centers_B_lda)
                regAccept_B = _group_cluster_values_by_region(acc_B_lda, ctr_B_lda, self.assigner.n_regions)
                regUpdate_B = _group_cluster_values_by_region(upd_B_lda, ctr_B_lda, self.assigner.n_regions)
                regC_B, regN_B = regC_B_lda, regN_B_lda
                cluster_to_region_B = ctr_B_lda
                summary_B = {
                    **_summarize_region_payload(regC_B_lda, regN_B_lda),
                    'streaming_visible_spikes': int(keep.sum()),
                    'streaming_detected_spikes': int(len(stream_ev['times'])),
                }
                cali_result['features'] = _apply_lda(feat_all, lda_mean, lda_scalings_q)
                _, cali_result['centers_60d'], _, _ = _compute_centers_from_labeled_features(
                    cali_result['features'], cali_result['all_labels'])
            else:
                print("[LDA] Could not fit LDA, falling back to 60d PCA features")

        elif lda_mode == 'direct':
            lda_n = getattr(config, 'CLUST_LDA_N_COMPONENTS', None)
            lda_min = int(getattr(config, 'CLUST_LDA_MIN_SAMPLES_PER_CLASS', 10))
            lda_prec = int(getattr(config, 'PRECISION_LDA', 6))
            n_direct_ch = getattr(config, 'CLUST_LDA_DIRECT_N_CHANNELS', 5)
            snippet_len = 2 * config.CLUST_SNIPPET_HW + 1
            input_dim = n_direct_ch * snippet_len
            print(f"[LDA] Mode: direct ({n_direct_ch}ch x {snippet_len} = {input_dim}d -> LDA)")

            raw_feat_B = extract_raw_waveform_features(
                whitened_chunk, stream_ev['times'][keep], channels_B,
                self.positions, snippet_len=snippet_len,
                n_nearby=n_direct_ch, nearby_idx=self._direct_nearby_idx)
            print(f"[LDA] Raw features B: {raw_feat_B.shape}")

            lda_mean, lda_scalings, lda_info = _fit_lda_projection(
                raw_feat_B, labels_B, n_components=lda_n, min_samples=lda_min)

            if lda_mean is not None:
                self.lda_mean_ = lda_mean
                self.lda_scalings_ = lda_scalings
                self.lda_enabled = True
                self.lda_precision_bits = lda_prec
                n_lda = lda_scalings.shape[1]
                print(f"[LDA] Projecting features: {input_dim}d -> {n_lda}d, "
                      f"weight quantization={lda_prec}-bit")

                lda_scalings_q = quantize_weights(lda_scalings, lda_prec)
                max_q_err = np.abs(lda_scalings - lda_scalings_q).max()
                print(f"[LDA] Scalings weight quant error: {max_q_err:.4e}")
                self.lda_scalings_q_ = lda_scalings_q

                feat_B_lda = _apply_lda(raw_feat_B, lda_mean, lda_scalings_q)
                labels_B_r, centers_B_lda, counts_B_r, _ = \
                    _compute_centers_from_labeled_features(feat_B_lda, labels_B)
                regC_B_lda, regN_B_lda, ctr_B_lda = map_clusters_to_regions(
                    labels_B_r, channels_B, centers_B_lda,
                    self.assigner.ch_to_region, self.assigner.n_regions, self.positions.shape[0])
                acc_B_lda, upd_B_lda = _compute_cluster_distance_thresholds(
                    feat_B_lda, labels_B_r, centers_B_lda)
                regAccept_B = _group_cluster_values_by_region(acc_B_lda, ctr_B_lda, self.assigner.n_regions)
                regUpdate_B = _group_cluster_values_by_region(upd_B_lda, ctr_B_lda, self.assigner.n_regions)
                regC_B, regN_B = regC_B_lda, regN_B_lda
                cluster_to_region_B = ctr_B_lda
                summary_B = {
                    **_summarize_region_payload(regC_B_lda, regN_B_lda),
                    'streaming_visible_spikes': int(keep.sum()),
                    'streaming_detected_spikes': int(len(stream_ev['times'])),
                }
                # Project ALL KS4 features for diagnostic
                raw_feat_all = extract_raw_waveform_features(
                    whitened_chunk, cali_result['all_times'], cali_result['all_channels'],
                    self.positions, snippet_len=snippet_len,
                    n_nearby=n_direct_ch, nearby_idx=self._direct_nearby_idx)
                cali_result['features'] = _apply_lda(raw_feat_all, lda_mean, lda_scalings_q)
                _, cali_result['centers_60d'], _, _ = _compute_centers_from_labeled_features(
                    cali_result['features'], cali_result['all_labels'])
            else:
                print("[LDA] Could not fit direct LDA, falling back to 60d PCA features")

        elif lda_mode == 'per_region':
            lda_n = getattr(config, 'CLUST_LDA_N_COMPONENTS', None)
            lda_min = int(getattr(config, 'CLUST_LDA_MIN_SAMPLES_PER_CLASS', 10))
            lda_prec = int(getattr(config, 'PRECISION_LDA', 6))
            print(f"[LDA] Mode: per_region (60d PCA -> per-region LDA)")

            ch_to_region = self.assigner.ch_to_region
            n_regions = self.assigner.n_regions
            region_B = ch_to_region[channels_B]

            self.lda_per_region_ = {}
            any_fitted = False

            region_lda_fits = {}
            max_n_lda = 0
            for r_id in range(n_regions):
                mask_r = region_B == r_id
                n_r = mask_r.sum()
                if n_r < 50:
                    continue
                r_mean, r_scalings, r_info = _fit_lda_projection(
                    feat_B[mask_r], labels_B[mask_r],
                    n_components=lda_n, min_samples=lda_min)
                if r_mean is None:
                    continue
                r_scalings_q = quantize_weights(r_scalings, lda_prec)
                region_lda_fits[r_id] = {'mean': r_mean, 'scalings_q': r_scalings_q}
                max_n_lda = max(max_n_lda, r_scalings_q.shape[1])

            if max_n_lda == 0:
                print("[LDA] No per-region LDA fitted, falling back to 60d PCA")
            else:
                feat_B_lda_all = np.zeros((len(feat_B), max_n_lda), dtype=np.float32)

                for r_id, fit in region_lda_fits.items():
                    self.lda_per_region_[r_id] = fit
                    any_fitted = True
                    mask_r = region_B == r_id
                    actual_n = fit['scalings_q'].shape[1]
                    feat_B_lda_all[mask_r, :actual_n] = _apply_lda(
                        feat_B[mask_r], fit['mean'], fit['scalings_q'])

                actual_n_lda = max_n_lda

            if any_fitted:
                self.lda_enabled = True
                self.lda_precision_bits = lda_prec
                n_fitted = len(self.lda_per_region_)
                print(f"[LDA] Per-region LDA fitted: {n_fitted}/{n_regions} regions, "
                      f"output={actual_n_lda}d")

                # Global fallback for regions without enough data
                if n_fitted < n_regions:
                    fb_mean, fb_scalings, _ = _fit_lda_projection(
                        feat_B, labels_B,
                        n_components=actual_n_lda, min_samples=lda_min)
                    if fb_mean is not None:
                        fb_scalings_q = quantize_weights(fb_scalings, lda_prec)
                        for r_id in range(n_regions):
                            if r_id not in self.lda_per_region_:
                                self.lda_per_region_[r_id] = {
                                    'mean': fb_mean, 'scalings_q': fb_scalings_q,
                                }
                                mask_r = region_B == r_id
                                if mask_r.any():
                                    feat_B_lda_all[mask_r, :actual_n_lda] = _apply_lda(
                                        feat_B[mask_r], fb_mean, fb_scalings_q)
                        print(f"[LDA] Global fallback applied to {n_regions - n_fitted} regions")

                # Recompute centers in per-region LDA space
                labels_B_r, centers_B_lda, counts_B_r, _ = \
                    _compute_centers_from_labeled_features(feat_B_lda_all, labels_B)
                regC_B_lda, regN_B_lda, ctr_B_lda = map_clusters_to_regions(
                    labels_B_r, channels_B, centers_B_lda,
                    ch_to_region, n_regions, self.positions.shape[0])
                acc_B_lda, upd_B_lda = _compute_cluster_distance_thresholds(
                    feat_B_lda_all, labels_B_r, centers_B_lda)
                regAccept_B = _group_cluster_values_by_region(acc_B_lda, ctr_B_lda, n_regions)
                regUpdate_B = _group_cluster_values_by_region(upd_B_lda, ctr_B_lda, n_regions)
                regC_B, regN_B = regC_B_lda, regN_B_lda
                cluster_to_region_B = ctr_B_lda
                summary_B = {
                    **_summarize_region_payload(regC_B_lda, regN_B_lda),
                    'streaming_visible_spikes': int(keep.sum()),
                    'streaming_detected_spikes': int(len(stream_ev['times'])),
                }
                # Project ALL KS4 features for diagnostic
                region_all = ch_to_region[cali_result['all_channels']]
                feat_all_lda = np.zeros((len(feat_all), actual_n_lda), dtype=np.float32)
                for r_id, fit in self.lda_per_region_.items():
                    mask_r = region_all == r_id
                    if mask_r.any():
                        actual_n = fit['scalings_q'].shape[1]
                        feat_all_lda[mask_r, :actual_n] = _apply_lda(
                            feat_all[mask_r], fit['mean'], fit['scalings_q'])
                cali_result['features'] = feat_all_lda
                _, cali_result['centers_60d'], _, _ = _compute_centers_from_labeled_features(
                    feat_all_lda, cali_result['all_labels'])
            else:
                print("[LDA] No per-region LDA fitted, falling back to 60d PCA")

        # --- Load centers ---
        print(f"[CALI] Version B: centers={summary_B['total_centers']}, "
              f"nonempty_regions={summary_B['nonempty_regions']}, "
              f"visible={summary_B.get('streaming_visible_spikes', 0)}/"
              f"{summary_B.get('streaming_detected_spikes', 0)}")

        self._cali_result = cali_result
        self.is_calibrated = True
        self.assigner.set_precomputed_centers(
            regC_B, regN_B,
            precision_bits=self.assign_precision_bits,
            region_accept_radii=regAccept_B,
            region_update_radii=regUpdate_B,
        )
        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self.all_times = []
        self.all_channels = []
        self.all_amplitudes = []
        self.all_templates = []
        self.all_scales = []
        self.all_labels = []
        self.all_distances = []
        self.all_features = []
        self._batch_diag_printed = False

    def process_batch(self, sa_events, whitened_chunk,
                      batch_time_offset=0):
        """Process one streaming batch: extract features -> assign -> update."""
        assert self.is_calibrated, "Not calibrated"
        n_spikes = len(sa_events['times'])
        if n_spikes == 0:
            return np.array([], dtype=np.int32)

        if self.lda_mode == 'direct' and self.lda_enabled and self.lda_scalings_q_ is not None:
            snippet_len = 2 * config.CLUST_SNIPPET_HW + 1
            raw_feat = extract_raw_waveform_features(
                whitened_chunk, sa_events['times'], sa_events['channels'],
                self.positions, snippet_len=snippet_len,
                nearby_idx=self._direct_nearby_idx)
            features = _apply_lda(raw_feat, self.lda_mean_, self.lda_scalings_q_)
        elif self.lda_mode == 'per_region' and self.lda_enabled and self.lda_per_region_:
            pca_feat = extract_temporal_features(
                whitened_chunk, sa_events['times'], sa_events['channels'],
                self.positions, self.temporal_pca, nearby_idx=self._nearby_idx)
            region_ids = self.assigner.ch_to_region[sa_events['channels']]
            n_lda = max(s['scalings_q'].shape[1] for s in self.lda_per_region_.values())
            features = np.zeros((n_spikes, n_lda), dtype=np.float32)
            for r_id, lda_state in self.lda_per_region_.items():
                mask = region_ids == r_id
                if mask.any():
                    proj = _apply_lda(
                        pca_feat[mask], lda_state['mean'], lda_state['scalings_q'])
                    actual_n = proj.shape[1]
                    features[mask, :actual_n] = proj
        else:
            features = extract_temporal_features(
                whitened_chunk, sa_events['times'], sa_events['channels'],
                self.positions, self.temporal_pca, nearby_idx=self._nearby_idx
            )
            if self.lda_mode == 'post_pca' and self.lda_enabled and self.lda_scalings_q_ is not None:
                features = _apply_lda(features, self.lda_mean_, self.lda_scalings_q_)

        labels, distances, update_mask = self.assigner.assign(
            features, sa_events['channels'])

        if not hasattr(self, '_batch_diag_printed') or not self._batch_diag_printed:
            self._batch_diag_printed = True
            print(f"[STREAM] First batch: mode={self.lda_mode}, "
                  f"lda_enabled={self.lda_enabled}, "
                  f"feat_shape={features.shape}, "
                  f"n_per_region={len(self.lda_per_region_) if self.lda_per_region_ else 0}")

        self.assigner.update_centers(
            features, sa_events['channels'], labels, update_mask=update_mask)

        self.all_times.append(sa_events['times'] + batch_time_offset)
        self.all_channels.append(sa_events['channels'])
        self.all_amplitudes.append(sa_events['amplitudes'])
        self.all_templates.append(sa_events['templates'])
        self.all_scales.append(sa_events['scales'])
        self.all_labels.append(labels)
        self.all_distances.append(distances)
        self.all_features.append(features)
        return labels

    def get_results(self):
        if not self.all_times:
            return None

        raw_times = np.concatenate(self.all_times)
        # Only correct for FIR group delay if FIR was actually applied
        fir_delay = getattr(config, 'FIR_GROUP_DELAY', 0)
        if getattr(self, '_skip_fir', False):
            fir_delay = 0
        corrected_times = raw_times - fir_delay

        counts = self.assigner.get_all_counts()
        centers = self.assigner.get_all_centers()
        active = self.assigner.get_active_clusters()
        n_total = self.assigner.n_clusters_total

        return {
            'times': corrected_times,
            'times_uncorrected': raw_times,
            'fir_group_delay': fir_delay,
            'channels': np.concatenate(self.all_channels),
            'amplitudes': np.concatenate(self.all_amplitudes),
            'templates': np.concatenate(self.all_templates),
            'scales': np.concatenate(self.all_scales),
            'labels': np.concatenate(self.all_labels),
            'distances': np.concatenate(self.all_distances),
            'features': np.vstack(self.all_features) if self.all_features else None,
            'centers': centers,
            'counts': counts,
            'active_clusters': active,
            'n_active': len(active),
            'pca_components': self.temporal_pca.basis.copy(),
            'pca_mean': self.temporal_pca.mean_.copy(),
            'pca_variance_ratio': None if self.temporal_pca.explained_variance_ratio_ is None
            else self.temporal_pca.explained_variance_ratio_.copy(),
            'n_components': self.temporal_pca.n_components,
            'n_clusters_total': n_total,
            'mode': 'temporal_regional',
            'n_regions': self.assigner.n_regions,
            'k_per_region': self.assigner.k_per_region,
            'cali_mode': 'ks4',
        }


# ==============================================================
# Save results
# ==============================================================

def save_clustering_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in ['times', 'channels', 'amplitudes', 'templates',
                 'scales', 'distances']:
        np.save(os.path.join(output_dir, f'spike_{name}.npy'), results[name])
    np.save(os.path.join(output_dir, 'spike_clusters.npy'), results['labels'])
    np.save(os.path.join(output_dir, 'cluster_centers.npy'), results['centers'])
    np.save(os.path.join(output_dir, 'cluster_counts.npy'), results['counts'])
    np.save(os.path.join(output_dir, 'pca_components.npy'),
            results['pca_components'])
    np.save(os.path.join(output_dir, 'pca_mean.npy'), results['pca_mean'])

    if 'is_good' in results:
        np.save(os.path.join(output_dir, 'cluster_is_good.npy'),
                results['is_good'])
    if 'contam_rate' in results:
        np.save(os.path.join(output_dir, 'cluster_contam_rate.npy'),
                results['contam_rate'])

    print(f"\n[SAVE] {len(results['times'])} spikes -> "
          f"{results['n_active']} active clusters "
          f"(K={results['n_clusters_total']})")
