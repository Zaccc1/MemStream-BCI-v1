"""
Spatial Aggregation Module
===========================
Two operating modes controlled by config.SA_DENSE:

Dense mode (SA_DENSE=True):
  B (T, n_ch, 6) -> Dense VMM: A = G[ch] @ B[t, neighbours, :]
  -> score_map (T, n_ch) = max(|A|) -> Threshold + NMS -> spike events

Sparse mode (SA_DENSE=False):
  B (T, n_ch, 6) -> Coarse screen -> Sparse VMM on candidates
  -> Threshold + NMS -> spike events

Signal path per VMM: DAC(B input) -> weight-quantized aggregation -> ADC(score)
"""

import numpy as np
from scipy.spatial.distance import cdist

import config
from preprocessing import (
    quantize_weights, quantize_signal, quantized_vmm,
    using_int_accum_quantization, using_uint8_input_quantization,
    prepare_quantized_weights, _fixed_range_for,
)


# ==============================================================
# Spatial Aggregator
# ==============================================================

class SpatialAggregator:
    """Spatial aggregation of template matching scores using Gaussian-weighted
    neighbour summation, implemented as a small VMM per channel."""

    def __init__(self, channel_positions, n_nearest=None, n_scales=None,
                 sigma_um=None, precision_bits=None):
        self.n_nearest = n_nearest or config.SA_NEAREST_CHANNELS
        self.n_scales = n_scales or config.SA_N_SCALES
        self.sigma_um = sigma_um or config.SA_SIGMA_UM
        self.precision_bits = precision_bits or config.PRECISION_SA
        self.positions = channel_positions.astype(np.float32)
        self.n_ch = self.positions.shape[0]

        if len(self.sigma_um) < self.n_scales:
            self.sigma_um = np.linspace(
                self.sigma_um[0], self.sigma_um[-1], self.n_scales
            ).tolist()
        self.sigma_um = self.sigma_um[:self.n_scales]

        dist_matrix = cdist(self.positions, self.positions).astype(np.float32)

        self.neighbour_idx = np.zeros((self.n_ch, self.n_nearest), dtype=np.int32)
        self.neighbour_dist = np.zeros((self.n_ch, self.n_nearest), dtype=np.float32)
        for ch in range(self.n_ch):
            idx = np.argsort(dist_matrix[ch])[:self.n_nearest]
            self.neighbour_idx[ch] = idx
            self.neighbour_dist[ch] = dist_matrix[ch, idx]

        self.G_full = np.zeros(
            (self.n_ch, self.n_scales, self.n_nearest), dtype=np.float32)
        for ch in range(self.n_ch):
            d = self.neighbour_dist[ch]
            for s in range(self.n_scales):
                sigma = self.sigma_um[s]
                w = np.exp(-d ** 2 / (2 * sigma ** 2))
                norm = np.linalg.norm(w)
                if norm > 1e-12:
                    w /= norm
                self.G_full[ch, s] = w

        self.G = quantize_weights(self.G_full, self.precision_bits)
        for ch in range(self.n_ch):
            for s in range(self.n_scales):
                norm = np.linalg.norm(self.G[ch, s])
                if norm > 1e-12:
                    self.G[ch, s] /= norm
        self.G = quantize_weights(self.G, self.precision_bits)
        self.G_prepared = [
            prepare_quantized_weights(self.G[ch], self.precision_bits)
            for ch in range(self.n_ch)
        ]

        quant_err = np.abs(self.G - self.G_full).max()
        print(f"[SA] {self.n_ch} channels, {self.n_nearest} neighbours, "
              f"{self.n_scales} scales")
        print(f"[SA] sigma = {self.sigma_um} um")
        print(f"[SA] {self.precision_bits}-bit weights, max quant error: {quant_err:.4e}")

    # ==============================================================
    # Dense aggregation
    # ==============================================================

    def dense_aggregate(self, B):
        """Compute aggregated score for ALL (t, ch) positions.

        Signal path: DAC(B) -> weight-quantized VMM -> ADC(scores)
        """
        T, n_ch, n_templates = B.shape
        score_map = np.zeros((T, n_ch), dtype=np.float32)
        template_map = np.zeros((T, n_ch), dtype=np.int32)
        scale_map = np.zeros((T, n_ch), dtype=np.int32)

        n_combo = self.n_scales * n_templates

        for ch in range(n_ch):
            nb_idx = self.neighbour_idx[ch]
            b = B[:, nb_idx, :]                        # (T, n_nearest, n_templates)
            if using_int_accum_quantization():
                x = b.transpose(0, 2, 1).reshape(T * n_templates, self.n_nearest)
                A = quantized_vmm(
                    x, self.G_prepared[ch], config.PRECISION_DAC,
                    self.precision_bits,
                    op_name="SA.dense")
                A = A.reshape(T, n_templates, self.n_scales).transpose(0, 2, 1)
            else:
                # DAC quantization on VMM input
                b = quantize_signal(b, config.PRECISION_DAC)
                # A[t, s, k] = sum_n G[ch][s,n] * b[t,n,k]
                A = np.einsum('sn, tnk -> tsk', self.G[ch], b)
            # ADC quantization on VMM output
            adc_name = "SA.dense.adc" if using_int_accum_quantization() else None
            A = quantize_signal(A, config.PRECISION_ADC, op_name=adc_name)
            abs_A = np.abs(A).reshape(T, n_combo)
            max_idx = np.argmax(abs_A, axis=1)
            score_map[:, ch] = abs_A[np.arange(T), max_idx]
            scale_map[:, ch] = max_idx // n_templates
            template_map[:, ch] = max_idx % n_templates

        return score_map, template_map, scale_map

    def dense_detect(self, B, final_th=None, nms_samples=None,
                     nms_channels=None, time_offset=None, stride=None,
                     verbose=True):
        """Dense detection: dense aggregation -> threshold -> NMS."""
        f_th = final_th if final_th is not None else config.SA_FINAL_THRESHOLD
        nms_t = nms_samples if nms_samples is not None else config.SA_NMS_SAMPLES
        nms_c = nms_channels if nms_channels is not None else config.SA_NMS_CHANNELS
        t_off = time_offset if time_offset is not None else config.MATCH_NT // 2
        s = stride if stride is not None else config.MATCH_STRIDE

        T, n_ch, n_templates = B.shape

        if T > 3000 * 2:
            return self._dense_detect_chunked(
                B, f_th, nms_t, nms_c, t_off, s, 3000)

        score_map, template_map, scale_map = self.dense_aggregate(B)

        above_mask = score_map > f_th
        n_above = int(above_mask.sum())
        if verbose:
            print(f"[SA] Dense mode: ({T}, {n_ch}) score map, "
                  f"{n_above} above Th={f_th}\u03c3")
        if n_above == 0:
            if verbose:
                print(f"[SA] After NMS: 0 spike events")
            return _empty_events()

        survived = self._dense_nms(score_map, above_mask, nms_t, nms_c)

        if len(survived) == 0:
            if verbose:
                print(f"[SA] After NMS: 0 spike events")
            return _empty_events()

        times = survived[:, 0].astype(np.int64)
        channels = survived[:, 1].astype(np.int32)

        events = {
            'times': times * s + t_off,
            'channels': channels,
            'amplitudes': score_map[times, channels].astype(np.float32),
            'templates': template_map[times, channels].astype(np.int32),
            'scales': scale_map[times, channels].astype(np.int32),
            'features': np.array([], dtype=np.float32),
        }
        if verbose:
            print(f"[SA] After NMS: {len(events['times'])} spike events")
        return events

    def _dense_detect_chunked(self, B, f_th, nms_t, nms_c, t_off, stride, chunk_size):
        """Process large B tensor in chunks, then merge events."""
        T, n_ch, n_templates = B.shape
        overlap = nms_t * 2
        all_events = {k: [] for k in ['times', 'channels', 'amplitudes',
                                       'templates', 'scales']}
        total_detected = 0
        n_chunks = 0

        start = 0
        while start < T:
            end = min(start + chunk_size + overlap, T)
            B_chunk = B[start:end]

            score_map, template_map, scale_map = self.dense_aggregate(B_chunk)
            above_mask = score_map > f_th
            n_above = int(above_mask.sum())
            total_detected += n_above

            if n_above > 0:
                survived = self._dense_nms(score_map, above_mask, nms_t, nms_c)
                if len(survived) > 0:
                    times_local = survived[:, 0]
                    channels_local = survived[:, 1]
                    if end < T:
                        valid = times_local < chunk_size
                    else:
                        valid = np.ones(len(times_local), dtype=bool)
                    if valid.sum() > 0:
                        t_sel = times_local[valid]
                        ch_sel = channels_local[valid]
                        all_events['times'].append(
                            (t_sel.astype(np.int64) + start) * stride + t_off)
                        all_events['channels'].append(ch_sel.astype(np.int32))
                        all_events['amplitudes'].append(
                            score_map[t_sel, ch_sel].astype(np.float32))
                        all_events['templates'].append(
                            template_map[t_sel, ch_sel].astype(np.int32))
                        all_events['scales'].append(
                            scale_map[t_sel, ch_sel].astype(np.int32))

            n_chunks += 1
            start += chunk_size

        print(f"[SA] Dense mode (chunked): {n_chunks} chunks, "
              f"{total_detected} above Th={f_th}\u03c3")

        if not all_events['times']:
            print(f"[SA] After NMS: 0 spike events")
            return _empty_events()

        events = {
            'times': np.concatenate(all_events['times']),
            'channels': np.concatenate(all_events['channels']),
            'amplitudes': np.concatenate(all_events['amplitudes']),
            'templates': np.concatenate(all_events['templates']),
            'scales': np.concatenate(all_events['scales']),
            'features': np.array([], dtype=np.float32),
        }

        # Cross-chunk boundary NMS
        n_pre = len(events['times'])
        if n_pre > 1:
            order = np.argsort(events['times'])
            for k in events:
                if hasattr(events[k], '__len__') and len(events[k]) == n_pre:
                    events[k] = events[k][order]
            keep = np.ones(n_pre, dtype=bool)
            nb_sets = {}
            for ch in range(self.n_ch):
                nb_sets[ch] = set(self.neighbour_idx[ch, :nms_c].tolist())
            for i in range(1, n_pre):
                if not keep[i]:
                    continue
                for j in range(i - 1, -1, -1):
                    if events['times'][i] - events['times'][j] > nms_t * stride:
                        break
                    if not keep[j]:
                        continue
                    ch_i = int(events['channels'][i])
                    ch_j = int(events['channels'][j])
                    if ch_i in nb_sets.get(ch_j, set()) or ch_j in nb_sets.get(ch_i, set()):
                        if events['amplitudes'][i] <= events['amplitudes'][j]:
                            keep[i] = False
                        else:
                            keep[j] = False
                        break
            if keep.sum() < n_pre:
                for k in events:
                    if hasattr(events[k], '__len__') and len(events[k]) == n_pre:
                        events[k] = events[k][keep]

        print(f"[SA] After NMS: {len(events['times'])} spike events")
        return events

    def _dense_nms(self, score_map, above_mask, nms_t, nms_c):
        """Spatiotemporal NMS on dense score map.

        Layer 1 (spatial): keep only if max among nms_c nearest channels.
        Layer 2 (temporal): keep only if max in +/-nms_t window.
        Layer 3 (cross-channel): greedy suppression of nearby detections.
        """
        T, n_ch = score_map.shape

        nb_idx = self.neighbour_idx[:, :nms_c]
        spatial_max = score_map[:, nb_idx].max(axis=2)
        spatial_pass = above_mask & (score_map >= spatial_max)

        from scipy.ndimage import maximum_filter1d
        temporal_max = maximum_filter1d(
            score_map, size=2 * nms_t + 1, axis=0, mode='constant', cval=0)
        both_pass = spatial_pass & (score_map >= temporal_max)

        cand_t, cand_ch = np.where(both_pass)
        if len(cand_t) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        scores_surv = score_map[cand_t, cand_ch]
        order = np.argsort(-scores_surv)
        final_keep = np.ones(len(order), dtype=bool)
        kept_list = []

        nb_sets = {}
        for ch in range(self.n_ch):
            nb_sets[ch] = set(self.neighbour_idx[ch, :nms_c].tolist())

        for rank in range(len(order)):
            idx = order[rank]
            if not final_keep[idx]:
                continue
            t_i = int(cand_t[idx])
            ch_i = int(cand_ch[idx])
            nb_i = nb_sets[ch_i]

            for t_k, ch_k in reversed(kept_list):
                if abs(t_i - t_k) > nms_t:
                    continue
                if ch_i in nb_sets.get(ch_k, set()) or ch_k in nb_i:
                    final_keep[idx] = False
                    break

            if final_keep[idx]:
                kept_list.append((t_i, ch_i))

        survived = np.column_stack([cand_t, cand_ch])
        return survived[final_keep]

    # ==============================================================
    # Sparse aggregation (with coarse screen)
    # ==============================================================

    def coarse_screen(self, B, threshold=None, peak_window=None, verbose=True):
        """Digital coarse screening: find candidate (ch, t) pairs.
        Pure digital operation, no crossbar involved."""
        th = threshold if threshold is not None else config.SA_COARSE_THRESHOLD
        pw = peak_window if peak_window is not None else config.SA_NMS_SAMPLES
        max_score = np.max(np.abs(B), axis=2)
        T, n_ch = max_score.shape

        all_t = []
        all_ch = []
        for ch in range(n_ch):
            sig = max_score[:, ch]
            above = np.where(sig > th)[0]
            if len(above) == 0:
                continue
            keep = np.ones(len(above), dtype=bool)
            for i in range(len(above)):
                if not keep[i]:
                    continue
                t_i = above[i]
                for j in range(i + 1, len(above)):
                    t_j = above[j]
                    if t_j - t_i > pw:
                        break
                    if sig[t_j] > sig[t_i]:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False
            peaks = above[keep]
            all_t.append(peaks)
            all_ch.append(np.full(len(peaks), ch, dtype=np.int32))

        if all_t:
            candidates = np.column_stack([
                np.concatenate(all_t), np.concatenate(all_ch)])
        else:
            candidates = np.zeros((0, 2), dtype=np.int32)

        n_total = T * n_ch
        pct = len(candidates) / max(n_total, 1) * 100
        if verbose:
            print(f"[SA] Coarse screen (Th={th}\u03c3): "
                  f"{len(candidates)} candidates ({pct:.4f}% of {n_total})")
        return candidates

    def sparse_aggregate(self, B, candidates):
        """Spatial aggregation VMM on candidate points only.

        Signal path: DAC(B neighbour scores) -> weight-quantized VMM -> ADC(scores)
        """
        if self._can_batch_sparse_int_accum():
            return self._sparse_aggregate_batched_int_accum(B, candidates)

        N = len(candidates)
        n_templates = int(B.shape[2])
        n_feat = self.n_scales * n_templates
        scores = np.zeros(N, dtype=np.float32)
        best_templates = np.zeros(N, dtype=np.int32)
        best_scales = np.zeros(N, dtype=np.int32)
        features = np.zeros((N, n_feat), dtype=np.float32)

        for i in range(N):
            t, ch = candidates[i]
            nb_idx = self.neighbour_idx[ch]
            b = B[t, nb_idx, :]
            if using_int_accum_quantization():
                A = quantized_vmm(
                    b.T, self.G_prepared[ch], config.PRECISION_DAC,
                    self.precision_bits,
                    op_name="SA.sparse").T
            else:
                # DAC quantization on VMM input
                b = quantize_signal(b, config.PRECISION_DAC)
                A = self.G[ch] @ b
            # ADC quantization on VMM output
            adc_name = "SA.sparse.adc" if using_int_accum_quantization() else None
            A = quantize_signal(A, config.PRECISION_ADC, op_name=adc_name)
            features[i] = A.ravel()
            abs_A = np.abs(A)
            flat_idx = np.argmax(abs_A)
            s_idx, k_idx = np.unravel_index(flat_idx, A.shape)
            scores[i] = abs_A[s_idx, k_idx]
            best_templates[i] = k_idx
            best_scales[i] = s_idx

        return scores, best_templates, best_scales, features

    def _can_batch_sparse_int_accum(self):
        return (
            using_int_accum_quantization() and
            using_uint8_input_quantization() and
            _fixed_range_for("SA.sparse.input") is not None and
            _fixed_range_for("SA.sparse.adc") is not None
        )

    def _sparse_aggregate_batched_int_accum(self, B, candidates):
        """Batch sparse SA VMM by channel when fixed ranges make it equivalent."""
        N = len(candidates)
        n_templates = int(B.shape[2])
        n_feat = self.n_scales * n_templates
        scores = np.zeros(N, dtype=np.float32)
        best_templates = np.zeros(N, dtype=np.int32)
        best_scales = np.zeros(N, dtype=np.int32)
        features = np.zeros((N, n_feat), dtype=np.float32)

        if N == 0:
            return scores, best_templates, best_scales, features

        cand_ch = candidates[:, 1].astype(np.int32, copy=False)
        cand_t = candidates[:, 0].astype(np.int32, copy=False)
        for ch in np.unique(cand_ch):
            idx = np.flatnonzero(cand_ch == ch)
            nb_idx = self.neighbour_idx[int(ch)]
            b = B[cand_t[idx]][:, nb_idx, :]
            x = b.transpose(0, 2, 1).reshape(len(idx) * n_templates,
                                             self.n_nearest)
            A = quantized_vmm(
                x, self.G_prepared[int(ch)], config.PRECISION_DAC,
                self.precision_bits,
                op_name="SA.sparse")
            A = A.reshape(len(idx), n_templates, self.n_scales).transpose(0, 2, 1)
            A = quantize_signal(A, config.PRECISION_ADC,
                                op_name="SA.sparse.adc")

            features[idx] = A.reshape(len(idx), n_feat)
            abs_A = np.abs(A).reshape(len(idx), n_feat)
            flat_idx = np.argmax(abs_A, axis=1)
            row = np.arange(len(idx))
            scores[idx] = abs_A[row, flat_idx]
            best_scales[idx] = flat_idx // n_templates
            best_templates[idx] = flat_idx % n_templates

        return scores, best_templates, best_scales, features

    def sparse_detect(self, B, coarse_th=None, final_th=None,
                      nms_samples=None, nms_channels=None,
                      time_offset=None, stride=None, verbose=True):
        """Sparse detection: coarse screen -> aggregate -> threshold -> NMS."""
        c_th = coarse_th if coarse_th is not None else config.SA_COARSE_THRESHOLD
        f_th = final_th if final_th is not None else config.SA_FINAL_THRESHOLD
        t_off = time_offset if time_offset is not None else config.MATCH_NT // 2
        s = stride if stride is not None else config.MATCH_STRIDE

        candidates = self.coarse_screen(B, threshold=c_th, verbose=verbose)
        if len(candidates) == 0:
            if verbose:
                print(f"[SA] No candidates after coarse screen")
            return _empty_events()

        scores, best_templates, best_scales, features = self.sparse_aggregate(
            B, candidates)

        above = scores > f_th
        candidates = candidates[above]
        scores = scores[above]
        best_templates = best_templates[above]
        best_scales = best_scales[above]
        features = features[above]
        if verbose:
            print(f"[SA] After final threshold (Th={f_th}\u03c3): {len(scores)} detections")

        if len(scores) == 0:
            return _empty_events()

        events = self.spatiotemporal_nms(
            candidates, scores, best_templates, best_scales,
            features=features,
            nms_samples=nms_samples, nms_channels=nms_channels)

        if len(events['times']) > 0:
            events['times'] = events['times'] * s + t_off

        if verbose:
            print(f"[SA] After NMS: {len(events['times'])} spike events")
        return events

    def spatiotemporal_nms(self, candidates, scores, best_templates, best_scales,
                           features=None, nms_samples=None, nms_channels=None):
        """Greedy spatio-temporal NMS for sparse mode. Pure digital."""
        nms_t = nms_samples if nms_samples is not None else config.SA_NMS_SAMPLES
        nms_c = nms_channels if nms_channels is not None else config.SA_NMS_CHANNELS
        N = len(scores)
        if N == 0:
            return _empty_events()

        nb_sets = {}
        for ch in np.unique(candidates[:, 1]):
            nb_sets[int(ch)] = set(self.neighbour_idx[ch, :nms_c].tolist())

        order = np.argsort(-scores)
        keep = np.ones(N, dtype=bool)
        kept_events = []

        for rank in range(N):
            i = order[rank]
            if not keep[i]:
                continue
            t_i = int(candidates[i, 0])
            ch_i = int(candidates[i, 1])
            nb_i = nb_sets.get(ch_i, {ch_i})

            suppressed = False
            for t_k, ch_k, _ in reversed(kept_events):
                if abs(t_i - t_k) > nms_t:
                    continue
                if ch_i in nb_sets.get(ch_k, {ch_k}) or ch_k in nb_i:
                    keep[i] = False
                    suppressed = True
                    break

            if not suppressed:
                kept_events.append((t_i, ch_i, i))

        sel = np.array([idx for _, _, idx in kept_events])
        if len(sel) == 0:
            return _empty_events()

        events = {
            'times': candidates[sel, 0].astype(np.int64),
            'channels': candidates[sel, 1].astype(np.int32),
            'amplitudes': scores[sel].astype(np.float32),
            'templates': best_templates[sel].astype(np.int32),
            'scales': best_scales[sel].astype(np.int32),
            'features': features[sel] if features is not None else np.array([]),
        }
        return events

    # ==============================================================
    # Unified entry point
    # ==============================================================

    def detect(self, B, coarse_th=None, final_th=None,
               nms_samples=None, nms_channels=None,
               time_offset=None, stride=None, dense=None, verbose=True):
        """Detect spikes from score tensor B.
        Dispatches to dense or sparse mode based on config.SA_DENSE."""
        use_dense = dense if dense is not None else getattr(
            config, 'SA_DENSE', False)

        if use_dense:
            return self.dense_detect(
                B, final_th=final_th,
                nms_samples=nms_samples, nms_channels=nms_channels,
                time_offset=time_offset, stride=stride, verbose=verbose)
        else:
            return self.sparse_detect(
                B, coarse_th=coarse_th, final_th=final_th,
                nms_samples=nms_samples, nms_channels=nms_channels,
                time_offset=time_offset, stride=stride, verbose=verbose)


def _empty_events():
    return {
        'times': np.array([], dtype=np.int64),
        'channels': np.array([], dtype=np.int32),
        'amplitudes': np.array([], dtype=np.float32),
        'templates': np.array([], dtype=np.int32),
        'scales': np.array([], dtype=np.int32),
        'features': np.array([], dtype=np.float32),
    }
