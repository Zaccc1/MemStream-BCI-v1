"""
Template Matching Module
=========================
Computes score matrices from whitened data using universal templates.
Used by spatial aggregation for spike detection.

Signal path: DAC(whitened input) -> weight-quantized convolution -> ADC(scores)
"""

import numpy as np
from pathlib import Path

import config
from preprocessing import (
    quantize_weights, quantize_signal, quantized_vmm,
    using_int_accum_quantization, prepare_quantized_weights,
    using_uint8_input_quantization, _fixed_range_for, _record_quant_range,
    _quantize_uint8_codes_with_stats, dequantize_uint8_codes,
    _quantize_signed_codes_with_stats, dequantize_signed_codes,
    quantization_report, _should_sample_quant_details, _quant_diag_enabled,
)


# ==============================================================
# Template I/O
# ==============================================================

WTEMP_URL = "https://raw.githubusercontent.com/MouseLand/Kilosort/main/kilosort/wTEMP.npz"
WTEMP_LOCAL = Path("wTEMP.npz")
_MATCH_TEMPLATE_CACHE = {}


def _match_template_cache_key(wtemp, precision_bits):
    arr = np.asarray(wtemp)
    return (id(arr), arr.shape, arr.dtype.str, int(precision_bits))


def _prepared_match_templates(wtemp, precision_bits):
    """Cache quantized template weights for repeated streaming batches."""
    key = _match_template_cache_key(wtemp, precision_bits)
    cached = _MATCH_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    templates = quantize_weights(np.asarray(wtemp, dtype=np.float32),
                                 int(precision_bits))
    prepared = prepare_quantized_weights(templates, int(precision_bits))
    if len(_MATCH_TEMPLATE_CACHE) >= 8:
        _MATCH_TEMPLATE_CACHE.clear()
    _MATCH_TEMPLATE_CACHE[key] = (templates, prepared)
    return templates, prepared


def _find_in_kilosort_package():
    try:
        from kilosort.utils import template_path
        p = Path(template_path())
        if p.exists():
            return p
    except ImportError:
        pass
    try:
        import kilosort
        p = Path(kilosort.__file__).parent / "wTEMP.npz"
        if p.exists():
            return p
    except ImportError:
        pass
    return None


def _download_templates(url=None, dest=None):
    url = url or WTEMP_URL
    dest = Path(dest) if dest else WTEMP_LOCAL
    if dest.exists():
        return True
    try:
        from urllib.request import urlopen, Request
        import ssl
        ctx = ssl.create_default_context()
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        print(f"[TEMPLATE] Downloading from GitHub ...")
        resp = urlopen(req, timeout=15, context=ctx)
        with open(dest, 'wb') as f:
            f.write(resp.read())
        print(f"[TEMPLATE] Saved to {dest}")
        return True
    except Exception as e:
        print(f"[TEMPLATE] Download failed: {e}")
        return False


def get_templates(whitened_data=None, source=None, filepath=None,
                  channel_positions=None):
    """Get universal templates.

    Priority when TEMPLATE_FROM_DATA=True (default):
      data -> file -> package -> download

    Priority when TEMPLATE_FROM_DATA=False:
      file -> package -> download -> data
    """
    fp = Path(filepath) if filepath else WTEMP_LOCAL
    from_data = getattr(config, 'TEMPLATE_FROM_DATA', False)

    if from_data and whitened_data is not None and source in (None, 'data'):
        print("[TEMPLATE] Extracting templates from whitened calibration data...")
        wTEMP, wPCA = compute_templates_from_data(
            whitened_data, channel_positions=channel_positions)
        return wTEMP, wPCA, 'data'

    if source == 'file' or (source is None and fp.exists() and not from_data):
        if fp.exists():
            wTEMP, wPCA = _load_from_file(fp)
            return wTEMP, wPCA, 'file'
        elif source == 'file':
            raise FileNotFoundError(f"{fp} not found")

    if source in ('package', None):
        pkg_path = _find_in_kilosort_package()
        if pkg_path is not None:
            wTEMP, wPCA = _load_from_file(pkg_path)
            return wTEMP, wPCA, 'package'

    if source in ('download', None):
        if _download_templates(dest=fp):
            wTEMP, wPCA = _load_from_file(fp)
            return wTEMP, wPCA, 'download'

    if whitened_data is not None:
        print("[TEMPLATE] Computing templates from data (fallback)...")
        wTEMP, wPCA = compute_templates_from_data(
            whitened_data, channel_positions=channel_positions)
        return wTEMP, wPCA, 'data'

    raise RuntimeError("Cannot get templates")


def _load_from_file(filepath):
    dd = np.load(filepath)
    wTEMP = dd['wTEMP'].astype(np.float32)
    wPCA = dd['wPCA'].astype(np.float32)
    norms = np.linalg.norm(wTEMP, axis=1, keepdims=True)
    wTEMP = wTEMP / (norms + 1e-12)
    print(f"[TEMPLATE] Loaded {wTEMP.shape[0]} templates, "
          f"length={wTEMP.shape[1]}, from {filepath}")
    return wTEMP, wPCA


def compute_templates_from_data(whitened_data, n_templates=None, nt=None,
                                th_single_ch=None, channel_positions=None):
    """Extract wTEMP and wPCA from whitened calibration data.

    Steps:
      1. Find single-channel peaks exceeding threshold
      2. Keep only isolated peaks (no larger peak in temporal+channel neighborhood)
      3. L2-normalise all clips
      4. KMeans -> wTEMP (universal waveform templates)
      5. TruncatedSVD -> wPCA (PCA basis for feature extraction)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    n_temp = n_templates or config.MATCH_N_TEMPLATES
    wlen = nt or config.MATCH_NT
    th = th_single_ch or config.TEMPLATE_TH_DETECT
    half = wlen // 2
    max_clips = config.TEMPLATE_MAX_CLIPS
    iso_t = config.TEMPLATE_ISOLATION_T
    iso_ch = config.TEMPLATE_ISOLATION_CH

    n_samples, n_ch = whitened_data.shape
    print(f"[TEMPLATE] Extracting from whitened data: shape={whitened_data.shape}, "
          f"Th={th}s, nt={wlen}, isolation=({iso_t}t, {iso_ch}ch)")

    # Precompute spatial neighbours for isolation check
    use_spatial_isolation = channel_positions is not None
    if use_spatial_isolation:
        from scipy.spatial.distance import cdist as _cdist
        dist_mat = _cdist(channel_positions, channel_positions)
        iso_neighbor_idx = np.argsort(dist_mat, axis=1)[:, :2 * iso_ch + 1]
        print(f"[TEMPLATE] Using spatial-distance isolation "
              f"(positions provided, {iso_neighbor_idx.shape[1]} neighbours)")

    clips = []
    clip_channels = []
    ch_order = np.random.default_rng(config.RANDOM_SEED).permutation(n_ch)

    for ch in ch_order:
        signal = whitened_data[:, ch]
        abs_sig = np.abs(signal)

        above = np.where(abs_sig > th)[0]
        if len(above) == 0:
            continue

        for p in above:
            if p < half or p >= n_samples - half - 1:
                continue

            local_win = abs_sig[max(0, p - 4): p + 5]
            if abs_sig[p] < local_win.max():
                continue

            t_lo = max(0, p - iso_t)
            t_hi = min(n_samples, p + iso_t + 1)

            if use_spatial_isolation:
                nb_chs = iso_neighbor_idx[ch]
                neighborhood = np.abs(whitened_data[t_lo:t_hi][:, nb_chs])
            else:
                ch_lo = max(0, ch - iso_ch)
                ch_hi = min(n_ch, ch + iso_ch + 1)
                neighborhood = np.abs(whitened_data[t_lo:t_hi, ch_lo:ch_hi])

            if abs_sig[p] < neighborhood.max():
                continue

            clip = signal[p - half: p + half + 1].copy()
            norm = np.linalg.norm(clip)
            if norm > 1e-6:
                clips.append(clip / norm)

            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break

    clips = np.array(clips, dtype=np.float32)
    print(f"[TEMPLATE] Found {len(clips)} isolated clips")

    if len(clips) < n_temp * 10:
        print(f"[TEMPLATE] WARNING: Only {len(clips)} clips, lowering threshold")
        clips_fallback = []
        for ch in range(n_ch):
            signal = whitened_data[:, ch]
            above = np.where(np.abs(signal) > th * 0.7)[0]
            for p in above:
                if p < half or p >= n_samples - half - 1:
                    continue
                local = np.abs(signal[max(0, p - 4):p + 5])
                if np.abs(signal[p]) < local.max():
                    continue
                clip = signal[p - half: p + half + 1]
                norm = np.linalg.norm(clip)
                if norm > 1e-6:
                    clips_fallback.append(clip / norm)
                if len(clips_fallback) >= max_clips:
                    break
            if len(clips_fallback) >= max_clips:
                break
        clips = np.array(clips_fallback, dtype=np.float32)
        print(f"[TEMPLATE] Fallback: {len(clips)} clips at Th={th*0.7:.1f}s")

    if len(clips) < n_temp * 2:
        raise ValueError(f"Only {len(clips)} clips found, cannot extract templates")

    km = KMeans(n_clusters=n_temp, n_init=10, random_state=config.RANDOM_SEED)
    km.fit(clips)
    wTEMP = km.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(wTEMP, axis=1, keepdims=True)
    wTEMP = wTEMP / (norms + 1e-12)

    counts = np.bincount(km.labels_, minlength=n_temp)
    print(f"[TEMPLATE] KMeans cluster sizes: "
          f"{', '.join(str(c) for c in sorted(counts, reverse=True))}")

    n_pcs = min(n_temp, len(clips), clips.shape[1])
    svd = TruncatedSVD(n_components=n_pcs, random_state=config.RANDOM_SEED)
    svd.fit(clips)
    wPCA = svd.components_.astype(np.float32)
    explained = svd.explained_variance_ratio_
    print(f"[TEMPLATE] PCA explained variance: "
          f"{sum(explained)*100:.1f}% ({', '.join(f'{v*100:.1f}%' for v in explained[:6])})")

    print(f"[TEMPLATE] Extracted {n_temp} templates and {n_pcs} PCs from data")
    return wTEMP, wPCA


# ==============================================================
# Template Matching Engine
# ==============================================================

class TemplateMatchingDetector:
    """Single-channel template matching on whitened data."""

    def __init__(self, wTEMP, stride=None, threshold=None,
                 nms_samples=None, precision_bits=None):
        self.stride = stride or config.MATCH_STRIDE
        self.threshold = threshold or config.MATCH_THRESHOLD
        self.nms_samples = nms_samples or config.MATCH_NMS_SAMPLES
        self.precision_bits = precision_bits or config.PRECISION_MATCH
        self.wTEMP_raw = wTEMP.copy()
        self.wTEMP = quantize_weights(wTEMP, self.precision_bits)
        self.n_templates, self.nt = self.wTEMP.shape
        self.half_nt = self.nt // 2

        err = np.abs(self.wTEMP - self.wTEMP_raw).max()
        print(f"[MATCH] {self.n_templates} templates, nt={self.nt}, "
              f"stride={self.stride}, Th={self.threshold}, "
              f"NMS={self.nms_samples}, {self.precision_bits}-bit weights")
        if self.precision_bits < 32:
            print(f"[MATCH] Template quant error: {err:.2e}")

        self._overlap = None
        self._chunk_index = 0
        self._samples_processed = 0

    def reset(self):
        self._overlap = None
        self._chunk_index = 0
        self._samples_processed = 0

    def _convolve_channel(self, signal):
        n_ext = len(signal)
        start = self.half_nt
        stop = n_ext - self.half_nt
        positions = np.arange(start, stop, self.stride)
        n_out = len(positions)
        scores = np.empty((self.n_templates, n_out), dtype=np.float32)
        for k in range(self.n_templates):
            full_conv = np.correlate(signal, self.wTEMP[k], mode='valid')
            scores[k] = full_conv[::self.stride][:n_out]
        return scores, positions

    def _detect_single_channel(self, scores, positions):
        best_template = np.argmax(np.abs(scores), axis=0)
        best_score = scores[best_template, np.arange(len(positions))]
        above = np.abs(best_score) > self.threshold
        if not above.any():
            return np.array([]), np.array([]), np.array([])

        cand_idx = np.where(above)[0]
        cand_pos = positions[cand_idx]
        cand_amp = best_score[cand_idx]
        cand_tmpl = best_template[cand_idx]

        keep = np.ones(len(cand_idx), dtype=bool)
        abs_amp = np.abs(cand_amp)
        for i in range(len(cand_idx)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(cand_idx)):
                if cand_pos[j] - cand_pos[i] > self.nms_samples:
                    break
                if abs_amp[j] > abs_amp[i]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
        return cand_pos[keep], cand_amp[keep], cand_tmpl[keep]

    def process_chunk(self, whitened_chunk):
        n_samples, n_ch = whitened_chunk.shape
        if self._overlap is not None:
            extended = np.vstack([self._overlap, whitened_chunk])
            overlap_len = self._overlap.shape[0]
        else:
            overlap_len = 0
            extended = whitened_chunk

        all_times, all_channels, all_amps, all_templates = [], [], [], []
        for ch in range(n_ch):
            scores, positions = self._convolve_channel(extended[:, ch])
            det_pos, det_amp, det_tmpl = self._detect_single_channel(
                scores, positions)
            if len(det_pos) == 0:
                continue
            new_data_start = overlap_len
            mask = det_pos >= new_data_start
            det_pos, det_amp, det_tmpl = det_pos[mask], det_amp[mask], det_tmpl[mask]
            if len(det_pos) == 0:
                continue
            global_times = self._samples_processed + (det_pos - new_data_start)
            all_times.append(global_times)
            all_channels.append(np.full(len(det_pos), ch, dtype=np.int32))
            all_amps.append(det_amp)
            all_templates.append(det_tmpl)

        self._overlap = whitened_chunk[-(self.nt - 1):].copy()
        self._samples_processed += n_samples
        self._chunk_index += 1

        if all_times:
            events = {
                'times': np.concatenate(all_times),
                'channels': np.concatenate(all_channels),
                'amplitudes': np.concatenate(all_amps),
                'templates': np.concatenate(all_templates),
            }
        else:
            events = {
                'times': np.array([], dtype=np.int64),
                'channels': np.array([], dtype=np.int32),
                'amplitudes': np.array([], dtype=np.float32),
                'templates': np.array([], dtype=np.int32),
            }
        return events

    def process_all_chunks(self, chunks):
        self.reset()
        all_events = {k: [] for k in ['times', 'channels', 'amplitudes', 'templates']}
        for i, chunk in enumerate(chunks):
            ev = self.process_chunk(chunk)
            for k in all_events:
                all_events[k].append(ev[k])
            total = len(chunks)
            milestones = {max(1, total // 4), max(1, total // 2),
                          max(1, 3 * total // 4), total}
            if (i + 1) in milestones:
                n_so_far = sum(len(v) for v in all_events['times'])
                print(f"[MATCH] {i + 1}/{total}: {n_so_far} events")
        return {k: np.concatenate(v) if v else np.array([])
                for k, v in all_events.items()}


# ==============================================================
# Score Matrix Computation (for spatial aggregation)
# ==============================================================

def compute_score_matrix(whitened_chunk, wTEMP, stride=None, precision_bits=None):
    """Compute full score tensor B(T_out, n_ch, n_templates).

    Signal path: DAC(whitened input) -> weight-quantized cross-correlation -> ADC(B)
    """
    from scipy.signal import fftconvolve

    s = stride or config.MATCH_STRIDE
    nb = precision_bits or config.PRECISION_MATCH
    if using_int_accum_quantization():
        templates, templates_vmm = _prepared_match_templates(wTEMP, nb)
    else:
        templates = quantize_weights(wTEMP, nb)
        templates_vmm = None
    n_templates, nt = templates.shape
    half = nt // 2
    n_samples, n_ch = whitened_chunk.shape

    start = half
    stop = n_samples - half
    positions = np.arange(start, stop, s)
    T_out = len(positions)

    if using_int_accum_quantization():
        B = _compute_score_matrix_int_accum_fast(
            whitened_chunk, templates, templates_vmm, s, T_out, nb)
    else:
        # DAC quantization on crossbar input
        chunk_dac = quantize_signal(whitened_chunk, config.PRECISION_DAC)

        B = np.empty((T_out, n_ch, n_templates), dtype=np.float32)
        for k in range(n_templates):
            template_flipped = templates[k, ::-1][:, np.newaxis]
            corr = fftconvolve(chunk_dac, template_flipped,
                               mode='valid', axes=0)
            B[:, :, k] = corr[::s][:T_out]

    # ADC quantization on crossbar output
    adc_name = "MATCH.adc" if using_int_accum_quantization() else None
    B = quantize_signal(B, config.PRECISION_ADC, op_name=adc_name)
    return B, positions


def _compute_score_matrix_int_accum_fast(whitened_chunk, templates,
                                         templates_vmm, stride, t_out,
                                         precision_bits):
    """Integer template matching without materializing all sliding windows.

    This computes the same integer-code dot products as ``quantized_vmm`` over
    the im2col window matrix.  It quantizes the full input once, then applies
    a strided window view and vectorized integer tensordot.
    """
    input_name = "MATCH.input"
    _record_quant_range(input_name, whitened_chunk)

    if using_uint8_input_quantization():
        value_range = _fixed_range_for(input_name)
        x_q, x_scale, x_zero_point, x_stats = \
            _quantize_uint8_codes_with_stats(
                whitened_chunk, int(config.PRECISION_DAC),
                value_range=value_range)
        x_centered = x_q.astype(np.int32) - int(x_zero_point or 0)
        x_range = (x_stats["qmin"], x_stats["qmax"])
    else:
        x_q, x_scale, x_stats = _quantize_signed_codes_with_stats(
            whitened_chunk, int(config.PRECISION_DAC))
        x_zero_point = None
        x_centered = x_q.astype(np.int32)
        x_range = (x_stats["qmin"], x_stats["qmax"])

    w_q = templates_vmm.q.astype(np.int32, copy=False)
    w_scale = templates_vmm.scale
    w_stats = templates_vmm.stats
    n_templates, nt = w_q.shape
    n_ch = whitened_chunk.shape[1]

    windows = np.lib.stride_tricks.sliding_window_view(
        x_centered, nt, axis=0)[::stride][:t_out]
    acc = np.tensordot(windows, w_q, axes=([2], [1])).astype(
        np.int32, copy=False)

    out = (acc.astype(np.float32) *
           float(x_scale or 0.0) * float(w_scale or 0.0))

    detail_name = "MATCH.output"
    _record_quant_range(detail_name, out)
    if _should_sample_quant_details(detail_name):
        if using_uint8_input_quantization():
            x_deq = dequantize_uint8_codes(x_q, x_scale, x_zero_point)
        else:
            x_deq = dequantize_signed_codes(x_q, x_scale)
        w_range = (w_stats["qmin"], w_stats["qmax"])
        quantization_report(
            input_name, whitened_chunk, x_deq,
            q=x_q, scale=x_scale, q_range=x_range,
            clip_fraction=x_stats["clip_fraction"],
            zero_point=x_zero_point,
            value_range=(x_stats.get("range_min"), x_stats.get("range_max"))
            if x_stats.get("range_min") is not None else None)
        quantization_report(
            "MATCH.weights", templates,
            dequantize_signed_codes(templates_vmm.q, w_scale),
            q=templates_vmm.q, scale=w_scale, q_range=w_range,
            clip_fraction=w_stats["clip_fraction"])
        windows = np.lib.stride_tricks.sliding_window_view(
            whitened_chunk, nt, axis=0)[::stride][:t_out]
        x_win = windows.reshape(t_out * n_ch, nt)
        ref = (x_win @ templates.T).astype(np.float32).reshape(
            t_out, n_ch, n_templates)
        quantization_report(detail_name, ref, out, accumulator=acc)
    elif _quant_diag_enabled():
        quantization_report("MATCH.accumulator", None, None,
                            accumulator=acc)

    return out.astype(np.float32)


def compute_score_matrix_chunks(whitened_chunks, wTEMP, stride=None,
                                 precision_bits=None):
    """Compute score matrices for a list of chunks and concatenate.

    Uses overlap between consecutive chunks to avoid losing detections
    at chunk boundaries.  Each chunk is extended with the last
    ``nt - 1`` samples of the previous chunk so that
    ``compute_score_matrix`` can evaluate every sample position.
    Duplicate positions in the overlap region are removed.
    """
    s = stride or config.MATCH_STRIDE
    nt = wTEMP.shape[1]
    overlap_len = nt - 1

    all_B = []
    all_pos = []
    offset = 0
    prev_tail = None

    for chunk in whitened_chunks:
        if prev_tail is not None:
            extended = np.vstack([prev_tail, chunk])
        else:
            extended = chunk

        B, pos = compute_score_matrix(extended, wTEMP, stride=s,
                                       precision_bits=precision_bits)

        if prev_tail is not None:
            # pos is relative to *extended*; shift to be relative to chunk
            # and keep only positions that fall within the current chunk
            overlap_samples = prev_tail.shape[0]
            mask = pos >= overlap_samples
            B = B[mask]
            pos = pos[mask] - overlap_samples

        all_pos.append(pos + offset)
        all_B.append(B)
        offset += chunk.shape[0]
        prev_tail = chunk[-overlap_len:].copy()

    return np.vstack(all_B), np.concatenate(all_pos)


# ==============================================================
# Extract spike waveforms
# ==============================================================

def extract_spike_waveforms(whitened_chunks, events, nt=None, max_spikes=None):
    """Extract waveform snippets around each detected spike."""
    wlen = nt or config.MATCH_NT
    half = wlen // 2
    max_spikes = max_spikes or 5000

    all_data = np.vstack(whitened_chunks)
    n_total, n_ch = all_data.shape
    n_events = len(events['times'])

    if n_events == 0:
        return np.zeros((0, wlen), dtype=np.float32), {
            k: np.array([]) for k in events}

    if n_events > max_spikes:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = np.sort(rng.choice(n_events, max_spikes, replace=False))
    else:
        idx = np.arange(n_events)

    waveforms = []
    valid_idx = []
    for i in idx:
        t = int(events['times'][i])
        ch = int(events['channels'][i])
        if t < half or t >= n_total - half - 1 or ch >= n_ch:
            continue
        snippet = all_data[t - half: t + half + 1, ch]
        if len(snippet) == wlen:
            waveforms.append(snippet)
            valid_idx.append(i)

    valid_idx = np.array(valid_idx)
    waveforms = np.array(waveforms, dtype=np.float32) if waveforms else \
        np.zeros((0, wlen), dtype=np.float32)

    meta = {k: events[k][valid_idx] if len(valid_idx) > 0 else np.array([])
            for k in events}
    return waveforms, meta
