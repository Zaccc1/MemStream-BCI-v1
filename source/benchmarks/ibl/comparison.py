"""
KS4 Reference Comparison Module
================================
Download IBL's spike sorting results (pykilosort/iblsorter) for the same
session, then compare against our pipeline output.

Automatically restricts the reference to the same time window that was
actually processed by the pipeline.

Usage (in main.py after merge):
    from comparison import compare_with_reference
    comp = compare_with_reference(one, eid, probe, clust_results,
                                  channel_positions=positions)
"""

import numpy as np
import re
import shutil
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import config


try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    njit = None
    prange = range
    _HAS_NUMBA = False


def _label_index_map(ids, labels):
    ids = np.asarray(ids, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    max_id = int(max(int(ids.max()) if ids.size else -1,
                     int(labels.max()) if labels.size else -1))
    out = np.full(max_id + 1, -1, dtype=np.int32)
    for i, k in enumerate(ids):
        if k >= 0:
            out[int(k)] = int(i)
    return out


def _map_labels(labels, label_to_index):
    labels = np.asarray(labels, dtype=np.int32)
    mapped = np.full(labels.shape, -1, dtype=np.int32)
    valid = (labels >= 0) & (labels < len(label_to_index))
    mapped[valid] = label_to_index[labels[valid]]
    return mapped


def _pack_cluster_trains(ids, trains, ch_trains):
    starts = np.zeros(len(ids) + 1, dtype=np.int64)
    total = 0
    for i, k in enumerate(ids):
        total += len(trains[k])
        starts[i + 1] = total
    times = np.empty(total, dtype=np.float64)
    channels = np.empty(total, dtype=np.int32)
    offset = 0
    for k in ids:
        t = np.asarray(trains[k], dtype=np.float64)
        c = np.asarray(ch_trains[k], dtype=np.int32)
        n = len(t)
        times[offset:offset + n] = t
        channels[offset:offset + n] = c
        offset += n
    return times, channels, starts


if _HAS_NUMBA:
    @njit(cache=True)
    def _candidate_pairs_numba(pipe_times, pipe_cluster_idx, pipe_channels,
                               ref_times, ref_cluster_idx, ref_channels,
                               positions, max_dt, max_dist, use_spatial,
                               n_pipe, n_ref):
        candidate = np.zeros((n_pipe, n_ref), np.bool_)
        j_start = 0
        max_dist2 = max_dist * max_dist
        n_ref_events = ref_times.size
        n_pos = positions.shape[0]
        for i in range(pipe_times.size):
            t = pipe_times[i]
            while j_start < n_ref_events and ref_times[j_start] < t - max_dt:
                j_start += 1
            pi = pipe_cluster_idx[i]
            if pi < 0:
                continue
            pch = pipe_channels[i]
            j = j_start
            while j < n_ref_events and ref_times[j] <= t + max_dt:
                ri = ref_cluster_idx[j]
                if ri >= 0:
                    ok = True
                    if use_spatial:
                        rch = ref_channels[j]
                        if pch < 0 or rch < 0 or pch >= n_pos or rch >= n_pos:
                            ok = False
                        else:
                            dx = positions[pch, 0] - positions[rch, 0]
                            dy = positions[pch, 1] - positions[rch, 1]
                            ok = dx * dx + dy * dy <= max_dist2
                    if ok:
                        candidate[pi, ri] = True
                j += 1
        return candidate


    @njit(cache=True, parallel=True)
    def _overlap_from_candidates_numba(pipe_times, pipe_channels, pipe_starts,
                                       ref_times, ref_channels, ref_starts,
                                       candidate, positions, max_dt, max_dist,
                                       use_spatial):
        n_pipe = pipe_starts.size - 1
        n_ref = ref_starts.size - 1
        overlap = np.zeros((n_pipe, n_ref), np.int64)
        max_dist2 = max_dist * max_dist
        n_pos = positions.shape[0]
        for i in prange(n_pipe):
            p0 = pipe_starts[i]
            p1 = pipe_starts[i + 1]
            for j in range(n_ref):
                if not candidate[i, j]:
                    continue
                r0 = ref_starts[j]
                r1 = ref_starts[j + 1]
                count = 0
                j_start = r0
                for p in range(p0, p1):
                    t = pipe_times[p]
                    while j_start < r1 and ref_times[j_start] < t - max_dt:
                        j_start += 1
                    q = j_start
                    while q < r1 and ref_times[q] <= t + max_dt:
                        ok = True
                        if use_spatial:
                            pch = pipe_channels[p]
                            rch = ref_channels[q]
                            if pch < 0 or rch < 0 or pch >= n_pos or rch >= n_pos:
                                ok = False
                            else:
                                dx = positions[pch, 0] - positions[rch, 0]
                                dy = positions[pch, 1] - positions[rch, 1]
                                ok = dx * dx + dy * dy <= max_dist2
                        if ok:
                            count += 1
                            j_start = q + 1
                            break
                        q += 1
                overlap[i, j] = count
        return overlap


def _compute_sparse_overlap_numba(pipe_times, pipe_labels, pipe_channels,
                                  ref_times, ref_labels, ref_channels,
                                  pipe_ids, ref_ids, pipe_trains, pipe_ch_trains,
                                  ref_trains, ref_ch_trains, channel_positions,
                                  max_dt, max_dist):
    if not _HAS_NUMBA:
        return None, {"backend": "legacy", "reason": "numba_unavailable"}
    use_spatial = channel_positions is not None
    positions = (np.asarray(channel_positions, dtype=np.float64)
                 if use_spatial else np.zeros((1, 2), dtype=np.float64))
    pipe_map = _label_index_map(pipe_ids, pipe_labels)
    ref_map = _label_index_map(ref_ids, ref_labels)

    pipe_idx = _map_labels(pipe_labels, pipe_map)
    ref_idx = _map_labels(ref_labels, ref_map)
    pipe_keep = pipe_idx >= 0
    ref_keep = ref_idx >= 0

    pt = np.asarray(pipe_times[pipe_keep], dtype=np.float64)
    pcidx = np.asarray(pipe_idx[pipe_keep], dtype=np.int32)
    pch = np.asarray(pipe_channels[pipe_keep], dtype=np.int32)
    rt = np.asarray(ref_times[ref_keep], dtype=np.float64)
    rcidx = np.asarray(ref_idx[ref_keep], dtype=np.int32)
    rch = np.asarray(ref_channels[ref_keep], dtype=np.int32)

    po = np.argsort(pt, kind="stable")
    ro = np.argsort(rt, kind="stable")
    pt = pt[po]
    pcidx = pcidx[po]
    pch = pch[po]
    rt = rt[ro]
    rcidx = rcidx[ro]
    rch = rch[ro]

    candidate = _candidate_pairs_numba(
        pt, pcidx, pch, rt, rcidx, rch, positions, float(max_dt),
        float(max_dist), bool(use_spatial), int(len(pipe_ids)), int(len(ref_ids)))
    candidate_pairs = int(np.count_nonzero(candidate))

    packed_pipe_times, packed_pipe_ch, pipe_starts = _pack_cluster_trains(
        pipe_ids, pipe_trains, pipe_ch_trains)
    packed_ref_times, packed_ref_ch, ref_starts = _pack_cluster_trains(
        ref_ids, ref_trains, ref_ch_trains)
    overlap = _overlap_from_candidates_numba(
        packed_pipe_times, packed_pipe_ch, pipe_starts,
        packed_ref_times, packed_ref_ch, ref_starts,
        candidate, positions, float(max_dt), float(max_dist), bool(use_spatial))
    return overlap, {
        "backend": "sparse_numba",
        "candidate_pairs": candidate_pairs,
        "total_pairs": int(len(pipe_ids) * len(ref_ids)),
    }



# ==============================================================
# Load IBL reference spike sorting
# ==============================================================

_UUID_PART_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MISSING_FILE_RE = re.compile(
    r"No such file or directory: ['\"]([^'\"]+)['\"]")


def _repair_one_uuid_cache_miss(exc):
    """Create a UUID-suffixed ALF cache alias from the base .npy/.pqt file."""
    match = _MISSING_FILE_RE.search(str(exc))
    if not match:
        return False

    raw_path = match.group(1)
    missing = Path(raw_path)
    if not missing.exists() and (chr(92) * 2) in raw_path:
        missing = Path(raw_path.replace(chr(92) * 2, chr(92)))
    if missing.exists():
        return False

    parts = missing.name.split(".")
    if len(parts) < 4 or not _UUID_PART_RE.match(parts[-2]):
        return False

    base_name = ".".join(parts[:-2] + parts[-1:])
    base_path = missing.with_name(base_name)
    if not base_path.exists():
        return False

    try:
        missing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, missing)
        print(f"[REF] Repaired ONE cache alias: {missing.name} <- {base_path.name}")
        return True
    except OSError as copy_error:
        print(f"[REF] Could not repair ONE cache alias {missing}: {copy_error}")
        return False


def _load_spike_sorting_with_cache_repair(ssl, spike_sorter, max_repairs=20):
    """Load sorting, retrying after repairing UUID-suffixed local cache aliases."""
    last_exc = None
    for _ in range(max_repairs + 1):
        try:
            return ssl.load_spike_sorting(spike_sorter=spike_sorter)
        except Exception as exc:
            last_exc = exc
            if not _repair_one_uuid_cache_miss(exc):
                raise
    raise last_exc

def load_reference(one, eid, pname, spike_sorter='iblsorter', raw_ind=None):
    """
    Download IBL spike sorting results via ONE API.

    Parameters
    ----------
    one : ONE instance
    eid : str — session EID
    pname : str — probe name (e.g. 'probe00')
    spike_sorter : str — 'iblsorter' or 'pykilosort'
    raw_ind : (n_ch,) int array or None — sorted→raw channel mapping

    Returns
    -------
    ref : dict with keys:
        'spike_times'    : (N,) float64 — spike times in seconds
        'spike_clusters' : (N,) int32 — cluster IDs
        'spike_channels' : (N,) int32 — peak channel per spike (raw index)
        'cluster_channels': (K,) int32 — peak channel per cluster (raw index)
        'cluster_labels' : (K,) — quality labels (1=good, else=noise/mua)
        'n_clusters'     : int
        'n_good'         : int
        'sorter'         : str
    """
    from brainbox.io.one import SpikeSortingLoader

    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one)

    # Try iblsorter first, fall back to pykilosort
    try:
        spikes, clusters, channels = _load_spike_sorting_with_cache_repair(
            ssl, spike_sorter=spike_sorter)
        sorter = spike_sorter
    except Exception:
        try:
            spikes, clusters, channels = _load_spike_sorting_with_cache_repair(
                ssl, spike_sorter='pykilosort')
            sorter = 'pykilosort'
        except Exception as e:
            print(f"[REF] Failed to load reference sorting: {e}")
            return None

    if getattr(config, 'DEBUG_VERBOSE', False):
        cluster_attrs = [a for a in dir(clusters) if not a.startswith('_')
                         and not callable(getattr(clusters, a, None))]
        print(f"[REF] clusters attributes: {cluster_attrs}")
        for attr in cluster_attrs:
            val = getattr(clusters, attr, None)
            if val is None:
                continue
            try:
                if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                    print(f"[REF]   clusters.{attr}: shape={val.shape}, "
                          f"dtype={val.dtype}, sample={val[:3] if len(val) > 0 else 'empty'}")
                elif hasattr(val, 'columns'):
                    print(f"[REF]   clusters.{attr}: DataFrame, "
                          f"shape={val.shape}, columns={list(val.columns)}")
                elif hasattr(val, '__len__'):
                    print(f"[REF]   clusters.{attr}: len={len(val)}, "
                          f"type={type(val).__name__}")
            except Exception as e:
                print(f"[REF]   clusters.{attr}: (error reading: {e})")

    spike_times = spikes.times
    spike_clusters = spikes.clusters.astype(np.int32)

    # Cluster info
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)

    # Peak channels per cluster
    cluster_channels = np.zeros(cluster_ids.max() + 1, dtype=np.int32)
    if hasattr(clusters, 'channels'):
        for k in cluster_ids:
            if k < len(clusters.channels):
                cluster_channels[k] = int(clusters.channels[k])
            else:
                mask = spike_clusters == k
                if mask.sum() > 0:
                    cluster_channels[k] = 0
    else:
        # Estimate from most common channel
        if hasattr(spikes, 'channels'):
            for k in cluster_ids:
                mask = spike_clusters == k
                if mask.sum() > 0:
                    chs = spikes.channels[mask].astype(int)
                    cluster_channels[k] = int(np.bincount(chs).argmax())

    # Quality labels — handle multiple IBL formats
    cluster_labels = np.zeros(cluster_ids.max() + 1, dtype=np.int32)
    label_source = None

    # Try different attribute names IBL uses
    raw_labels = None
    for attr in ['label', 'labels', '_phy_annotation', 'group']:
        if hasattr(clusters, attr):
            candidate = getattr(clusters, attr)
            if candidate is not None and hasattr(candidate, '__len__') and len(candidate) > 0:
                raw_labels = candidate
                label_source = attr
                break

    if raw_labels is not None:
        n_nan = 0
        sample_vals = []
        for k in cluster_ids[:10]:
            if k < len(raw_labels):
                sample_vals.append(repr(raw_labels[k]))

        for k in cluster_ids:
            if k < len(raw_labels):
                val = raw_labels[k]
                # Handle NaN
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    n_nan += 1
                    continue
                # Handle string labels
                if isinstance(val, (str, np.str_)):
                    val_lower = str(val).strip().lower()
                    if val_lower == 'good':
                        cluster_labels[k] = 1
                    elif val_lower == 'mua':
                        cluster_labels[k] = 2
                    elif val_lower == 'noise':
                        cluster_labels[k] = 3
                    else:
                        cluster_labels[k] = 0
                else:
                    # IBL float convention: 1.0=good, 0.667=mua,
                    # 0.333/0.0=noise
                    fval = float(val)
                    if fval > 0.8:
                        cluster_labels[k] = 1  # good
                    elif fval > 0.5:
                        cluster_labels[k] = 2  # mua
                    else:
                        cluster_labels[k] = 0  # noise

        n_by_label = {}
        for k in cluster_ids:
            v = int(cluster_labels[k])
            n_by_label[v] = n_by_label.get(v, 0) + 1
        label_names = {0: 'noise', 1: 'good', 2: 'mua'}
        named_dist = {label_names.get(k, f'?{k}'): v
                      for k, v in sorted(n_by_label.items())}
        print(f"[REF] Cluster labels from '{label_source}': {named_dist}")
        print(f"[REF]   Sample values: {sample_vals}, NaN count: {n_nan}")
    else:
        avail = [a for a in dir(clusters) if not a.startswith('_')
                 and not callable(getattr(clusters, a, None))]
        print(f"[REF] WARNING: No cluster labels from SpikeSortingLoader.")
        print(f"[REF]   Available cluster attributes: {avail}")

        # Try extracting from metrics DataFrame
        if hasattr(clusters, 'metrics') and clusters.metrics is not None:
            metrics = clusters.metrics
            if hasattr(metrics, 'columns'):
                print(f"[REF]   metrics columns: {list(metrics.columns)}")
                for col in ['label', 'ks2_label', 'bitwise_label',
                            'quality', 'group']:
                    if col in metrics.columns:
                        raw_col = metrics[col].values
                        print(f"[REF]   Found '{col}' in metrics: "
                              f"dtype={raw_col.dtype}, "
                              f"unique={np.unique(raw_col[:50])}")
                        # Map using cluster_id column if available
                        cid_col = metrics['cluster_id'].values \
                            if 'cluster_id' in metrics.columns else None
                        for idx in range(len(raw_col)):
                            cid = int(cid_col[idx]) if cid_col is not None \
                                else idx
                            if cid > cluster_ids.max():
                                continue
                            val = raw_col[idx]
                            if isinstance(val, (float, np.floating)):
                                if np.isnan(val):
                                    continue
                                # IBL convention: 1.0=good, 0.667=mua,
                                # 0.333/0.0=noise
                                if val > 0.8:
                                    cluster_labels[cid] = 1  # good
                                elif val > 0.5:
                                    cluster_labels[cid] = 2  # mua
                                else:
                                    cluster_labels[cid] = 0  # noise
                            elif isinstance(val, (str, np.str_)):
                                vl = str(val).strip().lower()
                                if vl == 'good':
                                    cluster_labels[cid] = 1
                                elif vl == 'mua':
                                    cluster_labels[cid] = 2
                                else:
                                    cluster_labels[cid] = 0
                            else:
                                cluster_labels[cid] = int(val)
                        n_by = {}
                        for k in cluster_ids:
                            v = int(cluster_labels[k])
                            n_by[v] = n_by.get(v, 0) + 1
                        label_names = {0: 'noise', 1: 'good', 2: 'mua'}
                        named_dist = {label_names.get(k, f'?{k}'): v
                                      for k, v in sorted(n_by.items())}
                        print(f"[REF]   Label distribution from "
                              f"metrics.{col}: {named_dist}")
                        label_source = f'metrics.{col}'
                        break

        # Last resort: try loading directly via ONE API
        if label_source is None:
            for collection in [f'alf/{pname}/pykilosort',
                               f'alf/{pname}/iblsorter',
                               f'alf/{pname}']:
                try:
                    label_data = one.load_dataset(
                        eid, 'clusters.label.npy',
                        collection=collection)
                    if label_data is not None and len(label_data) > 0:
                        print(f"[REF] Found clusters.label via ONE "
                              f"in '{collection}': "
                              f"shape={label_data.shape}, "
                              f"dtype={label_data.dtype}, "
                              f"sample={label_data[:5]}")
                        for k in cluster_ids:
                            if k < len(label_data):
                                val = label_data[k]
                                if isinstance(val, (float, np.floating)):
                                    if np.isnan(val):
                                        continue
                                    if val > 0.8:
                                        cluster_labels[k] = 1  # good
                                    elif val > 0.5:
                                        cluster_labels[k] = 2  # mua
                                    else:
                                        cluster_labels[k] = 0  # noise
                                elif isinstance(val, (str, np.str_)):
                                    vl = str(val).strip().lower()
                                    if vl == 'good':
                                        cluster_labels[k] = 1
                                    elif vl == 'mua':
                                        cluster_labels[k] = 2
                                    else:
                                        cluster_labels[k] = 0
                                else:
                                    cluster_labels[k] = int(val)
                        n_by = {}
                        for k in cluster_ids:
                            v = int(cluster_labels[k])
                            n_by[v] = n_by.get(v, 0) + 1
                        label_names = {0: 'noise', 1: 'good', 2: 'mua'}
                        named_dist = {label_names.get(k2, f'?{k2}'): v2
                                      for k2, v2 in sorted(n_by.items())}
                        print(f"[REF]   Label distribution: {named_dist}")
                        label_source = f'ONE:{collection}'
                        break
                except Exception as e:
                    print(f"[REF]   clusters.label not in "
                          f"'{collection}': {e}")

    # Convert cluster channels from sorted-index to raw-index
    if raw_ind is not None:
        n_sorted = len(raw_ind)
        converted = 0
        for k in cluster_ids:
            ch = cluster_channels[k]
            if ch < n_sorted:
                cluster_channels[k] = raw_ind[ch]
                converted += 1
        print(f"[REF] Converted {converted} cluster channels "
              f"sorted→raw using rawInd ({n_sorted} entries)")

    # Spike channels (from cluster -> channel mapping)
    spike_channels = cluster_channels[spike_clusters]

    # Cluster amplitudes (from metrics if available)
    cluster_amplitudes = None
    if hasattr(clusters, 'metrics') and clusters.metrics is not None:
        metrics = clusters.metrics
        if hasattr(metrics, 'columns') and 'amp_median' in metrics.columns:
            cluster_amplitudes = np.zeros(cluster_ids.max() + 1, dtype=np.float32)
            if 'cluster_id' in metrics.columns:
                # Use cluster_id column as key
                for _, row in metrics.iterrows():
                    cid = int(row['cluster_id'])
                    if cid <= cluster_ids.max():
                        cluster_amplitudes[cid] = float(row['amp_median'])
            else:
                # Fallback: assume index IS cluster_id
                for k in cluster_ids:
                    if k < len(metrics):
                        cluster_amplitudes[k] = float(
                            metrics['amp_median'].iloc[k])
            print(f"[REF] Loaded amp_median for {len(cluster_ids)} clusters, "
                  f"range=[{cluster_amplitudes[cluster_ids].min():.4f}, "
                  f"{cluster_amplitudes[cluster_ids].max():.4f}]")

    n_good = int((cluster_labels[cluster_ids] == 1).sum())
    n_mua = int((cluster_labels[cluster_ids] == 2).sum())
    n_noise = len(cluster_ids) - n_good - n_mua

    print(f"[REF] Loaded {sorter}: {len(spike_times)} spikes, "
          f"{n_clusters} clusters ({n_good} good, {n_mua} mua, {n_noise} noise)")

    return {
        'spike_times': spike_times,
        'spike_clusters': spike_clusters,
        'spike_channels': spike_channels,
        'cluster_channels': cluster_channels,
        'cluster_labels': cluster_labels,
        'cluster_amplitudes': cluster_amplitudes,
        'cluster_ids': cluster_ids,
        'n_clusters': n_clusters,
        'n_good': n_good,
        'sorter': sorter,
    }


# ==============================================================
# Filter reference to pipeline time window
# ==============================================================

def _get_pipeline_time_window(clust_results):
    """
    Infer the time window (in seconds) actually processed by the pipeline.

    Note: clust_results['times'] are already corrected for FIR group delay
    (subtracted in SpikeClustering.get_results()).

    Returns
    -------
    t_start, t_end : float — in seconds (absolute)
    """
    times = clust_results['times']
    if len(times) == 0:
        return 0.0, 0.0
    t_start_s = config.CALIBRATION_DURATION_S
    t_min_sample = float(times.min())
    t_max_sample = float(times.max())
    t_start = t_start_s + t_min_sample / config.SAMPLE_RATE
    t_end = t_start_s + t_max_sample / config.SAMPLE_RATE
    return t_start, t_end


def filter_reference(ref, t_start, t_end):
    """
    Restrict reference spikes to [t_start, t_end] seconds.

    Returns
    -------
    filtered : dict — same structure, restricted
    """
    mask = (ref['spike_times'] >= t_start) & (ref['spike_times'] <= t_end)
    st = ref['spike_times'][mask]
    sc = ref['spike_clusters'][mask]
    sch = ref['spike_channels'][mask]

    # Recount clusters
    active_ids = np.unique(sc)
    n_good = 0
    n_mua = 0
    for k in active_ids:
        if k < len(ref['cluster_labels']):
            if ref['cluster_labels'][k] == 1:
                n_good += 1
            elif ref['cluster_labels'][k] == 2:
                n_mua += 1

    n_noise = len(active_ids) - n_good - n_mua
    print(f"[REF] Filtered to [{t_start:.2f}, {t_end:.2f}] s: "
          f"{len(st)} spikes, {len(active_ids)} clusters "
          f"({n_good} good, {n_mua} mua, {n_noise} noise)")

    return {
        'spike_times': st,
        'spike_clusters': sc,
        'spike_channels': sch,
        'cluster_channels': ref['cluster_channels'],
        'cluster_labels': ref['cluster_labels'],
        'cluster_amplitudes': ref.get('cluster_amplitudes', None),
        'cluster_ids': active_ids,
        'n_clusters': len(active_ids),
        'n_good': n_good,
        'sorter': ref['sorter'],
        't_start': t_start,
        't_end': t_end,
    }


# ==============================================================
# Spike-level matching (Hungarian algorithm)
# ==============================================================

def match_units(pipeline_results, ref, channel_positions=None,
                max_dt_ms=0.5, max_dist_um=100.0):
    """
    Match pipeline clusters to reference clusters based on spike time overlap
    with optional spatial constraint.

    Parameters
    ----------
    pipeline_results : dict — our clust_results
    ref : dict — filtered reference
    channel_positions : (n_ch, 2) float array or None — if provided, adds
        spatial constraint: only count coincidence if channels within max_dist_um
    max_dt_ms : float — maximum time difference to count as match (ms)
    max_dist_um : float — maximum spatial distance for coincidence (μm)

    Returns
    -------
    matches : list of (pipe_id, ref_id, n_matched, accuracy, FP, FN)
    summary : dict with aggregate metrics
    """
    max_dt = max_dt_ms / 1000.0  # convert to seconds

    # Pipeline spike times in seconds (absolute)
    # Times are already FIR-corrected in get_results()
    pipe_times = pipeline_results['times'].astype(np.float64) / config.SAMPLE_RATE
    pipe_times += config.CALIBRATION_DURATION_S
    pipe_labels = pipeline_results['labels']
    pipe_channels = pipeline_results['channels']

    ref_times = ref['spike_times']
    ref_labels = ref['spike_clusters']
    ref_channels = ref['spike_channels']

    # Get active cluster IDs (exclude -1 = unassigned)
    pipe_ids = np.unique(pipe_labels)
    pipe_ids = pipe_ids[pipe_ids >= 0]
    ref_ids = ref['cluster_ids']

    # Only consider ref clusters with enough spikes
    ref_counts = {k: (ref_labels == k).sum() for k in ref_ids}
    ref_ids = np.array([k for k in ref_ids if ref_counts[k] >= 5])

    pipe_counts = {k: (pipe_labels == k).sum() for k in pipe_ids}
    pipe_ids = np.array([k for k in pipe_ids if pipe_counts[k] >= 5])

    if len(pipe_ids) == 0 or len(ref_ids) == 0:
        print("[MATCH] No clusters to match")
        return [], {'n_matched': 0}

    # Diagnostic: spike count distributions
    pipe_sizes = np.array([pipe_counts[k] for k in pipe_ids])
    ref_sizes = np.array([ref_counts[k] for k in ref_ids])
    total_pipe_spikes_all = pipe_sizes.sum()
    total_ref_spikes_all = ref_sizes.sum()

    spatial_str = f", dist < {max_dist_um} μm" if channel_positions is not None else ""
    print(f"[MATCH] Matching {len(pipe_ids)} pipeline clusters vs "
          f"{len(ref_ids)} reference clusters (dt < {max_dt_ms} ms{spatial_str})")
    print(f"[MATCH]   Pipeline:  {total_pipe_spikes_all} total spikes, "
          f"median={np.median(pipe_sizes):.0f}, "
          f"mean={np.mean(pipe_sizes):.0f}, "
          f"range=[{pipe_sizes.min()}, {pipe_sizes.max()}] per cluster")
    print(f"[MATCH]   Reference: {total_ref_spikes_all} total spikes, "
          f"median={np.median(ref_sizes):.0f}, "
          f"mean={np.mean(ref_sizes):.0f}, "
          f"range=[{ref_sizes.min()}, {ref_sizes.max()}] per cluster")
    print(f"[MATCH]   Pipeline time range: "
          f"[{pipe_times.min():.3f}, {pipe_times.max():.3f}] s")
    print(f"[MATCH]   Reference time range: "
          f"[{ref_times.min():.3f}, {ref_times.max():.3f}] s")

    # Build spike trains + channels per cluster
    pipe_trains = {}
    pipe_ch_trains = {}
    for k in pipe_ids:
        mask = pipe_labels == k
        order = np.argsort(pipe_times[mask])
        pipe_trains[k] = pipe_times[mask][order]
        pipe_ch_trains[k] = pipe_channels[mask][order]

    ref_trains = {}
    ref_ch_trains = {}
    for k in ref_ids:
        mask = ref_labels == k
        order = np.argsort(ref_times[mask])
        ref_trains[k] = ref_times[mask][order]
        ref_ch_trains[k] = ref_channels[mask][order]

    # Count coincidences.  The legacy implementation checked every
    # pipeline/ref cluster pair.  The fast path first finds cluster pairs that
    # have at least one event-level time/space candidate, then computes the
    # exact same greedy coincidence count only for those sparse pairs.
    n_pipe = len(pipe_ids)
    n_ref = len(ref_ids)
    match_backend = {"backend": "legacy"}
    overlap = None
    if getattr(config, "MATCH_UNITS_FAST_ENABLE", True):
        try:
            overlap, match_backend = _compute_sparse_overlap_numba(
                pipe_times, pipe_labels, pipe_channels,
                ref_times, ref_labels, ref_channels,
                pipe_ids, ref_ids, pipe_trains, pipe_ch_trains,
                ref_trains, ref_ch_trains, channel_positions,
                max_dt, max_dist_um)
        except Exception as e:
            overlap = None
            match_backend = {"backend": "legacy", "reason": str(e)}

    if overlap is None:
        overlap = np.zeros((n_pipe, n_ref), dtype=np.int64)
        for i, pk in enumerate(pipe_ids):
            pt = pipe_trains[pk]
            pc = pipe_ch_trains[pk]
            for j, rk in enumerate(ref_ids):
                rt = ref_trains[rk]
                rc = ref_ch_trains[rk]
                overlap[i, j] = _count_coincidences(
                    pt, rt, max_dt,
                    ch1=pc, ch2=rc,
                    positions=channel_positions,
                    max_dist=max_dist_um)
    print(f"[MATCH]   Backend: {match_backend.get('backend', 'legacy')}"
          f", candidate_pairs={match_backend.get('candidate_pairs', n_pipe * n_ref)}"
          f"/{n_pipe * n_ref}")

    # Hungarian matching (maximize overlap -> minimize negative)
    cost = -overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    total_TP = 0
    total_pipe_spikes = 0
    total_ref_spikes = 0

    for i, j in zip(row_ind, col_ind):
        pk = pipe_ids[i]
        rk = ref_ids[j]
        n_matched = overlap[i, j]
        n_pipe_k = pipe_counts[pk]
        n_ref_k = ref_counts[rk]

        if n_matched < 2:
            continue

        # Standard spike sorting metrics (SpikeInterface / SpikeForest convention)
        # TP = matched, FP = n_pipe - TP, FN = n_ref - TP
        # Accuracy (Jaccard) = TP / (TP + FP + FN) = TP / (n_pipe + n_ref - TP)
        TP = n_matched
        FP = n_pipe_k - TP
        FN = n_ref_k - TP
        accuracy = TP / max(n_pipe_k + n_ref_k - TP, 1)
        precision = TP / max(n_pipe_k, 1)   # 1 - FP_rate
        recall = TP / max(n_ref_k, 1)       # 1 - FN_rate

        is_ref_good = (rk < len(ref['cluster_labels'])
                       and ref['cluster_labels'][rk] == 1)

        matches.append({
            'pipe_id': int(pk),
            'ref_id': int(rk),
            'n_matched': int(n_matched),
            'n_pipe': int(n_pipe_k),
            'n_ref': int(n_ref_k),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'FP': int(FP),
            'FN': int(FN),
            'ref_is_good': is_ref_good,
        })
        total_TP += n_matched
        total_pipe_spikes += n_pipe_k
        total_ref_spikes += n_ref_k

    # Sort by n_matched (most overlap first) for diagnostics
    matches.sort(key=lambda m: -m['n_matched'])

    # Print top 10 pairs for diagnostics
    print(f"\n[MATCH] Top 10 pairs by overlap:")
    print(f"  {'pipe':>5s} {'ref':>5s}  {'matched':>7s} {'n_pipe':>6s} "
          f"{'n_ref':>5s}  {'prec':>5s} {'rec':>5s} {'acc':>5s}")
    for m in matches[:10]:
        print(f"  {m['pipe_id']:5d} {m['ref_id']:5d}  {m['n_matched']:7d} "
              f"{m['n_pipe']:6d} {m['n_ref']:5d}  "
              f"{m['precision']:.3f} {m['recall']:.3f} {m['accuracy']:.3f}")

    # Re-sort by accuracy for output
    matches.sort(key=lambda m: -m['accuracy'])

    # Aggregate
    accuracies = [m['accuracy'] for m in matches]
    good_matches = [m for m in matches if m['ref_is_good']]
    good_accuracies = [m['accuracy'] for m in good_matches]

    # Global recall: total TP / total ref spikes (includes unmatched ref units)
    # This is the key metric: what fraction of all GT spikes did we recover?
    global_recall = total_TP / max(total_ref_spikes_all, 1)

    # Good-unit recall: restrict to ref units labelled "good"
    good_TP = sum(m['n_matched'] for m in good_matches)
    good_ref_total = sum(ref_counts[k] for k in ref_ids
                         if k < len(ref['cluster_labels'])
                         and ref['cluster_labels'][k] == 1)
    good_global_recall = good_TP / max(good_ref_total, 1)
    good_pipe_total = sum(m['n_pipe'] for m in good_matches)
    good_global_precision = good_TP / max(good_pipe_total, 1)

    print(f"\n[MATCH] Global recall: {total_TP}/{total_ref_spikes_all} "
          f"= {global_recall:.4f}")
    print(f"[MATCH] Good-unit recall: {good_TP}/{good_ref_total} "
          f"= {good_global_recall:.4f}")
    print(f"[MATCH] Good-unit precision: {good_TP}/{good_pipe_total} "
          f"= {good_global_precision:.4f}")

    summary = {
        'n_matched_pairs': len(matches),
        'n_pipe_clusters': len(pipe_ids),
        'n_ref_clusters': len(ref_ids),
        'n_ref_good': ref['n_good'],
        'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0,
        'median_accuracy': float(np.median(accuracies)) if accuracies else 0,
        'mean_accuracy_good': float(np.mean(good_accuracies))
            if good_accuracies else 0,
        'n_good_matched': len(good_matches),
        'n_high_accuracy': sum(1 for a in accuracies if a > 0.8),
        'total_TP': total_TP,
        'total_pipe_spikes': int(total_pipe_spikes_all),
        'total_ref_spikes': int(total_ref_spikes_all),
        'global_recall': float(global_recall),
        'good_global_recall': float(good_global_recall),
        'good_global_precision': float(good_global_precision),
        'good_TP': int(good_TP),
        'good_ref_total': int(good_ref_total),
        'good_pipe_total': int(good_pipe_total),
        'mean_precision': float(np.mean([m['precision'] for m in matches]))
            if matches else 0,
        'mean_precision_good': float(np.mean([m['precision'] for m in good_matches]))
            if good_matches else 0,
        'mean_recall': float(np.mean([m['recall'] for m in matches]))
            if matches else 0,
        'best_overlap': matches[0]['n_matched'] if matches else 0,
        'match_backend': match_backend.get('backend', 'legacy'),
        'candidate_pairs': int(match_backend.get('candidate_pairs', n_pipe * n_ref)),
        'total_pairs': int(n_pipe * n_ref),
    }

    return matches, summary


def _count_coincidences(st1, st2, max_dt, ch1=None, ch2=None,
                        positions=None, max_dist=100.0):
    """Count spike coincidences between two sorted spike trains.

    Each spike in st1 (pipeline) and st2 (reference) is matched at most once
    (greedy, nearest-first). If ch1/ch2/positions provided, also requires
    spatial proximity < max_dist μm.

    Returns TP count.
    """
    use_spatial = (ch1 is not None and ch2 is not None
                   and positions is not None)
    count = 0
    j_start = 0
    for i in range(len(st1)):
        while j_start < len(st2) and st2[j_start] < st1[i] - max_dt:
            j_start += 1
        j = j_start
        while j < len(st2) and st2[j] <= st1[i] + max_dt:
            if use_spatial:
                d = np.sqrt(((positions[int(ch1[i])]
                              - positions[int(ch2[j])]) ** 2).sum())
                if d > max_dist:
                    j += 1
                    continue
            count += 1
            j_start = j + 1  # consume this ref spike
            break
    return count


# ==============================================================
# Channel-level firing rate comparison
# ==============================================================

def compare_firing_rates(pipeline_results, ref, channel_positions):
    """
    Compare per-channel firing rates between pipeline and reference.

    Returns
    -------
    rate_comparison : dict with per-channel rates and correlation
    """
    n_ch = channel_positions.shape[0]
    t_start, t_end = _get_pipeline_time_window(pipeline_results)
    duration = max(t_end - t_start, 0.001)

    # Pipeline rates.  Some filtered comparisons keep the original arrays but
    # mark rejected spikes with label=-1, so the rate diagnostic must honor
    # labels when they are present.
    pipe_ch = pipeline_results['channels']
    pipe_labels = pipeline_results.get('labels', None)
    if pipe_labels is not None and len(pipe_labels) == len(pipe_ch):
        pipe_mask = pipe_labels >= 0
        pipe_ch = pipe_ch[pipe_mask]
    pipe_rates = np.bincount(pipe_ch.astype(int), minlength=n_ch) / duration

    # Reference rates
    ref_ch = ref['spike_channels']
    ref_rates = np.bincount(ref_ch.astype(int), minlength=n_ch) / duration

    # Correlation
    valid = (pipe_rates > 0) | (ref_rates > 0)
    if valid.sum() > 2:
        corr = float(np.corrcoef(pipe_rates[valid], ref_rates[valid])[0, 1])
    else:
        corr = 0.0

    ratio = pipe_rates.sum() / max(ref_rates.sum(), 1e-6)

    print(f"[RATES] Pipeline total: {pipe_rates.sum():.1f} Hz, "
          f"Reference total: {ref_rates.sum():.1f} Hz "
          f"(ratio={ratio:.2f})")
    print(f"[RATES] Per-channel correlation: {corr:.3f}")

    return {
        'pipe_rates': pipe_rates,
        'ref_rates': ref_rates,
        'correlation': corr,
        'ratio': ratio,
        'duration': duration,
        'n_pipe_spikes': int(len(pipe_ch)),
        'n_ref_spikes': int(len(ref_ch)),
    }


def matched_pair_rate_comparison(matches, duration):
    """
    Compare firing rates only across matched pipeline/reference unit pairs.

    This is the right diagnostic for validation-filtered units: each selected
    pipeline cluster is compared only with its assigned reference cluster,
    instead of comparing the selected subset to the whole reference population.
    """
    duration = max(float(duration), 0.001)
    if not matches:
        return {
            'n_matched_pairs': 0,
            'duration_s': duration,
            'matched_pair_rate_correlation': 0.0,
            'matched_pair_count_correlation': 0.0,
            'pipe_ref_rate_ratio_over_matched_pairs': 0.0,
            'matched_ref_recall_over_matched_pairs': 0.0,
            'median_pair_rate_ratio': 0.0,
            'mean_pair_rate_ratio': 0.0,
            'min_pair_rate_ratio': 0.0,
            'max_pair_rate_ratio': 0.0,
        }

    n_pipe = np.asarray([m['n_pipe'] for m in matches], dtype=np.float64)
    n_ref = np.asarray([m['n_ref'] for m in matches], dtype=np.float64)
    n_matched = np.asarray([m['n_matched'] for m in matches], dtype=np.float64)
    pipe_rates = n_pipe / duration
    ref_rates = n_ref / duration

    if len(matches) > 1 and np.std(pipe_rates) > 0 and np.std(ref_rates) > 0:
        rate_corr = float(np.corrcoef(pipe_rates, ref_rates)[0, 1])
        count_corr = float(np.corrcoef(n_pipe, n_ref)[0, 1])
    else:
        rate_corr = 0.0
        count_corr = 0.0

    ratios = pipe_rates / np.maximum(ref_rates, 1e-12)
    return {
        'n_matched_pairs': int(len(matches)),
        'duration_s': duration,
        'matched_pair_rate_correlation': rate_corr,
        'matched_pair_count_correlation': count_corr,
        'sum_pipe_spikes_in_matched_pairs': int(n_pipe.sum()),
        'sum_ref_spikes_in_matched_pairs': int(n_ref.sum()),
        'sum_matched_spikes': int(n_matched.sum()),
        'pipe_ref_rate_ratio_over_matched_pairs':
            float(n_pipe.sum() / max(n_ref.sum(), 1.0)),
        'matched_ref_recall_over_matched_pairs':
            float(n_matched.sum() / max(n_ref.sum(), 1.0)),
        'median_pair_rate_ratio': float(np.median(ratios)),
        'mean_pair_rate_ratio': float(np.mean(ratios)),
        'min_pair_rate_ratio': float(np.min(ratios)),
        'max_pair_rate_ratio': float(np.max(ratios)),
        'pipe_ids': [int(m['pipe_id']) for m in matches],
        'ref_ids': [int(m['ref_id']) for m in matches],
    }


def save_matched_pair_rate_comparison(matches, duration, output_dir):
    """Save matched-pair rate diagnostics as JSON, CSV, and NumPy table."""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    summary = matched_pair_rate_comparison(matches, duration)

    dtype = [
        ('pipe_id', 'i4'), ('ref_id', 'i4'),
        ('n_pipe', 'i4'), ('n_ref', 'i4'), ('n_matched', 'i4'),
        ('pipe_rate_hz', 'f8'), ('ref_rate_hz', 'f8'),
        ('pipe_ref_rate_ratio', 'f8'),
        ('accuracy', 'f4'), ('precision', 'f4'), ('recall', 'f4'),
        ('ref_is_good', '?'),
    ]
    arr = np.empty(len(matches), dtype=dtype)
    duration = max(float(duration), 0.001)
    for i, m in enumerate(matches):
        pipe_rate = float(m['n_pipe']) / duration
        ref_rate = float(m['n_ref']) / duration
        arr[i] = (
            int(m['pipe_id']), int(m['ref_id']),
            int(m['n_pipe']), int(m['n_ref']), int(m['n_matched']),
            pipe_rate, ref_rate, pipe_rate / max(ref_rate, 1e-12),
            float(m['accuracy']), float(m['precision']), float(m['recall']),
            bool(m['ref_is_good']),
        )

    np.save(os.path.join(output_dir, 'matched_pair_rates.npy'), arr)
    with open(os.path.join(output_dir, 'matched_pair_rates.csv'), 'w') as f:
        f.write(','.join(arr.dtype.names) + '\n')
        for row in arr:
            f.write(','.join(str(row[name].item()) for name in arr.dtype.names)
                    + '\n')
    with open(os.path.join(output_dir, 'matched_pair_rate_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] Matched-pair rate comparison -> {output_dir}")
    return summary


# ==============================================================
# Main comparison entry point
# ==============================================================

def compare_with_reference(one, eid, pname, clust_results,
                           channel_positions=None, raw_ind=None,
                           verbose=True):
    """
    Full comparison: load reference, filter to time window, match units,
    compare firing rates.

    Parameters
    ----------
    one : ONE instance
    eid : str
    pname : str — probe name
    clust_results : dict — pipeline output (post-merge)
    channel_positions : (n_ch, 2) or None
    raw_ind : (n_ch,) int array or None — sorted→raw channel mapping

    Returns
    -------
    comp : dict with all comparison results, or None if reference unavailable
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"  REFERENCE COMPARISON")
        print(f"{'='*50}")

    # 1. Load reference
    ref = load_reference(one, eid, pname, raw_ind=raw_ind)
    if ref is None:
        return None

    # 2. Filter to pipeline time window
    t_start, t_end = _get_pipeline_time_window(clust_results)
    ref_filtered = filter_reference(ref, t_start, t_end)

    pipe_labels = clust_results['labels']
    pipe_assigned_count = int((pipe_labels >= 0).sum())
    if getattr(config, 'DEBUG_VERBOSE', False):
        valid = pipe_labels >= 0
        counts = np.bincount(pipe_labels[valid]) if valid.any() else np.array([])
        print("[DEBUG COMPARE_INPUT]")
        print("  total    =", len(pipe_labels))
        print("  assigned =", int(valid.sum()))
        print("  rejected =", int((pipe_labels < 0).sum()))
        print("  max_id   =", int(pipe_labels.max()) if len(pipe_labels) else -1)
        print("  max_size =", int(counts.max()) if len(counts) else 0)

    # 3. Unit matching (with spatial constraint if positions available)
    matches, summary = match_units(clust_results, ref_filtered,
                                   channel_positions=channel_positions)

    # 4. Firing rate comparison
    rate_comp = None
    if channel_positions is not None:
        rate_comp = compare_firing_rates(
            clust_results, ref_filtered, channel_positions)

    # 5. Print summary
    if verbose:
        print(f"\n  — Comparison Summary —")
        print(f"    Reference sorter: {ref['sorter']}")
        print(f"    Time window:      [{t_start:.2f}, {t_end:.2f}] s "
              f"({t_end - t_start:.1f} s)")
        print(f"    Pipeline:         {summary['n_pipe_clusters']} clusters, "
              f"{pipe_assigned_count} assigned spikes")
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
        print(f"    High accuracy:    {summary['n_high_accuracy']} "
              f"pairs > 0.8")
        if rate_comp:
            print(f"    Rate correlation:  {rate_comp['correlation']:.3f}")
            print(f"    Rate ratio:        {rate_comp['ratio']:.2f} "
                  f"(pipe/ref)")

        # Top matches
        if matches:
            print(f"\n    Top matched units:")
            for m in matches[:10]:
                good_str = " [GOOD]" if m['ref_is_good'] else ""
                print(f"      pipe={m['pipe_id']:3d} <-> "
                      f"ref={m['ref_id']:3d}{good_str}: "
                      f"acc={m['accuracy']:.3f}, "
                      f"matched={m['n_matched']}/{m['n_ref']} "
                      f"(prec={m['precision']:.2f}, rec={m['recall']:.2f})")

        # All good units detail
        good_matches = [m for m in matches if m['ref_is_good']]
        # Find unmatched good ref units
        matched_ref_ids = set(m['ref_id'] for m in matches)
        all_good_ref = [k for k in ref_filtered['cluster_ids']
                        if k < len(ref_filtered['cluster_labels'])
                        and ref_filtered['cluster_labels'][k] == 1]
        unmatched_good = [k for k in all_good_ref
                          if k not in matched_ref_ids]

        print(f"\n    === ALL GOOD UNITS ({len(all_good_ref)} total) ===")
        print(f"    {'ref':>5s} {'pipe':>5s} {'n_ref':>6s} {'matched':>7s} "
              f"{'prec':>5s} {'rec':>5s} {'acc':>5s} {'peak_ch':>7s}")

        # Sort good matches by accuracy descending
        good_matches_sorted = sorted(good_matches,
                                     key=lambda m: -m['accuracy'])
        for m in good_matches_sorted:
            rk = m['ref_id']
            ch = ref_filtered['cluster_channels'][rk] \
                if rk < len(ref_filtered['cluster_channels']) else -1
            print(f"    {rk:5d} {m['pipe_id']:5d} {m['n_ref']:6d} "
                  f"{m['n_matched']:7d} "
                  f"{m['precision']:.3f} {m['recall']:.3f} "
                  f"{m['accuracy']:.3f} {ch:7d}")

        for rk in sorted(unmatched_good):
            n_ref = int((ref_filtered['spike_clusters'] == rk).sum())
            ch = ref_filtered['cluster_channels'][rk] \
                if rk < len(ref_filtered['cluster_channels']) else -1
            print(f"    {rk:5d}     - {n_ref:6d}       -     -     - "
                  f"    - {ch:7d}  (unmatched)")

        # Summary stats for good units
        if good_matches_sorted:
            ga = [m['accuracy'] for m in good_matches_sorted]
            gr = [m['recall'] for m in good_matches_sorted]
            gp = [m['precision'] for m in good_matches_sorted]
            print(f"    ---")
            print(f"    Good units matched: {len(good_matches_sorted)}"
                  f"/{len(all_good_ref)}")
            print(f"    Accuracy:  mean={np.mean(ga):.3f}, "
                  f"median={np.median(ga):.3f}, "
                  f">0.5: {sum(1 for a in ga if a > 0.5)}, "
                  f">0.8: {sum(1 for a in ga if a > 0.8)}")
            print(f"    Recall:    mean={np.mean(gr):.3f}, "
                  f"median={np.median(gr):.3f}")
            print(f"    Precision: mean={np.mean(gp):.3f}, "
                  f"median={np.median(gp):.3f}")
        print(f"    === END GOOD UNITS (ref) ===")

        # Our pipeline's good clusters
        pipe_is_good = clust_results.get('is_good', None)
        pipe_contam = clust_results.get('contam_rate', None)
        if pipe_is_good is not None:
            pipe_good_ids = np.where(pipe_is_good)[0]
            matched_pipe_ids = {m['pipe_id']: m for m in matches}

            print(f"\n    === OUR GOOD CLUSTERS ({len(pipe_good_ids)} total) ===")
            print(f"    {'pipe':>5s} {'ref':>5s} {'ref_q':>5s} "
                  f"{'n_pipe':>6s} {'matched':>7s} "
                  f"{'prec':>5s} {'rec':>5s} {'acc':>5s} "
                  f"{'Q12':>5s}")

            pipe_good_details = []
            for pk in pipe_good_ids:
                pk = int(pk)
                q12 = float(pipe_contam[pk]) if pipe_contam is not None else -1
                if pk in matched_pipe_ids:
                    m = matched_pipe_ids[pk]
                    rk = m['ref_id']
                    ref_label = ref_filtered['cluster_labels'][rk] \
                        if rk < len(ref_filtered['cluster_labels']) else 0
                    label_map = {0: 'noise', 1: 'good', 2: 'mua'}
                    ref_q = label_map.get(int(ref_label), '?')
                    pipe_good_details.append((pk, m, ref_q, q12))
                else:
                    pipe_good_details.append((pk, None, '-', q12))

            # Sort by accuracy descending (matched first, then unmatched)
            pipe_good_details.sort(
                key=lambda x: -(x[1]['accuracy'] if x[1] else -1))

            for pk, m, ref_q, q12 in pipe_good_details:
                if m:
                    print(f"    {pk:5d} {m['ref_id']:5d} {ref_q:>5s} "
                          f"{m['n_pipe']:6d} {m['n_matched']:7d} "
                          f"{m['precision']:.3f} {m['recall']:.3f} "
                          f"{m['accuracy']:.3f} {q12:.3f}")
                else:
                    n_pipe = int((clust_results['labels'] == pk).sum())
                    print(f"    {pk:5d}     - {'-':>5s} "
                          f"{n_pipe:6d}       -     -     - "
                          f"    - {q12:.3f}  (no ref match)")

            # Summary
            matched_details = [x for x in pipe_good_details if x[1]]
            if matched_details:
                pa = [x[1]['accuracy'] for x in matched_details]
                ref_labels = [x[2] for x in matched_details]
                n_ref_good = sum(1 for r in ref_labels if r == 'good')
                n_ref_mua = sum(1 for r in ref_labels if r == 'mua')
                n_ref_noise = sum(1 for r in ref_labels if r == 'noise')
                print(f"    ---")
                print(f"    Our good matched: {len(matched_details)}"
                      f"/{len(pipe_good_ids)}")
                print(f"    Matched to ref: good={n_ref_good}, "
                      f"mua={n_ref_mua}, noise={n_ref_noise}")
                print(f"    Accuracy: mean={np.mean(pa):.3f}, "
                      f"median={np.median(pa):.3f}, "
                      f">0.5: {sum(1 for a in pa if a > 0.5)}, "
                      f">0.8: {sum(1 for a in pa if a > 0.8)}")
            print(f"    === END OUR GOOD CLUSTERS ===")

    comp = {
        'ref_full': ref,
        'ref_filtered': ref_filtered,
        'matches': matches,
        'summary': summary,
        'rate_comparison': rate_comp,
        't_start': t_start,
        't_end': t_end,
    }
    return comp


# ==============================================================
# Save comparison results
# ==============================================================

def save_comparison(comp, output_dir):
    """Save comparison results to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    summary = comp['summary']
    matches = comp['matches']

    # Save summary as text
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nt_start: {comp['t_start']:.4f}\n")
        f.write(f"t_end: {comp['t_end']:.4f}\n")
        f.write(f"sorter: {comp['ref_filtered']['sorter']}\n")

    # Save matches
    if matches:
        dtype = [('pipe_id', 'i4'), ('ref_id', 'i4'), ('n_matched', 'i4'),
                 ('n_pipe', 'i4'), ('n_ref', 'i4'),
                 ('accuracy', 'f4'), ('precision', 'f4'), ('recall', 'f4'),
                 ('ref_is_good', '?')]
        arr = np.array([(m['pipe_id'], m['ref_id'], m['n_matched'],
                         m['n_pipe'], m['n_ref'],
                         m['accuracy'], m['precision'], m['recall'],
                         m['ref_is_good']) for m in matches], dtype=dtype)
        np.save(os.path.join(output_dir, 'unit_matches.npy'), arr)

    # Save rate comparison
    if comp.get('rate_comparison'):
        rc = comp['rate_comparison']
        np.save(os.path.join(output_dir, 'pipe_firing_rates.npy'),
                rc['pipe_rates'])
        np.save(os.path.join(output_dir, 'ref_firing_rates.npy'),
                rc['ref_rates'])
        import json
        rate_summary = {
            'correlation': float(rc['correlation']),
            'ratio': float(rc['ratio']),
            'duration': float(rc['duration']),
            'n_pipe_spikes': int(rc.get('n_pipe_spikes', 0)),
            'n_ref_spikes': int(rc.get('n_ref_spikes', 0)),
        }
        with open(os.path.join(output_dir, 'rate_summary.json'), 'w') as f:
            json.dump(rate_summary, f, indent=2)

    print(f"[SAVE] Comparison results -> {output_dir}")
