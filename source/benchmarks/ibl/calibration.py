"""
Calibration Utilities
======================
Region mapping for cluster centers produced by official KS4 backend.
"""

import numpy as np


def map_clusters_to_regions(cluster_labels, spike_channels, centers_60d,
                            ch_to_region, n_regions, n_ch):
    """Assign each calibration cluster to a streaming region.

    Each cluster -> mode of its spikes' peak channels -> ch_to_region -> region_id.

    Returns
    -------
    region_centers : dict {region_id: (K_r, D) float32}
    region_counts  : dict {region_id: (K_r,) int64}
    cluster_to_region : dict {cluster_id: region_id}
    """
    n_clusters = centers_60d.shape[0]
    cluster_to_region = {}

    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        chs = spike_channels[mask].astype(int)
        peak_ch = np.argmax(np.bincount(chs, minlength=n_ch))
        cluster_to_region[cid] = int(ch_to_region[peak_ch])

    region_centers = {}
    region_counts = {}
    for r_id in range(n_regions):
        cids_in_region = [c for c, r in cluster_to_region.items() if r == r_id]
        if cids_in_region:
            region_centers[r_id] = centers_60d[cids_in_region].copy()
            region_counts[r_id] = np.array(
                [(cluster_labels == c).sum() for c in cids_in_region],
                dtype=np.int64)
        else:
            region_centers[r_id] = np.zeros((0, centers_60d.shape[1]),
                                             dtype=np.float32)
            region_counts[r_id] = np.array([], dtype=np.int64)

    for r_id in range(n_regions):
        k_r = region_centers[r_id].shape[0]
        n_sp = region_counts[r_id].sum() if len(region_counts[r_id]) > 0 else 0
        if k_r > 0:
            print(f"  Region {r_id}: {k_r} clusters, {n_sp} spikes")

    return region_centers, region_counts, cluster_to_region
