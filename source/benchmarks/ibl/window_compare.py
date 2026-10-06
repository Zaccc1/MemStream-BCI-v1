import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from comparison import load_reference, filter_reference, _count_coincidences, save_comparison


def _match_units_absolute(abs_times_s, labels, channels, ref, channel_positions=None,
                          max_dt_ms=0.5, max_dist_um=100.0):
    """Match cluster labels to reference using absolute spike times in seconds.

    Parameters are analogous to comparison.match_units, except abs_times_s are
    already absolute in the session timeline.
    """
    max_dt = max_dt_ms / 1000.0

    pipe_times = np.asarray(abs_times_s, dtype=np.float64)
    pipe_labels = np.asarray(labels, dtype=np.int32)
    pipe_channels = np.asarray(channels, dtype=np.int32)

    ref_times = ref['spike_times']
    ref_labels = ref['spike_clusters']
    ref_channels = ref['spike_channels']

    pipe_ids = np.unique(pipe_labels)
    pipe_ids = pipe_ids[pipe_ids >= 0]
    ref_ids = ref['cluster_ids']

    ref_counts = {k: int((ref_labels == k).sum()) for k in ref_ids}
    ref_ids = np.array([k for k in ref_ids if ref_counts[k] >= 5], dtype=np.int32)
    pipe_counts = {k: int((pipe_labels == k).sum()) for k in pipe_ids}
    pipe_ids = np.array([k for k in pipe_ids if pipe_counts[k] >= 5], dtype=np.int32)

    if len(pipe_ids) == 0 or len(ref_ids) == 0:
        return [], {
            'n_matched_pairs': 0,
            'n_pipe_clusters': len(pipe_ids),
            'n_ref_clusters': len(ref_ids),
            'n_ref_good': ref.get('n_good', 0),
            'mean_accuracy': 0.0,
            'median_accuracy': 0.0,
            'mean_accuracy_good': 0.0,
            'n_good_matched': 0,
            'n_high_accuracy': 0,
            'total_TP': 0,
            'total_pipe_spikes': int((pipe_labels >= 0).sum()),
            'total_ref_spikes': int(len(ref_times)),
            'global_recall': 0.0,
            'good_global_recall': 0.0,
            'good_TP': 0,
            'good_ref_total': 0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'best_overlap': 0,
        }

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

    overlap = np.zeros((len(pipe_ids), len(ref_ids)), dtype=np.int64)
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
                max_dist=max_dist_um
            )

    cost = -overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    total_TP = 0
    total_pipe_spikes_all = int(sum(pipe_counts[k] for k in pipe_ids))
    total_ref_spikes_all = int(sum(ref_counts[k] for k in ref_ids))
    good_ref_total = int(sum(ref_counts[k] for k in ref_ids
                             if k < len(ref['cluster_labels']) and ref['cluster_labels'][k] == 1))
    good_TP = 0

    for i, j in zip(row_ind, col_ind):
        pk = int(pipe_ids[i])
        rk = int(ref_ids[j])
        TP = int(overlap[i, j])
        n_pipe = int(pipe_counts[pk])
        n_ref = int(ref_counts[rk])
        if TP < 2:
            continue
        FP = n_pipe - TP
        FN = n_ref - TP
        accuracy = TP / max(n_pipe + n_ref - TP, 1)
        precision = TP / max(n_pipe, 1)
        recall = TP / max(n_ref, 1)
        ref_is_good = bool(rk < len(ref['cluster_labels']) and ref['cluster_labels'][rk] == 1)
        matches.append({
            'pipe_id': pk,
            'ref_id': rk,
            'n_matched': TP,
            'n_pipe': n_pipe,
            'n_ref': n_ref,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'FP': int(FP),
            'FN': int(FN),
            'ref_is_good': ref_is_good,
        })
        total_TP += TP
        if ref_is_good:
            good_TP += TP

    matches.sort(key=lambda m: -m['accuracy'])
    accuracies = [m['accuracy'] for m in matches]
    good_accuracies = [m['accuracy'] for m in matches if m['ref_is_good']]

    summary = {
        'n_matched_pairs': len(matches),
        'n_pipe_clusters': len(pipe_ids),
        'n_ref_clusters': len(ref_ids),
        'n_ref_good': ref.get('n_good', 0),
        'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
        'median_accuracy': float(np.median(accuracies)) if accuracies else 0.0,
        'mean_accuracy_good': float(np.mean(good_accuracies)) if good_accuracies else 0.0,
        'n_good_matched': int(sum(1 for m in matches if m['ref_is_good'])),
        'n_high_accuracy': int(sum(1 for m in matches if m['accuracy'] > 0.8)),
        'total_TP': int(total_TP),
        'total_pipe_spikes': int(total_pipe_spikes_all),
        'total_ref_spikes': int(total_ref_spikes_all),
        'global_recall': float(total_TP / max(total_ref_spikes_all, 1)),
        'good_global_recall': float(good_TP / max(good_ref_total, 1)),
        'good_TP': int(good_TP),
        'good_ref_total': int(good_ref_total),
        'mean_precision': float(np.mean([m['precision'] for m in matches])) if matches else 0.0,
        'mean_recall': float(np.mean([m['recall'] for m in matches])) if matches else 0.0,
        'best_overlap': int(matches[0]['n_matched']) if matches else 0,
    }
    return matches, summary


def _print_summary(title, t_start, t_end, comp):
    s = comp['summary']
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Time window:      [{t_start:.2f}, {t_end:.2f}] s ({t_end - t_start:.1f} s)")
    print(f"  Pipeline/KS4:     {s['n_pipe_clusters']} clusters, {s['total_pipe_spikes']} spikes")
    print(f"  Reference:        {s['n_ref_clusters']} clusters ({s['n_ref_good']} good), {s['total_ref_spikes']} spikes")
    print(f"  Matched pairs:    {s['n_matched_pairs']}")
    print(f"  Global recall:    {s['global_recall']:.4f} ({s['total_TP']}/{s['total_ref_spikes']})")
    print(f"  Good-unit recall: {s['good_global_recall']:.4f} ({s['good_TP']}/{s['good_ref_total']})")
    print(f"  Mean accuracy:    {s['mean_accuracy']:.3f} (all), {s['mean_accuracy_good']:.3f} (good units)")
    print(f"  High accuracy:    {s['n_high_accuracy']} pairs > 0.8")
    if comp['matches']:
        print("\n  Top matched units:")
        for m in comp['matches'][:10]:
            good_str = ' [GOOD]' if m['ref_is_good'] else ''
            print(f"    pipe={m['pipe_id']:3d} <-> ref={m['ref_id']:3d}{good_str}: "
                  f"acc={m['accuracy']:.3f}, matched={m['n_matched']}/{m['n_ref']} "
                  f"(prec={m['precision']:.2f}, rec={m['recall']:.2f})")


def compare_pipeline_first_window(one, eid, probe, clust_results, channel_positions,
                                  raw_ind=None, window_s=120.0, output_dir=None):
    """Compare only the first `window_s` of streaming output against IBL reference."""
    sr = float(config.SAMPLE_RATE)
    max_samples = int(round(window_s * sr))
    mask = clust_results['times'] < max_samples
    sub = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(mask) else v)
           for k, v in clust_results.items()}
    sub['n_active'] = int(np.unique(sub['labels'][sub['labels'] >= 0]).size)

    from comparison import compare_with_reference
    comp = compare_with_reference(one, eid, probe, sub, channel_positions=channel_positions,
                                  raw_ind=raw_ind, verbose=False)
    t0 = config.CALIBRATION_DURATION_S
    _print_summary(f'PIPELINE vs IBL — first {window_s:.0f}s of streaming', t0, t0 + window_s, comp)
    if output_dir:
        save_comparison(comp, os.path.join(output_dir,
                                           f'compare_pipeline_first_{int(window_s)}s'))
    return comp


def compare_ks4_calibration_to_reference(one, eid, probe, cali_result, channel_positions,
                                         raw_ind=None, window_s=120.0, output_dir=None):
    """Compare official KS4 calibration output against IBL reference on [0, window_s]."""
    ref = load_reference(one, eid, probe, raw_ind=raw_ind)
    if ref is None:
        return None
    t_start, t_end = 0.0, float(window_s)
    ref_filtered = filter_reference(ref, t_start, t_end)

    # Use raw times here because reference spikes are in raw session time.
    ks_times_s = np.asarray(cali_result['all_times_raw'], dtype=np.float64) / float(config.SAMPLE_RATE)
    mask = (ks_times_s >= t_start) & (ks_times_s <= t_end)
    ks_times_s = ks_times_s[mask]
    ks_labels = np.asarray(cali_result['all_labels'], dtype=np.int32)[mask]
    ks_channels = np.asarray(cali_result['all_channels'], dtype=np.int32)[mask]

    matches, summary = _match_units_absolute(
        ks_times_s, ks_labels, ks_channels, ref_filtered,
        channel_positions=channel_positions,
        max_dt_ms=0.5,
        max_dist_um=100.0,
    )
    comp = {
        'reference': ref_filtered,
        'matches': matches,
        'summary': summary,
        'rate_comparison': None,
    }
    _print_summary(f'OFFICIAL KS4(calibration) vs IBL — first {window_s:.0f}s', t_start, t_end, comp)
    if output_dir:
        # save simple json and text summary
        txt_path = os.path.join(output_dir, f'compare_ks4_calibration_first_{int(window_s)}s.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'summary': summary, 'top_matches': matches[:50]}, indent=2))
        print(f'[SAVE] {txt_path}')
    return comp


def run_first120_checks(one, eid, probe, clust_results, cali_result,
                        channel_positions, raw_ind=None, output_dir=None,
                        window_s=120.0):
    pipe_comp = compare_pipeline_first_window(
        one, eid, probe, clust_results, channel_positions,
        raw_ind=raw_ind, window_s=window_s, output_dir=output_dir,
    )
    ks4_comp = None
    if cali_result is not None:
        ks4_comp = compare_ks4_calibration_to_reference(
            one, eid, probe, cali_result, channel_positions,
            raw_ind=raw_ind, window_s=window_s, output_dir=output_dir,
        )
    return pipe_comp, ks4_comp
