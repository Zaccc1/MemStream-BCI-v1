"""
Neural Decoding: Choice & Feedback
=====================================
Decode task variables from spike-sorted neural data.
Compare: our pipeline (all/filtered/per-region) vs IBL reference (KS4).

Usage:
    python decode_task.py [--results_dir results_ibl]
                          [--val_duration 120]
                          [--acc_threshold 0.8]

Requires: streaming results + IBL ONE API access
"""

import sys, os, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================
# IBL behavioral data loading
# ==============================================================

def load_trials(one, eid):
    """Load IBL trial data."""
    trials = one.load_object(eid, 'trials')
    print(f"  Trials loaded: {len(trials['choice'])} trials")

    # Key fields
    choice = trials['choice']           # -1=left, +1=right
    feedback = trials['feedbackType']   # +1=correct, -1=incorrect
    stim_on = trials['stimOn_times']    # stimulus onset (seconds)
    feedback_t = trials['feedback_times']
    first_movement = trials.get('firstMovement_times',
                                np.full(len(choice), np.nan))
    go_cue = trials.get('goCue_times', None)
    contrast_L = trials.get('contrastLeft', np.full(len(choice), np.nan))
    contrast_R = trials.get('contrastRight', np.full(len(choice), np.nan))

    # Filter valid trials
    valid = (np.isfinite(stim_on) & np.isfinite(feedback_t) &
             np.isin(choice, [-1, 1]) & np.isin(feedback, [-1, 1]))
    n_valid = valid.sum()
    print(f"  Valid trials: {n_valid}/{len(choice)}")
    print(f"  Choice: L={( choice[valid] == -1).sum()}, R={(choice[valid] == 1).sum()}")
    print(f"  Feedback: correct={(feedback[valid] == 1).sum()}, "
          f"incorrect={(feedback[valid] == -1).sum()}")
    print(f"  Time range: [{stim_on[valid].min():.1f}, {stim_on[valid].max():.1f}] s")

    return {
        'choice': choice, 'feedback': feedback,
        'stim_on': stim_on, 'feedback_times': feedback_t,
        'first_movement': first_movement,
        'go_cue': go_cue,
        'contrast_left': contrast_L, 'contrast_right': contrast_R,
        'valid': valid, 'n_trials': len(choice),
    }


def get_choice_event_times(trials, allow_stim_fallback=True):
    """Return the event times used for choice decoding.

    Preference order:
      1. first movement / wheel movement
      2. explicit choice times
      3. go cue
      4. stimulus onset (optional fallback)
    """
    candidates = [
        ('first movement', trials.get('first_movement')),
        ('choice event', trials.get('choice_times')),
        ('go cue', trials.get('go_cue')),
    ]
    if allow_stim_fallback:
        candidates.append(('stimulus onset',
                           trials.get('stim_on', trials.get('stim_on_times'))))

    for label, arr in candidates:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size > 0 and np.isfinite(arr).any():
            return arr, label
    return None, None


# ==============================================================
# Spike data loading (our pipeline + IBL reference)
# ==============================================================

def load_pipeline_spikes(results_dir):
    """Load our pipeline's spike sorting results."""
    for subdir in ['clustering', os.path.join('version_B', 'clustering')]:
        d = os.path.join(results_dir, subdir)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'spike_clusters.npy')):
            clust_dir = d
            break
    else:
        raise FileNotFoundError("No clustering results found")

    times_samples = np.load(os.path.join(clust_dir, 'spike_times.npy'))
    labels = np.load(os.path.join(clust_dir, 'spike_clusters.npy'))
    channels = np.load(os.path.join(clust_dir, 'spike_channels.npy'))

    is_good = None
    ig_path = os.path.join(clust_dir, 'cluster_is_good.npy')
    if os.path.exists(ig_path):
        is_good = np.load(ig_path)

    # Convert to seconds (absolute)
    times_s = times_samples.astype(np.float64) / config.SAMPLE_RATE
    times_s += config.CALIBRATION_DURATION_S

    print(f"  Pipeline spikes: {len(times_s)}, "
          f"clusters: {len(np.unique(labels[labels >= 0]))}")
    if is_good is not None:
        print(f"  ACG good: {is_good.sum()}")

    return {
        'times': times_s, 'labels': labels, 'channels': channels,
        'is_good': is_good,
    }


def load_reference_spikes(one, eid, probe, t_start, t_end):
    """Load IBL reference spike sorting (KS4) for the same time window."""
    from comparison import load_reference, filter_reference

    raw_ind = None
    try:
        from data_loader import load_channel_geometry
        _, raw_ind = load_channel_geometry(one, eid, probe)
    except:
        pass

    ref = load_reference(one, eid, probe, raw_ind=raw_ind)
    if ref is None:
        print("  WARNING: Reference spikes not available")
        return None

    ref_f = filter_reference(ref, t_start, t_end)

    # Only keep "good" units
    good_mask = ref['cluster_labels'] == 1
    good_ids = set(np.where(good_mask)[0])
    spike_mask = np.array([c in good_ids for c in ref_f['spike_clusters']])

    print(f"  Reference good spikes: {spike_mask.sum()}, "
          f"good clusters: {len(good_ids & set(ref_f['cluster_ids']))}")

    return {
        'times': ref_f['spike_times'][spike_mask],
        'labels': ref_f['spike_clusters'][spike_mask],
        'channels': ref_f['spike_channels'][spike_mask],
        'cluster_labels': ref['cluster_labels'],
        'good_ids': good_ids,
    }


def load_validation_accuracy(results_dir):
    """Load per-cluster accuracy from comparison results."""
    search_paths = [
        os.path.join(results_dir, 'comparison', 'unit_matches.npy'),
        os.path.join(results_dir, 'version_B', 'comparison', 'unit_matches.npy'),
    ]
    for path in search_paths:
        if os.path.exists(path):
            matches = np.load(path, allow_pickle=True)
            return {int(m['pipe_id']): float(m['accuracy']) for m in matches}
    return {}


# ==============================================================
# Feature extraction: trial-binned firing rates
# ==============================================================

def compute_firing_rates(spike_times, spike_labels, cluster_ids,
                         trial_times, window):
    """Compute per-cluster firing rate in a time window around each trial.

    Parameters
    ----------
    spike_times : (N,) float64 — absolute seconds
    spike_labels : (N,) int32 — cluster labels
    cluster_ids : list of int — which clusters to include
    trial_times : (n_trials,) float64 — event times in seconds
    window : (pre, post) float — seconds before/after event

    Returns
    -------
    rates : (n_trials, n_clusters) float32 — firing rate in Hz
    """
    pre, post = window
    duration = post - pre
    n_trials = len(trial_times)
    n_clusters = len(cluster_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    rates = np.zeros((n_trials, n_clusters), dtype=np.float32)

    for t_idx, t_event in enumerate(trial_times):
        t0 = t_event + pre
        t1 = t_event + post
        mask = (spike_times >= t0) & (spike_times < t1)
        trial_labels = spike_labels[mask]
        for lab in trial_labels:
            if lab in cid_to_idx:
                rates[t_idx, cid_to_idx[lab]] += 1

    rates /= max(duration, 1e-6)  # convert to Hz
    return rates


# ==============================================================
# Decoding
# ==============================================================

def _quantize_symmetric(x, n_bits):
    """Symmetric uniform quantization, matching memristor precision."""
    if n_bits >= 32:
        return x.copy()
    n_levels = 2 ** n_bits
    abs_max = np.abs(x).max()
    if abs_max < 1e-12:
        return np.zeros_like(x)
    scale = abs_max / (n_levels // 2 - 1)
    return (np.round(x / scale) * scale)


def decode_logistic(X, y, n_splits=None, n_shuffle=None, random_state=42,
                    quant_bits=None):
    """L1 logistic regression with cross-validation, shuffle null,
    and quantized-inference sweep.

    Full-precision training, then inference with quantized weights
    to simulate memristor crossbar deployment.

    Returns
    -------
    result : dict with real_acc, quant_accs, null distribution, etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    # Defaults from config
    if n_splits is None:
        n_splits = getattr(config, 'DECODE_N_SPLITS', 5)
    if n_shuffle is None:
        n_shuffle = getattr(config, 'DECODE_N_SHUFFLE', 200)
    if quant_bits is None:
        quant_bits = list(getattr(config, 'DECODE_QUANT_BITS', [2, 4, 6, 8]))

    # Remove NaN rows
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]

    if len(np.unique(y)) < 2 or len(y) < 20:
        empty = {'real_acc': 0.5, 'p_value': 1.0, 'n_trials': len(y),
                 'null_mean': 0.5, 'null_std': 0, 'n_features': X.shape[1],
                 'quant_accs': {b: 0.5 for b in quant_bits}}
        return empty

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    # === Full-precision decoding ===
    real_scores = []
    coefs = []
    quant_scores = {b: [] for b in quant_bits}

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(penalty='l1', solver='liblinear',
                                  C=1.0, max_iter=1000,
                                  random_state=random_state)
        clf.fit(X_train, y[train_idx])

        # Full precision inference
        y_pred = clf.predict(X_test)
        real_scores.append(balanced_accuracy_score(y[test_idx], y_pred))
        coefs.append(clf.coef_[0])

        # Quantized inference: quantize W, b, scaler params, then predict
        W = clf.coef_[0]         # (n_features,)
        b = clf.intercept_[0]    # scalar
        s_mean = scaler.mean_    # (n_features,)
        s_scale = scaler.scale_  # (n_features,)

        for n_bits in quant_bits:
            # Quantize all parameters
            W_q = _quantize_symmetric(W, n_bits)
            b_q = _quantize_symmetric(np.array([b]), n_bits)[0]
            s_mean_q = _quantize_symmetric(s_mean, n_bits)
            s_scale_q = s_scale.copy()  # keep scale full precision (division)

            # Quantized forward pass: z = X_scaled @ W_q + b_q
            X_test_raw = X[test_idx]
            X_scaled_q = (X_test_raw - s_mean_q) / np.maximum(s_scale_q, 1e-12)
            # Quantize the scaled input (memristor input precision)
            X_scaled_q = _quantize_symmetric(X_scaled_q, n_bits)
            # Matrix multiply (this is the crossbar operation)
            logits = X_scaled_q @ W_q + b_q
            y_pred_q = (logits >= 0).astype(int)
            quant_scores[n_bits].append(
                balanced_accuracy_score(y[test_idx], y_pred_q))

    real_acc = np.mean(real_scores)
    mean_coefs = np.mean(coefs, axis=0)
    n_nonzero = (np.abs(mean_coefs) > 1e-6).sum()
    quant_accs = {b: float(np.mean(s)) for b, s in quant_scores.items()}

    # === Shuffle null (full precision only) ===
    rng = np.random.default_rng(random_state)
    null_accs = []
    for _ in range(n_shuffle):
        y_shuf = rng.permutation(y)
        shuf_scores = []
        for train_idx, test_idx in cv.split(X, y_shuf):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(penalty='l1', solver='liblinear',
                                      C=1.0, max_iter=1000,
                                      random_state=random_state)
            clf.fit(X_train, y_shuf[train_idx])
            y_pred = clf.predict(X_test)
            shuf_scores.append(balanced_accuracy_score(y_shuf[test_idx], y_pred))
        null_accs.append(np.mean(shuf_scores))

    null_accs = np.array(null_accs)
    p_value = float((null_accs >= real_acc).sum() + 1) / (len(null_accs) + 1)

    return {
        'real_acc': float(real_acc),
        'quant_accs': quant_accs,
        'p_value': p_value,
        'null_mean': float(null_accs.mean()),
        'null_std': float(null_accs.std()),
        'n_trials': len(y),
        'n_features': X.shape[1],
        'n_nonzero': int(n_nonzero),
        'cv_scores': real_scores,
        'mean_coefs': mean_coefs,
    }


# ==============================================================
# Main
# ==============================================================

def run_decoding_suite(spike_data, cluster_ids, trials, label,
                       t_start_valid, results):
    """Run choice + feedback decoding for a set of clusters."""
    valid_base = trials['valid'].copy()
    valid_base &= trials['stim_on'] >= t_start_valid
    valid_base &= np.isfinite(trials['stim_on'])
    valid_base &= np.isfinite(trials['feedback_times'])

    choice_event_times, choice_event_label = get_choice_event_times(trials)
    choice_idx = np.array([], dtype=np.int64)
    if choice_event_times is not None:
        choice_valid = valid_base & np.isfinite(choice_event_times)
        choice_idx = np.where(choice_valid)[0]
    feedback_idx = np.where(valid_base)[0]

    if len(choice_idx) < 30 and len(feedback_idx) < 30:
        print(f"  [{label}] Only {len(choice_idx)} choice trials and "
              f"{len(feedback_idx)} feedback trials, skipping")
        return

    # Choice decoding: movement-aligned when possible
    choice_window = getattr(config, 'DECODE_CHOICE_WINDOW', (-0.1, 0.0))
    if len(choice_idx) >= 30 and choice_event_times is not None:
        choice = trials['choice'][choice_idx]
        choice_times = choice_event_times[choice_idx]
        print(f"\n  [{label}] Choice decoding ({len(cluster_ids)} clusters, "
              f"{len(choice_idx)} trials, align={choice_event_label}, "
              f"window={choice_window})...")
        X_choice = compute_firing_rates(
            spike_data['times'], spike_data['labels'], cluster_ids,
            choice_times, window=choice_window)

        # Remove constant columns
        col_std = X_choice.std(axis=0)
        active_cols = col_std > 1e-6
        if active_cols.sum() < 2:
            print(f"    Only {active_cols.sum()} active clusters, skipping")
        else:
            y_choice = (choice == 1).astype(int)
            res_choice = decode_logistic(X_choice[:, active_cols], y_choice)
            results[f'{label}_choice'] = res_choice
            sig = '***' if res_choice['p_value'] < 0.001 else \
                  '**' if res_choice['p_value'] < 0.01 else \
                  '*' if res_choice['p_value'] < 0.05 else 'n.s.'
            print(f"    Full prec: {res_choice['real_acc']:.3f} "
                  f"(null: {res_choice['null_mean']:.3f}±{res_choice['null_std']:.3f}) "
                  f"p={res_choice['p_value']:.4f} {sig}")
            print(f"    Features: {active_cols.sum()}, "
                  f"L1 nonzero: {res_choice['n_nonzero']}")
            qa = res_choice.get('quant_accs', {})
            if qa:
                q_str = '  '.join(f"{b}b={a:.3f}" for b, a in sorted(qa.items()))
                print(f"    Quantized: {q_str}")
    else:
        print(f"\n  [{label}] Choice decoding skipped "
              f"(align={choice_event_label or 'unavailable'}, "
              f"n_trials={len(choice_idx)})")

    # Feedback decoding: feedback-aligned
    fb_window = getattr(config, 'DECODE_FEEDBACK_WINDOW', (0.0, 0.2))
    if len(feedback_idx) >= 30:
        feedback = trials['feedback'][feedback_idx]
        feedback_t = trials['feedback_times'][feedback_idx]
        print(f"  [{label}] Feedback decoding "
              f"({len(feedback_idx)} trials, window={fb_window})...")
        X_feedback = compute_firing_rates(
            spike_data['times'], spike_data['labels'], cluster_ids,
            feedback_t, window=fb_window)

        col_std = X_feedback.std(axis=0)
        active_cols = col_std > 1e-6
        if active_cols.sum() < 2:
            print(f"    Only {active_cols.sum()} active clusters, skipping")
        else:
            y_feedback = (feedback == 1).astype(int)
            res_feedback = decode_logistic(X_feedback[:, active_cols], y_feedback)
            results[f'{label}_feedback'] = res_feedback
            sig = '***' if res_feedback['p_value'] < 0.001 else \
                  '**' if res_feedback['p_value'] < 0.01 else \
                  '*' if res_feedback['p_value'] < 0.05 else 'n.s.'
            print(f"    Full prec: {res_feedback['real_acc']:.3f} "
                  f"(null: {res_feedback['null_mean']:.3f}±{res_feedback['null_std']:.3f}) "
                  f"p={res_feedback['p_value']:.4f} {sig}")
            print(f"    Features: {active_cols.sum()}, "
                  f"L1 nonzero: {res_feedback['n_nonzero']}")
            qa = res_feedback.get('quant_accs', {})
            if qa:
                q_str = '  '.join(f"{b}b={a:.3f}" for b, a in sorted(qa.items()))
                print(f"    Quantized: {q_str}")
    else:
        print(f"  [{label}] Feedback decoding skipped "
              f"(n_trials={len(feedback_idx)})")


def main(results_dir):
    val_duration = getattr(config, 'DECODE_VAL_DURATION_S', 120.0)
    acc_threshold = getattr(config, 'DECODE_ACC_THRESHOLD', 0.8)
    quant_bits = list(getattr(config, 'DECODE_QUANT_BITS', [2, 4, 6, 8]))

    print(f"{'=' * 70}")
    print(f"  NEURAL DECODING: CHOICE & FEEDBACK")
    print(f"{'=' * 70}")
    print(f"  Validation duration: {val_duration}s")
    print(f"  Accuracy threshold: {acc_threshold}")
    print(f"  Quantization bits: {quant_bits}")
    t0 = time.time()

    # 1. Load behavioral data
    print(f"\n{'─' * 70}")
    print(f"  Loading behavioral data")
    print(f"{'─' * 70}")
    from data_loader import connect_one, load_channel_geometry, load_channel_regions
    one = connect_one()
    eid = config.IBL_SESSION_EID
    probe = config.IBL_PROBE_LABEL
    trials = load_trials(one, eid)
    positions, raw_ind = load_channel_geometry(one, eid, probe)
    channel_regions = load_channel_regions(one, eid, probe)

    # 2. Load pipeline spikes
    print(f"\n{'─' * 70}")
    print(f"  Loading spike data")
    print(f"{'─' * 70}")
    pipe_spikes = load_pipeline_spikes(results_dir)

    # Time boundaries
    cali_end = config.CALIBRATION_DURATION_S
    val_end = cali_end + val_duration
    stream_end = pipe_spikes['times'].max()

    # ========================================================
    # SECTION 1: SPIKE SORTING SUMMARY
    # ========================================================
    print(f"\n{'=' * 70}")
    print(f"  SPIKE SORTING SUMMARY")
    print(f"{'=' * 70}")

    labels = pipe_spikes['labels']
    is_good = pipe_spikes['is_good']
    all_active = sorted(set(labels[labels >= 0]))

    if is_good is not None:
        good_ids = [cid for cid in all_active if cid < len(is_good) and is_good[cid]]
    else:
        good_ids = all_active

    n_total_spikes = len(labels)
    n_assigned = (labels >= 0).sum()
    n_rejected = (labels == -1).sum()

    print(f"  Total spikes: {n_total_spikes}")
    print(f"  Assigned: {n_assigned} ({n_assigned / n_total_spikes * 100:.1f}%)")
    print(f"  Rejected: {n_rejected} ({n_rejected / n_total_spikes * 100:.1f}%)")
    print(f"  Active clusters: {len(all_active)}")
    print(f"  ACG good clusters: {len(good_ids)}")
    print(f"  Time range: {cali_end:.0f}-{stream_end:.0f}s "
          f"({stream_end - cali_end:.0f}s streaming)")
    print(f"  Calibration: 0-{cali_end:.0f}s")
    print(f"  Validation: {cali_end:.0f}-{val_end:.0f}s")
    print(f"  Test: {val_end:.0f}-{stream_end:.0f}s")

    # Per-region cluster counts
    region_clusters = {}
    if channel_regions is not None:
        from collections import Counter
        for cid in good_ids:
            cid_mask = labels == cid
            if cid_mask.sum() == 0:
                continue
            chs = pipe_spikes['channels'][cid_mask]
            peak_ch = Counter(chs).most_common(1)[0][0]
            if peak_ch < len(channel_regions):
                region = channel_regions[peak_ch]
                if region not in region_clusters:
                    region_clusters[region] = []
                region_clusters[region].append(cid)

        print(f"\n  Good clusters by brain region:")
        for region, cids in sorted(region_clusters.items(),
                                    key=lambda x: -len(x[1])):
            print(f"    {region:<15s}: {len(cids)} clusters")

    # ========================================================
    # SECTION 2: CLUSTER FILTERING (VALIDATION-BASED)
    # ========================================================
    print(f"\n{'=' * 70}")
    print(f"  CLUSTER FILTERING (validation acc >= {acc_threshold})")
    print(f"{'=' * 70}")

    val_acc = load_validation_accuracy(results_dir)
    if val_acc:
        # Show per-cluster validation accuracy for good clusters
        good_with_acc = [(cid, val_acc[cid]) for cid in good_ids if cid in val_acc]
        good_with_acc.sort(key=lambda x: -x[1])

        filtered_ids = [cid for cid, acc in good_with_acc if acc >= acc_threshold]

        va = [acc for _, acc in good_with_acc]
        print(f"  Good clusters with ref match: {len(good_with_acc)}")
        print(f"  Accuracy distribution:")
        for t in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            n = sum(1 for a in va if a >= t)
            marker = " ← threshold" if abs(t - acc_threshold) < 0.01 else ""
            print(f"    acc >= {t:.1f}: {n:>4d} clusters{marker}")

        print(f"\n  SELECTED for decoding: {len(filtered_ids)} clusters")
        print(f"  Rejected: {len(good_with_acc) - len(filtered_ids)} clusters")
        if filtered_ids:
            fa = [val_acc[cid] for cid in filtered_ids]
            print(f"  Selected accuracy: mean={np.mean(fa):.3f}, "
                  f"median={np.median(fa):.3f}, "
                  f"min={np.min(fa):.3f}")

        # Show selected clusters by region
        if channel_regions is not None:
            filt_regions = {}
            for cid in filtered_ids:
                for region, cids in region_clusters.items():
                    if cid in cids:
                        if region not in filt_regions:
                            filt_regions[region] = []
                        filt_regions[region].append(cid)
                        break
            print(f"\n  Selected clusters by brain region:")
            for region, cids in sorted(filt_regions.items(),
                                        key=lambda x: -len(x[1])):
                total_in_region = len(region_clusters.get(region, []))
                print(f"    {region:<15s}: {len(cids)}/{total_in_region} "
                      f"({len(cids) / max(total_in_region, 1) * 100:.0f}%)")

        # Top 10 clusters by validation accuracy
        print(f"\n  Top 10 selected clusters:")
        print(f"    {'Cluster':>8s} {'ValAcc':>8s} {'Region':>15s}")
        for cid, acc in good_with_acc[:10]:
            if acc < acc_threshold:
                break
            reg = '?'
            for region, cids in region_clusters.items():
                if cid in cids:
                    reg = region
                    break
            print(f"    {cid:>8d} {acc:>8.3f} {reg:>15s}")
    else:
        filtered_ids = good_ids
        print(f"  No validation accuracy available, using all {len(good_ids)} good clusters")

    # ========================================================
    # SECTION 3: DECODING
    # ========================================================

    # 4. Load reference spikes
    ref_spikes = load_reference_spikes(one, eid, probe, val_end, stream_end)

    # 5. Run decoding
    print(f"\n{'=' * 70}")
    print(f"  DECODING (test period: {val_end:.0f}-{stream_end:.0f}s)")
    print(f"{'=' * 70}")

    results = {}

    # a) Our pipeline — all good clusters
    run_decoding_suite(pipe_spikes, good_ids, trials, "pipe_good",
                       val_end, results)

    # b) Our pipeline — filtered clusters
    if len(filtered_ids) >= 5:
        run_decoding_suite(pipe_spikes, filtered_ids, trials, "pipe_filtered",
                           val_end, results)

    # c) IBL reference (KS4 good units)
    if ref_spikes is not None:
        ref_good_ids = sorted(ref_spikes['good_ids'] &
                              set(np.unique(ref_spikes['labels'])))
        if len(ref_good_ids) >= 5:
            run_decoding_suite(ref_spikes, ref_good_ids, trials, "ref_ks4",
                               val_end, results)

    # d) Per-region (top 2 regions by cluster count)
    top_regions = sorted(region_clusters.items(), key=lambda x: -len(x[1]))[:3]
    for region, cids in top_regions:
        if len(cids) >= 5:
            run_decoding_suite(pipe_spikes, cids, trials,
                               f"region_{region}", val_end, results)

    # 6. Summary table
    print(f"\n{'=' * 70}")
    print(f"  DECODING SUMMARY")
    print(f"{'=' * 70}")
    qb_headers = ''.join(f" {b:>4d}b" for b in quant_bits)
    print(f"  {'Condition':<25s} {'Task':>10s} {'Full':>7s}{qb_headers} "
          f"{'Null':>12s} {'p':>8s} {'Sig':>5s} {'N_tr':>6s} {'L1nz':>5s}")
    print(f"  {'─' * (85 + 6 * len(quant_bits))}")

    for key, res in sorted(results.items()):
        parts = key.rsplit('_', 1)
        cond = parts[0]
        task = parts[1] if len(parts) > 1 else key
        sig = '***' if res['p_value'] < 0.001 else \
              '**' if res['p_value'] < 0.01 else \
              '*' if res['p_value'] < 0.05 else 'n.s.'
        qa = res.get('quant_accs', {})
        qb_vals = ''.join(f" {qa.get(b, 0.5):>5.3f}" for b in quant_bits)
        print(f"  {cond:<25s} {task:>10s} {res['real_acc']:>7.3f}{qb_vals} "
              f"{res['null_mean']:>5.3f}±{res['null_std']:.3f} "
              f"{res['p_value']:>8.4f} {sig:>5s} "
              f"{res['n_trials']:>6d} {res['n_nonzero']:>5d}")

    # 7. Save results
    save_path = os.path.join(results_dir, 'decoding_results.npz')
    np.savez(save_path, **{k: v for k, v in results.items()})
    print(f"\n  Results saved: {save_path}")

    # 8. Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        tasks = ['choice', 'feedback']
        conditions = []
        for key in results:
            cond = key.rsplit('_', 1)[0]
            if cond not in conditions:
                conditions.append(cond)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = {'pipe_good': '#3266ad', 'pipe_filtered': '#1D9E75',
                  'ref_ks4': '#D85A30'}
        # Add region colors
        for i, (region, _) in enumerate(top_regions[:3]):
            colors[f'region_{region}'] = ['#7F77DD', '#D4537E', '#BA7517'][i]

        for t_idx, task in enumerate(tasks):
            ax = axes[t_idx]
            x_pos = np.arange(len(conditions))
            for i, cond in enumerate(conditions):
                key = f'{cond}_{task}'
                if key in results:
                    res = results[key]
                    color = colors.get(cond, '#888780')
                    bar = ax.bar(i, res['real_acc'], color=color, alpha=0.8,
                                width=0.7, label=cond)
                    # Null distribution as error region
                    ax.hlines(res['null_mean'], i - 0.35, i + 0.35,
                             color='gray', alpha=0.5, linewidth=1)
                    # Significance star
                    if res['p_value'] < 0.05:
                        ax.text(i, res['real_acc'] + 0.01,
                               '***' if res['p_value'] < 0.001 else
                               '**' if res['p_value'] < 0.01 else '*',
                               ha='center', fontsize=10)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([c.replace('_', '\n') for c in conditions],
                              fontsize=8, rotation=0)
            ax.set_ylabel('Balanced accuracy')
            ax.set_title(f'{task.capitalize()} decoding')
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
            ax.set_ylim(0.4, max(0.7, ax.get_ylim()[1] + 0.05))

        plt.suptitle('Neural Decoding: Pipeline vs Reference', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(results_dir, 'decoding_comparison.png')
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_ibl")
    args = parser.parse_args()
    main(args.results_dir)
