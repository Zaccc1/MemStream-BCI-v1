"""
Visualization Module
Each function produces one combined figure with multiple subplots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config


def plot_fir_response(coeffs, save_path='fir_response.png'):
    from preprocessing import fir_frequency_response
    freqs, mag_db = fir_frequency_response(coeffs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(freqs, mag_db, 'b-', lw=1)
    axes[0].axvline(config.FIR_HIGHPASS_FREQ, color='r', ls='--',
                    label=f'Cutoff={config.FIR_HIGHPASS_FREQ}Hz')
    axes[0].axhline(-3, color='gray', ls=':', alpha=0.5, label='-3 dB')
    axes[0].set(xlim=[0, 2000], ylim=[-80, 5],
                xlabel='Freq (Hz)', ylabel='Mag (dB)')
    axes[0].set_title(f'FIR HP — {len(coeffs)} taps, {config.FIR_WINDOW} '
                      f'(group delay={config.FIR_GROUP_DELAY} samples)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(coeffs, 'b-', lw=0.5)
    axes[1].axvline(config.FIR_GROUP_DELAY, color='r', ls='--', lw=0.8,
                    alpha=0.6, label=f'Group delay={config.FIR_GROUP_DELAY}')
    axes[1].set(xlabel='Tap', ylabel='Coeff', title='Impulse Response')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_whitening_matrices(whitening_ops, save_path='whitening_matrices.png',
                            n_show=4):
    n = min(n_show, len(whitening_ops))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = axes[:, None]

    for i in range(n):
        op = whitening_ops[i]
        W = op["W_full"]
        im = axes[0, i].imshow(W, cmap='RdBu_r', aspect='equal',
                                vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        axes[0, i].set_title(f'W — G{i}')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

        axes[1, i].semilogy(op["eigenvalues"][::-1], 'b.-', ms=3)
        axes[1, i].set(xlabel='Component', ylabel='Eigenvalue',
                       title=f'Eigenvals G{i}')
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Local ZCA Whitening Matrices'); plt.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_pipeline_stages(raw, car, filtered, whitened, channels=None,
                         save_path='pipeline_stages.png'):
    chs = (channels or [0, 50, 100, 200])[:4]
    n_samp = min(1500, raw.shape[0], filtered.shape[0], whitened.shape[0])
    t = np.arange(n_samp) / config.SAMPLE_RATE * 1000

    stages = [("Raw", raw), ("After CAR", car),
              ("After FIR HP", filtered), ("After Whitening", whitened)]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for row, (title, data) in enumerate(stages):
        ax = axes[row]
        for j, ch in enumerate(chs):
            if ch < data.shape[1]:
                std = max(np.std(data[:n_samp, ch]), 1e-6)
                ax.plot(t, data[:n_samp, ch] + j * std * 5, lw=0.4,
                        label=f'Ch{ch}')
        ax.set_ylabel('µV'); ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (ms)')
    plt.suptitle('Preprocessing Pipeline Stages'); plt.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_channel_groups(positions, groups, save_path='channel_groups.png'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(groups))))

    for i, g in enumerate(groups):
        c = colors[i % 20]
        pos_in = positions[g["input_channels"]]
        ax.scatter(pos_in[:, 0], pos_in[:, 1], c=[c], s=6, alpha=0.2)
        pos_out = positions[g["output_channels"]]
        ax.scatter(pos_out[:, 0], pos_out[:, 1], c=[c], s=18, marker='s',
                  edgecolors='k', linewidth=0.3)

    ax.set(xlabel='x (µm)', ylabel='y (µm)')
    ax.set_title(f'Groups ({config.WHITEN_MODE}, n={config.WHITEN_NRANGE}, '
                 f'ov={config.WHITEN_OVERLAP})')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


# ==============================================================
# Template Matching Visualizations
# ==============================================================

def plot_templates(wTEMP, save_path='templates.png', fs=None):
    sr = fs or config.SAMPLE_RATE
    n_temp, nt = wTEMP.shape
    t_ms = (np.arange(nt) - nt // 2) / sr * 1000

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_temp))

    for k in range(min(n_temp, 6)):
        ax = axes[k // 3, k % 3]
        ax.plot(t_ms, wTEMP[k], color=colors[k], lw=1.5)
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.axvline(0, color='gray', ls=':', lw=0.5)
        ax.set_title(f'Template {k}')
        ax.grid(True, alpha=0.2)

    for j in range(3):
        axes[1, j].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude (a.u.)')
    axes[1, 0].set_ylabel('Amplitude (a.u.)')

    plt.suptitle(f'Universal Templates ({n_temp} × {nt} samples)')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_detected_spikes(whitened_chunks, events, wTEMP,
                         save_path='detected_spikes.png', fs=None,
                         max_waveforms=100, n_example_channels=6):
    from template_matching import extract_spike_waveforms

    sr = fs or config.SAMPLE_RATE
    nt = wTEMP.shape[1]
    half = nt // 2
    t_ms = (np.arange(nt) - half) / sr * 1000
    n_temp = wTEMP.shape[0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, max(n_temp, 1)))

    waveforms, meta = extract_spike_waveforms(
        whitened_chunks, events, nt=nt, max_spikes=max_waveforms)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Individual waveforms
    ax = axes[0, 0]
    if len(waveforms) > 0:
        for i in range(len(waveforms)):
            tid = int(meta['templates'][i]) if len(meta['templates']) > i else 0
            ax.plot(t_ms, waveforms[i], color=colors[tid % len(colors)],
                    alpha=0.15, lw=0.5)
        for k in range(n_temp):
            mask = meta['templates'] == k
            if mask.any():
                ax.plot([], [], color=colors[k], lw=2,
                        label=f'T{k} ({mask.sum()})')
        ax.legend(fontsize=7, loc='upper right')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set(xlabel='Time (ms)', ylabel='Amplitude (σ)',
           title=f'Detected Spike Waveforms (n={len(waveforms)})')
    ax.grid(True, alpha=0.2)

    # (0,1) Mean waveform per template
    ax = axes[0, 1]
    if len(waveforms) > 0:
        for k in range(n_temp):
            mask = meta['templates'] == k
            if mask.sum() > 2:
                mean_wav = waveforms[mask].mean(axis=0)
                std_wav = waveforms[mask].std(axis=0)
                ax.fill_between(t_ms, mean_wav - std_wav, mean_wav + std_wav,
                                color=colors[k], alpha=0.15)
                ax.plot(t_ms, mean_wav, color=colors[k], lw=1.5,
                        label=f'Mean T{k} (n={mask.sum()})')
            if mask.any():
                scale = np.abs(meta['amplitudes'][mask]).mean()
                ax.plot(t_ms, wTEMP[k] * scale, color=colors[k], lw=1,
                        ls='--', alpha=0.6)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set(xlabel='Time (ms)', ylabel='Amplitude (σ)',
           title='Mean Waveform vs Template (dashed)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # (1,0) Amplitude histogram
    ax = axes[1, 0]
    if len(events['amplitudes']) > 0:
        amps = np.abs(events['amplitudes'])
        ax.hist(amps, bins=50, color='#3498db', alpha=0.7,
                edgecolor='white', lw=0.3)
        ax.axvline(config.MATCH_THRESHOLD, color='red', ls='--', lw=1.5,
                   label=f'Threshold={config.MATCH_THRESHOLD}σ')
        ax.set(xlabel='|Amplitude| (σ)', ylabel='Count',
               title=f'Amplitude Distribution (n={len(amps)})')
        ax.legend(); ax.grid(True, alpha=0.2)

    # (1,1) Raster plot
    ax = axes[1, 1]
    if len(events['times']) > 0:
        t_sec = events['times'] / sr
        ch = events['channels']
        tmpl = events['templates']
        for k in range(n_temp):
            mask = tmpl == k
            if mask.any():
                ax.scatter(t_sec[mask], ch[mask], s=1, c=[colors[k]],
                           alpha=0.5, label=f'T{k}')
        ax.set(xlabel='Time (s)', ylabel='Channel',
               title=f'Spike Raster ({len(events["times"])} events)')
        ax.legend(fontsize=6, markerscale=5, loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Template Matching — Detection Results')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_sa_detected_spikes(events, B, channel_positions, wTEMP,
                            save_path='sa_detected_spikes.png', fs=None):
    sr = fs or config.SAMPLE_RATE
    scale_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Raster colored by scale
    ax = axes[0, 0]
    if len(events['times']) > 0:
        t_sec = events['times'] / sr
        ch = events['channels']
        scales = events.get('scales',
                            np.zeros(len(events['times']), dtype=np.int32))
        n_scales = int(scales.max()) + 1 if len(scales) > 0 else 1
        for s in range(n_scales):
            mask = scales == s
            if mask.any():
                c = scale_colors[s % len(scale_colors)]
                ax.scatter(t_sec[mask], ch[mask], s=2, c=c,
                           alpha=0.6, label=f'Scale {s}')
        ax.set(xlabel='Time (s)', ylabel='Channel',
               title=f'SA Spike Raster ({len(events["times"])} events)')
        ax.legend(fontsize=7, markerscale=5, loc='upper right')
    else:
        ax.set_title('SA Spike Raster (no events)')
    ax.grid(True, alpha=0.2)

    # (0,1) Amplitude histogram
    ax = axes[0, 1]
    if len(events['amplitudes']) > 0:
        amps = events['amplitudes']
        ax.hist(amps, bins=50, color='#3498db', alpha=0.7,
                edgecolor='white', lw=0.3)
        ax.axvline(config.SA_FINAL_THRESHOLD, color='red', ls='--', lw=1.5,
                   label=f'Th={config.SA_FINAL_THRESHOLD}σ')
        ax.set(xlabel='Score (σ)', ylabel='Count',
               title=f'Amplitude Distribution (n={len(amps)})')
        ax.legend(); ax.grid(True, alpha=0.2)
    else:
        ax.set_title('Amplitude Distribution (no events)')

    # (1,0) Scale distribution
    ax = axes[1, 0]
    if len(events.get('scales', [])) > 0:
        scales = events['scales']
        n_scales = int(scales.max()) + 1
        counts = [np.sum(scales == s) for s in range(n_scales)]
        cols = [scale_colors[s % len(scale_colors)] for s in range(n_scales)]
        ax.bar(range(n_scales), counts, color=cols, edgecolor='white')
        ax.set_xticks(range(n_scales))
        sigma_labels = config.SA_SIGMA_UM[:n_scales]
        ax.set_xticklabels([f'σ={sig:.0f}µm' for sig in sigma_labels])
        ax.set(xlabel='Scale', ylabel='Count', title='Detections per Scale')
        for i, c in enumerate(counts):
            ax.text(i, c + max(counts) * 0.02, str(c), ha='center',
                    fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.set_title('Scale Distribution (no events)')

    # (1,1) Channel activity
    ax = axes[1, 1]
    if len(events['channels']) > 0:
        n_ch = channel_positions.shape[0]
        ch_counts = np.bincount(events['channels'], minlength=n_ch)
        active = ch_counts > 0
        ax.scatter(channel_positions[~active, 0],
                   channel_positions[~active, 1],
                   s=3, c='#cccccc', alpha=0.3, label='Silent')
        sc = ax.scatter(channel_positions[active, 0],
                        channel_positions[active, 1],
                        s=10, c=ch_counts[active], cmap='hot', alpha=0.8,
                        label='Active')
        plt.colorbar(sc, ax=ax, fraction=0.046, label='Spike count')
        ax.set(xlabel='x (µm)', ylabel='y (µm)',
               title=f'Channel Activity ({active.sum()}/{n_ch} active)')
    else:
        ax.set_title('Channel Activity (no events)')
    ax.grid(True, alpha=0.2)

    plt.suptitle('Spatial Aggregation — Detection Results')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_clustering_summary(results, channel_positions=None,
                            save_path='clustering_summary.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = results['labels']
    counts = results['counts']
    active = results['active_clusters']
    pca_var = results['pca_variance_ratio']

    # --- Panel 1: Spikes by cluster ---
    ax = axes[0, 0]
    if results['n_components'] >= 2:
        n_active = min(len(active), 20)
        top_clusters = active[np.argsort(-counts[active])][:n_active]
        cmap = plt.cm.get_cmap('tab20', n_active)
        for i, k in enumerate(top_clusters):
            mask = labels == k
            if mask.sum() > 0:
                ax.scatter(results['channels'][mask],
                          results['amplitudes'][mask],
                          s=3, alpha=0.3, color=cmap(i), label=f'C{k}')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Amplitude (σ)')
        ax.set_title(f'Spikes by cluster (top {n_active})')
        ax.legend(fontsize=6, ncol=2, loc='upper right', markerscale=3)
    else:
        ax.text(0.5, 0.5, 'Need ≥2 PCA components',
               ha='center', va='center', transform=ax.transAxes)

    # --- Panel 2: Cluster sizes ---
    ax = axes[0, 1]
    sorted_counts = np.sort(counts[active])[::-1]
    ax.bar(range(len(sorted_counts)), sorted_counts, color='steelblue',
           edgecolor='none')
    ax.set_xlabel('Cluster rank')
    ax.set_ylabel('Spike count')
    ax.set_title(f'Cluster sizes ({len(active)} active / '
                f'{results["n_clusters_total"]} total)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: PCA explained variance ---
    ax = axes[1, 0]
    if pca_var is not None and len(pca_var) > 0:
        cumvar = np.cumsum(pca_var) * 100
        ax.bar(range(1, len(pca_var) + 1), pca_var * 100, color='coral',
               edgecolor='none', alpha=0.7, label='Per component')
        ax.plot(range(1, len(pca_var) + 1), cumvar, 'k-o', ms=4,
               label='Cumulative')
        ax.set_xlabel('PCA component')
        ax.set_ylabel('Explained variance (%)')
        ax.set_title(f'PCA variance ({cumvar[-1]:.1f}% total)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.58, 'External wPCA basis in use',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, fontweight='bold')
        ax.text(0.5, 0.42, 'Explained variance is not available\n'
                'for imported template PCA bases.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
        ax.set_axis_off()

    # --- Panel 4: Spike depth vs time (corrected) ---
    ax = axes[1, 1]
    if channel_positions is not None:
        # Times are FIR-corrected sample indices; convert to seconds
        times_s = results['times'] / config.SAMPLE_RATE
        depths = channel_positions[results['channels'], 1]
        n_show = min(len(active), 15)
        top_k = active[np.argsort(-counts[active])][:n_show]
        cmap = plt.cm.get_cmap('tab20', n_show)
        for i, k in enumerate(top_k):
            mask = labels == k
            if mask.sum() > 0:
                ax.scatter(times_s[mask], depths[mask],
                          s=2, alpha=0.3, color=cmap(i))
        ax.set_xlabel('Time (s, from stream start)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Spike raster by cluster')
    else:
        ax.text(0.5, 0.5, 'No positions', ha='center', va='center',
               transform=ax.transAxes)

    plt.suptitle('Spike Clustering Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_merge_summary(results, save_path='merge_summary.png'):
    n_merges = results.get('n_merges', 0)
    pre_n = results.get('pre_merge_n_clusters', 0)
    post_n = results.get('n_active', 0)
    counts = results.get('counts', np.array([]))
    active = results.get('active_clusters', np.array([]))
    is_good = results.get('is_good', None)
    contam_rate = results.get('contam_rate', None)
    merge_map = results.get('merge_map', {})

    n_panels = 2
    if is_good is not None:
        n_panels += 1
    if len(merge_map) > 0:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    idx = 0

    # Panel 1: Before / After
    ax = axes[idx]; idx += 1
    bars = ax.bar(['Pre-merge', 'Post-merge'], [pre_n, post_n],
                  color=['#6baed6', '#2171b5'], edgecolor='white')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'Cluster Merging ({n_merges} merges)')
    for bar, val in zip(bars, [pre_n, post_n]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontweight='bold')

    # Panel 2: Size distribution
    ax = axes[idx]; idx += 1
    if len(active) > 0 and len(counts) > 0:
        ac = counts[active]
        ax.hist(ac, bins=min(30, max(5, len(ac) // 3)), color='#74c476',
                edgecolor='white', alpha=0.85)
        ax.axvline(np.median(ac), color='red', ls='--', lw=1.5,
                   label=f'median={int(np.median(ac))}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Spikes per Cluster'); ax.set_ylabel('Count')
    ax.set_title('Post-merge Size Distribution')

    # Panel 3: Quality
    if is_good is not None and contam_rate is not None:
        ax = axes[idx]; idx += 1
        valid = ~np.isnan(contam_rate)
        if valid.sum() > 0:
            colors_q = ['#2ca25f' if g else '#e34a33' for g in is_good[valid]]
            ax.bar(np.arange(valid.sum()), contam_rate[valid],
                   color=colors_q, edgecolor='white', width=1.0)
            ax.axhline(0.2, color='gray', ls='--', lw=1, alpha=0.7,
                       label='ACG threshold')
            ax.legend(fontsize=8)
        ax.set_xlabel('Cluster'); ax.set_ylabel('Q12 (contamination)')
        n_good = is_good.sum() if is_good is not None else 0
        ax.set_title(f'Unit Quality ({n_good} good / {post_n})')

    # Panel 4: Merge map
    if len(merge_map) > 0:
        ax = axes[idx]; idx += 1
        sources = list(merge_map.keys())
        targets = [merge_map[s] for s in sources]
        n_show = min(len(sources), 25)
        y_pos = np.arange(n_show)
        ax.barh(y_pos, [1] * n_show, color='#fc9272', edgecolor='white')
        labels_txt = [f'{sources[i]} → {targets[i]}' for i in range(n_show)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_txt, fontsize=7)
        ax.set_title(f'Merge Map (top {n_show})')
        ax.invert_yaxis(); ax.set_xticks([])

    plt.suptitle('Cluster Merging Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


# ==============================================================
# NEW: Diagnostic Visualizations
# ==============================================================

def plot_fn_analysis(fn_info, comp, clust_results, positions,
                     save_path='fn_analysis.png'):
    """
    Four-panel FN diagnostic figure:
      (0,0) FN locations on probe (colored by reason)
      (0,1) FN reason pie chart
      (1,0) FN rate per ref neuron (histogram)
      (1,1) FN breakdown by amplitude
    """
    ref_f = comp['ref_filtered']
    ref_channels = ref_f['spike_channels']
    ref_clusters = ref_f['spike_clusters']
    reason = fn_info['reason']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    reason_colors = {0: '#2ecc71', 1: '#e74c3c', 2: '#f39c12', 3: '#95a5a6'}
    reason_names = {0: 'Matched', 1: 'Ch displaced',
                    2: 'Time displaced', 3: 'Sub-threshold'}

    # (0,0) FN locations on probe
    ax = axes[0, 0]
    fn_mask = reason > 0
    if fn_mask.sum() > 0 and positions is not None:
        fn_ch = ref_channels[fn_mask].astype(int)
        fn_r = reason[fn_mask]
        for rval in [3, 1, 2]:  # plot sub-threshold first
            m = fn_r == rval
            if m.sum() > 0:
                valid_ch = fn_ch[m]
                valid_ch = valid_ch[valid_ch < len(positions)]
                if len(valid_ch) > 0:
                    ax.scatter(positions[valid_ch, 0], positions[valid_ch, 1],
                               s=4, alpha=0.3, c=reason_colors[rval],
                               label=f'{reason_names[rval]} ({m.sum()})')
    ax.set(xlabel='x (µm)', ylabel='y (µm)')
    ax.set_title('FN Spike Locations on Probe')
    ax.legend(fontsize=7, markerscale=3); ax.grid(True, alpha=0.2)

    # (0,1) Pie chart
    ax = axes[0, 1]
    fn_counts = fn_info['counts']
    labels_pie = []
    sizes_pie = []
    colors_pie = []
    for rval in [3, 1, 2]:
        name = reason_names[rval]
        n = fn_counts[fn_info['labels_map'][rval]]
        if n > 0:
            labels_pie.append(f'{name}\n({n})')
            sizes_pie.append(n)
            colors_pie.append(reason_colors[rval])
    if sizes_pie:
        ax.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
               autopct='%1.1f%%', startangle=90)
    ax.set_title('FN Root-Cause Breakdown')

    # (1,0) Per-neuron FN rate histogram
    ax = axes[1, 0]
    unique_ids = np.unique(ref_clusters)
    fn_rates = []
    for nid in unique_ids:
        m = ref_clusters == nid
        if m.sum() >= 5:
            fn_rates.append((reason[m] > 0).mean())
    if fn_rates:
        ax.hist(fn_rates, bins=20, color='#e74c3c', alpha=0.7,
                edgecolor='white')
        ax.axvline(np.median(fn_rates), color='k', ls='--',
                   label=f'median={np.median(fn_rates):.2f}')
        ax.legend(fontsize=8)
    ax.set(xlabel='FN Rate', ylabel='# Neurons',
           title='Per-neuron FN Rate')
    ax.grid(True, alpha=0.2)

    # (1,1) FN rate vs quality
    ax = axes[1, 1]
    ref_labels = ref_f['cluster_labels']
    quality_names = {1: 'good', 2: 'mua', 0: 'noise'}
    quality_fn = {}
    for lval, lname in quality_names.items():
        ids = [nid for nid in unique_ids
               if nid < len(ref_labels) and ref_labels[nid] == lval]
        if not ids:
            continue
        q_mask = np.isin(ref_clusters, ids)
        n_total = q_mask.sum()
        n_fn = (fn_mask & q_mask).sum()
        quality_fn[lname] = n_fn / n_total if n_total > 0 else 0

    if quality_fn:
        names = list(quality_fn.keys())
        vals = [quality_fn[n] for n in names]
        qcolors = {'good': '#2ecc71', 'mua': '#f39c12', 'noise': '#95a5a6'}
        ax.bar(names, vals, color=[qcolors.get(n, '#999') for n in names],
               edgecolor='white')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)
    ax.set(ylabel='FN Rate', title='FN Rate by Quality Label')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('False-Negative Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def plot_firing_rate_comparison(rate_comp, positions,
                                save_path='firing_rate_comparison.png'):
    """
    Compare per-channel firing rates between pipeline and reference.
    Two panels: scatter plot + probe heatmap of rate ratio.
    """
    pipe_rates = rate_comp['pipe_rates']
    ref_rates = rate_comp['ref_rates']
    n_ch = min(len(pipe_rates), len(ref_rates), len(positions))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: pipe vs ref
    ax = axes[0]
    valid = (pipe_rates[:n_ch] > 0) | (ref_rates[:n_ch] > 0)
    if valid.sum() > 0:
        ax.scatter(ref_rates[:n_ch][valid], pipe_rates[:n_ch][valid],
                   s=8, alpha=0.5, c='steelblue')
        mx = max(ref_rates[:n_ch][valid].max(), pipe_rates[:n_ch][valid].max())
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.5, label='y=x')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No non-zero firing rates',
                ha='center', va='center', transform=ax.transAxes)
    ax.set(xlabel='Reference rate (Hz)', ylabel='Pipeline rate (Hz)',
           title=f'Per-channel firing rates (r={rate_comp["correlation"]:.3f})')
    ax.grid(True, alpha=0.2)

    # Probe heatmap: log ratio
    ax = axes[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log2((pipe_rates[:n_ch] + 0.1) / (ref_rates[:n_ch] + 0.1))
    sc = ax.scatter(positions[:n_ch, 0], positions[:n_ch, 1],
                    c=ratio, cmap='RdBu_r', s=10, vmin=-3, vmax=3, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log₂(pipe/ref)')
    ax.set(xlabel='x (µm)', ylabel='y (µm)',
           title='Rate ratio on probe')
    ax.grid(True, alpha=0.2)

    plt.suptitle('Firing Rate Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close()
    print(f"[VIS] → {save_path}")


def compute_drift_metrics(results, channel_positions, window_s=30.0):
    """Estimate probe drift from spike depth trajectory."""
    times_s = results['times'].astype(np.float64) / config.SAMPLE_RATE
    channels = np.asarray(results['channels'], dtype=int)
    channels = np.clip(channels, 0, len(channel_positions) - 1)
    depths = channel_positions[channels, 1].astype(np.float64)

    if len(times_s) == 0:
        return {
            'total_drift_um': 0.0, 'max_drift_um': 0.0,
            'drift_rate_um_per_min': 0.0, 'n_drift_events': 0,
            'recording_duration_s': 0.0,
            'drift_trace_um': np.array([]),
            'drift_trace_time_s': np.array([]),
        }

    t_min, t_max = times_s.min(), times_s.max()
    duration = t_max - t_min
    edges = np.arange(t_min, t_max + window_s, window_s)
    n_bins = len(edges) - 1

    median_depth = np.full(n_bins, np.nan)
    bin_centers = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (times_s >= edges[i]) & (times_s < edges[i + 1])
        bin_centers[i] = (edges[i] + edges[i + 1]) / 2
        if mask.sum() > 10:
            median_depth[i] = np.median(depths[mask])

    valid = ~np.isnan(median_depth)
    if valid.sum() < 2:
        return {
            'total_drift_um': 0.0, 'max_drift_um': 0.0,
            'drift_rate_um_per_min': 0.0, 'n_drift_events': 0,
            'recording_duration_s': duration,
            'drift_trace_um': median_depth[valid],
            'drift_trace_time_s': bin_centers[valid],
        }

    trace = median_depth[valid]
    trace_t = bin_centers[valid]
    total_drift = float(trace.max() - trace.min())
    diffs = np.abs(np.diff(trace))
    max_drift = float(diffs.max()) if len(diffs) > 0 else 0.0
    drift_events = int((diffs > 10.0).sum())
    rate = total_drift / (duration / 60.0) if duration > 0 else 0.0

    return {
        'total_drift_um': round(total_drift, 2),
        'max_drift_um': round(max_drift, 2),
        'drift_rate_um_per_min': round(rate, 2),
        'n_drift_events': drift_events,
        'recording_duration_s': round(duration, 1),
        'drift_trace_um': trace,
        'drift_trace_time_s': trace_t,
    }


def plot_drift_map(results, channel_positions, drift_metrics=None,
                   save_path='drift_map.png', max_spikes=200_000):
    """Three-panel drift map: scatter, heatmap, depth histogram."""
    times_s = results['times'].astype(np.float64) / config.SAMPLE_RATE
    channels = np.asarray(results['channels'], dtype=int)
    channels = np.clip(channels, 0, len(channel_positions) - 1)
    depths = channel_positions[channels, 1]
    amps = results.get('amplitudes', np.ones_like(times_s))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 1.2, 0.8],
                          hspace=0.35, wspace=0.35)

    # Panel 1: Spike depth x time scatter (spans both rows, left)
    ax_scatter = fig.add_subplot(gs[:, 0])
    n = len(times_s)
    if n == 0:
        ax_scatter.text(0.5, 0.5, 'No spikes available for drift map',
                        ha='center', va='center', transform=ax_scatter.transAxes)
        ax_scatter.set_axis_off()
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_heat.text(0.5, 0.5, 'No spike density',
                     ha='center', va='center', transform=ax_heat.transAxes)
        ax_heat.set_axis_off()
        ax_trace = fig.add_subplot(gs[1, 1])
        ax_trace.text(0.5, 0.5, 'No drift metrics',
                      ha='center', va='center', transform=ax_trace.transAxes)
        ax_trace.set_axis_off()
        ax_hist = fig.add_subplot(gs[:, 2])
        ax_hist.text(0.5, 0.5, 'No depth distribution',
                     ha='center', va='center', transform=ax_hist.transAxes)
        ax_hist.set_axis_off()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIS] -> {save_path}")
        return

    if n > max_spikes:
        idx = np.random.default_rng(42).choice(n, max_spikes, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    sizes = np.clip(amps[idx] * 0.5, 0.5, 5.0)
    ax_scatter.scatter(times_s[idx], depths[idx], s=sizes,
                       c=depths[idx], cmap='viridis', alpha=0.15,
                       edgecolors='none', rasterized=True)

    if drift_metrics is not None:
        trace = drift_metrics.get('drift_trace_um', np.array([]))
        trace_t = drift_metrics.get('drift_trace_time_s', np.array([]))
        if len(trace) > 1:
            ax_scatter.plot(trace_t, trace, 'r-', lw=2, alpha=0.8,
                           label='Median depth')
            ax_scatter.legend(fontsize=8, loc='upper right')

    ax_scatter.set_xlabel('Time (s)')
    ax_scatter.set_ylabel('Depth (um)')
    title = 'Drift Map'
    if drift_metrics:
        title += (f"  |  total={drift_metrics['total_drift_um']:.1f} um"
                  f"  max_step={drift_metrics['max_drift_um']:.1f} um"
                  f"  rate={drift_metrics['drift_rate_um_per_min']:.1f} um/min")
    ax_scatter.set_title(title, fontsize=10)
    ax_scatter.grid(True, alpha=0.2)

    # Panel 2 top: Depth-time density heatmap
    ax_heat = fig.add_subplot(gs[0, 1])
    t_min, t_max = times_s.min(), times_s.max()
    d_min, d_max = depths.min(), depths.max()
    n_tbins = min(100, max(20, int((t_max - t_min) / 10)))
    n_dbins = min(100, max(20, int((d_max - d_min) / 20)))
    h, xe, ye = np.histogram2d(times_s, depths, bins=[n_tbins, n_dbins])
    ax_heat.imshow(h.T, origin='lower', aspect='auto',
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap='hot', interpolation='bilinear')
    ax_heat.set_xlabel('Time (s)')
    ax_heat.set_ylabel('Depth (um)')
    ax_heat.set_title('Spike density', fontsize=10)

    # Panel 2 bottom: Drift trace
    ax_trace = fig.add_subplot(gs[1, 1])
    if drift_metrics is not None:
        trace = drift_metrics.get('drift_trace_um', np.array([]))
        trace_t = drift_metrics.get('drift_trace_time_s', np.array([]))
        if len(trace) > 1:
            baseline = trace[0]
            ax_trace.plot(trace_t, trace - baseline, 'b-', lw=1.5)
            ax_trace.axhline(0, color='gray', ls='--', alpha=0.5)
            ax_trace.set_xlabel('Time (s)')
            ax_trace.set_ylabel('Drift from start (um)')
            ax_trace.set_title('Drift trajectory', fontsize=10)
            ax_trace.grid(True, alpha=0.3)
        else:
            ax_trace.text(0.5, 0.5, 'Insufficient data',
                         ha='center', va='center', transform=ax_trace.transAxes)
    else:
        ax_trace.text(0.5, 0.5, 'No drift metrics',
                     ha='center', va='center', transform=ax_trace.transAxes)

    # Panel 3: Depth distribution histogram (spans both rows, right)
    ax_hist = fig.add_subplot(gs[:, 2])
    ax_hist.hist(depths, bins=80, orientation='horizontal',
                 color='steelblue', edgecolor='none', alpha=0.7)
    ax_hist.set_xlabel('Spike count')
    ax_hist.set_ylabel('Depth (um)')
    ax_hist.set_title('Depth distribution', fontsize=10)
    ax_hist.grid(True, alpha=0.2)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] -> {save_path}")
